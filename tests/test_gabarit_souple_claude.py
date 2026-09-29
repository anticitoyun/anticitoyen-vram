"""Pièce claude (poste6, 29/09) — llamacpp-mistral-small-4-119b : 500 à chaque requête de Claude Code,
« Only user, assistant and tool roles are supported, got system. » (gabarit Mistral Small 4, ligne 273). Capturé à sec
sur un faux serveur : Claude Code envoie son invite système en tête ET un message de rôle `system` APRÈS le premier
message utilisateur (rôles ['user', 'system']) ; llama.cpp les convertit en [system, user, system] et le gabarit
n'accepte le système qu'en position 0. Rejeu à sec (llama-server, Qwen3-0.6B, gabarit Mistral) : 500 identique ;
gabarit assoupli : 200. `parc/bin/gabarit-souple` (ex ~/.local/bin, règle Qwen3.x) gagne la règle Mistral et un
lecteur GGUF sans bibliothèque."""
import importlib.machinery
import importlib.util
import pathlib
import struct

import jinja2
import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "parc" / "bin" / "gabarit-souple"


@pytest.fixture(scope="module")
def gs():
    loader = importlib.machinery.SourceFileLoader("gabarit_souple", str(SCRIPT))   # fichier sans .py
    spec = importlib.util.spec_from_loader("gabarit_souple", loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# gabarit réduit au motif de Mistral Small 4 : système consommé en position 0, `else` qui refuse tout autre rôle
MISTRAL = (
    "{%- if messages[0]['role'] == 'system' %}{{- '[SYSTEM_PROMPT]' + messages[0]['content'] + '[/SYSTEM_PROMPT]' }}"
    "{%- set loop_messages = messages[1:] %}{%- else %}{%- set loop_messages = messages %}{%- endif %}"
    "{%- for message in loop_messages %}"
    "{%- if message['role'] == 'user' %}{{- '[INST]' + message['content'] + '[/INST]' }}"
    "{%- elif message['role'] == 'assistant' %}{{- message['content'] + '</s>' }}"
    "{%- elif message['role'] == 'tool' %}{{- '[TOOL_RESULTS]' + message['content'] + '[/TOOL_RESULTS]' }}"
    "{%- else %}\n"
    "        {{- raise_exception('Only user, assistant and tool roles are supported, got ' + message['role'] + '.') }}"
    "{%- endif %}{%- endfor %}"
)
QWEN = "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('System message must be at the beginning.') }}{% endif %}{% endfor %}"
MESSAGES = [{"role": "system", "content": "tête"}, {"role": "user", "content": "salut"},
            {"role": "system", "content": [{"type": "text", "text": "crochet"}]}]


def _rendre(texte: str) -> str:
    def raise_exception(msg):
        raise jinja2.TemplateError(msg)
    env = jinja2.Environment()
    env.globals["raise_exception"] = raise_exception
    return env.from_string(texte).render(messages=MESSAGES)


def test_le_gabarit_mistral_refuse_le_systeme_hors_position_0():
    with pytest.raises(jinja2.TemplateError, match="got system"):
        _rendre(MISTRAL)


def test_assoupli_mistral_rend_le_second_systeme_en_bloc(gs):
    s = gs.assouplir(MISTRAL)
    assert s is not None and gs.MISTRAL_ASSERT in s      # l'assertion reste pour les autres rôles
    out = _rendre(s)
    assert out.count("[SYSTEM_PROMPT]") == 2 and "[SYSTEM_PROMPT]crochet[/SYSTEM_PROMPT]" in out, out
    assert "[INST]salut[/INST]" in out


def test_assoupli_mistral_refuse_encore_un_role_inconnu(gs):
    s = gs.assouplir(MISTRAL)
    with pytest.raises(jinja2.TemplateError, match="got developer"):
        env = jinja2.Environment()
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
        env.from_string(s).render(messages=[{"role": "user", "content": "a"}, {"role": "developer", "content": "b"}])


def test_regle_qwen_inchangee(gs):
    s = gs.assouplir(QWEN)
    assert s is not None and gs.QWEN_ASSERT not in s and "<|im_start|>system" in s


def test_sans_assertion_connue_rien(gs):
    assert gs.assouplir("{% for m in messages %}{{ m.content }}{% endfor %}") is None


# ---- lecteur GGUF sans bibliothèque : le gabarit est lu APRÈS un tableau de chaînes et des scalaires -------------

def _gguf(path: pathlib.Path, kv: list) -> None:
    def s(b: bytes) -> bytes:
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for cle, t, v in kv:
        out += s(cle.encode()) + struct.pack("<I", t)
        if t == 8:
            out += s(v.encode())
        elif t == 4:
            out += struct.pack("<I", v)
        elif t == 9:                                   # tableau (type d'élément, n, éléments) : chaînes ou i32
            et, vals = v
            out += struct.pack("<IQ", et, len(vals))
            for x in vals:
                out += s(x.encode()) if et == 8 else struct.pack("<i", x)
    path.write_bytes(out)


def test_lecteur_sans_bibliotheque_trouve_le_gabarit_apres_les_tableaux(gs, tmp_path):
    g = tmp_path / "x.gguf"
    _gguf(g, [("general.architecture", 8, "llama"), ("llama.block_count", 4, 2),
              ("tokenizer.ggml.tokens", 9, (8, ["<s>", "</s>", "a"])), ("tokenizer.ggml.token_type", 9, (5, [3, 3, 1])),
              ("tokenizer.chat_template", 8, MISTRAL), ("general.name", 8, "x")])
    assert gs._gabarit_sans_bibliotheque(str(g)) == MISTRAL
    assert gs._gabarit_sans_bibliotheque(str(g)) is not None and gs.gabarit_du_gguf(str(g)) == MISTRAL


def test_lecteur_sans_gabarit_rend_none(gs, tmp_path):
    g = tmp_path / "y.gguf"
    _gguf(g, [("general.architecture", 8, "llama")])
    assert gs._gabarit_sans_bibliotheque(str(g)) is None


def test_cli_ecrit_gabarit_souple_a_cote_du_gguf(gs, tmp_path):
    g = tmp_path / "m.gguf"
    _gguf(g, [("tokenizer.chat_template", 8, MISTRAL)])
    assert gs.main(["gabarit-souple", str(g)]) == 0
    sortie = tmp_path / "gabarit-souple.jinja"
    assert sortie.is_file() and "[SYSTEM_PROMPT]" in sortie.read_text(encoding="utf-8")
    assert gs.main(["gabarit-souple", str(tmp_path)]) == 0     # dossier : réutilise le fichier écrit
