"""evp (poste1, 02/10) : gpt-oss répond en harmony — raisonnement, réponse et appels d'outils séparés par des
jetons SPÉCIAUX (<|channel|>, <|message|>, <|call|>…). Le décodage les retirait (skip_special_tokens=True) : le
client recevait « analysis…assistantfinal… », raisonnement compris, et aucun appel d'outil lisible. Correctif :
`Tokenizer.harmony` garde ces jetons au décodage, `FluxHarmony` rend le canal analysis en reasoning_content, le
canal final en content, et réécrit « to=functions.X » en ``<tool_call>`` que le chemin commun relit.
Le vrai gabarit levait aussi UndefinedError sur `strftime_now` (globale de transformers) → repli ChatML : invite
fausse pour gpt-oss, corrigée dans le même commit.
Cassures éprouvées : `decode` qui retire toujours les spéciaux → 1 rouge (test_tokeniseur_harmony_garde_les_marques) ;
canal analysis rendu en texte → 7 rouges ; sans `strftime_now` → 1 rouge (test_gabarit_gpt_oss_…)."""
import asyncio
import json
import pathlib
from types import SimpleNamespace

import pytest

from acvram.server.chat import (FiltreAppels, FluxHarmony, Tokenizer, blocs_anthropic, extraire_appels,
                                messages_pour_gabarit, normaliser_harmony)
from acvram.server.protocol import ChatMessage

GABARIT = (pathlib.Path(__file__).parent / "gabarits" / "gpt-oss-20b.jinja").read_text(encoding="utf-8")

REPONSE = ("<|channel|>analysis<|message|>L'utilisateur salue. Répondre brièvement.<|end|>"
           "<|start|>assistant<|channel|>final<|message|>Bonjour ! Comment puis-je aider ?<|return|>")
APPEL = ("<|channel|>analysis<|message|>Il faut la météo.<|end|>"
         "<|start|>assistant<|channel|>commentary to=functions.meteo <|constrain|>json<|message|>"
         '{"ville": "Paris", "jours": 2}<|call|>')


def test_normaliser_separe_raisonnement_et_reponse():
    raison, texte = normaliser_harmony(REPONSE)
    assert raison == "L'utilisateur salue. Répondre brièvement."
    assert texte == "Bonjour ! Comment puis-je aider ?"


def test_normaliser_appel_lu_par_le_chemin_commun():
    raison, texte = normaliser_harmony(APPEL)
    assert raison == "Il faut la météo."
    reste, appels = extraire_appels(texte)
    assert reste == "" and len(appels) == 1
    assert appels[0]["function"]["name"] == "meteo"
    assert json.loads(appels[0]["function"]["arguments"]) == {"ville": "Paris", "jours": 2}
    blocs, a_appel = blocs_anthropic(texte, True)
    assert a_appel and blocs[-1]["type"] == "tool_use" and blocs[-1]["input"] == {"ville": "Paris", "jours": 2}


def test_destinataire_dans_l_entete_de_start():
    """Forme alternative du format : « <|start|>assistant to=functions.X<|channel|>commentary json<|message|> »."""
    sortie = ("<|channel|>analysis<|message|>a<|end|><|start|>assistant to=functions.lire<|channel|>commentary json"
              '<|message|>{"chemin": "a.py"}<|call|>')
    _, appels = extraire_appels(normaliser_harmony(sortie)[1])
    assert [a["function"]["name"] for a in appels] == ["lire"]


@pytest.mark.parametrize("pas", [1, 2, 3, 5, 7, 13])
@pytest.mark.parametrize("sortie", [REPONSE, APPEL])
def test_flux_coupe_partout_egal_au_tout(sortie, pas):
    """Les marques arrivent coupées entre deux deltas : le flux doit rendre exactement le résultat d'un bloc."""
    f, raison, texte = FluxHarmony(), "", ""
    for i in range(0, len(sortie), pas):
        r, t = f.pousser(sortie[i:i + pas])
        raison, texte = raison + r, texte + t
        assert "<|" not in t and "<|" not in r, (i, t, r)
    r, t = f.finir()
    assert (raison + r, texte + t) == normaliser_harmony(sortie)


def test_reponse_finale_sort_au_fil_de_l_eau():
    """Pas de rétention de tout le texte jusqu'à la fin : le canal final sort dès qu'il est lu."""
    f = FluxHarmony()
    f.pousser("<|channel|>final<|message|>Bon")
    assert f.pousser("jour")[1] == "jour"


def test_appel_tronque_reste_du_texte():
    """max_tokens coupe l'appel : pas de <|call|>, JSON illisible → texte, jamais un appel inventé."""
    texte = normaliser_harmony('<|channel|>commentary to=functions.meteo <|constrain|>json<|message|>{"ville": "Pa')[1]
    assert extraire_appels(texte)[1] == [] and '{"ville": "Pa' in texte


def test_sortie_sans_harmony_rendue_telle_quelle():
    assert normaliser_harmony("juste du texte") == ("", "juste du texte")


def test_message_suivant_sans_end():
    """Un modèle qui enchaîne <|start|> sans <|end|> : l'analyse se ferme quand même."""
    assert normaliser_harmony("<|channel|>analysis<|message|>x<|start|>assistant<|channel|>final<|message|>y<|return|>") == ("x", "y")


def _backend(speciaux):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers
    vocab = {"[UNK]": 0, "bonjour": 1}
    tok = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(speciaux)
    return tok


def test_tokeniseur_harmony_garde_les_marques():
    b = _backend(["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>"])
    tok = Tokenizer(b, {}, None, "test")
    assert tok.harmony
    ids = [b.token_to_id("<|channel|>"), b.token_to_id("bonjour"), b.token_to_id("<|message|>")]
    assert tok.decode(ids).replace(" ", "") == "<|channel|>bonjour<|message|>"


def test_tokeniseur_ordinaire_retire_toujours_les_speciaux():
    b = _backend(["<|im_start|>", "<|im_end|>"])
    tok = Tokenizer(b, {}, None, "test")
    assert not tok.harmony
    assert tok.decode([b.token_to_id("bonjour"), b.token_to_id("<|im_end|>")]).strip() == "bonjour"
    assert not Tokenizer(None, {}, None, "test").harmony          # backend de test sans vocabulaire


def test_gabarit_gpt_oss_rend_outils_et_tour_d_appel():
    """Le vrai gabarit (tests/gabarits/, copié du dossier converti) : outils décrits, appel rejoué, résultat rendu,
    invite ouverte sur « <|start|>assistant » — d'où une sortie qui commence par <|channel|>."""
    tok = Tokenizer(None, {}, GABARIT, "test")
    outils = [{"type": "function", "function": {"name": "meteo", "description": "météo",
                                                 "parameters": {"type": "object",
                                                                "properties": {"ville": {"type": "string"}}}}}]
    tour = [ChatMessage(role="user", content="météo à Paris ?"),
            ChatMessage(role="assistant", content="", tool_calls=[{"id": "c1", "type": "function",
                        "function": {"name": "meteo", "arguments": '{"ville": "Paris"}'}}]),
            ChatMessage(role="tool", content="12 °C", tool_call_id="c1")]
    texte = tok.apply_chat_template(messages_pour_gabarit(tour), True, {"tools": outils})
    assert tok.gabarit_effectif == "jinja", texte[:300]
    assert "namespace functions" in texte and "type meteo" in texte
    assert "to=functions.meteo" in texte and "12 °C" in texte
    assert texte.endswith("<|start|>assistant")


# -- route /v1/chat/completions, en flux et hors flux ---------------------------------------------------------------
class _Service:
    def __init__(self, morceaux):
        self.morceaux = morceaux

    async def collect(self, request_id, q):
        for i, m in enumerate(self.morceaux):
            fin = i == len(self.morceaux) - 1
            yield SimpleNamespace(text_delta=m, completion_tokens=i + 1, finished=fin,
                                  finish_reason="stop" if fin else "")


def _flux(sortie, outils, harmony=True, pas=4):
    from acvram.server.app import _stream_chat
    morceaux = [sortie[i:i + pas] for i in range(0, len(sortie), pas)]

    async def lire():
        return [x async for x in _stream_chat(_Service(morceaux), "r", None, "m", 3, False, outils, harmony)]
    deltas, fin = [], None
    for ligne in asyncio.run(lire()):
        if not ligne.startswith("data: {"):
            continue
        ch = json.loads(ligne[6:])["choices"]
        if ch:
            deltas.append(ch[0]["delta"])
            fin = ch[0].get("finish_reason") or fin
    return deltas, fin


def test_flux_chat_raisonnement_a_part():
    deltas, fin = _flux(REPONSE, outils=False)
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "L'utilisateur salue. Répondre brièvement."
    assert "".join(d.get("content", "") for d in deltas) == "Bonjour ! Comment puis-je aider ?"
    assert fin == "stop"


def test_flux_chat_appel_d_outil():
    deltas, fin = _flux(APPEL, outils=True)
    appels = [a for d in deltas for a in d.get("tool_calls", [])]
    assert [a["function"]["name"] for a in appels] == ["meteo"] and fin == "tool_calls"
    assert "".join(d.get("content", "") for d in deltas) == ""


def test_flux_chat_sans_harmony_inchange():
    """Témoin : un modèle ordinaire garde son chemin, octet pour octet."""
    deltas, _ = _flux("<think>x</think>Bonjour", outils=False, harmony=False)
    assert "".join(d.get("content", "") for d in deltas) == "<think>x</think>Bonjour"
    assert not any("reasoning_content" in d for d in deltas)


def test_hors_flux_reasoning_content_et_appel():
    from acvram.server.app import _choix_final
    message, fin = _choix_final(REPONSE, "stop", False, harmony=True)
    assert message.content == "Bonjour ! Comment puis-je aider ?"
    assert message.reasoning_content == "L'utilisateur salue. Répondre brièvement." and fin == "stop"
    message, fin = _choix_final(APPEL, "stop", True, harmony=True)
    assert fin == "tool_calls" and message.tool_calls[0]["function"]["name"] == "meteo"
    message, _ = _choix_final("texte", "stop", False, harmony=False)
    assert message.reasoning_content is None and message.content == "texte"


def test_filtre_anthropic_apres_harmony():
    """/v1/messages en flux : FluxHarmony puis FiltreAppels — l'analyse ne fuit pas dans le texte."""
    fh, filtre, sortie = FluxHarmony(), FiltreAppels(), ""
    for i in range(0, len(APPEL), 3):
        sortie += filtre.pousser(fh.pousser(APPEL[i:i + 3])[1])
    sortie += filtre.pousser(fh.finir()[1])
    assert "météo" not in sortie and "analysis" not in sortie
    blocs, a_appel = blocs_anthropic(filtre.total, True)
    assert a_appel and blocs[-1]["name"] == "meteo"
