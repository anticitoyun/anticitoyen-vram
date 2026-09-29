"""Pièce claude (poste6, 29/09) — acvram-gemma4-12b-heretic-srci1q4km : claude sans réponse en 300 s, kimi rend un
texte qui enchaîne les tours. Cause : generation_config.json de la conversion GGUF = [1, 212] — 212 est « </s> »,
jeton ORDINAIRE (type 1) du vocabulaire Gemma 4 pris pour une fin sur son seul nom ; <turn|> (106, contrôle), la
vraie fin de tour, absent de la liste. Deux correctifs : la conversion ne retient que les jetons de contrôle et
connaît <turn|> ; au chargement, un marqueur spécial de tokenizer.json cité par le gabarit de chat rejoint les eos
(les dossiers déjà convertis, comme celui-ci, en profitent sans reconversion)."""
import json
import pathlib
import struct

import numpy as np
import pytest

from acvram.engine.config import _fins_de_tour_du_gabarit, load_model_spec
from acvram.quant.gguf import GGUFFile


def _dossier(tmp_path, *, gabarit="…<turn|>…", tokenizer=True, gen=(1, 212), jinja=False):
    d = tmp_path / "m"; d.mkdir(parents=True)
    json.dump({"architectures": ["Qwen2ForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
               "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 256,
               "eos_token_id": 1}, open(d / "config.json", "w"))
    json.dump({"eos_token_id": list(gen)}, open(d / "generation_config.json", "w"))
    if tokenizer:
        json.dump({"added_tokens": [{"id": 1, "content": "<eos>", "special": True},
                                    {"id": 106, "content": "<turn|>", "special": True},
                                    {"id": 212, "content": "</s>", "special": False},
                                    {"id": 7, "content": "<|im_end|>", "special": True}]},
                  open(d / "tokenizer.json", "w"))
    if jinja:
        (d / "chat_template.jinja").write_text(gabarit, encoding="utf-8")
        json.dump({}, open(d / "tokenizer_config.json", "w"))
    else:
        json.dump({"chat_template": gabarit}, open(d / "tokenizer_config.json", "w"))
    return d


def test_le_marqueur_special_cite_par_le_gabarit_rejoint_les_eos(tmp_path):
    d = _dossier(tmp_path)
    assert _fins_de_tour_du_gabarit(str(d)) == [106]            # <|im_end|> spécial mais absent du gabarit ; </s> non spécial
    assert load_model_spec(str(d)).eos_token_id == [1, 106, 212]


def test_gabarit_en_fichier_jinja(tmp_path):
    d = _dossier(tmp_path, gabarit="{{ '<|im_end|>' }}<turn|>", jinja=True)
    assert _fins_de_tour_du_gabarit(str(d)) == [7, 106]


def test_sans_tokenizer_ni_gabarit_rien_ne_change(tmp_path):
    assert _fins_de_tour_du_gabarit(str(_dossier(tmp_path, tokenizer=False))) == []
    assert _fins_de_tour_du_gabarit(str(_dossier(tmp_path / "b", gabarit="aucun marqueur"))) == []
    assert load_model_spec(str(_dossier(tmp_path / "c", gabarit="rien", gen=(1,)))).eos_token_id == [1]


# ---- conversion GGUF : fins de tour = jetons de contrôle seulement, <turn|> connu -------------------------------

def _gguf(path: pathlib.Path, kv: list) -> None:
    """GGUF v3 sans tenseur : scalaires u32/str et tableaux de chaînes ou d'i32 (ce que le lecteur des tests n'écrit pas)."""
    def s(b: bytes) -> bytes:
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for cle, v in kv:
        out += s(cle.encode())
        if isinstance(v, str):
            out += struct.pack("<I", 8) + s(v.encode())
        elif isinstance(v, int):
            out += struct.pack("<II", 4, v)
        elif v and isinstance(v[0], str):
            out += struct.pack("<IIQ", 9, 8, len(v)) + b"".join(s(x.encode()) for x in v)
        else:
            out += struct.pack("<IIQ", 9, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v)
    out += b"\0" * ((len(out) + 31) // 32 * 32 - len(out))
    path.write_bytes(out)


def _gen_apres_export(tmp_path, tokens, types):
    g = tmp_path / "g.gguf"
    kv = [("general.architecture", "qwen3"), ("qwen3.block_count", 1), ("qwen3.embedding_length", 16),
          ("qwen3.feed_forward_length", 32), ("qwen3.attention.head_count", 4), ("qwen3.attention.head_count_kv", 2),
          ("qwen3.context_length", 128), ("qwen3.vocab_size", len(tokens)), ("tokenizer.ggml.eos_token_id", 1),
          ("tokenizer.ggml.tokens", tokens)]
    if types is not None:
        kv.append(("tokenizer.ggml.token_type", types))
    _gguf(g, kv)
    out = tmp_path / "out"
    GGUFFile(str(g)).export_sidecars(str(out))
    return json.load(open(out / "generation_config.json"))["eos_token_id"]


def test_export_ne_prend_que_les_jetons_de_controle_et_connait_turn(tmp_path):
    tokens = ["<pad>", "<eos>", "<bos>", "</s>", "<turn|>", "<|im_end|>", "a"]
    assert _gen_apres_export(tmp_path, tokens, [3, 3, 3, 1, 3, 4, 1]) == [1, 4, 5]     # </s> ordinaire écarté


def test_export_sans_types_garde_le_nom_seul(tmp_path):
    tokens = ["<pad>", "<eos>", "<bos>", "</s>", "<turn|>"]
    assert _gen_apres_export(tmp_path, tokens, None) == [1, 3, 4]
