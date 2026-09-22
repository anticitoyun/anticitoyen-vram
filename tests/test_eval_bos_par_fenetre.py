"""verdict-ppl-31b-ab-v2-22-09 : `acvram eval` rendait sur Gemma 4 31B des PPL
25-45 × trop hautes, identiques avant et après le correctif d exil — aucune
fenêtre ne commençait par le BOS (le gabarit le pose, pas tokenizer.json).
Faux modèle à PPL CONNUE : il prédit la cible avec un logit A quand la
fenêtre commence par le BOS, 0 sinon → PPL = (V − 1 + e^A) / e^A avec BOS,
V sans ; `perplexity` doit reproduire ces deux chiffres selon que le
tokeniseur a un BOS ou non — et ne rien changer, au bit, pour un tokeniseur
sans BOS (Qwen)."""
import math
import types

import pytest
import torch

V, A, BOS = 50, 8.0, 7


class _Modele(torch.nn.Module):
    """h[i] = (cible i+1, fenêtre commence par BOS) ; logits : A sur la cible si BOS."""
    def forward(self, batch, return_hidden=False):
        t = batch.tokens
        avec = float(t[0].item() == BOS)
        cibles = torch.cat([t[1:], t[:1]]).to(torch.float32)
        return torch.stack([cibles, torch.full_like(cibles, avec)], dim=1)

    def _tete(self, h):
        logits = torch.zeros(h.shape[0], V)
        logits[torch.arange(h.shape[0]), h[:, 0].long()] = A * h[:, 1]
        return logits

    def _logits_finaux(self, x):
        return x

    nbytes = 0

    def nbytes_detail(self):
        return {}


class _Tok:
    def __init__(self, bos):
        self._bos = bos
        self.chunks = []

    def encode(self, text, add_special_tokens=False):
        return [(i * 7919 + 3) % (V - 10) + 10 for i in range(600)]      # jamais BOS, jamais < 10

    def bos_id(self):
        return self._bos


def _lancer(monkeypatch, bos, tmp_path):
    from acvram import evaluate
    from acvram.engine import loader
    from acvram.server import chat
    modele = _Modele()
    charge = types.SimpleNamespace(model=modele, spec=types.SimpleNamespace(total_params=1),
                                   manifest={"tensors": {}}, plan=None)
    monkeypatch.setattr(loader, "load_model", lambda *a, **k: charge)
    monkeypatch.setattr(chat, "load_tokenizer", lambda p: _Tok(bos))
    corpus = tmp_path / "c.txt"
    corpus.write_text("x")
    return evaluate.perplexity(str(tmp_path), str(corpus), window=64, stride=64, min_context=8, device="cpu")


def test_ppl_connue_avec_et_sans_bos(monkeypatch, tmp_path):
    ppl_bos = (V - 1 + math.exp(A)) / math.exp(A)
    r = _lancer(monkeypatch, BOS, tmp_path)
    assert r.bos == BOS and abs(r.perplexity - ppl_bos) < 1e-4, (r.perplexity, ppl_bos)
    r0 = _lancer(monkeypatch, None, tmp_path)
    assert r0.bos is None and abs(r0.perplexity - V) < 1e-3, r0.perplexity        # sans BOS : uniforme = V
    assert r0.windows == r.windows


def test_bos_id_depuis_le_gabarit_pas_le_post_traitement():
    """Gemma 4 : tokenizer.json ne pose pas le BOS, tokenizer_config le nomme →
    `bos_id` le trouve ; Qwen (bos_token None) → None."""
    from acvram.server.chat import Tokenizer
    backend = types.SimpleNamespace(token_to_id=lambda s: {"<bos>": 2}.get(s))
    assert Tokenizer(backend, {"bos_token": "<bos>"}, None, "x").bos_id() == 2
    assert Tokenizer(backend, {"bos_token": {"content": "<bos>"}}, None, "x").bos_id() == 2
    assert Tokenizer(backend, {"bos_token": None}, None, "x").bos_id() is None
    assert Tokenizer(backend, {}, None, "x").bos_id() is None
