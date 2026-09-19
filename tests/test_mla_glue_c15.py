"""C15 (chantier-c15-19-09) : la glue torch du décodage MLA à b=1 sous
ACVRAM_MLA_GLUE=1 rend la MÊME sortie au bit que le chemin inchangé, sur 32
pas greedy d'un jouet GLM (MLA bas rang + RoPE « norm » + MLP, 2 couches,
processeur, bf16), et lance moins d'opérations par pas.

À sec : le processeur suit les mêmes formulations torch que la carte hors
noyaux (RoPE, cat, normes en repli, résidu) — ce que la fusion touche. Ce
qu'il ne prouve PAS : l'égalité de ``rmsnorm_bf16`` avec résidu (le .cu) et
le niveau 2 (mla_prep_batch) — protocole carte dans la fiche.
"""
import copy
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from acvram.engine import mla as M_mla
from acvram.engine import model as M_model
from acvram.engine.layers import RMSNorm, RotaryEmbedding
from acvram.engine.mla import MLAttention
from acvram.engine.model import ACVRamModel, DecoderLayerGDN, MLP

torch.set_num_threads(min(8, torch.get_num_threads()))

H, NH, NOPE, ROPE, RANK, DV, QA, INTER, VOCAB = 64, 2, 16, 8, 32, 16, 24, 96, 50
L, BUCKET, PAS = 2, 64, 32
DT = torch.bfloat16


class _Tete:
    """lm_head minimal : ``qweight`` = tenseur (comme un poids non quantifié),
    logits fp32 — ce que ``ACVRamModel._tete`` attend d'une tête bf16."""

    def __init__(self, w):
        self.qweight = w

    def __call__(self, x):
        return F.linear(x, self.qweight.to(x.dtype))


def _lin(o, i):
    return nn.Linear(i, o, bias=False).to(DT)


def _jouet(seed: int) -> ACVRamModel:
    torch.manual_seed(seed)
    rope = RotaryEmbedding(ROPE, 4096, dtype=DT)
    couches = []
    for i in range(L):
        la = MLAttention(_lin(NH * (NOPE + ROPE), H), _lin(RANK + ROPE, H), _lin(H, NH * DV),
                         (torch.rand(RANK) + 0.5).to(DT),
                         (torch.randn(NH, RANK, NOPE) * 0.1).to(DT),
                         (torch.randn(NH, DV, RANK) * 0.1).to(DT),
                         NH, NOPE, ROPE, RANK, DV, eps=1e-5,
                         q_a_proj=_lin(QA, H), q_a_norm=(torch.rand(QA) + 0.5).to(DT),
                         q_b_proj=_lin(NH * (NOPE + ROPE), QA), rope=rope)
        mlp = MLP(_lin(INTER, H), _lin(INTER, H), _lin(H, INTER))
        c = DecoderLayerGDN(i, la, mlp, RMSNorm((torch.rand(H) + 0.5).to(DT), 1e-5),
                            RMSNorm((torch.rand(H) + 0.5).to(DT), 1e-5), torch.device("cpu"))
        c.statics.append(la.new_static(torch.device("cpu"), BUCKET + 16, DT))
        c.static_owners.append(0)
        c.static_bucket = BUCKET
        couches.append(c)
    spec = SimpleNamespace(vocab_size=VOCAB, logits_scaling=1.0, final_logit_softcapping=None)
    embed = (torch.randn(VOCAB, H) * 0.5).to(DT)
    tete = _Tete(torch.randn(VOCAB, H) * 0.2)
    return ACVRamModel(spec, embed, couches, RMSNorm((torch.rand(H) + 0.5).to(DT), 1e-5), tete, {}, DT)


_VUES = {"aten.view", "aten._unsafe_view", "aten.reshape", "aten.split", "aten.split_with_sizes",
         "aten.slice", "aten.select", "aten.unsqueeze", "aten.squeeze", "aten.t", "aten.transpose",
         "aten.permute", "aten.expand", "aten.detach", "aten.alias", "aten.narrow", "aten.as_strided",
         "aten.unbind", "aten.lift_fresh", "aten._local_scalar_dense", "aten.empty", "aten.empty_like",
         "aten.empty_strided", "aten.is_same_size"}


class _Compteur(TorchDispatchMode):
    """Opérations aten qui calculent ou copient (les vues exclues) : à sec,
    l'équivalent d'un lancement de noyau par opération."""

    def __init__(self):
        super().__init__()
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        nom = str(func).rsplit(".", 1)[0]
        res = func(*args, **(kwargs or {}))
        # `.to`/`.contiguous` qui rendent leur entrée ne lancent rien
        if nom not in _VUES and not (nom in ("aten.to", "aten.contiguous") and res is args[0]):
            self.ops[nom] += 1
        return res

    @property
    def total(self) -> int:
        return sum(self.ops.values())


def _decoder(modele: ACVRamModel, pas: int = PAS, compter_au: int = -1):
    """``pas`` jetons greedy depuis le jeton 1 ; rend (jetons, logits par
    pas, ops comptés au pas ``compter_au``)."""
    jeton = 1
    jetons, logits_tous, ops = [], [], None
    dev = torch.device("cpu")
    vide = torch.zeros(1, dtype=torch.long)
    with torch.inference_mode():
        for p in range(pas):
            x = modele.embed_tokens[jeton].view(1, H)
            if p == compter_au:
                with _Compteur() as c:
                    logits = modele.decode_fixed(x, vide, vide, vide, vide, BUCKET, 1)
                ops = c.ops
            else:
                logits = modele.decode_fixed(x, vide, vide, vide, vide, BUCKET, 1)
            logits_tous.append(logits.clone())
            jeton = int(logits.argmax(-1))
            jetons.append(jeton)
    return jetons, logits_tous, ops


def _bras(monkeypatch, glue: int, seed: int = 15):
    monkeypatch.setattr(M_mla, "_MLA_GLUE", glue)
    m = _jouet(seed)
    return m, _decoder(m, compter_au=PAS - 1)


def test_glue_1_bit_identique_sur_32_pas_et_lance_moins(monkeypatch):
    m0, (j0, l0, ops0) = _bras(monkeypatch, 0)
    m1, (j1, l1, ops1) = _bras(monkeypatch, 1)
    assert m1._res_differe() and not m0._res_differe(), "le résidu différé n'a pas basculé avec la glue"
    assert j0 == j1
    assert all(torch.equal(a, b) for a, b in zip(l0, l1)), "logits différents au bit"
    for c0, c1 in zip(m0.layers, m1.layers):
        assert torch.equal(c0.statics[0]["cache"], c1.statics[0]["cache"])
        assert torch.equal(c0.statics[0]["len"], c1.statics[0]["len"])
    n0, n1 = sum(ops0.values()), sum(ops1.values())
    # par couche, à sec : v_b fp32 (1), cat kvp (1), stack RoPE (1), index
    # sin (1) = 4. Les deux add_norm par couche (add + rmsnorm → un
    # lancement) ne se voient PAS à sec : le repli processeur fait l'addition
    # puis la norme, comme avant — c'est le .cu qui fusionne (carte, nsys).
    attendu = 4 * L
    print(f"\nops/pas glue=0 : {n0}, glue=1 : {n1}, écart {n0 - n1} (attendu {attendu})")
    assert n0 - n1 == attendu, (n0, n1, ops0 - ops1, ops1 - ops0)
    assert (ops0 - ops1) == Counter({"aten.to": L, "aten.cat": L, "aten.stack": L, "aten.index": L})


def test_temoins_qui_cassent(monkeypatch):
    """Un contrôle qui peut rendre faux : la même épreuve, avec une faute
    réintroduite dans chaque fusion, diverge du chemin inchangé."""
    _, (j0, l0, _) = _bras(monkeypatch, 0)
    # (a) demi-tables : cos et sin permutés
    orig = RotaryEmbedding.tables_demi

    def permute(self, positions, device, dtype, max_pos):
        cs = orig(self, positions, device, dtype, max_pos)
        return cs if cs is None else cs[:, [1, 0]]
    monkeypatch.setattr(RotaryEmbedding, "tables_demi", permute)
    _, (ja, la_, _) = _bras(monkeypatch, 1)
    monkeypatch.setattr(RotaryEmbedding, "tables_demi", orig)
    assert not all(torch.equal(a, b) for a, b in zip(l0, la_))
    # (b) résidu différé : le delta de la couche précédente compté deux fois
    orig_res = DecoderLayerGDN.decode_fixed_res

    def double(self, x, delta, *a, **k):
        return orig_res(self, x, None if delta is None else delta * 2, *a, **k)
    monkeypatch.setattr(DecoderLayerGDN, "decode_fixed_res", double)
    _, (jb, lb, _) = _bras(monkeypatch, 1)
    assert not all(torch.equal(a, b) for a, b in zip(l0, lb))


def test_glue_0_est_le_chemin_d_avant(monkeypatch):
    """Témoin : à glue 0, ni ``tables_demi`` ni ``decode_fixed_res`` ne sont
    appelées — le défaut reste le chemin inchangé."""
    appels = Counter()
    for cls, nom in ((RotaryEmbedding, "tables_demi"), (DecoderLayerGDN, "decode_fixed_res")):
        o = getattr(cls, nom)

        def wrap(self, *a, _o=o, _n=nom, **k):
            appels[_n] += 1
            return _o(self, *a, **k)
        monkeypatch.setattr(cls, nom, wrap)
    _bras(monkeypatch, 0)
    assert not appels
    _bras(monkeypatch, 1)
    assert appels["tables_demi"] == L * PAS and appels["decode_fixed_res"] == L * PAS
