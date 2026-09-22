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

pytestmark = pytest.mark.sans_extension   # tests à sec (device cpu) : repli torch même carte visible (T4 20/09)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from acvram.engine import mla as M_mla
from acvram.engine import model as M_model
from acvram.engine.layers import RMSNorm, RotaryEmbedding
from acvram.engine.mla import MLAttention
from acvram.engine.model import ACVRamModel, DecoderLayerGDN, MLP, MoEBlock
from acvram.kernels import route_prep as M_rp

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
    l'équivalent d'un lancement de noyau par opération. ``pause`` : ce qui
    tient lieu de noyau (référence torch d'un GEMV groupé) n'est pas compté."""

    def __init__(self):
        super().__init__()
        self.ops = Counter()
        self.pause = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        nom = str(func).rsplit(".", 1)[0]
        res = func(*args, **(kwargs or {}))
        # `.to`/`.contiguous` qui rendent leur entrée ne lancent rien
        if (not self.pause and nom not in _VUES
                and not (nom in ("aten.to", "aten.contiguous") and res is args[0])):
            self.ops[nom] += 1
        return res

    @property
    def total(self) -> int:
        return sum(self.ops.values())


_ACTIF: list = []          # le compteur en cours, pour les crochets par couche


def _par_couche(monkeypatch, releve: dict):
    """Crochets sur ``decode_fixed`` / ``decode_fixed_res`` : les ops du
    compteur actif entre l'entrée et la sortie de chaque couche."""
    for nom in ("decode_fixed", "decode_fixed_res"):
        orig = getattr(DecoderLayerGDN, nom)

        def wrap(self, *a, _o=orig, **k):
            if not _ACTIF:
                return _o(self, *a, **k)
            avant = _ACTIF[-1].ops.copy()
            res = _o(self, *a, **k)
            releve[self.index] = _ACTIF[-1].ops - avant
            return res
        monkeypatch.setattr(DecoderLayerGDN, nom, wrap)


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
                c = _Compteur()
                _ACTIF.append(c)
                try:
                    with c:
                        logits = modele.decode_fixed(x, vide, vide, vide, vide, BUCKET, 1)
                finally:
                    _ACTIF.pop()
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


# ---------------------------------------------------------------------------
# Compte à sec par couche et par niveau (chantier-c15-19-09 § « Compte à sec
# par couche ») : ops aten hors vues au dernier pas, par couche MLA du jouet,
# et hors couches (plongement, norme finale, tête). Les comptes sont exacts :
# un changement de chemin qui ajoute ou retire une opération casse ici.
# ---------------------------------------------------------------------------
# ops par couche [couche 0, couche 1], hors couches (norme finale en repli
# torch = 6, tête, résidu différé absorbé hors couche à glue ≥ 1) et total,
# au dernier pas. À glue=1 la couche 1 garde un add de plus que la couche 0 :
# c'est la somme résiduelle différée de la couche 0 (add_norm à sec = add puis
# norme), et le dernier delta passe hors couche (+1) — le total tombe de 8.
COMPTE = {0: ([72, 72], 10, 154), 1: ([67, 68], 11, 146), 2: ([67, 68], 11, 146)}
RETIRE_0_VERS_1 = {0: Counter({"aten.to": 1, "aten.cat": 1, "aten.stack": 1, "aten.index": 1, "aten.add": 1}),
                   1: Counter({"aten.to": 1, "aten.cat": 1, "aten.stack": 1, "aten.index": 1})}


def _niveau(monkeypatch, glue: int, releve: dict):
    releve.clear()
    m, (j, l, ops) = _bras(monkeypatch, glue)
    return m, j, l, ops, {i: c.copy() for i, c in releve.items()}


def test_compte_a_sec_par_couche_par_niveau(monkeypatch):
    releve = {}
    _par_couche(monkeypatch, releve)               # un seul crochet, relevé vidé entre niveaux
    res = {g: _niveau(monkeypatch, g, releve) for g in (0, 1, 2)}
    lignes = []
    for g, (m, j, l, ops, rel) in res.items():
        assert sorted(rel) == list(range(L))
        par_couche = [sum(rel[i].values()) for i in range(L)]
        hors = sum(ops.values()) - sum(par_couche)
        lignes.append(f"glue={g} : couches {par_couche}, hors couches {hors}, total {sum(ops.values())} ; "
                      f"couche 0 : {dict(sorted(rel[0].items()))}")
        assert (par_couche, hors, sum(ops.values())) == COMPTE[g], (g, par_couche, hors, rel)
    print("\n" + "\n".join(lignes))
    # les ops retirées entre niveaux, nommées, par couche
    for i in range(L):
        r0, r1, r2 = (res[g][4][i] for g in (0, 1, 2))
        assert (r0 - r1) == RETIRE_0_VERS_1[i] and not (r1 - r0), (i, r0 - r1, r1 - r0)
        assert r1 == r2, (i, r1 - r2, r2 - r1)
    # à sec le niveau 2 retombe sur le niveau 1 (mla_prep_batch est un noyau
    # CUDA ; model.py `_la_decode` le refuse sans extension) : mêmes ops, mêmes
    # bits. La tolérance du niveau 2 (q_abs ≤ 1 ulp bf16, PPL ± 0,001) ne se
    # juge que sur la carte — chaîne scratchpad/c15-carte-19-09.
    j0, j1, j2 = (res[g][1] for g in (0, 1, 2))
    assert j0 == j1 == j2
    l0, l1, l2 = (res[g][2] for g in (0, 1, 2))
    assert all(torch.equal(a, b) for a, b in zip(l0, l1))
    assert all(torch.equal(a, b) for a, b in zip(l1, l2))
    for c0, c1, c2 in zip(res[0][0].layers, res[1][0].layers, res[2][0].layers):
        assert torch.equal(c0.statics[0]["cache"], c1.statics[0]["cache"])
        assert torch.equal(c1.statics[0]["cache"], c2.statics[0]["cache"])


# ---------------------------------------------------------------------------
# Glue MoE (model.py `_forward_grouped`, § Reste de la fiche) : tok int64
# servi d'avance, eid converti une fois, x[tok] rassemblé une fois, tok_g =
# seq. Le GEMV groupé est CUDA seul : ici une référence torch (non comptée)
# tient sa place, les gathers, conversions et divisions AWQ sont ceux du
# code servi — c'est eux que le niveau 1 touche, et eux seuls.
# ---------------------------------------------------------------------------
E, KM, IM, TOPK = 5, 32, 48, 2


def _moe_jouet(distinct: bool, seed: int = 7) -> MoEBlock:
    torch.manual_seed(seed)
    bloc = MoEBlock(None, [MLP(_lin(IM, KM), _lin(IM, KM), _lin(KM, IM))], TOPK)
    bloc._stacks = {"gate_proj": ("ref", torch.randn(E, IM, KM) * 0.1),
                    "up_proj": ("ref", torch.randn(E, IM, KM) * 0.1),
                    "down_proj": ("ref", torch.randn(E, KM, IM) * 0.1)}
    tg = (torch.rand(E, KM) + 0.5).to(DT)
    tu = (torch.rand(E, KM) + 0.5).to(DT) if distinct else tg
    bloc._stacks_awq = {"gate_proj": tg, "up_proj": tu, "up_distinct": distinct,
                        "down_proj": (torch.rand(E, IM) + 0.5).to(DT), "hadamard": {}}
    bloc._stack_state = "oui"
    return bloc


def _grouped_ref(self, x32, pile, eid, tok, tri=None):
    c = _ACTIF[-1] if _ACTIF else None
    if c is not None:
        c.pause += 1
    try:
        return torch.einsum("gk,gmk->gm", x32[tok.long()], pile[1][eid.long()])
    finally:
        if c is not None:
            c.pause -= 1


def _moe_pas(monkeypatch, glue: int, distinct: bool, servi: bool, t: int = 1):
    """``t`` jetons par ``_forward_grouped`` ; ``servi`` = eid/tok déjà réservés
    (la forme route_prep / route_fusee, celle de la carte à b=1)."""
    monkeypatch.setattr(M_mla, "_MLA_GLUE", glue)
    monkeypatch.setattr(MoEBlock, "_grouped", _grouped_ref)
    bloc = _moe_jouet(distinct)
    torch.manual_seed(3)
    x = (torch.randn(t, KM) * 0.7).to(DT)
    topw = torch.rand(t, TOPK)
    topi = torch.tensor([[4, 1], [0, 2], [3, 3]][:t], dtype=torch.int32)
    eid = topi.reshape(-1).to(torch.int32) if servi else None
    if servi:                                     # tables réservées une fois (premier pas), hors compte
        M_rp.index_jetons(t, TOPK, x.device)
        M_rp.index_jetons_long(t, TOPK, x.device)
    c = _Compteur()
    _ACTIF.append(c)
    try:
        with torch.inference_mode(), c:
            y = bloc._forward_grouped(x, topw, topi, eid=eid)
    finally:
        _ACTIF.pop()
    return y, c.ops


# ce que le niveau 1 retire par couche, nommé : (distinct, servi) → ops
RETIRE_MOE = {
    (False, True): Counter({"aten.to": 2, "aten.arange": 1}),                  # Marlin : tok, eid ×1, tok_g
    (True, True): Counter({"aten.to": 4, "aten.index": 1, "aten.arange": 1}),  # distinct : tok ×2, eid ×2, x[tok], tok_g
    (False, False): Counter({"aten.to": 1}),                                   # eid non servi : eid ×1 seulement
    (True, False): Counter({"aten.to": 3, "aten.index": 1}),
}


@pytest.mark.parametrize("distinct", [False, True])
@pytest.mark.parametrize("servi", [True, False])
def test_moe_glue_tok_eid_int64_au_bit(monkeypatch, distinct, servi):
    y0, o0 = _moe_pas(monkeypatch, 0, distinct, servi)
    y1, o1 = _moe_pas(monkeypatch, 1, distinct, servi)
    assert y0.dtype == DT and torch.equal(y0, y1), "sortie MoE différente au bit"
    print(f"\nMoE distinct={distinct} servi={servi} : ops glue=0 {sum(o0.values())}, glue=1 {sum(o1.values())}, "
          f"retiré {dict(o0 - o1)}")
    assert (o0 - o1) == RETIRE_MOE[(distinct, servi)] and not (o1 - o0), (o0 - o1, o1 - o0)


def test_moe_glue_temoin_qui_casse(monkeypatch):
    """La même épreuve avec `tok` int64 servi FAUX (décalé d'un jeton) diverge
    dès deux jetons : le contrôle lit bien la table servie d'avance."""
    y_ref, _ = _moe_pas(monkeypatch, 0, True, True, t=2)
    y_vrai, _ = _moe_pas(monkeypatch, 1, True, True, t=2)
    orig = M_rp.index_jetons_long
    monkeypatch.setattr(M_rp, "index_jetons_long", lambda t, k, d: (orig(t, k, d) + 1) % t)
    y_faux, _ = _moe_pas(monkeypatch, 1, True, True, t=2)
    assert torch.equal(y_vrai, y_ref) and not torch.equal(y_faux, y_ref)


def test_le_defaut_est_le_niveau_2():
    """0.6.32 : niveau 2 tenu sur carte (M3 2a-bis 4/4, Manon 9ee0c00a) → défaut ; la table de régime,
    le module et la ligne lisent le même défaut ; 1 = témoin nommé."""
    from pathlib import Path
    from acvram import regime
    v = {x.env: x for x in regime.VARIABLES}["ACVRAM_MLA_GLUE"]
    assert v.defaut == "2" and v.torch == "0"
    src = (Path(__file__).resolve().parents[1] / "acvram" / "engine" / "mla.py").read_text()
    assert 'os.environ.get("ACVRAM_MLA_GLUE", "2")' in src
    assert M_mla.regime_glue_texte() == {2: "mla_glue=2", 1: "mla_glue=1(temoin)", 0: "mla_glue=0"}[M_mla._MLA_GLUE]
