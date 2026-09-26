"""Pièce 209 (a) : une pile d'experts NVFP4 qu'un facteur Marlin commun écraserait (sous-normales e4m3 à côté de 448 —
Qwen3-Coder-30B couches 0, 1, 2, 4, 157) est préparée avec un facteur PAR LIGNE D'EXPERT et une échelle globale par
(expert, colonne) [E, N] : le dépaquetage rend EXACTEMENT `dequantize_nvfp4` ; sans le facteur par ligne (témoin
`par_ligne=False`) il ne le rend pas (bras cassant). Une pile sans écrasement garde AU BIT la préparation d'avant.
À sec (repack et dépaquetage torch) ; les piles RÉELLES du Coder si l'alias est présent."""
import os

import pytest
import torch

E4 = torch.float8_e4m3fn
ALIAS = os.environ.get("ACVRAM_ALIAS_CODER", "/mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c")


def _pile_synthetique(E, N, K, graine, minuscules=()):
    """Experts randn ; pour e dans `minuscules`, la moitié des lignes × 1e-5 (échelles sous-normales e4m3 > 0)."""
    from acvram.quant.nvfp4 import quantize_nvfp4
    g = torch.Generator().manual_seed(graine)
    ts = []
    for e in range(E):
        w = torch.randn(N, K, generator=g)
        if e in minuscules:
            w[N // 2:] *= 1e-5                                   # rapport 10⁵ ≈ 2^16,6 : sous-normales e4m3 > 0 (montage de la 157)
        ts.append(quantize_nvfp4(w.to(torch.bfloat16)))
    qw = torch.stack([t.qweight for t in ts]); bs = torch.stack([t.block_scale for t in ts])
    gs = torch.stack([t.global_scale.reshape(()) for t in ts]).float()
    return ts, qw, bs, gs


def _depaquete(MP, qw, bs, gs, K, N, par_ligne):
    w, s_, g = MP.preparer_pile(qw, bs, gs, repack=MP.repack_torch, par_ligne=par_ligne)
    return w, s_, g, MP.depaqueter_marlin(w, s_, g, K, N, noyau="torch")


def test_pile_sans_ecrasement_au_bit_d_avant():
    from acvram.kernels import marlin_port as MP
    ts, qw, bs, gs = _pile_synthetique(4, 128, 256, 209)
    assert MP.echelles_ecrasees(bs) == 0
    a = _depaquete(MP, qw, bs, gs, 256, 128, True)
    b = _depaquete(MP, qw, bs, gs, 256, 128, False)
    assert a[2].shape == (4,) and all(torch.equal(x, y) for x, y in zip(a, b))       # w, s, g, dépaquetage : identiques


def test_pile_a_sous_normales_forcees_exacte_et_cassante():
    from acvram.kernels import marlin_port as MP
    from acvram.quant.nvfp4 import dequantize_nvfp4
    E, N, K = 4, 128, 256
    ts, qw, bs, gs = _pile_synthetique(E, N, K, 2090, minuscules=(1, 3))
    assert MP.echelles_ecrasees(bs) > 0, "montage : le facteur commun doit écraser"
    ref = torch.stack([dequantize_nvfp4(t, torch.bfloat16) for t in ts])
    w, s_, g, W = _depaquete(MP, qw, bs, gs, K, N, True)
    assert g.shape == (E, N)
    assert torch.equal(W, ref), f"{int((W != ref).sum())} valeurs fausses avec le facteur par ligne"
    _, _, g0, W0 = _depaquete(MP, qw, bs, gs, K, N, False)                  # témoin : l'ancien facteur commun
    assert g0.shape == (E,) and not torch.equal(W0, ref), "le témoin sans facteur par ligne devrait être faux"
    assert int((W0 != ref).sum()) >= 16 * MP.echelles_ecrasees(bs) // 2      # au moins la moitié des blocs écrasés changent


@pytest.mark.skipif(not os.path.isdir(ALIAS), reason="alias Coder absent")
@pytest.mark.parametrize("couche,mat", [(0, "gate_proj"), (0, "up_proj"), (1, "gate_proj"), (1, "up_proj"),
                                        (2, "gate_proj"), (2, "up_proj"), (4, "gate_proj"), (4, "up_proj")])
def test_piles_reelles_du_coder_au_bit(couche, mat):
    """Les 8 piles réelles refusées par la 157 : experts touchés (+ 2 témoins non touchés), dépaquetage au bit de la référence."""
    import json
    from safetensors import safe_open
    from acvram.kernels import marlin_port as MP
    from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4
    m = json.load(open(os.path.join(ALIAS, "acvram_manifest.json")))
    wm = m["weight_map"]
    lecteurs = {}

    def lire(nom):
        f = wm[nom]
        if f not in lecteurs:
            lecteurs[f] = safe_open(os.path.join(ALIAS, f), "pt", device="cpu")
        return lecteurs[f].get_tensor(nom)
    base = f"model.layers.{couche}.mlp.experts.{{}}.{mat}.weight"
    E = 128
    bs_tous = torch.stack([lire(base.format(e) + ".block_scale").view(E4) for e in range(E)])
    assert MP.echelles_ecrasees(bs_tous) > 0, "cette pile n'écrase plus : le test ne juge rien"
    touches = [e for e in range(E) if MP.echelles_ecrasees(bs_tous[e:e + 1]) > 0]
    temoins = [e for e in range(E) if e not in touches][:2]
    experts = touches + temoins
    ts = []
    for e in experts:
        q = lire(base.format(e) + ".qweight"); b = bs_tous[e]; gsc = lire(base.format(e) + ".global_scale").float()
        ts.append(NVFP4Tensor(q, b, gsc, (q.shape[0], q.shape[1] * 2), q.shape[1] * 2))
    qw = torch.stack([t.qweight for t in ts]); bs = torch.stack([t.block_scale for t in ts])
    gs = torch.stack([t.global_scale.reshape(()) for t in ts]).float()
    N, K = qw.shape[1], qw.shape[2] * 2
    ref = torch.stack([dequantize_nvfp4(t, torch.bfloat16) for t in ts])
    w, s_, g, W = _depaquete(MP, qw, bs, gs, K, N, True)
    assert g.shape == (len(experts), N)
    assert torch.equal(W, ref), f"couche {couche} {mat} : {int((W != ref).sum())} valeurs fausses sur {len(touches)} experts touchés"
    if couche == 0 and mat == "gate_proj":
        _, _, _, W0 = _depaquete(MP, qw, bs, gs, K, N, False)
        assert not torch.equal(W0, ref), "témoin : le facteur commun devrait rendre la pile fausse"


def test_232_le_facteur_par_ligne_est_a_la_demande_et_0_le_defaut(monkeypatch):
    """Pièce 232 b (26/09, chef) : la 209 est À LA DEMANDE. Historique : 209 au défaut (25/09) → 220 opt-in sur le banc 217 → 226 b
    au défaut (artefact du banc 217 démontré, +12,8 % en salve unique à invites réelles) → 229 (poste3) : en débit SOUTENU le 1 perd
    −15,3 % / +25,4 % J sur le Coder pur — écart entre protocoles non expliqué, la release garde l'ancien comportement.
    Ce test casse si le défaut revient à 1 (code, table des variables) ou si, au défaut, une pile à sous-normales est préparée
    par ligne (g [E, N]) au lieu d'être refusée comme depuis la 157 ; 1 doit encore servir la pile (à la demande)."""
    from acvram import regime
    from acvram.engine import moe
    from acvram.kernels import marlin_port as MP
    monkeypatch.delenv("ACVRAM_MARLIN_PAR_LIGNE", raising=False)
    assert moe.MARLIN_PAR_LIGNE_DEFAUT == "0"
    assert os.environ.get("ACVRAM_MARLIN_PAR_LIGNE", moe.MARLIN_PAR_LIGNE_DEFAUT) != "1"
    v = next(v for v in regime.VARIABLES if v.nom == "MARLIN_PAR_LIGNE")
    assert v.defaut == "0" and v.torch == "1"
    E, N, K = 4, 128, 256
    _, qw, bs, gs = _pile_synthetique(E, N, K, 2320, minuscules=(1, 3))
    assert MP.echelles_ecrasees(bs) > 0, "montage : le facteur commun doit écraser"
    par_ligne_defaut = os.environ.get("ACVRAM_MARLIN_PAR_LIGNE", moe.MARLIN_PAR_LIGNE_DEFAUT) == "1"
    assert MP.echelles_ecrasees(bs, par_ligne=par_ligne_defaut) > 0, "au défaut, la pile doit être REFUSÉE (moe.py:490)"
    assert MP.preparer_pile(qw, bs, gs, repack=MP.repack_torch, par_ligne=par_ligne_defaut)[2].shape == (E,)
    monkeypatch.setenv("ACVRAM_MARLIN_PAR_LIGNE", "1")
    demande = os.environ["ACVRAM_MARLIN_PAR_LIGNE"] == "1"
    assert MP.echelles_ecrasees(bs, par_ligne=demande) == 0, "à la demande, la pile doit être servie"
    assert MP.preparer_pile(qw, bs, gs, repack=MP.repack_torch, par_ligne=demande)[2].shape == (E, N)
