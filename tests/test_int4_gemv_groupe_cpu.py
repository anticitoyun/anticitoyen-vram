"""Banc d'équivalence pour `int4_gemv_grouped` (W4A16 groupé, MoE), écrit
AVANT le nouveau noyau de Laurine — même motif que `nvfp4_gemv_grouped`, qui
a déjà un noyau `_warp` optimisé sans que le noyau INT4 en ait un.

`int4_gemv_grouped_kernel` (acvram_kernels.cu:733) existe déjà et est lié,
mais sans épreuve numérique : ce fichier pose la référence float32 — pile
d'experts empilée comme le fait `Layer._try_build_stacks` (torch.stack sur
qweight/scales/zeros, engine/model.py:786), routée par (expert_ids,
token_ids) comme le fait `Layer._grouped` — pour que le prochain noyau
(warp, gate+up fusionné) ait un juge dès son premier lancement, pas après.

Deux paliers, comme `test_nvfp4_echelle_par_ligne_cpu.py` :
  1. le témoin : la référence Python/PyTorch doit elle-même savoir rendre
     « faux » sur un expert corrompu (REGLES §5) — sinon l'épreuve du noyau
     ne prouverait rien.
  2. le noyau CUDA (`kernels.int4_gemv_grouped`), sauté proprement — pas en
     échec — tant que la carte ou le noyau lui-même sont absents.
"""
from __future__ import annotations

import pytest
import torch

from acvram.quant.int4 import INT4Tensor, dequantize_int4, quantize_int4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")

GROUP = 32


def _pile_experts(n_experts, m, k, graine):
    """Empile `n_experts` projections INT4 indépendantes, comme le fait
    `Layer._try_build_stacks` pour le chemin groupé (torch.stack, pas de
    copie de vue ici puisqu'aucun modèle réel ne lit ces tenseurs)."""
    g = torch.Generator().manual_seed(graine)
    qts = []
    for e in range(n_experts):
        w = (torch.randn(m, k, generator=g) * (0.02 * (e + 1))).to(torch.float32)
        qts.append(quantize_int4(w, group_size=GROUP))
    qw = torch.stack([t.qweight for t in qts]).contiguous()
    sc = torch.stack([t.scales for t in qts]).contiguous()
    zr = torch.stack([t.zeros for t in qts]).contiguous()
    return qts, qw, sc, zr


def _reference_fp32(qts, expert_ids, token_ids, x):
    """Ce que le noyau groupé prétend approcher : pour chaque paire
    (expert, jeton), le produit du poids DÉQUANTIFIÉ (référence de
    `dequantize_int4`, celle que le noyau CUDA fusionné doit suivre) contre
    la ligne d'activation, en float32 des deux côtés — mélanger un format
    plus étroit mesurerait l'arrondi du format, pas l'échelle (leçon du
    banc NVFP4 voisin)."""
    sorties = []
    for e, tok in zip(expert_ids.tolist(), token_ids.tolist()):
        w = dequantize_int4(qts[e], torch.float32)          # [m, k]
        sorties.append(w @ x[tok])
    return torch.stack(sorties)


def test_le_temoin_distingue_un_expert_corrompu():
    """Un expert dont l'échelle est doublée DOIT sortir faux — sinon la
    référence elle-même ne prouverait rien (REGLES §5, même motif que
    `test_le_temoin_montre_que_l_epreuve_peut_echouer` pour NVFP4)."""
    m, k, n_experts = 16, 128, 3
    qts, _, _, _ = _pile_experts(n_experts, m, k, graine=11)
    g = torch.Generator().manual_seed(12)
    x = torch.randn(4, k, generator=g)
    expert_ids = torch.tensor([0, 1, 2, 0])
    token_ids = torch.tensor([0, 1, 2, 3])

    correct = _reference_fp32(qts, expert_ids, token_ids, x)

    qts_corrompu = list(qts)
    abime = qts_corrompu[1]
    qts_corrompu[1] = INT4Tensor(abime.qweight, abime.scales * 2.0,
                                 abime.zeros, abime.group_size,
                                 abime.shape, abime.padded_in)
    faux = _reference_fp32(qts_corrompu, expert_ids, token_ids, x)

    assert not torch.allclose(correct[1], faux[1], atol=1e-2), (
        "le témoin ne diverge pas sur l'expert corrompu : la référence "
        "ne prouverait rien")
    assert torch.allclose(correct[0], faux[0]), "l'expert non corrompu ne doit pas bouger"
    assert torch.allclose(correct[2], faux[2]), "l'expert non corrompu ne doit pas bouger"


@CUDA
def test_le_noyau_cuda_sur_une_pile_groupee_est_juste():
    """Palier réel : `kernels.int4_gemv_grouped` contre la référence
    float32. Sauté (pas en échec) si l'extension n'a pas encore ce
    symbole ou le refuse — cas normal avant l'arrivée du nouveau noyau
    de Laurine, `int4_gemv_grouped` existant restant le seul juge."""
    from acvram import kernels

    m, k, n_experts, n_tokens, n_routes = 16, 128, 3, 5, 8
    qts, qw, sc, zr = _pile_experts(n_experts, m, k, graine=21)

    g = torch.Generator().manual_seed(22)
    x = torch.randn(n_tokens, k, generator=g)
    expert_ids = torch.randint(0, n_experts, (n_routes,), generator=g)
    token_ids = torch.randint(0, n_tokens, (n_routes,), generator=g)

    dev = torch.device("cuda")
    qw, sc, zr = qw.to(dev), sc.to(dev), zr.to(dev)
    expert_ids, token_ids = expert_ids.to(dev).to(torch.int32), token_ids.to(dev).to(torch.int32)
    x32 = x.to(dev, torch.float32)

    obtenu = kernels.int4_gemv_grouped(x32, qw, sc, zr, expert_ids, token_ids,
                                       k, GROUP)
    if obtenu is None:
        pytest.skip("kernels.int4_gemv_grouped indisponible (extension non "
                    "chargée, ou noyau pas encore lié) — banc prêt, pas de "
                    "juge encore présent")
    obtenu = obtenu[:, :m].to(torch.float32).cpu()

    attendu = _reference_fp32(qts, expert_ids.cpu(), token_ids.cpu(), x)
    ecart = (obtenu - attendu).abs().max().item()
    echelle = attendu.abs().max().item()
    assert ecart <= 1e-3 * max(echelle, 1e-6), (
        f"le GEMV groupé INT4 derive de {ecart:.3e} sur une pile de "
        f"{n_experts} experts (echelle {echelle:.3e})")
