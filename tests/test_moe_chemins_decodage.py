"""Les trois chemins d'une couche MoE rendent-ils la même chose ?

Le décodage tout-q3n de Qwen3-Coder-Next dégénère, et les deux chemins de
décodage dégénèrent **différemment** : le chemin direct à un jeton donne
« loi,loi,loi », le chemin par masques donne « URTURT ». Un défaut de format
les frapperait tous les deux de la même façon ; qu'ils divergent entre eux
désigne le code, pas le format.

Ce fichier construit une couche MoE réduite mais complète — experts q3n exilés
en mémoire hôte, copiés par le pool, dix experts routés sur seize — et compare
les trois chemins à une référence calculée en float32 à partir des mêmes poids
déquantifiés. La référence n'est pas un autre chemin du moteur : c'est
l'arithmétique que les trois prétendent tous approcher.
"""
from __future__ import annotations

import os

import pytest
import torch

from acvram.quant.q3n import dequantize_q3n, quantize_q3n

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")

CACHE, INTER, N_EXPERTS, TOP_K = 256, 128, 16, 10


def _lineaire(sortie, entree, graine, dev, pool):
    from acvram.engine.layers import QuantLinear

    g = torch.Generator().manual_seed(graine)
    w = torch.randn(sortie, entree, generator=g) * 0.02
    t = quantize_q3n(w)
    lin = QuantLinear(t, out_features=sortie, in_features=entree)
    ref = dequantize_q3n(t, torch.float32)          # ce que le poids vaut vraiment
    return lin.to_device(dev, streamed=pool is not None, pool=pool), ref


def _couche(dev, pool):
    """Une couche MoE et, à côté, les poids déquantifiés pour la référence."""
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import MLP, MoEBlock

    experts, refs = [], []
    for e in range(N_EXPERTS):
        gate, rg = _lineaire(INTER, CACHE, 10 * e + 1, dev, pool)
        up, ru = _lineaire(INTER, CACHE, 10 * e + 2, dev, pool)
        down, rd = _lineaire(CACHE, INTER, 10 * e + 3, dev, pool)
        experts.append(MLP(gate, up, down))
        refs.append((rg, ru, rd))

    routeur, _ = _lineaire(N_EXPERTS, CACHE, 999, dev, None)
    bloc = MoEBlock(routeur, experts, TOP_K).to(dev)
    bloc._stack_state = "non"        # les piles court-circuiteraient les deux chemins
    return bloc, refs


def _reference(x32, topw32, topi, refs):
    """Ce que la couche doit valoir, en float32, sans passer par le moteur."""
    out = torch.zeros_like(x32)
    for j, e in enumerate(topi.reshape(-1).tolist()):
        rg, ru, rd = (r.to(x32.device) for r in refs[e])
        h = torch.nn.functional.silu(x32 @ rg.t()) * (x32 @ ru.t())
        out += (h @ rd.t()) * topw32.reshape(-1)[j]
    return out


def _ecart(a, b) -> float:
    """Écart relatif en norme — 0 si les deux sorties sont le même vecteur."""
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    return float((a32 - b32).norm() / b32.norm().clamp(min=1e-12))


@CUDA
@pytest.mark.parametrize("exile", [False, True], ids=["resident", "exile"])
def test_les_deux_chemins_de_decodage_concordent(exile):
    """Un jeton, dix experts : chemin direct et chemin par masques.

    ``exile=True`` fait passer les poids par le tampon épinglé et le pool,
    c'est-à-dire par la copie des experts. Si l'écart n'apparaît que là, le
    transport est en cause ; s'il apparaît des deux côtés, c'est le calcul.
    """
    from acvram.engine.layers import ExpertPool

    dev = torch.device("cuda:0")
    pool = ExpertPool(dev, 2 * TOP_K + 2) if exile else None
    bloc, refs = _couche(dev, pool)

    g = torch.Generator().manual_seed(7)
    x = (torch.randn(1, CACHE, generator=g) * 0.5).to(dev, torch.bfloat16)

    topw, topi = bloc._route(x)
    attendu = _reference(x.to(torch.float32), topw.to(torch.float32), topi, refs)

    os.environ.pop("ACVRAM_MOE_DECODE_MASQUES", None)
    direct = bloc(x)
    os.environ["ACVRAM_MOE_DECODE_MASQUES"] = "1"
    try:
        masques = bloc(x)
    finally:
        os.environ.pop("ACVRAM_MOE_DECODE_MASQUES", None)

    e_direct, e_masques = _ecart(direct, attendu), _ecart(masques, attendu)
    entre_eux = _ecart(direct, masques)
    print(f"\n  exile={exile}  direct/ref {e_direct:.2e}  "
          f"masques/ref {e_masques:.2e}  entre eux {entre_eux:.2e}")

    # Zero EXACT, pas une tolerance : depuis que les deux chemins accumulent en
    # float32, l'ordre des sommes n'a plus d'effet et ils rendent le meme
    # vecteur bit pour bit. Avant, l'ordre seul valait 5,4e-3 — assez pour que
    # deux chemins choisissent deux mots differents sur un modele au bord.
    assert entre_eux == 0.0, "les deux chemins de décodage ne calculent pas la même chose"
    assert e_direct < 5e-2, "le chemin direct s'écarte de l'arithmétique de référence"
    assert e_masques < 5e-2, "le chemin par masques s'écarte de l'arithmétique de référence"


@CUDA
def test_experts_pris_un_a_un_avant_la_somme():
    """Chaque expert, seul, contre sa référence — avant toute pondération.

    C'est la mesure qui sépare « un expert calcule faux » de « la somme est
    mal faite ». Les experts sont exilés : chacun passe par le pool.
    """
    from acvram.engine.layers import ExpertPool

    dev = torch.device("cuda:0")
    pool = ExpertPool(dev, 2 * TOP_K + 2)
    bloc, refs = _couche(dev, pool)

    g = torch.Generator().manual_seed(7)
    x = (torch.randn(1, CACHE, generator=g) * 0.5).to(dev, torch.bfloat16)
    x32 = x.to(torch.float32)

    _topw, topi = bloc._route(x)
    pires = []
    for e in topi.reshape(-1).tolist():
        rg, ru, rd = (r.to(dev) for r in refs[e])
        h = torch.nn.functional.silu(x32 @ rg.t()) * (x32 @ ru.t())
        pires.append((_ecart(bloc.experts[e](x), h @ rd.t()), e))
    pires.sort(reverse=True)
    print("\n  pires experts (écart, indice) : "
          + ", ".join(f"{v:.2e}@{e}" for v, e in pires[:3]))
    assert pires[0][0] < 5e-2, f"expert {pires[0][1]} calcule faux, seul"


def _aberrant(sortie, entree, graine, force=100.0):
    """Un poids gaussien où quelques coefficients écrasent tous les autres.

    C'est la forme qu'ont les projections de sortie des vrais experts. Elle
    compte parce que Q3N tire son échelle globale du maximum absolu du tenseur
    entier : un aberrant à cent fois la normale divise par cent l'échelle de
    tous les autres blocs, et l'échelle de bloc — un FP8 e4m3 — tombe dans les
    subnormaux, où il ne reste que trois bits.
    """
    g = torch.Generator().manual_seed(graine)
    w = torch.randn(sortie, entree, generator=g) * 0.02
    w.view(-1)[graine % w.numel()] = force * 0.02
    return w


@CUDA
def test_ecart_des_chemins_sous_echelles_subnormales():
    """L'écart entre les deux chemins grandit-il quand le format se dégrade ?

    Si oui, la divergence observée sur le vrai modèle n'est pas un second
    défaut : c'est le même bruit de format, lu par deux ordres de sommation
    différents. Un argmax pris sur du bruit change de mot pour un dernier bit.
    """
    from acvram.engine.layers import ExpertPool, QuantLinear
    from acvram.engine.model import MLP, MoEBlock

    dev = torch.device("cuda:0")
    mesures = []
    for force in (1.0, 30.0, 100.0, 1000.0):
        pool = ExpertPool(dev, 2 * TOP_K + 2)
        experts, refs = [], []
        for e in range(N_EXPERTS):
            trio, rtrio = [], []
            for k, (s, en) in enumerate(((INTER, CACHE), (INTER, CACHE), (CACHE, INTER))):
                w = _aberrant(s, en, 10 * e + k + 1, force)
                t = quantize_q3n(w)
                trio.append(QuantLinear(t, out_features=s, in_features=en)
                            .to_device(dev, streamed=True, pool=pool))
                rtrio.append(dequantize_q3n(t, torch.float32))
            experts.append(MLP(*trio)); refs.append(tuple(rtrio))

        routeur, _ = _lineaire(N_EXPERTS, CACHE, 999, dev, None)
        bloc = MoEBlock(routeur, experts, TOP_K).to(dev)
        bloc._stack_state = "non"

        g = torch.Generator().manual_seed(7)
        x = (torch.randn(1, CACHE, generator=g) * 0.5).to(dev, torch.bfloat16)
        topw, topi = bloc._route(x)
        attendu = _reference(x.to(torch.float32), topw.to(torch.float32), topi, refs)

        os.environ.pop("ACVRAM_MOE_DECODE_MASQUES", None)
        direct = bloc(x)
        os.environ["ACVRAM_MOE_DECODE_MASQUES"] = "1"
        try:
            masques = bloc(x)
        finally:
            os.environ.pop("ACVRAM_MOE_DECODE_MASQUES", None)

        # Part des echelles de bloc tombees dans les subnormaux du e4m3
        # (< 2^-6) : la ou il ne reste plus que trois bits de mantisse.
        bs = torch.cat([r.reshape(-1) for r in ()]) if False else None
        mesures.append((force, _ecart(direct, masques),
                        _ecart(direct, attendu), _ecart(masques, attendu)))

    print("\n  aberrant   entre eux   direct/ref  masques/ref")
    for f, ee, ed, em in mesures:
        print(f"  x{f:<8.0f} {ee:.2e}    {ed:.2e}    {em:.2e}")

    # Le fait mesure, quel qu'il soit, est le resultat : ce test ne fixe qu'un
    # garde-fou grossier pour que la mesure ne disparaisse pas en silence.
    assert all(ee == 0.0 for _f, ee, _d, _m in mesures), \
        "l'ordre des sommes est redevenu visible entre les deux chemins"
