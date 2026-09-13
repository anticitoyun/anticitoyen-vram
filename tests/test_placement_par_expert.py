"""Placement par expert (bead anticitoyen-vram-pds, point 4 — poste7 §4) :
sortie bit-identique entre une couche 100 % résidente et la MÊME couche à
moitié exilée PAR EXPERT (pas par couche entière). « Placement = vitesse,
jamais sémantique » — REGLES.md : ce test EST cette règle, appliquée au
grain le plus fin que ce chantier introduit.

Style repris de test_moe_chemins_decodage.py (couche réduite mais réelle,
poids NVFP4 identiques entre les deux configurations, comparaison au chemin
`t == 1` du décodage) ; NVFP4 ici plutôt que q3n, parce que c'est le format
du contrat de table (`memory/table_adresses.py`) et du noyau de poste4.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu_requis

CACHE, INTER, N_EXPERTS, TOP_K = 256, 128, 8, 4


def _lineaire(sortie, entree, graine, dev, pool):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4

    g = torch.Generator().manual_seed(graine)
    w = torch.randn(sortie, entree, generator=g) * 0.02
    t = quantize_nvfp4(w)
    lin = QuantLinear(t, out_features=sortie, in_features=entree)
    return lin.to_device(dev, streamed=pool is not None, pool=pool)


def _couche(dev, residents: "set[int] | None"):
    """`residents=None` : tout résident (comportement d'aujourd'hui).
    Sinon : les experts HORS `residents` sont streamés via un `ExpertPool`,
    exactement le chemin qu'un `mlp_storage` de couche entière emprunte déjà
    — appliqué ici par expert."""
    from acvram.engine.layers import ExpertPool
    from acvram.engine.model import MLP, MoEBlock

    pool = ExpertPool(dev, 2 * TOP_K + 2) if residents is not None else None
    experts = []
    for e in range(N_EXPERTS):
        froid = residents is not None and e not in residents
        p = pool if froid else None
        gate = _lineaire(INTER, CACHE, 10 * e + 1, dev, p)
        up = _lineaire(INTER, CACHE, 10 * e + 2, dev, p)
        down = _lineaire(CACHE, INTER, 10 * e + 3, dev, p)
        experts.append(MLP(gate, up, down))

    g = torch.Generator().manual_seed(999)
    routeur_w = torch.randn(N_EXPERTS, CACHE, generator=g) * 0.02
    routeur = _lineaire2(routeur_w, dev)
    bloc = MoEBlock(routeur, experts, TOP_K)
    bloc._stack_state = "non"          # isole le placement du choix pile/boucle
    return bloc


def _lineaire2(w, dev):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4
    t = quantize_nvfp4(w)
    return QuantLinear(t, out_features=w.shape[0],
                       in_features=w.shape[1]).to_device(dev)


@pytest.mark.parametrize("residents", [
    {0, 2, 4, 6},               # les moitiés paire/impaire, arbitraire
    {1, 2, 3},                  # 5/8 exilés
])
def test_moitie_exilee_par_expert_est_bit_identique(residents):
    """`t == 1` (décodage) : le chemin qui appelle `self.experts[e](x)`
    directement, celui que `MoEBlock.forward` emprunte pour un seul jeton."""
    dev = torch.device("cuda")
    tout_resident = _couche(dev, None)
    moitie = _couche(dev, residents)

    torch.manual_seed(42)
    x = torch.randn(1, CACHE, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        y1 = tout_resident(x.clone())
        y2 = moitie(x.clone())

    assert torch.equal(y1, y2), (
        f"placement par expert a changé la sortie (écart max "
        f"{(y1 - y2).abs().max().item()}) — violation de « placement = "
        f"vitesse, jamais sémantique »")


def test_tous_exiles_est_aussi_bit_identique():
    """Cas limite : `residents` vide (aucun résident). Doit rester correct —
    équivalent à l'exil de couche entière d'aujourd'hui, à ce détail près
    que le placement est décrit par expert plutôt que par `mlp_storage`."""
    dev = torch.device("cuda")
    tout_resident = _couche(dev, None)
    tout_exile = _couche(dev, set())

    torch.manual_seed(7)
    x = torch.randn(1, CACHE, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        y1 = tout_resident(x.clone())
        y2 = tout_exile(x.clone())
    assert torch.equal(y1, y2)
