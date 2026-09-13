"""Compteur de routage : bincount accumulé sur device, jamais lu au pas.

Chantier colibrì (`revue/colibri-lecture-code.md`), bead anticitoyen-vram-jt5 :
le premier compteur qu'un cache par expert demanderait, avant toute décision
de placement par expert. Ce fichier teste UNIQUEMENT l'accumulation, sa
capturabilité par un graphe CUDA et la mesure de concentration ; le chemin de
routage lui-même (top_k, softmax, sigmoïde) est déjà couvert par
tests/test_moe_chemins_decodage.py.
"""
from __future__ import annotations

import pytest
import torch

from acvram.engine.model import MoEBlock
from acvram.memory.expert_usage import concentration_top_fraction


def _bloc(n_experts: int, top_k: int) -> MoEBlock:
    """Un MoEBlock réduit au strict nécessaire pour `_compter_routage` : ni
    routeur ni experts réels ne sont invoqués dans ces tests, seul
    `len(self.experts)` compte."""
    experts = [torch.nn.Identity() for _ in range(n_experts)]
    return MoEBlock(torch.nn.Identity(), experts, top_k)


# --------------------------------------------------------------------------
# accumulation, sans GPU
# --------------------------------------------------------------------------

def test_rien_avant_le_premier_appel():
    assert _bloc(4, 1)._usage_routage is None


def test_compte_s_accumule_entre_appels():
    bloc = _bloc(n_experts=8, top_k=2)
    bloc._compter_routage(torch.tensor([[0, 1], [1, 2]]))
    bloc._compter_routage(torch.tensor([[1, 1], [0, 3]]))
    # expert 0: 2, expert 1: 4, expert 2: 1, expert 3: 1, le reste à 0
    assert bloc._usage_routage.tolist() == [2, 4, 1, 1, 0, 0, 0, 0]


def test_creneau_fantome_n_est_pas_compte():
    """-1 (créneau fantôme du remplissage godet `bucket_batch`, bead pds
    14/09) : ni compté, ni confondu avec l'expert 0 par `clamp(min=0)`."""
    bloc = _bloc(n_experts=4, top_k=2)
    bloc._compter_routage(torch.tensor([[0, 1], [-1, -1]]))
    assert bloc._usage_routage.tolist() == [1, 1, 0, 0]


def test_le_tampon_ne_change_pas_d_adresse():
    """Condition de capturabilité par un graphe CUDA : `+=` doit muter le
    tenseur existant, jamais en créer un nouveau — sinon un graphe capturé
    écrirait au replay dans une adresse que la capture n'a jamais vue (même
    défaut que RotaryEmbedding, voir
    test_improvements.py::test_rope_ne_se_realloue_pas_sous_capture)."""
    bloc = _bloc(n_experts=4, top_k=1)
    bloc._compter_routage(torch.tensor([[0]]))
    tampon = bloc._usage_routage
    bloc._compter_routage(torch.tensor([[1]]))
    assert bloc._usage_routage.data_ptr() == tampon.data_ptr()
    assert bloc._usage_routage is tampon


def test_index_hors_domaine_leve_sur_cpu():
    """Règle 5 : un contrôle doit pouvoir rendre faux — sur CPU. `scatter_add_`
    y refuse un index >= n_experts (bogue amont, jamais produit par le
    routage réel) par une exception plutôt qu'en silence. Sur CUDA ce même cas
    est un comportement NON DÉFINI de `scatter_add_` (voir la docstring de
    `_compter_routage`) : ce test ne couvre QUE le chemin CPU."""
    bloc = _bloc(n_experts=4, top_k=1)
    with pytest.raises(RuntimeError):
        bloc._compter_routage(torch.tensor([[4]]))     # experts valides : 0..3


# --------------------------------------------------------------------------
# concentration — fonction pure, sans GPU
# --------------------------------------------------------------------------

def test_concentration_un_seul_expert_chaud():
    compte = torch.zeros(8, dtype=torch.int64)
    compte[3] = 100
    assert concentration_top_fraction(compte, fraction=0.10) == pytest.approx(1.0)


def test_concentration_uniforme():
    """Routage parfaitement uniforme : les 10 % les plus chauds ne captent
    que ~10 % des sélections, aucun expert n'étant plus chaud qu'un autre."""
    compte = torch.full((100,), 7, dtype=torch.int64)
    assert concentration_top_fraction(compte, fraction=0.10) == pytest.approx(0.10)


def test_concentration_sans_selection_est_zero():
    compte = torch.zeros(8, dtype=torch.int64)
    assert concentration_top_fraction(compte) == 0.0


def test_concentration_vide_est_zero():
    assert concentration_top_fraction(torch.zeros(0, dtype=torch.int64)) == 0.0


# --------------------------------------------------------------------------
# capturabilité par un graphe CUDA — nécessite la carte
# --------------------------------------------------------------------------

@pytest.mark.gpu_requis
def test_compteur_capturable_par_un_graphe_cuda():
    """Aucune allocation, forme fixe pendant la capture : le tampon existe
    déjà (échauffement), la capture ne fait que muter dedans."""
    dev = torch.device("cuda")
    bloc = _bloc(n_experts=8, top_k=2).to(dev)
    topi = torch.zeros(4, 2, dtype=torch.int64, device=dev)
    topi[:, 1] = 1

    bloc._compter_routage(topi)                        # réserve le tampon
    avant = bloc._usage_routage.clone()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        bloc._compter_routage(topi)                     # échauffement du graphe
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        bloc._compter_routage(topi)                      # ENREGISTRÉ, pas exécuté :
    for _ in range(5):                                   # capturer un noyau ne le
        g.replay()                                       # lance pas — seul replay()
    torch.cuda.synchronize()                              # l'exécute, ici 5 fois

    pas = torch.zeros(8, dtype=torch.int64, device=dev).scatter_add_(
        0, topi.reshape(-1), torch.ones_like(topi.reshape(-1)))
    # 1 échauffement (stream s) + 5 replays = 6 passes de plus qu'avant ; la
    # capture elle-même n'en ajoute aucune (elle enregistre, sans exécuter)
    assert torch.equal(bloc._usage_routage, avant + 6 * pas)


@pytest.mark.gpu_requis
def test_reallocation_sous_capture_refusee():
    """Un bloc jamais échauffé, sollicité pour la première fois PENDANT une
    capture, doit refuser plutôt que réserver un tampon que le graphe ne
    reverra jamais au replay."""
    dev = torch.device("cuda")
    bloc = _bloc(n_experts=4, top_k=1).to(dev)
    topi = torch.zeros(1, 1, dtype=torch.int64, device=dev)
    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="capture"):
        with torch.cuda.graph(g):
            bloc._compter_routage(topi)                 # jamais réservé hors capture
