"""Le chemin FP4 tensor cores dans une capture de graphe : deux défauts.

Mesuré le 10/09/2026 sur Qwen3-4B-nvfp4, 5090, sonde `outils/sonder-seuil-graphes.py` :

    seqs  libre_depart  avant_pas  captures  vivants  verdict
    8     30,85 G       23,46 G    1         1        OK
    12    30,85 G       23,46 G    0         0        ECHEC AcceleratorError

Même marge au moment de la capture dans les deux cas : ce n'est pas la place.
`captures = 0` : l'échec précède toute capture. Le seul seuil de lot du paquet
vaut `> 8` — à douze séquences le décodage prend le backend `fp4-tensorcores`,
donc `torch._scaled_mm`, donc cuBLASLt, dont le premier appel sur un flux
réserve un espace de travail : interdit sous capture.

Ces tests comparent des COMPORTEMENTS, pas du texte source, et tournent sans
GPU : la condition de capture court-circuite avant toute requête au pilote.
"""
import pytest
import torch

from acvram.kernels import backends as _bk


def _backend_fp4():
    for b in _bk._REGISTRY:
        if b.name == "fp4-tensorcores":
            return b
    pytest.skip("backend fp4-tensorcores absent du registre")


class _Faux:
    """Assez de tenseur pour la condition, pas assez pour toucher le pilote."""
    def __init__(self, rangees, colonnes=128):
        self.shape = (rangees, colonnes)
    def reshape(self, *_a):
        return self


def test_le_backend_fp4_se_retire_sous_capture(monkeypatch):
    """Le correctif : douze rangées ne suffisent plus, il faut aussi être
    hors capture. Sans quoi la capture est invalidée avant d'exister."""
    b = _backend_fp4()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert b.matmul(_Faux(12), object()) is None, \
        "sous capture, le chemin cuBLASLt doit etre ecarte quel que soit le lot"


def test_huit_rangees_restent_ecartees_hors_capture(monkeypatch):
    """Contrôle : le seuil de lot n'a pas été supprimé, seulement complété.
    Huit rangées ne prennent pas ce chemin — c'est pourquoi la manche à huit
    séquences réussissait."""
    b = _backend_fp4()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert b.matmul(_Faux(8), object()) is None, "8 > 8 est faux"


def test_l_echec_sous_capture_n_eteint_pas_le_processus(monkeypatch):
    """Le second défaut, plus grave que le premier : l'extinction était
    PERMANENTE. Un serveur ayant vu une seule fois plus de huit séquences
    perdait les tensor cores FP4 pour toutes ses requêtes suivantes, sans
    autre trace qu'un avertissement. Une capture en cours n'est pas une panne
    du chemin : l'erreur se relaie, elle ne s'enregistre pas."""
    from acvram.kernels import fp4_gemm as f

    monkeypatch.setattr(f, "_ETEINT_PAR_CAPACITE", {})
    monkeypatch.setattr(f, "fp4_mm_available", lambda d=None: True)
    monkeypatch.setattr(f, "_capacite", lambda d: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    class _Poids:
        global_scale_rows = None
        padded_in = 128
        qweight = torch.zeros(64, 64, dtype=torch.uint8)

    def _explose(*_a, **_k):
        raise RuntimeError("operation not permitted during stream capture")
    monkeypatch.setattr("acvram.quant.nvfp4.quantize_nvfp4", _explose)

    with pytest.raises(RuntimeError):
        f.nvfp4_mm_tensorcore(torch.zeros(12, 128), _Poids())
    assert f._ETEINT_PAR_CAPACITE == {}, \
        "un accident de capture ne doit pas devenir un etat permanent"
