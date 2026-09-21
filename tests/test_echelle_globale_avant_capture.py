"""L'échelle globale NVFP4 doit être lue AVANT la capture, pas pendant.

`NVFP4Tensor.global_scale_float()` mémorise sa valeur dans `_gs_f`, mais le
premier appel fait `.item()` — une synchronisation hôte, donc
`cudaErrorStreamCaptureUnsupported` si elle tombe dans une capture de graphe.

Mesuré le 10/09/2026 sur Qwen3-4B-nvfp4, 5090 :

    seqs  avant_pas  captures  exiles  actifs  verdict     (avant / apres)
    8     23,46 G    1 / 1     0       oui     OK   / OK
    12    23,46 G    0 / 1     0       oui     ECHEC/ OK
    16    23,46 G      / 1     0       oui         / OK

A huit séquences le seuil de lot n'est pas franchi, le GEMV ne lit pas cette
échelle-là, la capture réussit. A douze, `v_proj` prend le chemin W4A8, qui
déquantifie et lit l'échelle pour la première fois — dans la capture.

Ce test ne demande pas de GPU : il vérifie que la mémoïsation existe et qu'une
lecture préalable la remplit, ce qui est exactement ce dont la capture dépend.
"""
import torch

from acvram.quant.nvfp4 import NVFP4Tensor


def _tenseur():
    return NVFP4Tensor(
        qweight=torch.zeros(32, 16, dtype=torch.uint8),
        block_scale=torch.ones(32, 2, dtype=torch.uint8),
        global_scale=torch.tensor([2.5]),
        shape=(32, 32), padded_in=32)


def test_la_premiere_lecture_memorise():
    """Sans mémoïsation, chaque GEMV synchroniserait : 62 % du temps de
    décodage au profil, et une capture impossible."""
    t = _tenseur()
    assert "_gs_f" not in t.__dict__, "rien n'est memorise avant la lecture"
    assert t.global_scale_float() == 2.5
    assert t.__dict__["_gs_f"] == 2.5, "la valeur doit survivre a l'appel"


def test_la_lecture_prealable_dispense_de_la_suivante(monkeypatch):
    """Le geste du correctif : lire une fois avant la capture suffit à ce que
    plus aucune lecture ne touche le périphérique ensuite. On le prouve en
    rendant `.item()` explosif APRES la lecture préalable."""
    t = _tenseur()
    t.global_scale_float()                    # la lecture d'avant-capture

    class _Piege:
        def item(self):
            raise AssertionError("un .item() pendant la capture : interdit")
    monkeypatch.setattr(t, "global_scale", _Piege(), raising=False)

    assert t.global_scale_float() == 2.5, \
        "la valeur memorisee doit repondre sans toucher le peripherique"


def test_sans_lecture_prealable_le_piege_se_declenche(monkeypatch):
    """Témoin : le piège du test précédent mord bien quand la lecture
    préalable n'a pas eu lieu. Sans lui, ce test passerait pour de mauvaises
    raisons — c'est le contrôle qui manquait a `33ab9d8`."""
    import pytest
    t = _tenseur()

    class _Piege:
        def item(self):
            raise AssertionError("un .item() pendant la capture : interdit")
    monkeypatch.setattr(t, "global_scale", _Piege(), raising=False)

    with pytest.raises(AssertionError):
        t.global_scale_float()
