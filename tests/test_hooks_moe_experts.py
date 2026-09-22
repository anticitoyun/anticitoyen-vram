"""Fake-quant sur les GEMV groupes MoE (gate/up/down des experts) — bead
anticitoyen-vram-brd, recadrage de Jerome 13/09 : le noyau MMA ne sert que
le chemin MoE groupe, qui n'appelle jamais QuantLinear.forward().

Aucun modele MoE jouet n'existe dans ce depot (conftest.py ne construit
qu'un Llama dense) : ces tests verifient le mecanisme de monkeypatch avec
des OBJETS FACTICES qui reproduisent l'interface exacte de MoEBlock._grouped
et de l'extension CUDA — pas un forward complet. La vraie mesure est sur
carte, Qwen3-Coder-30B-A3B-nvfp4 (revue/prediction-moe-experts-a4.md).
"""
import torch

from outils.hooks_activations_a4 import installer_hooks_moe_experts, retirer_hooks


class _FauxMoEBlock:
    """Reproduit juste assez de MoEBlock pour le monkeypatch : le nom de
    classe (`type(m).__name__.startswith("MoEBlock")`, verifie par
    `installer_hooks_moe_experts`) et `_grouped(x32, pile, eid, tok)`."""

    def __init__(self):
        self.appels = []

    def _grouped(self, x32, pile, expert_ids, token_ids):
        self.appels.append(x32.clone())
        return x32 @ pile.t()

    def modules(self):
        return [self]


class MoEBlockFactice(_FauxMoEBlock):
    pass


class _FauxExtension:
    """Reproduit l'extension CUDA : `nvfp4_gemv_grouped_gateup` present,
    pour verifier qu'`installer_hooks_moe_experts` le masque bien."""

    def nvfp4_gemv_grouped_gateup(self):
        return "chemin fusionne — ne doit jamais etre atteint pendant le hook"

    def moe_reduce(self):
        return "autre fonction de l'extension, doit rester accessible"


def test_installer_hooks_moe_experts_fake_quantifie_l_entree_de_grouped(monkeypatch):
    import acvram.kernels as kernels_module
    ext = _FauxExtension()
    monkeypatch.setattr(kernels_module, "get_extension", lambda *a, **kw: ext)

    modele = MoEBlockFactice()

    class _Model:
        def modules(self):
            return [modele]

    handles, blocs = installer_hooks_moe_experts(_Model(), "a4")
    assert blocs == [modele]
    assert len(handles) == 1

    x = torch.randn(4, 32)
    pile = torch.randn(8, 32)
    modele._grouped(x, pile, None, None)
    retirer_hooks(handles)

    assert len(modele.appels) == 1
    assert not torch.equal(modele.appels[0], x), (
        "l'entree recue par le vrai _grouped n'a pas ete fake-quantifiee")


def test_installer_hooks_moe_experts_masque_le_noyau_fusionne_gateup(monkeypatch):
    """Pendant le hook, `kernels.get_extension()` ne doit PLUS exposer
    `nvfp4_gemv_grouped_gateup` (sinon gate/up bypasseraient `_grouped` et
    ne seraient jamais fake-quantifies) ; les autres attributs de
    l'extension restent accessibles."""
    import acvram.kernels as kernels_module
    ext = _FauxExtension()
    monkeypatch.setattr(kernels_module, "get_extension", lambda *a, **kw: ext)

    modele = MoEBlockFactice()

    class _Model:
        def modules(self):
            return [modele]

    handles, _ = installer_hooks_moe_experts(_Model(), "a4")
    ext_pendant = kernels_module.get_extension()
    assert not hasattr(ext_pendant, "nvfp4_gemv_grouped_gateup")
    assert ext_pendant.moe_reduce() == "autre fonction de l'extension, doit rester accessible"
    retirer_hooks(handles)

    ext_apres = kernels_module.get_extension()
    assert hasattr(ext_apres, "nvfp4_gemv_grouped_gateup"), (
        "l'extension n'a pas ete restauree apres retirer_hooks")


def test_restaurer_remet_le_grouped_original(monkeypatch):
    """`bloc._grouped` accede sans attribut d'instance cree un nouveau
    wrapper de methode liee a chaque appel (comportement normal de
    Python) : comparer `is` sur deux acces distincts serait donc toujours
    faux, meme sans monkeypatch. Le test verifie le COMPORTEMENT — apres
    restauration, l'entree n'est plus fake-quantifiee — pas l'identite
    d'objet."""
    import acvram.kernels as kernels_module
    monkeypatch.setattr(kernels_module, "get_extension", lambda *a, **kw: _FauxExtension())

    modele = MoEBlockFactice()

    class _Model:
        def modules(self):
            return [modele]

    x = torch.randn(4, 32)
    pile = torch.randn(8, 32)

    handles, _ = installer_hooks_moe_experts(_Model(), "a4")
    modele._grouped(x, pile, None, None)
    assert not torch.equal(modele.appels[-1], x), "pas encore fake-quantifie pendant le hook"
    retirer_hooks(handles)

    modele._grouped(x, pile, None, None)
    assert torch.equal(modele.appels[-1], x), "toujours fake-quantifie APRES retirer_hooks"


def test_aucun_moe_block_ne_leve_pas_rend_des_listes_vides():
    class _ModelSansMoE:
        def modules(self):
            return []

    handles, blocs = installer_hooks_moe_experts(_ModelSansMoE(), "a4")
    assert handles == []
    assert blocs == []


def test_regime_inconnu_leve():
    class _Model:
        def modules(self):
            return []
    import pytest
    with pytest.raises(ValueError, match="inconnu"):
        installer_hooks_moe_experts(_Model(), "a2")
