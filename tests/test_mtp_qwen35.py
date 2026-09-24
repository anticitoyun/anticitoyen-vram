"""Pièce 105 : têtes MTP de la famille Qwen3.5 (noms HF ``mtp.*``) reconnues au chargement, trois normes hors
couche décalées (+1), et repli ``auto`` → n-gram NOMMÉ sur la ligne de régime. À sec."""
import torch

from acvram.engine import mtp as M
from acvram.engine.runner import _speculation_texte


def _man(*noms, **extra):
    return {"tensors": {n: {} for n in noms}, **extra}


QWEN35 = ("mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
          "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.input_layernorm.weight")


def test_cles_des_deux_conventions():
    assert M.cles_mtp(_man(*QWEN35)) == [0]
    assert M.cles_mtp(_man("model.mtp.0.eh_proj.weight", "model.mtp.1.enorm.weight")) == [0, 1]
    assert M.cles_mtp(_man("model.layers.0.self_attn.q_proj.weight", "lm_head.weight")) == []


def test_noms_selon_la_convention():
    q = M.noms_mtp(_man(*QWEN35), 0)
    assert (q["convention"], q["bloc"], q["eh"], q["enorm"], q["hnorm"], q["fin"]) == (
        "qwen35", "mtp.layers.0.", "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight", ("mtp.norm.weight",))
    d = M.noms_mtp(_man("model.mtp.0.eh_proj.weight"), 0)
    assert (d["convention"], d["bloc"], d["eh"]) == ("deepseek", "model.mtp.0.", "model.mtp.0.eh_proj.weight")


def test_trois_normes_qwen35_decalees_et_elles_seules():
    z = torch.zeros(4, dtype=torch.bfloat16)
    for nom in M.NORMES_QWEN35_A_DECALER:
        assert torch.equal(M.norme_mtp(nom, z, "qwen35", {}), torch.ones(4, dtype=torch.bfloat16)), nom
    # déjà décalée par la conversion (suffixe input_layernorm) : intacte
    assert torch.equal(M.norme_mtp("mtp.layers.0.input_layernorm.weight", z, "qwen35", {}), z)
    # convention DeepSeek/GGUF : intacte ; manifeste qui déclare le décalage fait : intacte
    assert torch.equal(M.norme_mtp("mtp.norm.weight", z, "deepseek", {}), z)
    assert torch.equal(M.norme_mtp("mtp.norm.weight", z, "qwen35", {"mtp_normes_decalees": True}), z)


def test_le_repli_auto_est_nomme_sur_la_ligne():
    from acvram.cli import repli_speculatif

    class Sans:
        mtp = None
        mtp_raison = "aucune tête dans le manifeste"

    class Avec:
        mtp = object()
    assert repli_speculatif(Avec()) == ("mtp", None)
    mode, repli = repli_speculatif(Sans())
    assert (mode, repli) == ("ngram", "mtp absent : aucune tête dans le manifeste")
    ligne = _speculation_texte({"mode": mode, "garde_active": True, "gain_moyen": None, "lot_max": 2, "repli": repli})
    assert ligne == " speculation=ngram(mtp absent : aucune tête dans le manifeste,on,lot_max=2)", ligne
    # sans repli, la ligne d'avant (aucune régression du format)
    assert _speculation_texte({"mode": "ngram", "garde_active": True, "gain_moyen": None, "lot_max": 2}) \
        == " speculation=ngram(on,lot_max=2)"


def test_le_chargeur_rend_la_raison(monkeypatch):
    from acvram.engine.loader import _charger_mtp
    assert _charger_mtp({"tensors": {}}, None, None, None, 128, torch.bfloat16, None, None, {}) == \
        ([], "aucune tête dans le manifeste")
    monkeypatch.setenv("ACVRAM_MTP", "non")
    assert _charger_mtp(_man(*QWEN35), None, None, None, 128, torch.bfloat16, None, None, {}) == ([], "ACVRAM_MTP=non")


def test_etat_cache_mtp_selon_la_convention(monkeypatch):
    """Pièce 105 : DeepSeek lit l'état BRUT (défaut, inchangé) ; Qwen3.5 l'état normalisé sous ACVRAM_MTP_ETAT=auto."""
    from acvram.engine.model import ACVRamModel

    class Faux:
        def __init__(self, convention):
            self.mtp = type("T", (), {"convention": convention})()
    f = ACVRamModel._mtp_normalise
    monkeypatch.delenv("ACVRAM_MTP_ETAT", raising=False)
    assert f(Faux("deepseek")) is False and f(Faux("qwen35")) is False      # défaut : brut pour tous
    monkeypatch.setenv("ACVRAM_MTP_ETAT", "auto")
    assert f(Faux("deepseek")) is False                                    # le chemin DeepSeek ne bouge pas
    assert f(Faux("qwen35")) is True
    monkeypatch.setenv("ACVRAM_MTP_ETAT", "norme")
    assert f(Faux("deepseek")) is True
