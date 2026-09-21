"""REPIN à chaud (bead anticitoyen-vram-pds, point 3 — Sage §4) : fonctions
pures de `memory/repin.py`, port de `tier.h`/`repin_pick` de colibrì. Aucune
carte, aucun `Engine` : ce sont des fonctions sur des dicts.
"""
from __future__ import annotations

import torch

from acvram.memory.repin import Echange, cadence_atteinte, choisir_echanges
from acvram.memory.table_adresses import adresse_expert


# --------------------------------------------------------------------------
# cadence_atteinte
# --------------------------------------------------------------------------

def test_cadence_pas_atteinte_avant_n():
    assert not cadence_atteinte(jetons_ecoules=63, n=64)


def test_cadence_atteinte_a_n():
    assert cadence_atteinte(jetons_ecoules=64, n=64)


def test_cadence_n_nul_desactive():
    # ACVRAM_REPIN=0 : jamais atteinte, même avec beaucoup de jetons écoulés
    # — sinon un n<=0 tournerait sans fin à chaque pas.
    assert not cadence_atteinte(jetons_ecoules=10_000, n=0)


# --------------------------------------------------------------------------
# choisir_echanges — hystérésis 25 % + 4
# --------------------------------------------------------------------------

def test_aucun_echange_si_rien_de_pinne():
    assert choisir_echanges({0: (set(), {0: 100, 1: 0})}) == []


def test_aucun_echange_si_aucun_candidat_hors_pin():
    # Tous les experts vus sont déjà pinnés : rien à faire entrer.
    assert choisir_echanges({0: ({0, 1}, {0: 100, 1: 50})}) == []


def test_echange_franchit_l_hysteresis():
    # Pinné le plus froid : expert 0 (heat 10). Non-pinné le plus chaud :
    # expert 5 (heat 100). Seuil = 10 + 10//4 + 4 = 16 ; 100 > 16 : échange.
    etat = {0: ({0, 1}, {0: 10, 1: 80, 5: 100})}
    r = choisir_echanges(etat)
    assert r == [Echange(couche=0, sortant=0, entrant=5, gain=90)]


def test_pas_d_echange_sous_l_hysteresis():
    # Seuil = 10 + 10//4 + 4 = 16 ; le plus chaud non pinné vaut 15 : reste
    # sous le seuil, pas de ping-pong pour un écart marginal.
    etat = {0: ({0}, {0: 10, 1: 15})}
    assert choisir_echanges(etat) == []


def test_pile_au_seuil_ne_bascule_pas():
    # Sage/colibrì : le test est `>`, pas `>=` — égal au seuil ne bascule pas
    # (tier.h:11, `tier_should_promote`).
    etat = {0: ({0}, {0: 10, 1: 16})}         # seuil exact = 16
    assert choisir_echanges(etat) == []


def test_budget_global_pas_par_couche():
    # Quatre couches ont chacune un échange qui franchit l'hystérésis ; avec
    # max_echanges=2, seules les DEUX au plus gros gain passent, toutes
    # couches confondues — pas un plafond par couche (repin_pick, colibri).
    etat = {
        c: ({0}, {0: 10, 1: 10 + gain + (10 + (10 >> 2) + 4)})
        for c, gain in enumerate([1, 50, 5, 30])
    }
    r = choisir_echanges(etat, max_echanges=2)
    assert [e.couche for e in r] == [1, 3]     # gains 50 puis 30, triés


def test_ne_modifie_pas_ses_entrees():
    pin = {0, 1}
    heat = {0: 10, 1: 80, 5: 100}
    choisir_echanges({0: (pin, heat)})
    assert pin == {0, 1}
    assert heat == {0: 10, 1: 80, 5: 100}


# --------------------------------------------------------------------------
# Engine._repin_pass — intégration légère, sans modèle réel
# --------------------------------------------------------------------------

def _bloc(n_experts: int, top_k: int):
    import torch
    from acvram.engine.model import MoEBlock
    return MoEBlock(torch.nn.Identity(),
                    [torch.nn.Identity() for _ in range(n_experts)], top_k)


def test_repin_pass_echange_et_journalise(capsys, monkeypatch):
    import torch
    from acvram.engine.runner import Engine

    bloc = _bloc(8, 2)
    bloc.index_couche = 0
    bloc._usage_routage = torch.tensor([10, 80, 5, 5, 5, 5, 5, 100])

    class FakeStats:
        decode_tokens = 64

    class FakeModel:
        def modules(self):
            return [bloc]

    monkeypatch.setenv("ACVRAM_REPIN", "64")
    eng = object.__new__(Engine)   # pas de modèle chargé : on teste _repin_pass seule
    eng.stats = FakeStats()
    eng.model = FakeModel()
    eng._pin = {0: {0, 1}}
    eng._dernier_repin = 0

    Engine._repin_pass(eng)

    sortie = capsys.readouterr().out
    assert "[REPIN] couche 0" in sortie
    assert "0 sort" in sortie and "7 entre" in sortie
    assert eng._pin[0] == {1, 7}          # 0 (froid, 10) sort, 7 (chaud, 100) entre
    assert eng._dernier_repin == 64


def test_repin_pass_sans_pin_ne_journalise_rien(capsys, monkeypatch):
    """`_pin` vide (point (2) pas câblé) : no-op de fait, pas de log — mais
    `_dernier_repin` avance quand même, pour ne pas re-tester la cadence à
    chaque jeton une fois la fenêtre passée."""
    from acvram.engine.runner import Engine

    class FakeStats:
        decode_tokens = 64

    class FakeModel:
        def modules(self):
            return []

    monkeypatch.setenv("ACVRAM_REPIN", "64")
    eng = object.__new__(Engine)
    eng.stats = FakeStats()
    eng.model = FakeModel()
    eng._pin = {}
    eng._dernier_repin = 0

    Engine._repin_pass(eng)

    assert capsys.readouterr().out == ""
    assert eng._dernier_repin == 64


def _lineaire_nvfp4(sortie, entree, graine):
    import torch
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4
    g = torch.Generator().manual_seed(graine)
    w = torch.randn(sortie, entree, generator=g) * 0.02
    return QuantLinear(quantize_nvfp4(w), out_features=sortie, in_features=entree)


# --------------------------------------------------------------------------
# _demote_expert / _promote_expert — aucune carte, CPU suffit (StreamedWeight
# fonctionne sans CUDA, voir layers.py)
# --------------------------------------------------------------------------

def test_demote_puis_promote_preserve_les_valeurs():
    from acvram.engine.runner import _demote_expert, _promote_expert
    from acvram.quant.nvfp4 import dequantize_nvfp4

    lin = _lineaire_nvfp4(32, 16, 1)
    ref = dequantize_nvfp4(lin.qweight, torch.float32)
    dev = torch.device("cpu")

    _demote_expert(lin, dev)
    assert lin.streamed is not None
    apres_demote = dequantize_nvfp4(lin.qweight, torch.float32)  # gabarit : 0 octet utile
    assert apres_demote.numel() == 0

    _promote_expert(lin, dev)
    assert lin.streamed is None
    apres_promote = dequantize_nvfp4(lin.qweight, torch.float32)
    assert torch.equal(apres_promote, ref)


def test_demote_libere_le_gros_tenseur_pas_seulement_le_nom():
    """Le gabarit après `_demote_expert` ne doit PAS être une vue sur
    l'ancien tenseur (même à 0 élément, une vue garderait tout le stockage
    original vivant — voir le commentaire de `_demote_expert`)."""
    from acvram.engine.runner import _demote_expert
    lin = _lineaire_nvfp4(32, 16, 2)
    ancien_stockage = lin.qweight.qweight.untyped_storage()
    _demote_expert(lin, torch.device("cpu"))
    assert lin.qweight.qweight.untyped_storage().data_ptr() != ancien_stockage.data_ptr()


def test_promote_reconstruit_meme_apres_plusieurs_cycles():
    """Deux allers-retours de suite : `_promote_expert` ne doit jamais
    dépendre d'un `qweight` de la toute première charge — le gabarit réduit
    de `_demote_expert` doit rester un gabarit VALIDE pour le cycle suivant."""
    from acvram.engine.runner import _demote_expert, _promote_expert
    from acvram.quant.nvfp4 import dequantize_nvfp4
    lin = _lineaire_nvfp4(16, 8, 3)
    ref = dequantize_nvfp4(lin.qweight, torch.float32)
    dev = torch.device("cpu")
    for _ in range(3):
        _demote_expert(lin, dev)
        _promote_expert(lin, dev)
    assert torch.equal(dequantize_nvfp4(lin.qweight, torch.float32), ref)


def test_adresse_expert_change_de_regime_pendant_le_froid():
    """Pendant le froid, `adresse_expert` lit le tampon épinglé — une adresse
    différente de la résidence d'origine. (Après re-promotion, l'allocateur
    peut légitimement réutiliser la même adresse que libère la démotion :
    ce n'est pas vérifié ici, ce serait vérifier l'allocateur, pas le code.)
    """
    from acvram.engine.runner import _demote_expert, _promote_expert
    lin = _lineaire_nvfp4(16, 8, 4)
    dev = torch.device("cpu")
    avant = adresse_expert(lin, "qweight")
    _demote_expert(lin, dev)
    pendant = adresse_expert(lin, "qweight")           # lit streamed.host
    assert pendant != avant
    _promote_expert(lin, dev)
    assert lin.streamed is None                        # re-résident, plus streamé


def test_repin_pass_avant_la_cadence_ne_fait_rien(monkeypatch):
    from acvram.engine.runner import Engine

    class FakeStats:
        decode_tokens = 10          # < 64

    eng = object.__new__(Engine)
    eng.stats = FakeStats()
    eng.model = None
    eng._pin = {0: {0}}
    eng._dernier_repin = 0
    monkeypatch.setenv("ACVRAM_REPIN", "64")

    Engine._repin_pass(eng)          # ne doit pas toucher a `.model` (None)

    assert eng._dernier_repin == 0   # inchange : la cadence n'est pas atteinte
