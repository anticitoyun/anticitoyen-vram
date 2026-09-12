"""Le rembourrage d'un godet hybride ne doit pas corrompre l'état exporté.

Quand le lot est plus petit que le godet (b réel < b godet), les créneaux
de rembourrage doivent être liés à un identifiant sentinelle. Sans cela,
``_la_decode`` avance l'état du créneau non lié avec des entrées factices,
puis ``static_bind`` exporte cet état corrompu sous l'identifiant du
propriétaire précédent.
"""

import torch
import torch.nn as nn
from acvram.engine.model import DecoderLayerGDN, _STATIC
from acvram.engine.graphs import bucket_batch


# -- simulacre minimal d'attention linéaire ----------------------------------

class _GDNFictif(nn.Module):
    """Mutation en place visible : chaque decode_static additionne 1 à S."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.dim = dim

    def forward(self, x, etat):
        return x, etat

    def new_static(self, device):
        return {"conv": torch.zeros(1, self.dim), "S": torch.zeros(self.dim, self.dim)}

    @staticmethod
    def static_load(st, etat):
        if etat is None:
            st["conv"].zero_(); st["S"].zero_()
            return
        st["conv"].copy_(etat[0]); st["S"].copy_(etat[1])

    @staticmethod
    def static_export(st):
        return (st["conv"].clone(), st["S"].clone())

    def decode_static(self, x, st):
        st["S"].add_(1.0)
        st["conv"].add_(0.5)
        return x


class _NormIdentite(nn.Module):
    def forward(self, x):
        return x


def _creer_couche(dim=4):
    gdn = _GDNFictif(dim)
    norm = _NormIdentite()
    return DecoderLayerGDN(index=0, gdn=gdn, mlp=None,
                           input_norm=norm, post_norm=norm,
                           device=torch.device("cpu"))


# -- patron : lier → décoder → relire → vérifier ----------------------------

def _lier_sids(couche, store, sids, godet, max_len=128):
    """Appelle static_bind pour chaque créneau du godet, sentinelle pour le
    rembourrage — reproduit le correctif de ``_bind_hybrid``."""
    n = len(sids)
    for slot in range(godet):
        sid = sids[slot] if slot < n else -1
        couche.static_bind(slot, sid, store, max_len, torch.float32)


def _lier_sids_ancien(couche, store, sids, _godet, max_len=128):
    """Ancien code : n'itère que sur les séquences réelles."""
    for slot, sid in enumerate(sids):
        couche.static_bind(slot, sid, store, max_len, torch.float32)


def _simuler_decode(couche, godet, dim=4):
    """Simule _la_decode : avance les ``godet`` premiers créneaux."""
    x = torch.zeros(1, dim)
    for i in range(godet):
        couche.linear_attn.decode_static(x, couche.statics[i])


# -- tests -------------------------------------------------------------------

def test_godet_rembourre_preserve_etat():
    """L'état exporté d'une séquence évincée par le rembourrage est propre."""
    couche = _creer_couche()
    store = {}
    godet = 4

    _lier_sids(couche, store, [10, 20, 30, 40], godet)
    _simuler_decode(couche, godet)

    etat_40_propre = couche.linear_attn.static_export(couche.statics[3])

    _lier_sids(couche, store, [10, 20, 30], godet)
    _simuler_decode(couche, godet)

    _lier_sids(couche, store, [10, 20, 30, 50], godet)

    assert 40 in store and store[40] is not _STATIC
    S_exporte = store[40][1]
    assert torch.equal(S_exporte, etat_40_propre[1]), (
        f"état corrompu : attendu S={etat_40_propre[1][0,:2].tolist()}, "
        f"obtenu S={S_exporte[0,:2].tolist()}")


def test_ancien_code_corrompt():
    """Preuve que l'ancien code (sans sentinelle) corrompt l'état exporté."""
    couche = _creer_couche()
    store = {}
    godet = 4

    _lier_sids_ancien(couche, store, [10, 20, 30, 40], godet)
    _simuler_decode(couche, godet)

    etat_40_propre = couche.linear_attn.static_export(couche.statics[3])

    _lier_sids_ancien(couche, store, [10, 20, 30], godet)
    _simuler_decode(couche, godet)

    _lier_sids_ancien(couche, store, [10, 20, 30, 50], godet)

    assert 40 in store and store[40] is not _STATIC
    S_exporte = store[40][1]
    assert not torch.equal(S_exporte, etat_40_propre[1]), (
        "l'ancien code aurait dû corrompre mais ne l'a pas fait — le test "
        "est invalide")


def test_godet_rembourre_etat_zero():
    """Le créneau de rembourrage repart à zéro après liaison sentinelle."""
    couche = _creer_couche()
    store = {}
    godet = 4

    _lier_sids(couche, store, [10, 20, 30, 40], godet)
    _simuler_decode(couche, godet)

    _lier_sids(couche, store, [10, 20, 30], godet)

    S_padding = couche.statics[3]["S"]
    assert S_padding.abs().max() == 0.0, "le créneau sentinelle devrait être à zéro"


def test_deux_pas_stabilite():
    """Deux pas consécutifs à b < godet ne font pas dériver l'état."""
    couche = _creer_couche()
    store = {}
    godet = 4

    _lier_sids(couche, store, [10, 20, 30, 40], godet)
    _simuler_decode(couche, godet)

    for _ in range(3):
        _lier_sids(couche, store, [10, 20, 30], godet)
        _simuler_decode(couche, godet)

    _lier_sids(couche, store, [10, 20, 30, 40], godet)

    etat_40 = couche.linear_attn.static_export(couche.statics[3])
    S = etat_40[1]
    attendu = torch.ones_like(S)
    assert torch.equal(S, attendu), (
        f"après 3 passes fantômes, l'état de 40 devrait valoir 1 partout, "
        f"obtenu {S[0,:2].tolist()}")


def test_casse_si_on_remet_sids():
    """Méta-test : vérifier que le scénario CASSE avec enumerate(sids)."""
    couche = _creer_couche()
    store = {}
    godet = 4

    _lier_sids(couche, store, [10, 20, 30, 40], godet)
    _simuler_decode(couche, godet)

    _lier_sids_ancien(couche, store, [10, 20, 30], godet)
    _simuler_decode(couche, godet)

    _lier_sids_ancien(couche, store, [10, 20, 30, 50], godet)

    S_exporte = store[40][1]
    assert S_exporte.sum() > 4 * 4, (
        "avec l'ancien code la somme devrait avoir dérivé (2 passes) "
        f"mais vaut {S_exporte.sum()}")


# -- tests bucket_batch -----------------------------------------------------

def test_bucket_batch_puissances_de_deux():
    """bucket_batch arrondit au godet supérieur."""
    assert bucket_batch(1) == 1
    assert bucket_batch(2) == 2
    assert bucket_batch(3) == 4
    assert bucket_batch(4) == 4
    assert bucket_batch(5) == 8
    assert bucket_batch(7) == 8
    assert bucket_batch(9) == 16


def test_bucket_batch_identite_sur_puissance():
    """Une puissance de deux ne change pas."""
    for p in [1, 2, 4, 8, 16, 32]:
        assert bucket_batch(p) == p
