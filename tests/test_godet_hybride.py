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
from acvram.engine.graphs import GraphRunner, bucket_batch, godet_hybride


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


# -- tests godet_hybride (bug du 13/09, bead anticitoyen-vram-x0s) ----------
#
# max_slots=12 (ACVRAM_HYBRID_SLOTS=12, notre réglage de campagne) n'est pas
# une puissance de deux. Avant le correctif, `bucket_batch(9..12) = 16 > 12`
# faisait tomber ces lots en eager en permanence : aucun test ne l'avait
# jamais couvert, parce qu'aucune campagne n'avait jamais fait tourner un lot
# concurrent RÉEL de 9 à 12 séquences sous ce réglage (les campagnes
# « slots=12 » antérieures généraient une seule séquence par tour).


def test_godet_hybride_lots_9_a_15_avec_max_slots_12():
    """Épreuve directe du bug : b_reel de 9 à 12 doit rester rejouable."""
    for b_reel in range(9, 13):
        godet = godet_hybride(b_reel, max_slots=12)
        assert godet is not None, (
            f"b_reel={b_reel} <= max_slots=12 : ne doit JAMAIS refuser le "
            "graphe (c'est exactement le bug du 13/09)")
        assert godet <= 12, f"le godet {godet} dépasse le plafond des tampons"
        assert godet >= b_reel, "le godet doit pouvoir contenir b_reel séquences"

    for b_reel in range(13, 16):
        assert godet_hybride(b_reel, max_slots=12) is None, (
            f"b_reel={b_reel} > max_slots=12 : refus attendu, "
            "les tampons ne le contiennent pas")


def test_godet_hybride_egal_au_bucket_sous_la_puissance_de_deux():
    """En dessous de la puissance de deux qui précède max_slots, inchangé."""
    for b_reel in range(1, 9):
        assert godet_hybride(b_reel, max_slots=12) == bucket_batch(b_reel)


def test_godet_hybride_max_slots_puissance_de_deux_inchange():
    """Avec max_slots déjà une puissance de deux, le comportement d'avant
    le correctif est préservé au bit près (pas de régression)."""
    for b_reel in range(1, 17):
        attendu = bucket_batch(b_reel) if bucket_batch(b_reel) <= 16 else None
        assert godet_hybride(b_reel, max_slots=16) == attendu


class _FauxModeleDtype:
    dtype = torch.float32


class _FauxRunnerHybride(GraphRunner):
    """Assez de surface pour appeler `_bind_hybrid` sans modèle réel."""

    def __init__(self, hybrid_layers, max_model_len=128):
        self.hybrid_layers = hybrid_layers
        self.model = _FauxModeleDtype()
        self.max_model_len = max_model_len


class _FauxBatchHybride:
    def __init__(self, seq_ids, gdn_store):
        self.seq_ids = seq_ids
        self.batch_size = len(seq_ids)
        self.gdn_store = gdn_store


def test_sentinelle_distincte_par_creneau_de_rembourrage():
    """Bug trouvé en écrivant cette épreuve (13/09, bead x0s) : `_bind_hybrid`
    donnait le MÊME sid (-1) à tous les créneaux de rembourrage. `static_bind`
    suppose qu'un sid ne vit que dans un seul créneau — le second appel avec
    sid=-1 croit que « -1 vit déjà ailleurs », exporte le premier créneau
    (perd son propriétaire, `static_owners -> None`) au lieu de lui laisser
    un état neuf à zéro. Rare tant qu'un seul créneau de rembourrage existait
    par pas ; devenu fréquent depuis `godet_hybride`, qui autorise plusieurs
    créneaux de rembourrage simultanés (b_reel=9, max_slots=12 : trois).
    Corrigé : un sid distinct par créneau de rembourrage (-1, -2, -3, ...).
    """
    couche = _creer_couche()
    sids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # b_reel = 10
    godet = godet_hybride(len(sids), max_slots=12)
    assert godet == 12                                 # trois creneaux de rembourrage
    runner = _FauxRunnerHybride([couche])
    batch = _FauxBatchHybride(sids, {})
    runner._bind_hybrid(batch, lb=0, b_godet=godet)

    for slot in range(len(sids)):
        assert couche.static_owners[slot] == sids[slot], (
            f"créneau réel {slot} : sid attendu {sids[slot]}, "
            f"obtenu {couche.static_owners[slot]}")
    rembourrage = couche.static_owners[len(sids):godet]
    assert rembourrage == [-1, -2], (
        f"sentinelles de rembourrage attendues [-1, -2], obtenu {rembourrage}")
    assert len(set(rembourrage)) == len(rembourrage), (
        "deux créneaux de rembourrage partagent le même sid — régression du bug")


def test_godet_hybride_ancien_code_aurait_refuse():
    """Méta-test : preuve que l'ancien calcul (bucket_batch(b) > max_slots
    sans plafonnement) refusait bien ces lots — sinon ce test ne prouve rien."""
    for b_reel in range(9, 13):
        ancien_refus = bucket_batch(b_reel) > 12
        assert ancien_refus, (
            f"b_reel={b_reel} : l'ancien calcul aurait dû refuser "
            "(bucket_batch > max_slots) — le scénario du bug est invalide")
