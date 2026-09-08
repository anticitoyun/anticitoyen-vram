"""Le format Q3N survit-il au flux des experts ?

Le décodage de Qwen3-Next en tout-q3n dégénère alors que le préremplissage est
sain, et les deux chemins de décodage divergent entre eux. Leur tronc commun
est la copie des experts : ``_emballer`` aplatit les tenseurs dans un tampon
uint8 épinglé, ``_decouper`` les rend par des vues, ``_rehydrate`` rebâtit
l'objet quantifié autour. Q3N a une géométrie qu'aucun autre format n'a —
charges en mots de trois octets, échelle de bloc en float8 rendue telle quelle
par ``state_dict`` là où NVFP4 la convertit en octets, échelle globale de forme
vide. Ce fichier fait l'aller-retour et compare la déquantification.

La comparaison est **exacte** : une copie d'octets qui change un seul bit du
résultat est un défaut, pas une tolérance.
"""
from __future__ import annotations

import pytest
import torch

from acvram.engine.layers import _decouper, _emballer, _rehydrate
from acvram.quant.q3n import dequantize_q3n, quantize_q3n

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")


def _expert(sortie: int = 512, entree: int = 2048, graine: int = 0):
    g = torch.Generator().manual_seed(graine)
    w = torch.randn(sortie, entree, generator=g) * 0.02
    return quantize_q3n(w)


def _identiques(a, b) -> None:
    """Les deux tenseurs quantifiés décrivent-ils exactement les mêmes poids ?"""
    assert a.shape == b.shape and a.block == b.block
    assert torch.equal(a.qweight, b.qweight.cpu()), "charges altérées"
    assert torch.equal(a.block_scale.view(torch.uint8),
                       b.block_scale.view(torch.uint8).cpu()), "échelles de bloc altérées"
    assert torch.equal(a.global_scale.cpu(), b.global_scale.cpu()), "échelle globale altérée"
    assert torch.equal(dequantize_q3n(a, torch.float32),
                       dequantize_q3n(b, torch.float32).cpu()), "déquantification altérée"


def test_aller_retour_hote():
    """``_emballer`` puis ``_decouper`` sans quitter l'hôte : rien ne doit bouger."""
    t = _expert()
    plat, decoupe = _emballer(t.state_dict())
    rebati = _rehydrate(t, _decouper(plat, decoupe))
    _identiques(t, rebati)


def test_dtypes_preserves_par_la_decoupe():
    """La découpe rend-elle les dtypes d'origine ?

    ``_rehydrate`` réinterprète ``block_scale`` en float8 ; si la découpe l'a
    déjà rendue en float8, la réinterprétation est sans effet et tout va bien.
    Si elle la rendait en octets, la réinterprétation la corrigerait. Les deux
    marchent — ce test fige lequel des deux est vrai, parce qu'un changement
    de ``state_dict`` casserait l'autre en silence.
    """
    t = _expert()
    _, decoupe = _emballer(t.state_dict())
    assert decoupe["qweight"][2] == torch.uint8
    assert decoupe["block_scale"][2] == torch.float8_e4m3fn
    assert decoupe["global_scale"][2] == torch.float32
    assert decoupe["global_scale"][1] == ()          # scalaire, forme vide


def test_alignement_des_decalages():
    """Chaque tenseur commence sur une frontière que sa réinterprétation admet."""
    t = _expert()
    _, decoupe = _emballer(t.state_dict())
    for cle, (off, _forme, dt, n) in decoupe.items():
        assert off % 256 == 0, cle
        assert n % dt.itemsize == 0, cle


@CUDA
def test_aller_retour_par_le_flux():
    """Le vrai chemin : tampon épinglé, copie asynchrone, double tampon."""
    from acvram.engine.layers import StreamedWeight

    dev = torch.device("cuda:0")
    t = _expert()
    sw = StreamedWeight(t.state_dict(), dev)
    for _ in range(4):                    # plusieurs tours : les deux tampons
        slot = sw.prefetch()
        rebati = _rehydrate(t, sw.wait(slot))
        _identiques(t, rebati)
        sw.release(slot)


@CUDA
def test_pool_sature_refuse_au_lieu_de_corrompre():
    """Plus de copies que d'emplacements : l'erreur doit etre dite, pas subie.

    Avant le 8/09/2026 le pool faisait tourner un curseur modulo ``n_slots`` et
    ne se protegeait que par l'evenement ``libre`` de l'emplacement vise. Tant
    qu'un emplacement n'a jamais ete rendu, cet evenement n'a jamais ete
    enregistre et l'attendre ne fait rien : les octets d'un expert ecrasaient
    ceux d'un autre, en silence. Mesure : dix experts, quatre emplacements,
    l'echelle globale relue appartenait a un voisin.
    """
    from acvram.engine.layers import ExpertPool, StreamedWeight

    dev = torch.device("cuda:0")
    pool = ExpertPool(dev, n_slots=4)
    experts = [_expert(graine=i) for i in range(10)]
    poids = [StreamedWeight(e.state_dict(), dev, pool=pool) for e in experts]

    with pytest.raises(RuntimeError, match="sature"):
        for p in poids:
            p.prefetch()


@CUDA
def test_couche_moe_entiere_dans_les_clous():
    """Le scenario de production : dix experts routes, trois lineaires chacun.

    ``n_slots = 2 x experts_par_jeton + 2`` (loader.py). Les projections
    d'entree partagent une geometrie et donc un jeu d'emplacements ; la
    projection de sortie a le sien. Toutes les copies sont lancees avant le
    premier calcul — c'est ce que fait le chemin de decodage a un jeton. Chaque
    expert doit se retrouver intact.
    """
    from acvram.engine.layers import ExpertPool, StreamedWeight

    dev = torch.device("cuda:0")
    top_k = 10
    pool = ExpertPool(dev, 2 * top_k + 2)
    # gate et up : [intermediaire, cache] ; down : [cache, intermediaire].
    entrees = [(_expert(512, 2048, graine=i), _expert(512, 2048, graine=100 + i))
               for i in range(top_k)]
    sorties = [_expert(2048, 512, graine=200 + i) for i in range(top_k)]

    flux = [(StreamedWeight(g.state_dict(), dev, pool=pool),
             StreamedWeight(u.state_dict(), dev, pool=pool),
             StreamedWeight(d.state_dict(), dev, pool=pool))
            for (g, u), d in zip(entrees, sorties)]

    slots = [(fg.prefetch(), fu.prefetch(), fd.prefetch()) for fg, fu, fd in flux]

    for (g, u), d, (fg, fu, fd), (sg, su, sd) in zip(entrees, sorties, flux, slots):
        _identiques(g, _rehydrate(g, fg.wait(sg)))
        _identiques(u, _rehydrate(u, fu.wait(su)))
        _identiques(d, _rehydrate(d, fd.wait(sd)))
        fg.release(sg); fu.release(su); fd.release(sd)


@CUDA
def test_l_echelle_memoisee_survit_au_transfert():
    """Un expert transfere ne doit pas relire son echelle globale par `.item()`.

    `global_scale_float()` memoise le scalaire parce que le relire synchronise
    le flux CUDA — cet appel pesait 62 % du temps de decodage au profil NVFP4,
    et une synchronisation rend le noyau incapturable dans un graphe. Mais
    `_rehydrate` construit un objet NEUF a chaque transfert : sans propagation
    du cache, la memoisation est annulee a chaque copie d'expert, soit une
    synchronisation par expert et par couche.

    La valeur est identique par construction — le tampon GPU est la copie du
    tenseur hote que le template decrit.
    """
    from acvram.engine.layers import StreamedWeight

    dev = torch.device("cuda:0")
    t = _expert()
    attendu = t.global_scale_float()          # remplit le cache du template
    assert "_gs_f" in t.__dict__

    sw = StreamedWeight(t.state_dict(), dev)
    slot = sw.prefetch()
    rebati = _rehydrate(t, sw.wait(slot))
    sw.release(slot)

    assert "_gs_f" in rebati.__dict__, \
        "cache perdu : le prochain GEMV synchronisera le flux"
    assert rebati.__dict__["_gs_f"] == attendu


def test_un_template_sans_cache_ne_fabrique_rien():
    """Et si le template n'a jamais lu son echelle, on n'invente pas de valeur.

    Un garde-fou qui remplit un cache avec autre chose que la vraie valeur
    serait pire que pas de cache du tout.
    """
    t = _expert()
    assert "_gs_f" not in t.__dict__          # jamais lu
    plat, decoupe = _emballer(t.state_dict())
    rebati = _rehydrate(t, _decouper(plat, decoupe))
    assert "_gs_f" not in rebati.__dict__
    # et la lecture reste correcte quand elle a lieu
    assert rebati.global_scale_float() == t.global_scale_float()
