"""L'épinglage hôte doit valoir la taille des poids, et ne pas s'accumuler.

Ces tests exigent une carte : `pin_memory()` passe par CUDA. Ils ne sont pas
lançables pendant qu'une mesure occupe la machine.

Ce qu'ils gardent, et pourquoi c'est un test plutôt qu'une relecture :
l'allocateur hôte épinglé de PyTorch **ne rend jamais au système** ce qu'il a
pris. Tout tenseur épinglé transitoirement reste donc verrouillé pour la vie
du processus — invisible à `Unevictable` comme à `Mlocked`, et donc invisible
à toute relecture qui s'appuierait sur ces compteurs. Seul l'écart entre
`reserved` et `allocated` de l'allocateur hôte le montre.
"""
import pytest
import torch

from acvram.engine.layers import StreamedWeight

besoin_carte = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="pin_memory() exige une carte")


def _epingle():
    """Mémoire hôte DMA-épinglée par toute la machine, en octets.

    `torch.cuda.host_memory_stats()` ne convient PAS ici, vérifié le 8/09/2026 :
    elle reste vide tant qu'aucun DMA n'a eu lieu, n'expose aucune clé
    `reserved`, et ses `allocated_bytes.current` / `active_bytes.current`
    restent égaux et ne redescendent jamais — ils ne séparent donc pas ce qui
    est référencé de ce qui dort en cache.

    `foll_pin` voit l'épinglage dès l'allocation, sans DMA (+520 Mio pour
    500 Mio épinglés), reste à 520 après `del` — le cache ne rend rien — et
    retombe à 8 après `torch._C._host_emptyCache()`.
    """
    acq = rel = 0
    for l in open("/proc/vmstat", encoding="utf-8"):
        if l.startswith("nr_foll_pin_acquired"):
            acq = int(l.split()[1])
        elif l.startswith("nr_foll_pin_released"):
            rel = int(l.split()[1])
    return max(0, acq - rel) * 4096


def _poids(n_tenseurs=8, mio=32):
    n = mio * 1024 * 1024 // 2
    return {f"t{i}": torch.empty(n, dtype=torch.bfloat16) for i in range(n_tenseurs)}


@besoin_carte
def test_la_source_n_est_pas_epinglee():
    """Un poids exilé épingle son tampon plat, et RIEN d'autre.

    Avant le correctif, chaque tenseur source était épinglé un à un avant
    l'emballage, puis déréférencé — et restait dans le cache de l'allocateur.
    Sur le témoin MoE à 30 couches : 46 Gio épinglés pour 33,75 attendus.
    """
    torch._C._host_emptyCache()
    avant = _epingle()
    poids = _poids()
    attendu = sum(t.numel() * t.element_size() for t in poids.values())

    sw = StreamedWeight(poids, torch.device("cpu"))
    pris = _epingle() - avant

    # Le tampon plat aligne chaque tenseur sur 256 octets : on tolère cet
    # écart, pas un facteur. Sans le correctif, `pris` valait ~2x `attendu`.
    assert pris <= attendu * 1.10, (
        f"{pris} octets épinglés pour {attendu} de poids : la source est "
        f"encore épinglée, ou une copie n'a pas été rendue")
    assert sw.plat.is_pinned(), "le tampon plat doit rester épinglé : c'est lui qui fait le DMA"


@besoin_carte
def test_l_epinglage_ne_s_accumule_pas_d_un_chargement_a_l_autre():
    """Deux poids successifs ne doivent pas doubler le verrouillage.

    Cas du service : `serve` change de modèle sans redémarrer. Si le cache
    épinglé retient un résidu à chaque chargement, il s'accumule pour la vie
    du processus et finit par prendre la machine — sans qu'aucun compteur
    système ne le montre.
    """
    torch._C._host_emptyCache()
    poids1 = _poids()
    sw1 = StreamedWeight(poids1, torch.device("cpu"))
    apres_1 = _epingle()
    del sw1, poids1

    poids2 = _poids()
    sw2 = StreamedWeight(poids2, torch.device("cpu"))
    apres_2 = _epingle()

    # Le second chargement doit réemployer le cache du premier, pas s'y ajouter.
    assert apres_2 <= apres_1 * 1.30, (
        f"réservé {apres_1} puis {apres_2} : le résidu épinglé s'accumule "
        f"d'un chargement à l'autre — problème de service, pas de mesure")
    assert sw2.plat.is_pinned()


@besoin_carte
def test_le_pool_supprime_l_arrondi_a_la_puissance_de_2():
    """Des tampons de 3 Mio coûtent 4 Mio chacun ; découpés dans une arène, 3.

    L'allocateur hôte arrondit chaque demande à la puissance de 2 supérieure.
    Nos experts font exactement 3,00 Mio, donc +33,3 % sur toute la mémoire
    verrouillée. Une arène dont la taille est une somme de puissances de 2 ne
    paie rien, et les tranches qu'on y prend restent épinglées.
    """
    from acvram.engine.layers import liberer_pool, reserver_pool

    n, mio = 64, 3
    utile = n * mio * 1024 * 1024          # 192 Mio = 128 + 64, exact

    def construire():
        return [StreamedWeight(_poids(n_tenseurs=1, mio=mio), torch.device("cpu"))
                for _ in range(n)]

    liberer_pool()
    avant = _epingle()
    sws = construire()
    sans_pool = _epingle() - avant
    del sws
    torch._C._host_emptyCache()

    avant = _epingle()
    pool = reserver_pool(utile)
    sws = construire()
    avec_pool = _epingle() - avant
    tranche_epinglee = sws[0].plat.is_pinned()
    del sws
    liberer_pool()

    assert pool is not None and pool.octets == utile, (
        f"l'arène devait faire {utile} octets, elle fait "
        f"{pool.octets if pool else 0} — la décomposition en puissances de 2 a échoué")
    assert tranche_epinglee, "une tranche prise dans l'arène doit rester épinglée"
    # Un surcoût FIXE d'environ 3,3 Mio accompagne la première allocation, quelle
    # que soit la taille : mesuré 3,2 / 3,5 / 3,3 Mio pour n = 64 / 128 / 256.
    # Il ne croît pas avec l'arène, donc il est négligeable à l'échelle réelle
    # (33,76 Gio) et ne doit pas faire échouer le test à petite échelle.
    assert avec_pool <= utile + 16 * 1024 * 1024, (
        f"avec arène : {avec_pool} octets épinglés pour {utile} utiles")
    assert avec_pool < sans_pool * 0.85, (
        f"sans arène {sans_pool}, avec arène {avec_pool} : le gain attendu est "
        f"d'un quart, il n'est pas là")
