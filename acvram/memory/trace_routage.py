# SPDX-FileCopyrightText: 2026 Anticitoyen
# SPDX-License-Identifier: Apache-2.0
"""Journal des experts routés, jeton par jeton et couche par couche.

Pourquoi ce fichier existe
--------------------------

Le planificateur décide où loger les experts d'un modèle creux, et il le fait
sans jamais savoir lesquels sont réellement demandés. `cached_expert_fraction`
n'est aujourd'hui qu'un rapport de capacité — combien d'octets tiennent dans le
cache, divisé par combien il y en a — c'est-à-dire le taux de succès qu'on
obtiendrait si le routage était uniforme. Il ne l'est pas : c'est toute la
raison d'être d'un cache.

Aucun taux de succès mesuré n'existe pour une pile à 512 experts routés 10.
Sans trace, tout cache est réglé sur une hypothèse.

Ce que ce journal enregistre
----------------------------

Une ligne par jeton et par couche : le rang du jeton, l'index de la couche, et
les experts choisis. **Dans l'ordre d'émission**, parce qu'un cache se juge sur
une suite, pas sur des comptes : savoir qu'un expert est demandé 8 % du temps
ne dit pas s'il l'est en rafale ou dispersé, et ces deux régimes ne donnent pas
le même taux de succès.

Le format est du texte compact, une ligne par entrée, relisible sans
bibliothèque :

    <jeton> <couche> <expert>,<expert>,...

Ce qu'il coûte quand il est éteint
----------------------------------

Un test de booléen par appel, et rien d'autre. La fonction sort avant de
toucher aux tenseurs. **Elle ne doit jamais provoquer de synchronisation
implicite** : un `.tolist()` sur un tenseur de carte force l'hôte à attendre le
calcul, ce qui fausserait à la fois la mesure de débit et celle d'énergie. Le
transfert est donc explicitement non bloquant, et la conversion différée à
l'écriture.

Activation
----------

    ACVRAM_TRACE_ROUTAGE=/chemin/du/journal.txt

Absente, tout est éteint. Le fichier est ouvert à la première écriture et
fermé par `fermer()` ou à la fin du processus.
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import Optional

__all__ = ["actif", "noter", "fermer", "chemin", "poser_modalites", "modalites_du_lot", "relire_modalites", "taux_de_succes",
           "taux_de_succes_par_couche", "taux_de_succes_pin"]

_CHEMIN: Optional[str] = os.environ.get("ACVRAM_TRACE_ROUTAGE") or None
_ACTIF: bool = _CHEMIN is not None
_FICHIER = None
_VERROU = threading.Lock()
_JETON = 0
_BASE: Optional[int] = None


def actif() -> bool:
    """Vrai si la trace est demandée. Un test de booléen, rien de plus."""
    return _ACTIF


def chemin() -> Optional[str]:
    return _CHEMIN


def _fichier():
    global _FICHIER
    if _FICHIER is None:
        # Tampon de 1 Mio : une trace fait des centaines de milliers de lignes,
        # et une écriture par ligne coûterait plus que le routage lui-même.
        _FICHIER = open(_CHEMIN, "w", buffering=1 << 20)
        _FICHIER.write("# jeton couche [modalite] experts[:poids]  (v2 : modalite et poids si le lot les porte)\n")
        atexit.register(fermer)
    return _FICHIER


# Pièce 27(a) : modalité par position de la passe courante — "t" (texte),
# "i" (image), "s" (spécial/autre), posée par `ACVRamModel.forward` depuis le
# lot RÉELLEMENT fourni (jetons image du processeur) quand la trace est
# active, effacée à la fin de la passe. None : trace v1, modalité inconnue.
_MODALITES: Optional[list] = None


def poser_modalites(modalites) -> None:
    """Le masque de modalité de la passe (une lettre par position du lot), ou
    None. Ne fait RIEN hors trace : coût nul quand personne ne trace."""
    global _MODALITES
    if not _ACTIF:
        return
    _MODALITES = list(modalites) if modalites is not None else None


def modalites_du_lot(tokens, image_token_id=None, plages=None) -> list:
    """Une lettre par position : "i" si la position est un jeton image (id du
    processeur, ou dans une plage [début, fin) rendue par le lot), "s" si
    l identifiant est un jeton spécial connu du lot (négatif ou hors vocab),
    "t" sinon. Pur, testable sans modèle."""
    ids = [int(x) for x in (tokens.tolist() if hasattr(tokens, "tolist") else tokens)]
    dans = [False] * len(ids)
    for d, f in (plages or []):
        for i in range(max(0, int(d)), min(len(ids), int(f))):
            dans[i] = True
    out = []
    for i, x in enumerate(ids):
        if dans[i] or (image_token_id is not None and x == int(image_token_id)):
            out.append("i")
        elif x < 0:
            out.append("s")
        else:
            out.append("t")
    return out


def noter(couche: int, indices, poids=None) -> None:
    """Enregistre les experts routés d'une couche pour le jeton courant.

    ``indices`` est le tenseur des index d'experts, de forme [jetons, top_k].
    Il est ramené sur l'hôte ici : cette fonction n'est appelée que sous
    trace, et le coût du transfert est le prix de la mesure. Hors trace, on
    n'arrive jamais ici.
    """
    if not _ACTIF:
        return
    try:
        # .tolist() synchronise. C'est assumé SOUS TRACE et seulement là : une
        # trace prise sans synchroniser mélangerait les couches, ce qui est
        # exactement ce qu'on veut mesurer.
        lignes = indices.detach().to("cpu").tolist()
    except Exception:                                   # noqa: BLE001
        return
    if lignes and not isinstance(lignes[0], list):
        lignes = [lignes]
    global _JETON, _BASE
    with _VERROU:
        f = _fichier()
        # Toutes les couches d'un même passage portent les MÊMES numéros de
        # jeton : la base est figée à la première couche et relue par les
        # suivantes. Une première version faisait avancer le compteur puis
        # repartait de sa valeur courante, ce qui numérotait la couche 1 en
        # 3, 4, 5 là où la couche 0 disait 0, 1, 2 — les mêmes jetons sous
        # deux noms, et un rejeu qui aurait cru voir deux fois plus de trafic.
        if couche == 0 or _BASE is None:
            _BASE = _JETON
            _JETON += len(lignes)
        base = _BASE
        pw = None
        if poids is not None:
            try:
                pw = poids.detach().to("cpu").tolist()
                if pw and not isinstance(pw[0], list):
                    pw = [pw]
            except Exception:                           # noqa: BLE001
                pw = None
        for i, experts in enumerate(lignes):
            # v1 : `jeton couche e1,e2,…` ; v2 (pièce 27) : `jeton couche m e1:w1,…`
            # — `relire` rend les deux, `relire_modalites` n accepte que v2.
            m = (_MODALITES[i] if _MODALITES is not None and i < len(_MODALITES) else None)
            corps = (",".join(f"{int(e)}:{pw[i][j]:.6g}" for j, e in enumerate(experts))
                     if pw is not None and i < len(pw) else ",".join(str(int(e)) for e in experts))
            f.write(f"{base + i} {couche} " + (f"{m} " if m else "") + corps + "\n")


def fermer() -> None:
    global _FICHIER
    with _VERROU:
        if _FICHIER is not None:
            _FICHIER.close()
            _FICHIER = None


def relire(chemin_journal: str):
    """Relit une trace, dans l'ordre d'émission.

    Rend des triplets ``(jeton, couche, [experts])``. L'ordre du fichier est
    l'ordre d'émission : c'est lui qui porte l'information, et le relecteur ne
    doit pas le trier.
    """
    with open(chemin_journal) as f:
        for ligne in f:
            if ligne.startswith("#") or not ligne.strip():
                continue
            a, b, c = ligne.split(None, 2)
            c = c.strip()
            if c[:1] in ("t", "i", "s") and (len(c) == 1 or c[1] == " "):
                c = c[2:].strip() if len(c) > 1 else ""      # v2 : la modalité précède les experts
            yield int(a), int(b), [int(e.split(":")[0]) for e in c.split(",") if e]


def relire_modalites(chemin_journal: str):
    """Trace v2 : ``(jeton, couche, modalité, [(expert, poids)])`` — poids de
    porte quand la trace les porte, 0.0 sinon. Lève sur une trace v1 : une
    table par modalité ne se reconstruit pas d une trace qui ne la porte pas
    (pièce 27(a) : jamais un chiffre inventé)."""
    with open(chemin_journal) as f:
        for n, ligne in enumerate(f):
            if ligne.startswith("#") or not ligne.strip():
                continue
            a, b, c = ligne.split(None, 2)
            c = c.strip()
            if not (c[:1] in ("t", "i", "s") and len(c) > 1 and c[1] == " "):
                raise ValueError(f"{chemin_journal}:{n + 1} : trace v1 (sans modalité) — "
                                 f"rejouer sous ACVRAM_TRACE_ROUTAGE avec un lot multimodal")
            m, reste = c[0], c[2:].strip()
            paires = []
            for e in reste.split(","):
                if not e:
                    continue
                t = e.split(":")
                paires.append((int(t[0]), float(t[1]) if len(t) > 1 else 0.0))
            yield int(a), int(b), m, paires


def taux_de_succes(chemin_journal: str, capacite: int,
                   politique: str = "lru") -> dict:
    """Rejoue la trace à travers un cache de ``capacite`` experts.

    C'est le chiffre que personne n'a publié pour 512 experts routés 10, et
    que le planificateur suppose aujourd'hui égal au rapport de capacité.

    ``politique`` : ``lru`` (le moins récemment servi sort) ou ``lfu`` (le
    moins fréquemment servi sort). Les deux sont fournies parce que le choix
    n'est pas évident : un routage en rafale favorise la première, un routage
    à experts chauds stables la seconde. **La trace tranchera, pas nous.**
    """
    from collections import OrderedDict, Counter
    cache: OrderedDict = OrderedDict()
    freq: Counter = Counter()
    succes = demandes = 0
    par_couche: dict[int, list[int]] = {}
    for _, couche, experts in relire(chemin_journal):
        s = d = 0
        for e in experts:
            cle = (couche, e)
            d += 1
            if cle in cache:
                s += 1
                cache.move_to_end(cle)
            else:
                if len(cache) >= capacite:
                    if politique == "lfu":
                        victime = min(cache, key=lambda k: freq[k])
                        cache.pop(victime)
                    else:
                        cache.popitem(last=False)
                cache[cle] = True
            freq[cle] += 1
        succes += s
        demandes += d
        acc = par_couche.setdefault(couche, [0, 0])
        acc[0] += s
        acc[1] += d
    return {
        "politique": politique,
        "capacite": capacite,
        "demandes": demandes,
        "succes": succes,
        "taux": succes / demandes if demandes else 0.0,
        "taux_par_couche": {c: (s / d if d else 0.0)
                            for c, (s, d) in sorted(par_couche.items())},
    }


def taux_de_succes_par_couche(chemin_journal: str, capacite: int,
                              politique: str = "lru") -> dict:
    """Comme `taux_de_succes`, mais un cache de ``capacite`` experts PAR
    COUCHE — chacune la sienne, pas un pool partagé entre toutes.

    C'est la capacité qui répond à une question de placement (bead jt5,
    revue/poste7-cache-experts-13-09.md §5, point manquant §3) : « si CETTE
    couche a C emplacements chauds, quel taux obtient-elle ? », pas « si tout
    le modèle partage un même réservoir de C·L emplacements ». Les deux ne
    coïncident pas : une couche à routage concentré et une à routage plat se
    disputeraient le même pool dans `taux_de_succes`.
    """
    from collections import OrderedDict, Counter
    caches: dict[int, OrderedDict] = {}
    freqs: dict[int, Counter] = {}
    succes = demandes = 0
    par_couche: dict[int, list[int]] = {}
    for _, couche, experts in relire(chemin_journal):
        cache = caches.setdefault(couche, OrderedDict())
        freq = freqs.setdefault(couche, Counter())
        s = d = 0
        for e in experts:
            d += 1
            if e in cache:
                s += 1
                cache.move_to_end(e)
            else:
                if len(cache) >= capacite:
                    if politique == "lfu":
                        victime = min(cache, key=lambda k: freq[k])
                        cache.pop(victime)
                    else:
                        cache.popitem(last=False)
                cache[e] = True
            freq[e] += 1
        succes += s
        demandes += d
        acc = par_couche.setdefault(couche, [0, 0])
        acc[0] += s
        acc[1] += d
    return {
        "politique": politique,
        "capacite": capacite,
        "demandes": demandes,
        "succes": succes,
        "taux": succes / demandes if demandes else 0.0,
        "taux_par_couche": {c: (s / d if d else 0.0)
                            for c, (s, d) in sorted(par_couche.items())},
    }


def taux_de_succes_pin(chemin_journal: str, capacites: list, entrainement: float = 0.5) -> dict:
    """Politique PIN : les ``capacite`` experts les plus demandés PAR COUCHE,
    appris sur la PREMIÈRE fraction ``entrainement`` de la trace (dans l'ordre
    d'émission des jetons), évalués sur le RESTE — jamais sur la moitié qui
    l'a appris (poste7 §5 : « jamais sur la moitié qui l'a appris »).

    C'est la politique d'`AUTOPIN` de colibrì (`.coli_usage`, épinglage figé au
    démarrage) — par opposition à `taux_de_succes_par_couche` qui est la LRU
    ou LFU de colibrì (`REPIN`, adaptative en cours de service). Un histogramme
    appris et évalué sur la MÊME moitié dirait « ce cache aurait marché sur les
    données qui l'ont construit », ce qui est vrai de n'importe quel cache assez
    grand — pas une mesure, une tautologie.

    Rend ``{capacite: {"taux": ..., "taux_par_couche": {...}}}`` pour chaque
    capacité de ``capacites`` — une seule lecture d'entraînement et une seule
    lecture d'évaluation, quel que soit le nombre de capacités demandées.
    """
    from collections import Counter

    jeton_min = jeton_max = None
    for jeton, _, _ in relire(chemin_journal):
        if jeton_min is None:
            jeton_min = jeton_max = jeton
        elif jeton > jeton_max:
            jeton_max = jeton
        elif jeton < jeton_min:
            jeton_min = jeton
    if jeton_min is None:
        vide = {"taux": 0.0, "taux_par_couche": {}}
        return {c: dict(vide) for c in capacites}
    seuil = jeton_min + (jeton_max - jeton_min) * entrainement

    freq: dict[int, Counter] = {}
    for jeton, couche, experts in relire(chemin_journal):
        if jeton > seuil:
            continue
        c = freq.setdefault(couche, Counter())
        for e in experts:
            c[e] += 1

    pins = {capacite: {couche: {e for e, _ in c.most_common(capacite)}
                       for couche, c in freq.items()}
           for capacite in capacites}

    resultats = {capacite: {"succes": 0, "demandes": 0, "par_couche": {}}
                for capacite in capacites}
    for jeton, couche, experts in relire(chemin_journal):
        if jeton <= seuil:
            continue
        for capacite in capacites:
            r = resultats[capacite]
            pin_couche = pins[capacite].get(couche, set())
            acc = r["par_couche"].setdefault(couche, [0, 0])
            for e in experts:
                s = 1 if e in pin_couche else 0
                r["succes"] += s
                r["demandes"] += 1
                acc[0] += s
                acc[1] += 1

    out = {}
    for capacite in capacites:
        r = resultats[capacite]
        out[capacite] = {
            "taux": r["succes"] / r["demandes"] if r["demandes"] else 0.0,
            "taux_par_couche": {c: (s / d if d else 0.0)
                                for c, (s, d) in sorted(r["par_couche"].items())},
        }
    return out


# ---------------------------------------------------------------------------
# M1 étendu (poste7-c9-119b-cache-experts-19-09 § 2) : ce que la trace doit
# rendre pour décider si un cache d'experts EXISTE avant d'écrire du code.
# ---------------------------------------------------------------------------

def pas_de_decodage(chemin_journal: str):
    """Regroupe la trace en PAS : une rafale de lignes consécutives de même
    couche = un pas de décodage (à b=12, `noter` reçoit [B, top_k] d'un coup,
    donc B lignes de suite pour la couche). Rend ``(couche, [experts par
    jeton])`` par pas, dans l'ordre d'émission. À b=1 un pas = un jeton."""
    couche_courante = None
    jeton_prec = None
    rafale: list = []
    for jeton, couche, experts in relire(chemin_journal):
        # nouveau pas : la couche change, ou le rang de jeton ne croît plus
        # (la couche suivante du même pas réécrit les mêmes rangs ; le pas
        # suivant repart à base + i). Une trace à UNE seule couche ne sépare
        # pas ses pas à b=1 : les rangs y croissent sans fin — tracer toutes
        # les couches, comme `noter` le fait en service.
        if rafale and (couche != couche_courante or (jeton_prec is not None and jeton <= jeton_prec)):
            yield couche_courante, rafale
            rafale = []
        couche_courante = couche
        jeton_prec = jeton
        rafale.append(experts)
    if rafale:
        yield couche_courante, rafale


def distincts_par_pas(chemin_journal: str) -> dict:
    """Experts DISTINCTS demandés par pas et par couche — c'est ce qui se
    paie en octets PCIe (un expert demandé par trois jetons du lot se lit une
    fois), pas la somme des top_k. Rend par couche : moyenne, médiane, max
    des distincts, la taille de lot moyenne, et le rapport distincts / (lot ×
    top_k) qui dit la part de recouvrement dans le lot."""
    import statistics
    par_couche: dict[int, list] = {}
    lots: dict[int, list] = {}
    for couche, rafale in pas_de_decodage(chemin_journal):
        u = set()
        for experts in rafale:
            u.update(experts)
        par_couche.setdefault(couche, []).append(len(u))
        lots.setdefault(couche, []).append((len(rafale), sum(len(e) for e in rafale)))
    out = {}
    for couche, d in par_couche.items():
        demandes = sum(t for _, t in lots[couche])
        out[couche] = {"pas": len(d), "moyenne": statistics.fmean(d), "mediane": statistics.median(d),
                       "max": max(d), "lot_moyen": statistics.fmean(b for b, _ in lots[couche]),
                       "recouvrement": 1.0 - (sum(d) / demandes if demandes else 0.0)}
    return out


def taux_de_succes_lru_juge(chemin_journal: str, capacite: int, entrainement: float = 0.5,
                            par_pas: bool = True) -> dict:
    """LRU PAR COUCHE de ``capacite`` experts, chauffée sur la première
    fraction ``entrainement`` des jetons et JUGÉE sur le reste seulement —
    le même partage que `taux_de_succes_pin`, pour comparer h_lru et h_pin
    sur les mêmes jetons. ``par_pas`` : un expert demandé par plusieurs jetons
    du même pas compte une demande (celle qui coûte des octets)."""
    from collections import OrderedDict
    jetons = [j for j, _, _ in relire(chemin_journal)]
    if not jetons:
        return {"taux": 0.0, "taux_par_couche": {}, "capacite": capacite}
    seuil = min(jetons) + (max(jetons) - min(jetons)) * entrainement
    caches: dict[int, OrderedDict] = {}
    succes = demandes = 0
    par_couche: dict[int, list] = {}
    # relecture par pas, en gardant le rang du premier jeton du pas
    rang = 0
    ordre = list(relire(chemin_journal))
    i = 0
    while i < len(ordre):
        jeton, couche, _ = ordre[i]
        j = i + 1
        while j < len(ordre) and ordre[j][1] == couche and ordre[j][0] > ordre[j - 1][0]:
            j += 1
        rafale = [e for _, _, e in ordre[i:j]]
        i = j
        cache = caches.setdefault(couche, OrderedDict())
        vus = set() if par_pas else None
        for experts in rafale:
            for e in experts:
                if vus is not None:
                    if e in vus:
                        continue
                    vus.add(e)
                hit = e in cache
                if hit:
                    cache.move_to_end(e)
                else:
                    cache[e] = True
                    if len(cache) > capacite:
                        cache.popitem(last=False)
                if jeton > seuil:
                    acc = par_couche.setdefault(couche, [0, 0])
                    succes += hit; demandes += 1
                    acc[0] += hit; acc[1] += 1
    return {"capacite": capacite, "taux": succes / demandes if demandes else 0.0,
            "demandes_jugees": demandes,
            "taux_par_couche": {c: (a / b if b else 0.0) for c, (a, b) in par_couche.items()}}


def rapport_m1(chemin_journal: str, capacites=(16, 32, 48, 64, 96), entrainement: float = 0.5,
               nb_experts: Optional[int] = None) -> dict:
    """M1 : h_pin(C) et h_lru(C) par couche, apprises/chauffées sur la
    première moitié et jugées sur la seconde, distincts par pas, et le
    critère de poste7 : Δh = h(E/2) − 0,5 (E = nombre d'experts vus) — < 0,10
    → pas de cache apprenant ; ≥ 0,25 → cache engagé. Un seul fichier lu."""
    E = 0
    n_lignes = 0
    for _, _, experts in relire(chemin_journal):
        n_lignes += 1
        if experts:
            E = max(E, max(experts) + 1)
    if nb_experts:
        E = max(E, int(nb_experts))            # E du config.json, pas « le plus grand index vu »
    pin = taux_de_succes_pin(chemin_journal, list(capacites), entrainement)
    lru = {c: taux_de_succes_lru_juge(chemin_journal, c, entrainement) for c in capacites}
    c_demi = max(1, E // 2)
    h_demi = {"pin": taux_de_succes_pin(chemin_journal, [c_demi], entrainement)[c_demi]["taux"],
              "lru": taux_de_succes_lru_juge(chemin_journal, c_demi, entrainement)["taux"]}
    meilleur = max(h_demi.values())
    return {"experts_vus": E, "lignes": n_lignes, "capacites": list(capacites),
            "h_pin": {c: pin[c]["taux"] for c in capacites},
            "h_lru": {c: lru[c]["taux"] for c in capacites},
            "h_pin_par_couche": {c: pin[c]["taux_par_couche"] for c in capacites},
            "h_lru_par_couche": {c: lru[c]["taux_par_couche"] for c in capacites},
            "distincts_par_pas": distincts_par_pas(chemin_journal),
            "capacite_demi": c_demi, "h_demi": h_demi, "delta_h": meilleur - 0.5,
            "verdict_poste7": ("cache engagé" if meilleur - 0.5 >= 0.25 else
                             "pas de cache apprenant" if meilleur - 0.5 < 0.10 else "bande 0,10-0,25 : poste7 tranche")}
