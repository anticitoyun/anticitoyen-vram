# SPDX-FileCopyrightText: 2026 Anticitoyen
# SPDX-License-Identifier: Apache-2.0
"""REPIN à chaud : entre deux pas, échanger les experts pinnés les plus
froids contre les non-pinnés les plus chauds (bead anticitoyen-vram-pds,
point (3) — Sage §4).

Distinct de l'AUTOPIN (`memory/expert_usage.choisir_residents`) : l'AUTOPIN
décide au chargement, sur un historique persistant et cumulatif
(`.acvram_usage`) ; REPIN décide EN COURS DE SERVICE, sur la chaleur récente,
pour suivre un changement de charge que l'historique cumulatif — qui ne fait
que grossir — ne peut plus refléter. Les deux coexistent, comme chez colibrì
(`.coli_usage`/`PIN=auto` vs `REPIN=n`, `revue/colibri-lecture-code.md`).

Les fonctions ci-dessous sont un port DIRECT de `tier.h` de colibrì
(`tier_lfru_score`, `tier_should_promote`), pures et testables sans carte.
Limite honnête, à corriger plus tard : `tier_lfru_score` prend une RÉCENCE
(dernier accès) que nous ne suivons pas encore par expert — seul
`MoEBlock._usage_routage` (un compte cumulatif, jamais décroissant) existe
côté acvram aujourd'hui. En son absence, `recence=0` partout dégrade la
politique en pur LFU (fréquence seule) : moins bon que le LFRU de colibrì,
mais un ordre de grandeur au-dessus de ne rien faire, et honnêtement étiqueté
comme tel plutôt que déguisé en LFRU complet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["Echange", "choisir_echanges", "cadence_atteinte"]

# 25 % + 4 : hystérésis anti-ping-pong de colibrì (tier.h:9-12,
# `tier_should_promote` — "widen before adding the margin"), reprise telle
# quelle. Rien ne nous dit qu'un autre réglage vaudrait mieux, et en inventer
# un ferait passer une supposition pour un choix mesuré.
_MARGE_RELATIVE = 0.25
_MARGE_FIXE = 4


def _devrait_promouvoir(chaud: int, froid: int) -> bool:
    """Port de `tier_should_promote` (tier.h:9-12) : élargir la marge avant
    de l'ajouter — un compteur saturé doit devenir collant, pas déborder le
    seuil et admettre un candidat plus froid par débordement."""
    seuil = froid + (froid >> 2) + _MARGE_FIXE
    return chaud > seuil


@dataclass(frozen=True)
class Echange:
    couche: int
    sortant: int          # expert pinné qui quitte la résidence
    entrant: int          # expert non pinné qui la prend
    gain: int             # chaud - froid, sert au tri global inter-couches


def _meilleur_echange_couche(couche: int, pin: dict, heat: dict) -> Optional[Echange]:
    """Le meilleur échange candidat pour UNE couche : le pinné le plus froid
    contre le non-pinné le plus chaud, si l'écart franchit l'hystérésis."""
    if not pin:
        return None
    froid_id = min(pin, key=lambda e: heat.get(e, 0))
    froid_v = heat.get(froid_id, 0)
    candidats = {e: h for e, h in heat.items() if e not in pin}
    if not candidats:
        return None
    chaud_id = max(candidats, key=candidats.get)
    chaud_v = candidats[chaud_id]
    if not _devrait_promouvoir(chaud_v, froid_v):
        return None
    return Echange(couche, froid_id, chaud_id, chaud_v - froid_v)


def choisir_echanges(etat: dict, max_echanges: int = 4) -> list:
    """Jusqu'à `max_echanges` échanges, choisis GLOBALEMENT (toutes couches
    confondues, triés par gain) — pas `max_echanges` PAR couche. Port de
    `repin_pick` (colibri.c:8271-8304) : un budget de copies partagé entre
    toutes les couches plutôt qu'un par couche, parce que le coût (une copie
    hôte→VRAM par échange) est le même partout et qu'une seule couche très
    déséquilibrée ne doit pas être bridée par le budget d'une autre qui ne
    l'est pas.

    `etat` : `{couche: (pin, heat)}` — `pin` l'ensemble des experts
    actuellement résidents pour cette couche, `heat` un `{expert: compte}`
    couvrant AU MOINS les experts de `pin` et leurs candidats (typiquement
    `MoEBlock._usage_routage`, converti en dict par l'appelant).

    Ne modifie rien : rend la LISTE des échanges à faire. L'appelant décide
    de l'écriture réelle (table d'adresses, `memory/table_adresses.py`) —
    cette fonction ne sait rien d'une carte.
    """
    candidats = []
    for couche, (pin, heat) in etat.items():
        e = _meilleur_echange_couche(couche, pin, heat)
        if e is not None:
            candidats.append(e)
    candidats.sort(key=lambda e: -e.gain)
    return candidats[:max_echanges]


def cadence_atteinte(jetons_ecoules: int, n: int) -> bool:
    """`jetons_ecoules` : jetons décodés depuis le DERNIER repin (pas depuis
    le début du service — c'est à l'appelant de faire cette soustraction, en
    général `stats.decode_tokens - dernier_repin`). `n <= 0` désactive REPIN
    (`ACVRAM_REPIN=0`) plutôt que de tourner sans fin à `n=0`."""
    return n > 0 and jetons_ecoules >= n
