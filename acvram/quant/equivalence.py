"""Critère d'équivalence par position, scellé par Sage le 15/09
(revue/sage-glm-equivalence-15-09.md § 2, précisé § 3) : un seuil absolu
(0,05) ne peut pas rendre « vrai » sous l'ulp bf16 lui-même — REGLES §4.
Remplacé par un critère relatif au régime, valable pour toute équivalence
future, pas seulement GLM-4.7-Flash.

Par position, CUMULATIF (§ 3 : le « ET » est bien un ET, pas une des deux
portes) :
  - `top-1` identique ET `delta ≤ 2 ulp bf16 de max_j |logit_ref,j|`
    (le maximum sur toute la LIGNE de référence — l'erreur d'accumulation
    sur le vocabulaire est fixée par les grands termes, pas par la valeur
    à l'indice du delta) ET `cos ≥ 0,9999`  →  passe ;
  - sinon, seulement si l'écart top-1/top-2 de la RÉFÉRENCE est
    `≤ 2 ulp bf16` (ex-aequo PROUVÉ, pas supposé)  →  passe, compté ;
  - sinon  →  échoue.

Global : toutes les positions non ex-aequo passent, ET au plus 2
positions sur 16 sont comptées ex-aequo. Pas de cosinus global : il
masque une position fausse derrière quinze bonnes (0,999556 passait
avec un delta de 1,815 avant ce correctif).

Le multiplicateur a été RECALÉ le 15/09 (pas laissé négocié) sur deux
témoins référence-contre-elle-même, tous deux HF bf16, mêmes 16 jetons
(revue/prediction-temoin-ulp-cpu-15-09.md,
revue/prediction-temoin-ulp-gpu-cpu-15-09.md) :
  - CPU eager vs sdpa       : plancher hors ex-aequo = 4,00 ulp
  - GPU (cuda:0) vs CPU     : plancher hors ex-aequo = 4,00 ulp (le
    plancher BRUT y monte à 14,69, mais entièrement porté par la même
    position déjà identifiée comme ex-aequo dans le premier témoin —
    exclue du calcul du plancher pour la même raison qu'elle serait
    exclue du critère lui-même, pas retenue comme bruit général)
Les deux témoins s'accordent sur 4 ulp. Seuil retenu = plancher + 1 = 5.

CE RÉGIME (`verdict_position`/`verdict_global`) NE VAUT QUE POUR LE
PREFILL À 2 COUCHES, où il a été calibré. Au DÉCODAGE, un nombre d'ulp
fixe ne juge pas : Laure (15/09, revue/verdict-narrow-voyants-15-09.md)
a mesuré que ce même seuil (5 ulp / cos≥0,9999) échoue 739/768 sur le
témoin lui-même (A-graphes vs A-eager, aucun bogue — juste plus de bruit
de sommation qu'en prefill 2 couches). Un seuil qu'un témoin sans bogue
ne peut pas passer n'est pas un contrôle (REGLES §4).

`verdict_decodage` (Sage §7.2, revue/sage-reprise-15-09-b.md) : le test
calcule son propre témoin à CHAQUE exécution (le même bras A, sous
graphes contre eager, mêmes noyaux et mêmes jetons — jamais un chiffre
d'une session antérieure) et exige que B/A le DOMINE, statistique par
statistique, plutôt que de comparer à un seuil absolu :
  - `med(B/A) ≤ 1,2 × med(témoin)`
  - `p90(B/A) ≤ 1,2 × p90(témoin)`
  - `max(B/A) ≤ max(témoin)`
  - toute divergence top-1 hors ex-aequo (B/A) ≤ `max(témoin)`
Garde sur l'INSTRUMENT, pas sur B/A : si le témoin lui-même dépasse
150 ulp en max ou cos < 0,998, c'est le chemin des graphes qui a bougé
depuis le calibrage — le test s'arrête et le dit (invalide), il ne rend
jamais "faux" sur un instrument qui a changé sous lui.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Recalé le 15/09 (voir ci-dessus) — un paramètre documenté, pas un
# nombre nu choisi à la main : 2 était une négociation avant mesure.
MULTIPLICATEUR_ULP = 5.0
MAX_EX_AEQUO = 2
COS_MIN = 0.9999

# -- mode décodage (Sage §7.2) : le témoin remplace le nombre fixe --------
FACTEUR_DOMINATION = 1.2          # med/p90 de B/A doivent rester sous ce facteur du témoin
TEMOIN_MAX_ULP_INVALIDE = 150.0   # au-dessus : les graphes ont bougé, pas un bogue de B/A
TEMOIN_COS_MIN_INVALIDE = 0.998


def ulp_bf16(v: float) -> float:
    """Le pas de représentation bf16 (1 signe, 8 exposant, 7 mantisse) au
    voisinage de ``v`` : 2**(exposant - 7). bf16 n'a que 7 bits de mantisse
    explicites, contre 23 en fp32 — c'est ce pas, pas un chiffre choisi à
    la main, qui borne ce qu'une comparaison bf16↔bf16 peut prouver."""
    v = abs(v)
    if v == 0.0:
        return 2.0 ** -133          # plancher dénormalisé, jamais atteint ici
    return 2.0 ** (math.floor(math.log2(v)) - 7)


@dataclass
class VerdictPosition:
    ok: bool
    ex_aequo: bool
    raison: str


def verdict_position(delta: float, echelle_ref: float,
                     top1_ref: int, top1_nous: int, cos: float,
                     ecart_top1_top2_ref: Optional[float],
                     multiplicateur: float = MULTIPLICATEUR_ULP) -> VerdictPosition:
    """``echelle_ref`` : ``max_j |logit_ref,j|`` sur TOUTE la ligne de
    référence (pas seulement le top-1, pas notre propre sortie — Sage
    § 3 : l'erreur d'accumulation sur le vocabulaire est fixée par les
    grands termes de la référence, jamais par la nôtre). ``ecart_top1_
    top2_ref`` : l'écart MESURÉ entre les deux meilleurs logits de la
    RÉFÉRENCE seule, ou ``None`` s'il n'a pas été relevé (jamais
    supposé). ``multiplicateur`` : par défaut le seuil recalé (5, voir
    le docstring du module) — un paramètre, pas une valeur cachée, pour
    qu'un futur recalage (nouveau témoin) n'oblige pas à modifier
    l'appelant."""
    seuil = multiplicateur * ulp_bf16(echelle_ref)
    if top1_ref == top1_nous:
        if delta <= seuil and cos >= COS_MIN:
            return VerdictPosition(True, False, "ok")
        return VerdictPosition(
            False, False,
            f"top-1 identique mais delta={delta:.4f} (seuil {seuil:.4f}) "
            f"ou cos={cos:.6f} (seuil {COS_MIN})")
    if ecart_top1_top2_ref is not None and ecart_top1_top2_ref <= seuil:
        return VerdictPosition(
            True, True,
            f"ex-aequo prouvé : écart référence {ecart_top1_top2_ref:.4f} "
            f"≤ seuil {seuil:.4f}")
    return VerdictPosition(
        False, False,
        f"top-1 différent (réf={top1_ref} nous={top1_nous}), "
        f"pas d'ex-aequo prouvé"
        + (f" (écart réf {ecart_top1_top2_ref:.4f} > seuil {seuil:.4f})"
           if ecart_top1_top2_ref is not None else " (écart non relevé)"))


def verdict_global(positions: list[VerdictPosition]) -> tuple[bool, str]:
    n_ex_aequo = sum(1 for p in positions if p.ex_aequo)
    fautives = [i for i, p in enumerate(positions) if not p.ok]
    if fautives:
        return False, f"positions en échec : {fautives}"
    if n_ex_aequo > MAX_EX_AEQUO:
        return False, f"{n_ex_aequo} ex-aequo > {MAX_EX_AEQUO} autorisés"
    return True, f"{n_ex_aequo} ex-aequo sur {len(positions)}, toutes les autres passent"


def _p90(valeurs: Sequence[float]) -> float:
    """90e centile, interpolation linéaire — pas de dépendance numpy pour
    une fonction pure appelée depuis des bancs qui n'en ont pas besoin
    ailleurs."""
    v = sorted(valeurs)
    if len(v) == 1:
        return v[0]
    pos = 0.9 * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    frac = pos - lo
    return v[lo] * (1 - frac) + v[hi] * frac


@dataclass
class VerdictDecodage:
    ok: bool
    invalide: bool
    raison: str
    temoin: dict = field(default_factory=dict)
    ba: dict = field(default_factory=dict)


def verdict_decodage(
    deltas_ulp_temoin: Sequence[float], cos_temoin: Sequence[float],
    deltas_ulp_ba: Sequence[float],
    divergences_top1_hors_ex_aequo_ba: Sequence[float] = (),
) -> VerdictDecodage:
    """Sage §7.2 : B/A doit DOMINER un témoin mesuré au même lancement
    (A sous graphes contre A eager, mêmes noyaux, mêmes jetons), pas un
    nombre d'ulp fixe — voir le docstring du module pour le pourquoi.

    ``deltas_ulp_temoin``/``cos_temoin`` : par jeton, du témoin.
    ``deltas_ulp_ba`` : par jeton, de B contre A (ce qui est sous test).
    ``divergences_top1_hors_ex_aequo_ba`` : le delta/ulp des SEULES
    positions de B/A où le top-1 diffère ET où ce n'est pas un ex-aequo
    prouvé (vide s'il n'y en a aucune) — chacune doit rester sous
    `max(témoin)`, comme n'importe quel autre point de la distribution.
    """
    max_temoin = max(deltas_ulp_temoin)
    cos_min_temoin = min(cos_temoin)
    if max_temoin > TEMOIN_MAX_ULP_INVALIDE or cos_min_temoin < TEMOIN_COS_MIN_INVALIDE:
        return VerdictDecodage(
            False, True,
            f"témoin hors bornes (max={max_temoin:.2f} ulp, "
            f"cos min={cos_min_temoin:.5f}) : le chemin des graphes a "
            f"bougé depuis le calibrage — mesure invalide, pas réfutée",
            temoin={"max": max_temoin, "cos_min": cos_min_temoin})

    med_temoin, p90_temoin = statistics.median(deltas_ulp_temoin), _p90(deltas_ulp_temoin)
    med_ba, p90_ba, max_ba = (statistics.median(deltas_ulp_ba), _p90(deltas_ulp_ba),
                              max(deltas_ulp_ba))
    temoin_r = {"med": med_temoin, "p90": p90_temoin, "max": max_temoin,
               "cos_min": cos_min_temoin}
    ba_r = {"med": med_ba, "p90": p90_ba, "max": max_ba}

    fautes = []
    if med_ba > FACTEUR_DOMINATION * med_temoin:
        fautes.append(f"med(B/A)={med_ba:.2f} > {FACTEUR_DOMINATION}×med(témoin)"
                      f"={FACTEUR_DOMINATION * med_temoin:.2f}")
    if p90_ba > FACTEUR_DOMINATION * p90_temoin:
        fautes.append(f"p90(B/A)={p90_ba:.2f} > {FACTEUR_DOMINATION}×p90(témoin)"
                      f"={FACTEUR_DOMINATION * p90_temoin:.2f}")
    if max_ba > max_temoin:
        fautes.append(f"max(B/A)={max_ba:.2f} > max(témoin)={max_temoin:.2f}")
    hors_bornes = [d for d in divergences_top1_hors_ex_aequo_ba if d > max_temoin]
    if hors_bornes:
        fautes.append(f"{len(hors_bornes)} divergence(s) top-1 hors ex-aequo "
                      f"au-dessus de max(témoin)={max_temoin:.2f} : {hors_bornes}")

    if fautes:
        return VerdictDecodage(False, False, " ; ".join(fautes), temoin_r, ba_r)
    return VerdictDecodage(True, False, "B/A domine le témoin sur les trois "
                          "statistiques, aucune divergence hors bornes",
                          temoin_r, ba_r)
