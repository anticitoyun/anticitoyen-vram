"""Critère d'équivalence par position, scellé par poste7 le 15/09
(revue/poste7-glm-equivalence-15-09.md § 2, précisé § 3) : un seuil absolu
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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Recalé le 15/09 (voir ci-dessus) — un paramètre documenté, pas un
# nombre nu choisi à la main : 2 était une négociation avant mesure.
MULTIPLICATEUR_ULP = 5.0
MAX_EX_AEQUO = 2
COS_MIN = 0.9999


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
    référence (pas seulement le top-1, pas notre propre sortie — poste7
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
