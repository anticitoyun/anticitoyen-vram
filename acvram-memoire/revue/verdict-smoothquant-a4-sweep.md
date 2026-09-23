# Verdict — SmoothQuant + A4 (balayage α) et isolation down_proj (13/09/2026)

Manon, bead `anticitoyen-vram-brd` étape 1, suite du verdict A4 nu
(PPL 5,7550, +2,58 %, seuil dépassé). Mesuré sur la 5090, carte prêtée par
Jérôme, `python outils/smoothquant-a4-sweep.py --pour-de-vrai`. Prédiction
et seuil scellés dans `revue/prediction-smoothquant-a4-sweep.md`.

## Résultat

| régime | PPL | écart vs A16 (5,6102) | seuil ≤ 5,6663 | durée |
|---|---|---|---|---|
| SmoothQuant α=0,50 | 5,8895 | **+4,978 %** | DÉPASSÉ | 91 s |
| SmoothQuant α=0,65 | 6,0190 | **+7,287 %** | DÉPASSÉ | 92 s |
| SmoothQuant α=0,80 | 6,9954 | **+24,691 %** | DÉPASSÉ | 92 s |
| down_proj seul (A4, reste A16) | 5,6415 | **+0,557 %** | **RESPECTÉ** | 61 s |

Pour mémoire (verdict précédent) : A4 nu (7 genres) +2,58 %, A8 nu
+0,13 %.

## Contre la prédiction scellée : réfutée sur SmoothQuant, confirmée sur down_proj

* **SmoothQuant : la prédiction est FAUSSE, dans TOUS les sens.**
  Prédit α=0,50 « échec probable, marge modeste » (5,68-5,75) — mesuré
  5,8895, hors fourchette et **pire que le A4 nu**. Prédit α=0,65
  « le plus prometteur » (5,63-5,70) — mesuré 6,0190, **encore pire**
  qu'α=0,50. Prédit α=0,80 « probablement pire » (5,70-5,80) — direction
  juste, magnitude ratée d'un facteur 3 (+24,69 % au lieu de +14 % au
  pire de la fourchette). **Le lissage SmoothQuant, tel qu'implémenté ici,
  DÉGRADE monotonement avec α et rend le A4 pire que sans lissage du
  tout.**
* **down_proj isolé : la prédiction tient.** Fourchette annoncée
  +0,3 % à +1,2 % ; mesuré +0,557 %, au milieu de la fourchette. **Sous
  le seuil**, meilleur régime A4 mesuré depuis le début de ce chantier
  (down_proj est le SEUL genre qui, seul en A4 avec le reste en A16, ne
  dépasse pas le seuil).

**L'issue nommée qui me gênerait ne s'est pas produite** : down_proj
isolé ne récupère PAS la quasi-totalité de l'écart du A4 complet
(+0,557 % sur +2,58 %, soit environ 22 % de l'écart total) — le reste de
la dégradation (les ~78 % restants) vient donc majoritairement des 6
AUTRES genres pris ensemble, pas de down_proj seul. **Ceci contredit
l'hypothèse initiale (Bridging Gap) selon laquelle down_proj serait LE
point de fragilité dominant** — au moins dans ce protocole de fake-quant
dynamique par appel, sans calibration.

## Pourquoi SmoothQuant dégrade ici (hypothèse à vérifier)

Le lissage classique répartit la difficulté PAR CANAL D'ENTRÉE
individuel. Le NVFP4 quantifie par BLOC DE 16 canaux d'entrée avec UNE
SEULE échelle E4M3 partagée par bloc. Une échelle SmoothQuant qui varie
fortement d'un canal à l'autre À L'INTÉRIEUR d'un même bloc de 16 peut
créer un déséquilibre que le format n'avait PAS avant le lissage : un
canal fortement mis à l'échelle domine l'`amax` du bloc entier et écrase
la résolution des 15 autres canaux du même bloc, qui n'avaient pourtant
pas besoin d'être touchés. Plus α est grand, plus s s'écarte de 1
canal-par-canal (poussé par `max|X|^α`, très hétérogène d'un canal à
l'autre), et plus ce déséquilibre intra-bloc s'aggrave — ce qui explique
la dégradation MONOTONE observée avec α croissant, à l'identique sur les
trois valeurs testées.

**Ce module de lissage n'est donc probablement PAS adapté à un format à
échelle de bloc fine (16) sans modification** : une variante qui LISSE
la variance de s À L'INTÉRIEUR de chaque bloc de 16 (par exemple un s
commun par bloc plutôt que par canal individuel, ou un plafond sur le
rapport max/min de s dans un même bloc) serait le prochain test naturel
— non fait ici, hors du périmètre demandé (le balayage α tel que posé).

## Conséquence pour le bead brd

* **SmoothQuant per-canal, tel quel, à ÉCARTER pour le NVFP4 bloc-16.**
  Ne pas creuser d'autres valeurs d'α sans d'abord résoudre le problème
  d'homogénéité intra-bloc identifié ci-dessus.
* **down_proj seul en A4 (reste en A16) est le régime le plus prometteur
  à ce jour** — mais c'est un régime MIXTE A4/A16, pas un W4A4 complet ;
  il ne donne PAS accès à la MMA native `mxf4nvf4` sur les 6 autres
  genres, qui resteraient alors sur le chemin W4A16 logiciel actuel. Son
  intérêt dépend de ce que down_proj représente en débit — à chiffrer
  séparément.
* **A8 nu (E4M3, +0,13 %) reste le meilleur résultat GLOBAL** parmi tous
  les régimes testés jusqu'ici pour un fake-quant appliqué aux 7 genres
  uniformément — mais Laurine a établi (13/09, ptxas 13.4 sm_120a) que
  `mxf8f6f4` natif exige des échelles UE8M0/bloc 32, incompatibles avec
  nos poids NVFP4 (UE4M3/bloc 16) : notre A8 logiciel n'a donc pas
  d'équivalent matériel direct sur cette carte.

## Ce qui reste

* Tester une variante de SmoothQuant à échelle homogénéisée par bloc de
  16 (hypothèse ci-dessus), pas seulement par canal.
- Isoler chaque genre restant un par un (q/k/v/o/gate/up), comme
  down_proj, pour localiser précisément d'où vient le reste de l'écart
  (~78 % non expliqué par down_proj seul).
- Chiffrer le débit d'un régime mixte (down_proj en A8 ou A16, les 6
  autres en A4/mxf4nvf4) une fois le noyau MMA de Laurine disponible pour
  un banc réel — cette note reste une mesure de PPL, pas de débit.
