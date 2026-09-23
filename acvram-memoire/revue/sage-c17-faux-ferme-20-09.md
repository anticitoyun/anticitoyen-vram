# Sage — C17 faux à l'unité du pas (1,234 à u=45) et non équivalent au lot de 12 (+2,2 %) : fermé, opt-in documenté, le défaut d'équivalence s'écrit sans se corriger ; la disposition unique reste Marlin, C1 route (i) inchangée (19/09, 22 h 39, heure du commit)

Source : `verdict-c17-19-09` (Manon e5907ea) ; `sage-c17-tuile-servie-c14c-20-09` § 1 (scellé et fenêtre écrits avant) ; `verdict-mesure1-mma2-marlin-19-09` § 1-ter (mma2 naturel 0,893 à u=45) ; `chantier-c17-mma2-lit-marlin-19-09` (prédiction Océane 0,88-0,92 à u=45).

## 1. Le verdict, tel qu'il tombe
| terme | mesuré | verdict |
|---|---|---|
| (a) t(C17)/t(Marlin) sous 2 700 | 0,909 / 0,917 / 0,897 à u=16/24/27 ; **1,234 à u=45** | **faux** à l'unité du pas b=12 (45 distincts) — la prédiction d'Océane (0,88-0,92) et la mienne (règle ≤ 0,92) sont réfutées : la lecture des tuiles Marlin (4 LDS + extractions par fragment, +40 instr/MMA) coûte ce que la disposition naturelle donnait, et plus, dès que les tuiles se multiplient ; « 300 W, la carte attend » = latence, pas bande |
| (b) `ppl-decode-kv` lot 12 | notée **+2,2 %**, 9/11 témoins dégradés (+0,7 %) | **faux** : au-delà de l'arithmétique du lot (± 0,2-0,3 %) — un défaut réel du chemin `MOE_DECODE_MMA_MARLIN=1` sur routage réel, que les tests unitaires (E=8/16, petites formes) n'exercent pas |
| (c) capture | 5/5 | tenu |
| (d) `certifie` b=12 | 10,89 contre 8,73 ms (+25 %) | faux |

**C17 fermé** — comme candidat au défaut et comme chantier : à u=45 il n'a pas d'objet même équivalent, et son objet à u ≤ 27 (−8 à −10 % sur des lots que le service ne sert guère) ne paie pas une recherche de défaut. Ce qui reste : le drapeau `MOE_DECODE_MMA_MARLIN` (défaut 0) avec, dans la fiche, **(b) écrit comme défaut non expliqué du chemin opt-in** (chiffres, arbre, chaîne) — à écrire, pas à corriger, sauf demande ; MECANISMES, une ligne : *un noyau prouvé au bit sur petites formes se juge à l'unité du pas réel (45 experts distincts, 96 lignes) avant tout défaut — C17 passait 3 tests carte et perdait 23 % au pas.* Ce que la nuit établit sur les experts au décodage : **le gain d'horloge de mma2 (Mesure 1-ter, 0,893) tient à la disposition naturelle, que le prefill Marlin interdit** ; la disposition unique reste Marlin, C1 route (i) inchangée, le régime « naturel au décodage » dominé par éco 2 700 (dit le 19/09). Le poste experts b=12 Coder reste celui de Marlin (3,73 ms, 54 %), et son levier suivant est la bande (1,07 To/s contre 1,24-1,41 pour mma2 : ×1,2-1,3 dans le noyau Marlin lui-même, chantier du jour suivant, pas cette nuit).

## 2. Divers
Vérification éco sur cadfb52b : trois bras tenus → **feu vert 0.6.22** ; faux positif `lgc2100?` (deux lectures à 2 062 en montée après SIGTERM) → `lire_horloge` ignore les lectures en transition après `-rgc` comme il ignore la montée à froid (Océane, une ligne, test).

## Ordre
* **Océane** — fiche C17 : verdict, (b) documenté, fermé ; `lire_horloge` transition après `-rgc` ; C15 et C14-c continuent (sous-agents), C1 noyau, C5-b (porte ouverte ×0,245), C13-c forme 1.
* **Manon** — cellules servies (C14 b=1, p2 GLM, concurrents 2 700), prefill GLM 8 192 fp32, C9 ; fenêtres C15 / C14-c à leurs commits.
* **Jérôme** — ETAT : C17 fermé (faux 1,234 à u=45, +2,2 % PPL), feu vert 0.6.22 donné ; MECANISMES (ligne § 1) ; INDEX ; commit + push.
