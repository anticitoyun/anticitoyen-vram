# Sage — Objectif « vitesse à qualité constante » : bilan chiffré, revendication à réécrire, et le poste suivant est b=1 (fusion des petits noyaux), le seul régime où un classé nous bat encore (17/09)

Entrée : Jérôme — 0.6.8 poussé (E défaut, C mixte défaut, B0 défaut), Coder b=12 997,8 t/s / 0,400 J, b=1 257,8 / 1,214 ; prefill 9 913 j/s.

## 1. Où en est le but du projet, Coder, classés seulement

| cellule | acvram matin | acvram 0.6.8 | meilleur classé | tout moteur |
|---|---|---|---|---|
| b=12 t/s / J | 743,4 / 0,538 | **997,8 / 0,400** | EXL3 855,7 / 0,362 | vLLM Marlin 2 031 / 0,197 |
| b=1 t/s / J | 233,9 / 1,365 | 257,8 / 1,214 | **llama.cpp 341,4 / 1,108** | idem |
| prefill j/s | 8 633 | 9 913 | **llama.cpp 15 717** | Marlin 20 988 |

Revendication Coder à réécrire (Jérôme, comparatif, note datée) : « acvram est le plus rapide des classés à b=12 (+ 17 % sur EXL3), EXL3 reste le plus économe (0,362 J) ; llama.cpp reste devant à b=1 (+ 32 %) et au prefill (+ 59 %) ». GLM : rien ne change (MLA hors E ; C et B0 ont leur ligne). Les régimes de 0.6.8 vont dans la ligne de chaque cellule éditée.

## 2. Poste suivant : b=1 — pas un noyau de plus, moins de noyaux

* b=1 est le régime d'un moteur domestique et la seule cellule de vitesse où un **classé** nous bat. Profil b=1 (Laure 6a4fd57, avant E) : experts 0,93 ms (58 % de bande passante, correct), dense 1,04 ms (correct), attention 0,85 (E l'a réduit), **glue MoE 0,43 + norm/rope/act 0,34 + élémentaire 0,41 = 1,17 ms** — 30 % du pas dans des noyaux qui ne déplacent presque rien. Octets par pas ≈ 2,3 Go → plancher ≈ 1,3 ms ; pas actuel ≈ 3,8 ms. Sous graphes, chaque noyau coûte encore 2-3 µs de lancement : à 10-20 noyaux par couche × 48, c'est 1-2 ms — le poste est le **nombre de lancements**, pas la bande passante.
* **Mesure d'abord (Laure, 20 min)** : nombre de noyaux par pas à b=1 (le profil l'a déjà dans `profil.json`, à compter) et temps GPU des trois postes ci-dessus après E. Prédictions : ≥ 600 lancements par pas ; glue + norm + élémentaire ≥ 28 % du GPU. Issue qui gênerait : < 400 lancements et < 20 % — alors b=1 est borné par experts + dense et le chantier n'a pas d'objet ; on note.
* **Chantier F (Laurine, à sec, après la mesure)** : fusions exactes — RMSNorm + rope dans un noyau, routeur (sigmoid, biais, top-k, normalisation, × 1,8) dans un noyau, act × up dans la GEMV (`gateup` fusionnée existe déjà au décodage), résidus fusionnés dans les épilogues ; cible ≤ 6 noyaux par couche. Scellé : Coder b=1 ≥ **300 t/s** (pas ≤ 3,3 ms), b=12 non dégradé (≥ 990), sortie = ± 2⁻⁸ par fusion (test dans chaque commit, un commit par fusion), régime dans `regime_ligne()`. Réfutation : < 285 t/s → on publie le gain obtenu et on clôt. Coût : Laurine 2-3 j à sec, ≤ 2 passes de carte de 30 min.

## 3. Campagne — deux corrections en passant

* Qwen3.8-27B : la cellule acvram 1,0282 vient d'un converti **`srcexl3` (EXL3 6 bpw → NVFP4)**, une double quantification ; le fait « NVFP4 dense + 2,7 % » est confondu par la source. Manon : reconversion depuis la source bf16 HF (calibration bras-A, sha256), 1 h carte dans un bloc de campagne ; Laure : PPL privé 20 min ; prédiction ≤ 1,015 (classé) ; sinon le fait tient et se scelle après la campagne. vLLM MIXED communautaire reste non classé, régime nommé.
* Paliers 0 et 1 : aucun état depuis l'ordre de carte (Katy inventaire, Océane `fiche-service.py` + juge de refus). Jérôme : une ligne d'ETAT — prouvé / en cours / bloqué, avec la raison.

## Ordre

1. Jérôme : ETAT — objectif traité sur 3 postes, bilan § 1, revendication réécrite, poste F ouvert après mesure ; état des paliers 0-1.
2. Laure (carte) : mesure § 2 (20 min) ; puis campagne (Qwen3.8 reconverti quand Manon livre) ; passes F en préemption.
3. Laurine (à sec) : F si la mesure tient, un commit par fusion avec son test.
4. Manon : reconversion Qwen3.8-27B depuis bf16, verdict avec sha256.
