# Pièce 47 — l'échelle AWQ des experts passée aux GEMV, au bit (conception + code, à sec) — 22/09 (Océane)

## Prédiction et seuils, écrits AVANT le code

* origine : pièce 42, verdict `oceane-piece42-glue-22-09.md` — 0,473 ms/pas et
  384 lancements/pas (8 par couche) de gathers, divisions et casts en torch
  devant les deux GEMV d'experts, payés par **tout** alias dont les experts ont
  des statistiques AWQ réelles (`moe.py:1186` et `1294`), quel que soit le
  format des projections d'attention.
* remède retenu (Jerome) : table d'échelles passée aux **deux** noyaux
  (`nvfp4_gemv_marlin_kernel<XT, 1>` down, `<XT, 2>` gate+up), division faite à
  la lecture de `x`, **au bit** contre le chemin torch actuel. Pas de repli à la
  conversion.
* **prédiction chiffrée** : −0,47 ms/pas (0,473 ± sa dispersion) sur tout alias
  calibré ; glue_torch de **591 → ≈ 207 lancements/pas** et de 0,964 à ≈ 0,49
  ms/pas ; aucun effet sur un alias sans échelles d'experts (l'officiel :
  exactement 0 lancement retiré).
* **seuils, et ce qu'ils rendraient si l'hypothèse était fausse** :
  1. **Équivalence au bit** : sur les formes réelles, `sha256` des sorties du
     GEMV identique entre le chemin torch et le chemin fusionné. Faux → la
     division dans le noyau n'a pas l'arrondi de PyTorch (ordre bf16 → fp32 →
     bf16) et le remède redevient « pas au bit », donc à juger par KL.
  2. **Le test doit pouvoir casser** : la faute réintroduite une fois (division
     retirée du noyau) doit le faire échouer. S'il passe encore, ce n'est pas un
     test d'équivalence et il ne compte pas.
  3. **Lancements** : `familles-noyaux` doit rendre ≈ 207 (± 10) sur l'alias
     alpha2. Au-dessus de 300, la branche torch est encore prise quelque part —
     le gain annoncé serait faux et il faudrait dire où.
  4. **Gain de temps** : ≥ 0,35 ms/pas mesuré. En dessous, les lancements
     retirés n'étaient pas le coût (le coût serait la bande, pas le lancement) —
     à publier tel quel, c'est un résultat.
* la compilation et la mesure attendent la carte (Gaelle, pièce 44) ; tout ce
  qui suit est écrit et relu à sec.

## Code écrit et relu (à sec, aucune compilation, aucune carte)

* **noyau** `acvram_kernels.cu:2022-2060` : deux paramètres de plus
  (`xsc`, `ld_sc`) et la division faite **au chargement de `x` en mémoire
  partagée** — une fois par élément de K, au même endroit que le cast qui
  existait déjà, sans un octet de plus à lire et sans lancement. Le pas de
  ligne `ld_sc` est celui de la table (`padded_in` ≥ K), pas K.
* **arrondi, le point qui décide du « au bit »** : PyTorch promeut bf16 en
  float, divise, et arrondit **une fois** au plus proche pair. Le noyau fait
  `__float2bfloat16_rn(v / s)` puis repromeut. Pour le `down`, l'entrée arrive
  en fp32 et `moe.py` l'arrondissait d'abord (`act.to(torch.bfloat16)`) : le
  noyau reproduit ce **premier** arrondi, et seulement quand une échelle est
  fournie — sans échelle, le chemin fp32 est inchangé.
* **hôtes** `acvram_kernels.cu:2187` et `2247` : `xscale` optionnel, refus
  explicite si la table n'est pas bf16, [E, ≥ K], à lignes contiguës.
  Bindings avec `py::arg("xscale") = c10::nullopt` : une extension neuve sert
  les anciens appels sans changement.
* **`moe.py`** : `fuse_ech` (avant le bloc AWQ) décide une fois par forward ;
  quand il est vrai, le gather `x[tok64]`, les divisions et les casts sont
  sautés et le GEMV reçoit `x` brut avec les **vrais** index de jetons — le
  noyau savait déjà les lire (`x + token_ids[g] * K`), le gather n'existait que
  pour la division. Conditions : pile Marlin servie, `not distinct` (le noyau
  fusionné n'a qu'un `x`), extension qui accepte `xscale`, `x` en bf16.
* **garde qui peut rendre faux** : si la division a été sautée et qu'un autre
  chemin que `gemv_marlin` est pris, le forward **lève** au lieu de servir des
  sorties non échelonnées en silence.
* **compatibilité** : `_gemv_marlin_porte_echelle(ext)` lit la signature du
  binaire chargé ; une extension d'avant la pièce 47 garde la division torch.
  Testé à sec sur deux extensions factices.
* **ligne de régime** : `echelle_awq=gemv(N/M) | torch(N/M) | aucune | mixte(…)`,
  lu sur les blocs au dernier forward (`runner.py:_regime_echelle_awq`) — la
  fusion se prouve dans le processus qui mesure, pas en comptant des lancements
  après coup.
* **tests** (`tests/test_gemv_marlin.py`, +110 lignes) : trois au bit
  (`torch.equal`, down, gate+up, table plus large que K), chacun doublé d'un
  contrôle que l'échelle n'est **pas** ignorée (`not torch.equal` sans elle),
  plus un test à sec de la garde de compatibilité.
* joué à sec : `test_moe_awq_pile`, `test_moe_chemins_decodage`,
  `test_gemv_marlin`, `test_chemin_moe_atteint` → **5 passés, 50 sautés**
  (carte requise), `nice 19`, 2 cœurs, 2,3 s. Le venv éditable pointant sur
  l'arbre principal, tout est lancé avec `PYTHONPATH=<worktree>` (vérifié :
  `acvram.engine.moe.__file__` est bien dans `travail/oceane-routage`).

## Reste, à la première carte libre

1. compiler l'extension depuis **ce** worktree et jouer les trois tests au bit ;
2. **réintroduire la faute une fois** — retirer `__float2bfloat16_rn` du
   quotient — et vérifier que les trois échouent ; un test qui passerait encore
   ne serait pas un test d'équivalence ;
3. `familles-noyaux` sur l'alias alpha2 : lancements de `glue_torch` et ms/pas
   contre les seuils 3 et 4 ci-dessus, la ligne portant `echelle_awq=gemv(48/48)`.

## Mesure sous carte — 22/09, trois prises (build 19:32, tests 19:35, mesure 20:41)

* instrument : la chaîne du contrôle 2 de Gaelle, inchangée
  (`frontiere-pas.py 12 60` sous `nsys --cuda-graph-trace=node`, puis
  `familles-noyaux --detail glue_torch`) ; journaux
  `scratchpad/oceane-p47-22-09/` ; commit mesuré **9bcb7037**, alias B
  `…-nvfp4-qkv-alpha2-22-09`, le même que le contrôle 2.
* régime : b=12, 60/60 pas pleins, `experts_layout=marlin`,
  `chemin_moe=mma-a4(atteint=gemv_marlin)`, `graphes=on captures=5 replays=99` ;
  **horloge médiane 2 692 MHz, température max 40 °C** — identique au contrôle 2
  (2 692 MHz, ≤ 43 °C), la comparaison est à horloge ≤ 3 % ; compute-apps début
  = fin (llama-server 8081 seul) ; 39 s de fenêtre, `tenue` sous les 30 min.
* **équivalence au bit : TENUE.** 6/6 verts (`torch.equal`), 19,4 s.
* **le test peut rendre faux : VÉRIFIÉ.** `__float2bfloat16_rn` retiré du
  quotient, extension recompilée : **5 échecs sur 5** tests au bit (le sixième,
  la garde de compatibilité, est à sec et ne dépend pas du noyau). Fichier
  restauré, arbre propre.

| | contrôle 2 (Gaelle) | pièce 47 | écart |
|---|---|---|---|
| `glue_torch` | 0,964 ms / **591** lanc. | 0,492 ms / **207** lanc. | **−0,472 ms / −384** |
| `experts_marlin` | 3,965 ms / 96 | **4,334 ms** / 96 | **+0,369 ms** |
| noyaux | 8,078 ms | 7,977 ms | −0,101 ms |
| **mur** | 8,571 ms | **8,361 ms** | **−0,210 ms (−2,4 %)** |
| lancements | 1 322 | 938 | −384 |

* **seuil 3 (≈ 207 ± 10 lancements) : TENU**, à l'unité — les 8 lancements par
  couche ont disparu, aucun autre site n'est resté sur la branche torche.
* **seuil 4 (gain ≥ 0,35 ms/pas) : NON TENU.** Le gain réel est 0,21 ms sur le
  mur (0,10 sur les noyaux), parce que **le GEMV paie ce que la glue économise**
  : la table `[E, K]` est relue par **chacun** des N/64 blocs de la grille,
  exactement comme `x`. Pour gate+up à K=2048, N=768 : 4 Ko × 12 blocs × 96
  paires × 48 couches ≈ 221 Mo par pas, soit ≈ 0,15 ms au plancher de bande —
  du bon ordre que les +0,369 mesurés. **Mon commentaire « sans un octet de plus
  à lire » était faux ; il est corrigé dans le noyau par cette mesure.**
* ce que ça dit, et que la pièce 42 disait déjà autrement : à M=12 ces noyaux
  sont **tenus par la bande**, pas par les lancements. Retirer 384 lancements ne
  rend que ce que les lancements coûtaient vraiment — 0,21 ms, pas 0,47.

## Suite proposée (la décision est à Jerome)

1. **Garder en l'état** : 0,21 ms/pas (−2,4 %) au bit, sur tout alias calibré,
   sans dette. C'est acquis et honnête.
2. **Supprimer la relecture** : replier l'échelle dans les échelles de bloc des
   poids à la conversion (option 1 du verdict 42) — plus **aucun** octet ni
   lancement, gain attendu ≈ 0,47 ms, mais **pas au bit** (up re-quantifié), à
   juger par KL (seuil 1,0, référence 0,519) et PPL à 2 SE.
3. Demi-mesure au bit : un seul lancement torch qui divise `x` une fois pour
   les deux projections (8 → 2 par couche au lieu de 8 → 0), sans relecture par
   bloc. Gain attendu entre les deux, à mesurer.
