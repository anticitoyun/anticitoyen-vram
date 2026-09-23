# Sage — P2 au défaut : six lignes tenues, le régime servi de Coder change (19/09, 19 h 45)

Source : `verdict-c11-19-09` (Manon, f78dbcd) ; `verdict-p2-decodage-19-09` (6ed21ca) ; `verdict-p2-ppl-19-09` (77ddd7d) ; `verdict-p2-moteur-19-09` ; `chantier-c11-19-09` (Océane, c17ea89).

## 1. Ce qui est tenu, tout au harnais égal, ABAB contre le défaut classé

| ligne | mesuré (Coder `-qkvo-i8c`, `PREFILL_INT8=cublas`) | défaut classé 0.6.13 | seuil | verdict |
|---|---|---|---|---|
| PPL 3 tranches privées | **1,0094** (1,0042 / 1,0112 / 1,0128) | 1,0155 | ≤ 1,020 | tenu, et **meilleur** |
| équivalence décodage/prefill | B 6/5/8 ≤ 2A + 2 ; distances ± 0,5 % | — | 2A + 2 | tenu |
| prefill 2 047 j/s | **18 850** | 16 426 | ≥ 17 500 | tenu (+14 %) |
| J/jeton prefill | 0,89-0,93 × A | 1 | ≤ 1,00 × | tenu |
| b=1 t/s · J brut | **396,0** · 0,7999 | 375 · 0,79 | ≥ 356 · ≤ 1,02 × | tenu (+5 %) |
| b=12 t/s · J net (C11) | **1 365** · 0,2243 | 1 349,6 · 0,2258 | ≥ 1 334 · ≤ 1,02 × | tenu (1,008 × · 0,993 ×), capture 4/4, chemin prouvé |

Aucune ligne fausse, chacune avec son témoin dans la même prise. **Décision : P2 est le défaut de Coder.** Mécanisme compris : l'échelle int8 par canal (depuis bf16) bat l'affine par groupes de 128 promue depuis NVFP4 sur des projections denses bien conditionnées (hors-moteur 0,9947, en moteur 1,0094 < 1,0155) ; au prefill, `torch._int_mm` (tensor cores int8) remplace la déquantification g128 → GEMM bf16 ; au décodage, C11 fait servir l'i8c par le même chemin étroit que le g128 (vue par groupes, codes partagés), donc mêmes octets, même temps.

## 2. Ce que « au défaut » veut dire, fichier par fichier

1. `regime.py:54` : `PREFILL_INT8` défaut `bf16` → **`cublas`** — éligible seulement si groupe = K, zéros = 128, M > 16, K et N multiples de 8 ; **sortie inchangée pour tout converti classé**, prouvée par un test qui asserte `CHEMINS_INT8['cublas'] == 0` sur un classé (Coder nvfp4, GLM) sous le nouveau défaut (Océane).
2. Catalogue (`~/TSV/*.tsv`, `config.toml`, GUI) : l'alias servi de Coder pointe sur le converti **`-qkvo-i8c`** ; le classé nvfp4 reste inscrit sous son propre alias (« défaut 0.6.13 », témoin) (Jérôme). S2 0/0 après.
3. Comparatif : ligne « **acvram défaut 0.6.15 (P2 : i8c, cublas, marlin/marlin, chemin_moe=mma)** » = les six cellules ci-dessus avec leurs commits ; la ligne 0.6.13 reste comme historique ; revendication Coder réécrite : devant llama.cpp en t/s partout (b=1 +16 %, b=12 +28 %, prefill +20 %), PPL 1,0094 contre 1,0103, J net b=12 0,2243 contre 0,2136 (+5 %, l'éco 2700 comble : à mesurer sur P2).
4. `.deb` : le bump 0.6.15 (épingle 13.0.*) embarque ce défaut — une seule version pour les deux changements ; l'utilisateur réinstalle quand il veut.
5. `REPRISE.md` § 2 : régime livré = P2 ; § 10 : P2 clos, C11 clos, retirer « opt-in ».
6. GLM : rien ne change (aucun tenseur i8c dans son converti ; P2 GLM fermé le 18/09, 1,0143 > 1,005) — la ligne de régime le montre, `CHEMINS_INT8` = 0.

## 3. Une cellule manque à la revendication complète : P2 + éco

Les cellules éco (`-lgc 2700` : −10 % J à b=12, −17 % J brut à b=1) ont été mesurées sur le classé. L'effet est celui de la carte (bornée par 400 W), pas du converti — mais « probablement transportable » n'est pas une cellule. Fenêtre **20 min** (Manon, fin de file) : P2 + `-lgc 2700`, b=1 et b=12, seuils = ceux d'E1/E1-bis (b=1 ≥ 340,1 ; b=12 ≥ 1 066 et J net ≤ 0,2136). Prédiction : b=1 ~375 / J −17 %, b=12 ~1 365 / J net ≈ 0,2015 → **devant llama.cpp en vitesse ET en énergie, à b=1 et b=12, sur le défaut livré**.

## Ordre

* **Océane** — § 2.1 maintenant (une ligne + table de régime + test du classé inchangé), commit `oceane-11`, pointeur à Jérôme ; C11 fusionnable avec (déjà sur main via c17ea89 : vérifier).
* **Jérôme** — fusion, § 2.2-2.5, 0.6.15 reconstruit après la fusion (épingle + P2), push ; `ETAT` : « P2 au défaut (six lignes) ».
* **Manon** — fin de file : P2 + éco 2700 (§ 3), verdict `revue/verdict-p2-eco-19-09.md`.
