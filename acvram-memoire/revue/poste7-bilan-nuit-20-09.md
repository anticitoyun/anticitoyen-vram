# poste7 — bilan de la nuit du 19 au 20/09 (demandé par l'utilisateur à 05 h 51, horloge machine) : ce qui est servi, ce qui est mesuré, ce qui est faux, ce qui reste — et la question 119B

Source : les 30 notes `poste7-*-19-09` / `-20-09` de la nuit et les verdicts qu'elles citent ; heures = horloge machine (les titres portent l'heure du commit après l'erratum de 00 h 10). Équipe : chef (fusions, ETAT, .deb), poste1 (code, 26 commits), poste2 (carte, 34 verdicts), poste7 (30 notes). Aucune mention interdite, GitLab seul, chaque commit poussé.

## 1. Ce qui est servi ce matin (défaut 0.6.24, main 6c7bb533 ; 0.6.25 en construction pour la ligne de régime)
| changement de défaut | preuve | effet |
|---|---|---|
| **éco 2 700** (décision utilisateur 20 h 22) : le processus qui sert pose `-lgc`, rend à l'arrêt ; sans sudo on sert bruyamment ; .deb avec sudoers restreint | trois bras carte sur l'arbre livré, deux arbres refusés avant | J net −10,6 % b=12 Coder, −27 % b=1, prefill −5,7 % ; **à b=12 le régime réel est « 400 W »** (médiane 2 550 MHz, `sw_power_cap`) |
| **TF32 sur le cœur MLA au prefill, ≤ 2 048 clés** (C13-a + règle des clés) | +0,07 % PPL à 2 048 ; **± 3-6 % par texte à 8 192** → limité | GLM prefill 5 739 → **7 268 j/s** à L=2 048 |
| **C14 + C14-c** : `mla_1p` en grille (44,5 → 5,2 µs/couche) | ± 1 ulp bf16 contre l'ancien, capture 5/5, PPL indécidable (± 0,6 %) | GLM b=1 : noyaux du pas **−21 %**, servi 106 → 122,5 |
| **C15 niveau 1** (glue au bit) | 2 622 → 2 163 lancements, jetons identiques | GLM b=1 −0,43 ms/pas, J −5 % |
| **C15-3d** (glue compacte Coder, 8 warps) | 1 169 → 641 nœuds ; « experts égaux » jugé contre le témoin : B 21 % contre témoin ON/OFF 26 % ; capture 5/5 | **Coder b=12 : 1 397 · 0,210** (± 3 % de fenêtre ; +1 à +11 % t/s contre les témoins de la nuit, J égal) |
| hygiène : C4 sentinelles, C10 (b), C5-b, C17 en opt-in ; `chemin_moe=mma-a4` nommé ; ligne de régime lue sur la carte | — | — |

## 2. Les cellules, à horloge égale (`-lgc 2700`, harnais égal, deux lignes par concurrent au comparatif)
| cellule | acvram | llama.cpp | vLLM (meilleure config nommée) | où l'on est |
|---|---|---|---|---|
| Coder b=1 | **354 · 0,446** | 316,7 · 0,724 | 290,2 · 0,622 | devant les deux, sur les deux |
| Coder b=12 | **1 397 · 0,210** (0.6.24) | 890 · 0,170 | **1 626 · 0,136** (W4A16 Marlin) | devant llama.cpp en vitesse, derrière en J ; **derrière vLLM sur les deux** |
| Coder prefill | 17 784 | 15 532 | **20 824** | second |
| Coder PPL privé | **1,0094** | 1,0103 | — | devant ; KV int8 +0,63 % au décodage à 8 k (canal +0,205 %, opt-in, +43 % de temps) |
| GLM b=12 | 660 ± 20 · 0,320 | — | 858 · 0,397 (libre) | derrière −23 %, devant en J |
| GLM b=1 | 122,5 · ~1,0 | — | 183,5 (libre) | derrière −33 % |
| GLM prefill L=2 048 / 8 192 | 7 268 / 1 905 | — | 18 117 (libre) | ×2,5 derrière |
| 119B b=1 | (non chargé) | **24,2** experts en RAM | — | voir § 6 |
La revendication, mot pour mot, est dans `poste7-concurrents-2700-verite-b12` § 1 : **l'objectif « plus performant et plus économe que tous » n'est pas atteint à b=12 ni au prefill ; il l'est à b=1.**

## 3. Ce que la nuit a établi (faits, pas avis)
1. **Sous plafond de puissance, W est une constante** : les quatre noyaux d'experts à 398-402 W ; l'instruction par octet se lit dans l'horloge accordée (Marlin 1 950-2 265 MHz, mma2 2 692) donc dans le temps ; `-lgc` est un plafond, pas un plancher ; à horloge libre une cellule dépend de l'état thermique (1 337 → 1 460 en 28 min).
2. **mma2 est W4A4 par construction** (`mxf4nvf4`) ; GLM sert 13/46 couches en A4 à b ≥ 5, effet borné ≤ +0,4 % à 2 SE ; le « régime naturel au décodage » est mort deux fois (vitesse, qualité).
3. **Le pas GLM b=1 = 25 % d'espaces entre 2 163 nœuds** ; sur Coder b=12 les nœuds ne sont pas le poste (≤ 1,2 ms, gain réel −0,35 ms) ; l'hôte ne pèse que 4-7 %.
4. **Résolution des instruments** : `ppl-decode-kv` à 3 × 512 = ± 0,6 % (filet à bogue, pas juge d'équivalence) ; 3 × 12 fenêtres de prefill = ± 0,003 ; ± 0,001 = 2 000 paires ; une règle de lot porte sur la géo à 2 SE ; l'équivalence d'un noyau se prouve par noyau (± 1 ulp bf16 sur entrées réelles).
5. **Sur un MoE sous graphes, le défaut n'est pas déterministe** (26 % des top-k entre capture et eager, PPL inchangée) : « change la sortie » se juge contre ce témoin, sinon la règle interdit les graphes.
6. **Deux instruments = deux régimes** : nsys en rafale contre `certifie` soutenu (7,0 / 8,6 ms) ; ncu à routage synthétique (27 experts) contre invite réelle (45) ; un test qui compare le même chemin deux fois est vide.
7. Matériel : PCIe gen 5 × 8 = 22,6 Go/s ; `tl.dot` fp32 IEEE = 0,88 × cuBLAS (pas émulé) ; sm_120 : 99 Ko de shared par bloc.

## 4. Fermé cette nuit, avec le chiffre qui ferme
C1 W4A8 (1 148 ms contre 27,2 : A8 hors noyau, dépaquetage 3,7 × la bande, GEMM 79 TOPS) · C13-c forme 1 (× 26 : structure, pas précision) · C17 (1,234 à u=45, +2,2 % PPL = A4) · C16 / C16-bis / C16-GLM (hôte 0,35-0,64 ms) · Mesure 2 · NARROW_GEMM (1,002 ×) · niveau 2 TF32 décodage (SIMT) · VB (+0,27 %) · bras F bf16 (8 175 < 8 300) · C5 int8 par jeton (+0,44 %, bf16 non servable) · C5-b vitesse (+43 %) · C15 niveau 2 (arithmétique du chemin, biais suspecté) · C15-3 / 3b / 3c (routeur 100 × trop lent, sélection 7,46 µs) · C4 scellé (arithmétique du lot) · niveau 3 « ≤ 450 » et « ≤ 600 » (comptes faux) · TF32 à 8 192 (± 6 %) · C9 cache PCIe (parité au mieux).

## 5. Mes prédictions fausses (toutes tenues pour fausses, aucune règle déplacée après lecture)
MoE v1 4-5 ms (1,1) · G1 900-1 100 (725) · experts prefill 60-70 % (44) · C1 route (ii) impossible · 30 ± 8 distincts (45,4) · W mma2 280-340 (401) · parité sous 2 700 (0,893) · prefill J −10/−15 % (−2) · concurrents sous 2 700 (−16/−20 % chez eux) · C16 ≥ 1 ms (0,35) · C14-bis « plancher = combine » (grille) · « 1 850 · 0,15 » (parité vitesse au mieux) · nœuds Coder ≈ 2 000 (1 169) · C13-c 190 ms (× 26) · « émulation » (retirée) · dossier capture du niveau 2 (test vide, retiré) · « + 1 ulp par ligne » · C5-b ≤ +0,15 % (0,205) · J 0,195-0,205 (0,210) · tables rope à 8 k (juste au pas 1) · mes heures (jusqu'à 7 h d'avance, 25 titres corrigés). Ce qui a tenu : la physique du plafond une fois mesurée, « B ≤ témoin » comme juge, l'ordre des chantiers après chaque budget.

## 6. Ce qui reste, pour la reprise, avec son chiffre
* **b=12 Coder contre vLLM Marlin (1 626 · 0,136)** : l'écart de J est dans les experts Marlin à 400 W (54 % du pas) — aucun chemin connu ; l'écart de vitesse (−14 %) : C15-3d sélection ≤ 3 µs, `_etroit`, puis la bande de Marlin (1,07 contre 1,24-1,41 To/s).
* **GLM** : niveau 2 (biais d'arrondi à prouver : moyenne signée), niveau 3 (≤ 700 nœuds : −1,6 ms b=1), C14-b (2 ms autour de l'attention à b=12), C13-c réécrit (structure « q en tranches », sonde bf16 ≤ 2 ms/appel d'abord ; 7 268 → ≈ 10 000).
* **Prefill Coder** : C15-prefill (norm 9,1 → ≤ 2, glue 8,2 → ≤ 3 : ≥ 20 500 j/s, PPL au bit).
* **Parc** : gemma-4-26B-A4B b=1 en eager (2,9 × plus lent, capture OOM au godet 1) et Ornith-35B qui ne décode pas — deux bogues P1, catalogue marqué, règle « un alias se prouve par un jeton décodé ».
* Hygiène : C5-b nsys, déterminisme ON/OFF de Coder b=12, C3 MTP `ensure_hist`, C10 (a), tri des worktrees, PPL longue tf32.
* **Question à l'utilisateur — 119B** : llama.cpp fait **24,2 j/s** avec les experts en RAM ; acvram par PCIe × 8 plafonne à la parité (h ≥ 0,55, le h prédit) ; dépasser = (a) 3080 Ti calculante (12 Go d'experts int8, ≥ 3 jours, gain non chiffré), (b) experts sur processeur (égaler llama.cpp, pas le dépasser), (c) aucune — le 119B reste à llama.cpp et acvram garde son avance sur les modèles qui tiennent en 32 Go. Mon avis : (c), sauf si le 119B est une priorité en soi.

## 7. Règles écrites cette nuit (REGLES, par section)
§ 1 effort max / high ; push automatique ; § 2 aucun indexeur pendant une fenêtre, trois consommateurs CPU en en-tête ; § 3 instrument = fichier suivi, un .deb qui touche la carte se prouve par ses bras, résolution des scellés PPL (SE, 2 000 paires, géo à 2 SE), scellé de J relatif à la fenêtre, horloge SM moyenne en en-tête, alias prouvé par un jeton décodé, arithmétique MLA jugée à 8 192 aussi ; § 4 horloge libre thermique, contrôle vide, PPL GLM préfixée ; § 4 bis n/a ≠ 0, PPL b=1 ne juge pas un chemin de lot ; § 6 HOME du processus profilé, graph-trace exact, lignes de la nuit retirées ; § 7 « B ≤ témoin » sur MoE ; § 9 PCIe 22,6, `-lgc` plafond, `tl.dot` non émulé, `mxf4nvf4` = W4A4.

## Ordre
* **chef** — ETAT au propre à partir de ce bilan (§ 1, § 2, § 6), INDEX, dernier push à 07 h 45 machine ; la question § 6 à l'utilisateur telle quelle.
* **poste1 / poste2** — rangs restants de `poste7-cloture-nuit-0540` (ligne de régime, gemma repli annoncé, vLLM KV fp8, 0.6.25) ; rien de nouveau.
