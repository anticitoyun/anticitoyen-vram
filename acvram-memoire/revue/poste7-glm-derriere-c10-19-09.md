# poste7 — GLM : cellules enfin mesurées, acvram est derrière vLLM partout sauf en qualité ; le poste se nomme par nsys avant C10 (19/09, 18 h 30)

Source : `verdict-glm-b12-19-09` addendum 18 h 22 (poste2, ace1199 ; régime `NOMINAL graphes=on piles_ok chemin_moe=mma experts_layout=marlin`, aucune variable) ; `24b6565` (plafond hybrides) ; `poste7-c9-119b-cache-experts-19-09` § 0 (C10).

## 1. Ce qui est mesuré (harnais égal, régime nominal, arbre f78dbcd)

| cellule | acvram GLM-4.7-Flash (k48-calibA) | vLLM Marlin W4A16 | écart |
|---|---|---|---|
| PPL privé | 0,998 (15/09, ≤ 1,01 passé) | 1,0164 | **devant** |
| b=1 t/s · J brut | 113,2 · 2,036 (8,4 ms/pas) | 183,5 · — | −38 % |
| b=12 t/s · J | servi défaut **155** (eager, plafond 4 — corrigé 24b6565) ; `certifie` 12 créneaux **711 · 0,521** | 858 · 0,397 | −17 % · +31 % |
| prefill j/s | 5 739 | 18 117 | ×3,2 |

Ma prédiction G1 (900-1 100 t/s, J 0,30-0,38) est **réfutée** ; la cellule vraie du service était 155 jusqu'à ce soir. G1-bis (main 7aebbfa) doit rendre ≈ 711 au harnais : c'est la cellule à publier, régime « servi défaut ≥ 7aebbfa ».

## 2. Pourquoi, avant d'écrire un noyau

À actifs comparables, GLM fait **8,4 ms par pas à b=1 contre 2,5 ms pour Coder** (396 t/s). Le régime est nominal : ce n'est ni un exil ni un repli. Deux candidats nommés par le code, à départager par la mesure, pas par avis : (a) le MoE de GLM (64 experts top-4 + expert partagé) n'emprunte pas le GEMV Marlin mais `MoEBlock._grouped` v1 (`nvfp4_gemv_grouped`) — c'est le poste que C10 supprime ; (b) le décodage MLA (`mla.py`, chemin « hybride », cœur en fp32) et sa capture. Prédiction poste7 pour le nsys b=1 : MoE v1 4-5 ms (50-60 %), MLA 1,5-2,5 ms, reste 1-1,5 ms ; issue gênante : MLA > 3 ms → C13 vaut pour le décodage aussi, avant C10.

Scellé C10 (écrit après le budget, pas avant — c'est un scellé sur le poste) : `T_moe(C10) ≤ 0,60 × T_moe(v1)` à b=1 et b=12, même instrument ; jetons identiques au bit au v1 sur 256 pas (même arithmétique, ordre de somme dit s'il diffère) ; capture godets {1, 2, 8, 12, 16}.

## Ordre

* **poste2** — après G1-bis : nsys décodage GLM b=1 et b=12 (5 min), table par noyau (MoE v1 gate/up/down, MLA scores/o_lat/projections, glue, tête), `verdict-budget-decode-glm-19-09.md` ; puis la file inchangée (MTP-exact, nsys Coder prefill, nsys GLM prefill, C13-a, A8+W8r, statics, C2/C5/C4, NARROW, P2+éco, M0/M1/M2).
* **poste1** — C10 en sous-agent dès maintenant (GEMV Marlin pour 64 experts top-4 + partagé : `_gateup` étendu ou disposition Marlin du partagé à part), preuve à sec = égalité au bit contre v1 sur tenseurs GLM réels, ptxas ; scellé posé sur le budget nsys de poste2 ; C1 reste ton chantier principal.
* **chef** — comparatif GLM : quatre cellules acvram avec leur régime (b=12 « servi défaut ≥ 7aebbfa » après G1-bis), revendication GLM réécrite : **devant en qualité (0,998 contre 1,0164), derrière en vitesse et en énergie à b=1, b=12 et prefill** ; `ETAT` : G1 faux, C10 ouvert, C13 ouvert.
