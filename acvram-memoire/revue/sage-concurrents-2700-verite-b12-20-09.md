# Sage — concurrents sous 2 700 : mes prédictions fausses (−16 % / −20 % de vitesse chez eux, J −20 % / −31 %) ; à horloge égale acvram est devant à b=1, second au prefill, et à b=12 derrière vLLM Marlin en vitesse ET en énergie — la revendication se réécrit telle quelle, et le chantier qui compte est C15 (nœuds + glue) pour les deux modèles, avant C1 (19/09, 23 h 55, heure du commit)

Source : `verdict-concurrents-2700-19-09` (Manon 447d715, mêmes instruments que le 17/09, harnais égal) ; `verdict-eco-2700-defaut-19-09` (acvram 2 700) ; `verdict-nsys-coder-b12-19-09` (Coder b=12 : Marlin 3,73, glue 1,58, `_etroit` 1,06, 1,9 ms hors noyaux) ; `sage-glm-b1-noeuds-c15-niveau3-20-09` (≈ 1 µs par nœud de graphe).

## 1. La table, à horloge égale (`-lgc 2700`, harnais égal)
| cellule | acvram (défaut 0.6.23) | llama.cpp Q4_K_M | vLLM W4A16 Marlin | verdict |
|---|---|---|---|---|
| Coder b=1 t/s · J net | **354 · 0,446** | 316,7 · 0,724 | 290,2 · 0,622 | **devant les deux, sur les deux** |
| Coder b=12 | 1 341 · 0,2005 | 890,4 · 0,170 | **1 626 · 0,136** | devant llama.cpp en vitesse (+51 %), derrière en J (+18 %) ; **derrière vLLM sur les deux (−18 % t/s, +47 % J)** |
| Coder prefill j/s | 17 784 | 15 532 | **20 824** | second (−15 % sur vLLM) |

Mes prédictions (« llama.cpp 1 040-1 065 · 0,19-0,20, vLLM 1 170-1 200 · 0,18-0,19 ») sont **fausses** : le verrou leur coûte 16-20 % de vitesse et leur rend 20-31 % de joules — plus qu'à nous (−1,8 % / −10,6 %) ; et la ligne vLLM qui compte est **W4A16 Marlin (2 031 à horloge libre)**, pas FP4/FlashInfer (1 198) : le comparatif nomme la configuration dans chaque ligne et la revendication se juge contre la meilleure. **Revendication Coder, mot pour mot** : *à horloge égale (2 700), devant llama.cpp et vLLM à b=1 en vitesse et en énergie ; à b=12, devant llama.cpp en vitesse (+51 %) et derrière en énergie (+18 %), derrière vLLM Marlin en vitesse (−18 %) et en énergie (+47 %) ; prefill second (−15 % sur vLLM) ; PPL 1,0094 < 1,0103.* Deux lignes par concurrent (libre, 2 700), rien d'autre. L'objectif de l'utilisateur (« plus performant et plus économe que tous ») **n'est pas atteint à b=12** : Jérôme le dit tel quel.

## 2. D'où vient l'écart b=12, et donc l'ordre du jour suivant
vLLM b=12 : 7,4 ms/pas à 221 W net ; nous : 8,95 ms à 269 W. Notre pas b=12 (nsys) : Marlin 3,73 (le même noyau que le leur) + glue 1,58 + `_etroit` 1,06 + attention/reste ≈ 0,5 + **1,9 ms hors noyaux** (≈ 2 000 nœuds × 1 µs, comme à b=1 GLM). Ce qu'ils n'ont pas : nos ~1 700 élémentaires et nos 2 000 nœuds (MoE fusionnée, ~10 nœuds par couche). **C15 niveaux 2-3 sur Coder** = glue −1,0 et nœuds −1,4 → ≈ 6,5 ms → **≈ 1 850 t/s, J ≈ 0,15** (prédiction, scellée à la fiche) : devant vLLM sur les deux à b=12, si Marlin tient (le poste experts est commun). C1 (prefill) vaut +25 % → ≈ 22 000 > 20 824, second levier. **Ordre du jour suivant (Océane) : C15 niveau 2 (test + fenêtre) → C15 niveau 3 Coder et GLM → C1 noyau → C13-c forme 1** ; C5-b et C14-b derrière. Rien de nouveau cette nuit : le bilan de 07 h 30 fixe les scellés.

## Ordre
* **Jérôme** — comparatif : table § 1 avec les deux lignes par concurrent et la configuration vLLM nommée ; revendication § 1 mot pour mot ; **à l'utilisateur, une ligne** : « à horloge égale, b=12 : derrière vLLM Marlin en vitesse (−18 %) et en énergie (+47 %) ; b=1 devant les deux ; prefill second — chantier C15 (glue + nœuds) prédit ≈ 1 850 t/s · 0,15 J » ; ETAT ; INDEX ; commit + push.
* **Manon** — test niveau 2 (3 min), C1 noyau à son commit, C9 ; rien de neuf.
* **Océane** — ordre du jour § 2 ; C15 niveau 2 dès le test.
