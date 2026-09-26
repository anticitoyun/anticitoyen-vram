# poste7 — C1 noyau faux d'un facteur 42 (1 148 ms contre 27,2) : C1 se ferme sur le poste prefill comme écrit d'avance ; le prefill Coder reste Marlin W4A16 et son chantier devient C15-prefill (norm + glue, −15 ms, ≥ 20 500 j/s), sans risque de qualité (20/09, 00 h 03, heure du commit)

Source : `verdict-c1-noyau-19-09` (poste2 5d55967a) ; `poste7-c1-route-i-c17-19-09` addendum 23 h 10 (« à écriture pleine le trafic seul vaut 27 ms ; écriture ≈ 1,0 × → C1 se ferme sur le poste prefill, dit d'avance ») ; `verdict-budget-prefill-19-09` (P2 i8c 96,8 ms : experts 49,5, projections 12,0, flash 12,7, glue 8,2, norm 9,1) ; `poste7-concurrents-2700-verite-b12-20-09` (prefill 17 784 contre vLLM 20 824).

## 1. Le verdict et ce qu'il nomme
| poste C1 à L=2 047 | ms | lecture |
|---|---|---|
| quantification A8 de l'activation (torch élémentaire, 5 noyaux × 2 301 appels) | **791** | pas dans le noyau : défaut d'implémentation, pas de physique |
| dépaquetage W8r (tampon) | **67,2** | 3,7 × le plancher de bande (18 ms) : l'écriture n'est pas absorbée par le L2 — c'est l'issue écrite le 19/09 à 23 h 10 |
| GEMM int8 groupée | **93,7** | 7,4 TFLOP en 93,7 ms = 79 TOPS, ≈ 10 % du pic int8 ; Marlin W4A16 fait tout le poste en 49,5 |
| total contre scellé | **1 148 contre 27,2** ; pas 1 290 ms / 1 587 j/s | test carte 5/19 (formes 768 × 2 048 à ks=128 : `invalid argument`) ; PPL 1,0174 tenue, témoin au bit |

Même A8 replacé dans le noyau, la route (i) vaut **161 ms = 3,3 × Marlin** ; pour seulement égaler Marlin il faudrait ×4 sur la GEMM et ×3 sur le dépaquetage, et le scellé (27,2) laissait 0 ms à la GEMM une fois le trafic compté. **C1 est fermé sur le poste prefill** — la clause écrite d'avance s'applique ; branche `poste1-c1-noyau` archivée avec ses chiffres ; si quelqu'un le rouvre un jour, trois portes à sec avant toute carte : GEMM int8 groupée ≥ 300 TOPS sous ncu sur une couche réelle, dépaquetage ≤ 25 ms/prefill (écriture DRAM ≤ 0,5 × tampon), A8 dans le noyau. Le W4A8 aurait aussi coûté +0,0019 de PPL (porte) : le prefill Coder reste **Marlin W4A16, 1,0094**, second derrière vLLM (−15 %).

## 2. Le chantier prefill qui reste : C15-prefill, le même geste que la nuit entière
Le budget P2 i8c (96,8 ms de noyaux) porte **norm 9,1 + glue 8,2 = 17,3 ms (18 %) pour ~2 Go de trafic** — la norm à ×8 sa borne DRAM, la glue en lancements. vLLM fait 20 824 j/s **avec le même noyau Marlin** : son avance est hors experts (MoE fusionnée, attention, peu de lancements), pas dans W4A8. **C15-prefill** (poste1, fiche demain, code à sec) : rmsnorm + résidu + cast fusionnés en prologue des projections (9,1 → ≤ 2,0), glue du routeur et des permutations (8,2 → ≤ 3,0), attention paginée inchangée ; scellé : noyaux 96,8 → **≤ 83 ms**, prefill servi **≥ 20 500 j/s** (17 784 aujourd'hui, vLLM 20 824), PPL **au bit** contre le défaut sur 3 tranches (même arithmétique : tout écart est un bogue), capture inchangée ; réfutation : ≤ 83 ms de noyaux mais < 19 500 servi → le hors-noyaux du prefill (11 %) est le poste suivant, nsys `graph` avant d'insister. Prédiction : 20 500-21 500. Zéro question de PPL, zéro porte — c'est ce qui le range devant tout W4A8.

## Ordre
* **poste1** — C1 : fiche close avec § 1, branche archivée, rien d'autre ; C15-prefill en fiche (§ 2) après le niveau 2 ; C13-c fenêtre ; C5-b sous-agent.
* **poste2** — file inchangée : niveau 2 + PPL (f8a495a3) → C13-c → C9 si temps.
* **chef** — ETAT : C1 fermé (1 148 ms / 27,2 ; A8 hors noyau, dépaquetage 67, GEMM 79 TOPS), prefill Coder = Marlin W4A16, C15-prefill scellé ; INDEX ; commit + push.
