# Verdict — 142, famille 24B mistral/llama : Cydonia-Magnum-Diamond-24B, disposition Marlin unique + v2 — 24/09 (poste1)

* **instrument** : `scratchpad/poste1-p142-24-09/prise-famille.sh` (copie figée ; kl : kl-chemins 5 invites neuves × 8 pas +
  témoin T2 ; eval PPL fenêtres de la 102 ; b8/b1 : certifie-b12 CERT_PUR, ABBA A B B A A B B A A B, CERT_CIBLE_S=15,
  -lgc 2700) ; banc par forme `banc-v2-mistral24.py` (copie du banc 130, v1 en tranches ≤ KMAX)
* **commit** : 46657c74 (prises), 865b10eb (banc) ; code = poste1-mtp après la 146 (garde B ≥ A)
* **régime** : RTX 5090, cpu-safe 100 → 100, compute-apps début = fin (llama-server sur la 3080 Ti) ; A et B NOMINAL,
  0/40 exilée ; B `marlin(doubles=0,seuls=161)` ; prises poste1-p142-mistral24-{kl,b8,b1} 13:56 → 14:19, banc 14:22
* **scellé** : `scratchpad/poste1-p142-24-09/scelle-mistral24.md` + addendum (avant le banc)
* **mesuré** (médianes de 5 lots) :

| | A | B | B/A | prédit | |
|---|---|---|---|---|---|
| b=8 pas | 19,335 ms, 413,8 t/s, 0,967 J/j, 400 W (bridé) | 10,187 ms, 785,4 t/s, 0,508 J/j | **0,527** (+89,8 %, J −47,5 %) | 0,55-0,75 | **TENU** (hors fourchette, côté favorable) |
| b=1 pas | 9,761 ms, 102,4 t/s, 3,90 J/j, 400 W (bridé, SM 2 260) | 10,191 ms, 98,1 t/s, 3,97 J/j, 390 W (SM 2 677) | **1,044** | 0,96-1,02 | **FAUX** (> 1,03) |
| KL | T2 0,001229 | 0,000731, argmax 40/40 | ≤ 0,002458 | | TENU |
| PPL | 3,2724 (11,7 s) | 3,2724 (12,46 s) | = ; eval +6,5 % | ≤ +0,5 % | TENU |

  Écart entre lots ≤ 0,5 % par bras.
* **banc b=1 par forme** (µs, naturel → v2 au réglage servi TPB 1 / S 0) : down 5 120 × 32 768 60,55 → 61,26 (+1,2 %) ;
  **gate‖up 65 536 × 5 120 116,46 → 134,85 (+15,8 %)**, 118,18 à TPB 2 ; qkv 14,65 → 15,07 (+2,9 %) ;
  **o 5 120 × 4 096 10,41 → 11,45 (+10 %)**, aucun réglage sous +9 % (queue du split-K, comme GDN out et o à la 130) ;
  tête 287,7 → 246,2 (−14,5 %). Projection au pas : réglage servi +9,3 % des GEMV, **réglage unique 2/0 +2,0 %**,
  meilleur par forme +1,0 %.
* **verdict** : b=8 TENU (+89,8 %, le plus fort gain des trois familles denses) ; KL et PPL TENUES ; **b=1 FAUX : +4,4 %
  de pas**. Ma prédiction de cause (down, K = 32 768) était FAUSSE (+1,2 %). L'excès vient de **gate‖up à TPB = 1** (N = 65 536,
  deux fois celle de Qwen3.8 où TPB 1 était le meilleur réglage) et de **o** (K = 4 096, queue du split-K).
* **correctif prédit (non codé)** : choisir TPB par forme, soit TPB 2 quand N ≥ 49 152 : gate‖up de +15,8 % à +1,5 % ;
  projection b=1 de +9,3 % à ≈ +2 % des GEMV, soit ≈ +1 % du pas mesuré ; plus le reste nommé de o. À mesurer :
  même ABBA b=1, prédit B/A ≤ 1,02 ; Qwen3.8 inchangé (N ≤ 34 816 → TPB 1).
* **incidence sur la bascule par défaut** : à b=1, les modèles de cette forme perdraient ≈ 4 % tant que TPB n'est pas
  choisi par forme.

## 142 bis — TPB par forme (d7b14777, prise poste1-p142bis-tpb 14:26 → 14:48, scellé `scelle-tpb.md` écrit avant)
* Code : `kernels._tpb_marlin(N)`, ACVRAM_GEMV_MARLIN_TPB=0 (nouveau défaut) → 2 si N ≥ 49 152, sinon 1. Sortie AU BIT de
  TPB 1 : `test_tpb_par_forme_au_bit_de_tpb1`, 5 formes ; 28 tests verts. Cassant (seuil 64) **ROUGE** sur (34 816, 5 120) :
  TPB 2 y lance 272 blocs, d'où S = 2 contre 1. Ma prédiction nommait les petites N, restées vertes, car S plafonne à 8 des
  deux côtés : détail faux, mais le cassant est rouge.
* ABBA b=1, médianes de 5 lots, A naturel, B unique + v2 + TPB par forme :
  * **24B Cydonia** : A 9,748 ms, B **9,542 ms**, **B/A 0,979** (prédit 0,99-1,02 ; mieux que prédit), J/j 3,90 → 3,82.
    Le b=1 FAUX (1,044) devient TENU (seuil du chef ≤ 1,02).
  * **Qwen3.8** : A 13,235 ms, B 13,183 ms, **B/A 0,996** (prédit 0,985-1,005 ; 130 : 0,9967), J/j 5,10 → 4,74 (−7,3 %). TENU.
* Conséquence : la configuration à basculer par défaut est unique + v2 + **TPB par forme** (défaut 0).
