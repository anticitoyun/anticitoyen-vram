# 4q8 étape 1 — à sec (poste1, 01/10) : projections int8 et tête au décodage, borne d'octets contre mesure ; l'écart du Coder est le plancher par appel déjà jugé, la branche NVFP4 plafonne à −0,20 ms : aucune des deux branches n'atteint le seuil utile

* instrument : manifestes lus (`acvram_manifest.json`, octets = Σ numel × bpw / 8) ; mesures reprises, aucune nouvelle :
  nsys familles du 22/09 (`verdict-nsys-familles-22-09`, Coder b=12), banc par appel de la 100 A (`poste1-piece100a-j1-marlin-dense-23-09`,
  L2 froid, graphe de 48 appels, M=12), cellule d'attribution 118 (`poste2-piece118-attribution-23-09`), 185 c (plancher par appel),
  188 et 195 (27B b=8). Code lu : `acvram/kernels/__init__.py`, `acvram/kernels/gemm_etroit.py`.
* commit : poste1-4q8 = origin/main a77970f45. · régime : sans objet (à sec). · carte : aucune.
* bande de lecture retenue : 1,2-1,45 To/s (REGLES § 9, motif tuilé ; 1 050 Go/s = celle de nos GEMV).
* seuil utile (écrit avant le calcul) : une branche s'ouvre si son plafond dépasse **3 % du mur b=12 du Coder (0,21 ms sur 7,083)**.
* verdict : **ni la branche noyau au bit, ni la branche NVFP4** (§ 4).

## 1. Ce qui sert aujourd'hui (fichier:ligne)
* Coder servi = `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c` (`outils/poste/alias-servis-20-09.txt:3`). q/k/v/o sont en int8 PAR CANAL (groupe = K), la tête est
  en int8 g128 et le routeur en bf16.
* Le noyau par canal 195 (K entier) ne sert que les formes de sa table (`gemm_etroit.py:365-375` : uniquement des formes du 27B). Une forme absente
  « reste sur le noyau servi » (`gemm_etroit.py:394-401`). Les formes du Coder (qkv 5 120 × 2 048, o 2 048 × 4 096, tête 151 936 × 2 048)
  passent donc par le noyau à tranches K sur la vue g128 (`int8_matmul` → `etroit_triton`, `kernels/__init__.py:1649`). Le découpage vise
  2 programmes par SM (`decouper_k`, `gemm_etroit.py:258-263`) : qkv 80 tuiles × 4 tranches, o 32 tuiles × 11 tranches.

## 2. Bornes d'octets à M=12 (indépendantes de M : poids lus une fois)
| modèle | tenseurs | Go | borne 1,45-1,2 To/s | mesuré | débit | écart à la borne |
|---|---|---|---|---|---|---|
| Coder i8c | qkv ×48 | 0,504 | 0,348-0,420 ms | 48 × 10,10 µs = 0,485 ms (100 A) | 1,04 To/s | +0,07-0,14 ms |
| | o ×48 | 0,403 | 0,278-0,336 ms | 48 × 11,29 µs = 0,542 ms (100 A) | 0,74 To/s | +0,21-0,26 ms |
| | tête | 0,318 | 0,220-0,265 ms | ≈ 0,21 ms (reste du nsys) | ≈ à la bande | ≈ 0 |
| | routeur bf16 ×48 | 0,025 | 0,017-0,021 ms | ≈ 0,10 ms (116 ter) | 0,25 To/s | +0,08 ms |
| | **Σ (145 lancements)** | **1,250** | **0,862-1,042 ms** | **1,335 ms (nsys 22/09)** | 0,94 To/s | **+0,29-0,47 ms = 4,1-6,7 % du mur** |
| Qwen3.8-27B mixte-i8c | attn 1,679 + GDN 5,540 + MLP int8 2,140 + tête 1,272 | 10,63 | 7,33-8,86 ms | 8,37 ms à b=8 avant 195 (188) ; ≈ 7,51 après (−0,86, 195) | 1,27 → ≈ 1,42 To/s | **≈ 0 (dans la bande)** |

Les deux mesures du Coder se recoupent : 100 A donne 1,027 ms (48 × 21,39 µs), plus la tête et le routeur, soit 1,34-1,39 ms contre 1,335 au nsys.
Pour le 27B, je n'ai aucune mesure à b=12. Les octets ne dépendent pas de M, mais le chiffre de b=8 n'est qu'une borne inférieure du temps à b=12.

## 3. Attribution de l'écart du Coder
* Plancher fixe par appel (185 c, mesuré : tout sauf la lecture des poids) : 3,3-4,0 µs. Les 97 appels int8 du Coder font 0,32-0,39 ms.
  **L'écart total (0,29-0,47 ms) tient tout entier dans ce plancher** : sur la partie qui lit les poids, les noyaux sont déjà à la bande.
* Seule o dépasse le plancher : 11,29 − (5,8 à 7,0) = 4,3-5,5 µs, contre 3,3-4,0, soit au plus ≈ 1-2 µs × 48 = **0,05-0,10 ms**.
  Explication probable : N = 2 048 ne donne que 32 tuiles, d'où 11 tranches de 3 groupes et une réduction plus longue.
* Le plancher a déjà été jugé au bit. En 185 c : BN 32/16 donne −0,058 ms/pas, l'épilogue ≈ 0 et les segments compacts −1,24 µs/appel, sous le
  seuil. En 218 (persistant), le gain est de 3-6 % pour ≈ 2 jours, avec « NON recommandé ». La 195 (K entier, hors bit ± 1 ulp) ne s'applique pas à o
  du Coder : à K entier, 2 048 / 32 = 64 programmes, pour 170 SM.

## 4. Décision entre les deux branches
* **Branche noyau int8 au bit — NON.** Son plafond, à partition et ordre de somme inchangés, est l'excès propre à o : 0,05-0,10 ms (0,7-1,4 %),
  sous le seuil de 0,21 ms. Le reste de l'écart est le plancher par appel, que seule une réduction du nombre de lancements atteindrait (fusion,
  persistant). Le persistant est déjà chiffré par la 218 et non recommandé.
* **Branche q/k/v/o NVFP4 W4A16, tête int8 — NON.**
  - Le chemin NVFP4 servi au défaut est le Marlin seul (`PROJ_MARLIN=1`, `kernels/__init__.py:1231`). Le Triton dense étroit (`:847`) ne sert qu'un
    poids que la passe Marlin n'a pas converti.
  - Mesuré par appel en 100 A à M=12 : qkv 7,59 µs, o 9,62 µs, contre 10,10 et 11,29 en int8, soit −4,18 µs × 48 = **−0,20 ms (−2,8 %)**.
    Le gain d'octets (−0,40 Go) se perd parce que le Marlin ne lit ces formes qu'à 0,49-0,78 To/s.
  - Sous le seuil, et à prouver en qualité : S9 (projections NVFP4 RTN, tête int8) est assemblé mais non qualifié. S8, qui avait aussi la tête
    en NVFP4, fait 1,064 de PPL, dont environ +5 % dû à la tête et rien aux projections (poste6, 23/09).
  - Le service le contredit : à b=12, S8 sert à −5,2 points de S1b (118 : 1 541,9 contre 1 645,6 t/s), alors qu'il lit moins d'octets.
  - A6 (+8,79 %) est déjà séparé par les variantes D/E du 15/09 (D, tête int8 seule : PPL 0,987) et par S8/S1b du 23/09. La tête est la coupable.
* Hors du champ b=12, à noter : **à b=1, S8 sert à +8,87 % d'i8c** (118). La lecture en GEMV y paie les octets. Une pièce « S9 à b=1 » (qualification
  complète, puis cellule b=1) serait la seule suite NVFP4 dont le plafond dépasse le seuil.

## 5. Ce qui reste des 19,8 %
Coder b=12 : 1,335 ms, dont 0,86-1,04 de lecture à la bande et 0,32-0,39 ms de plancher par appel. Le 27B est dans la bande. Aucun levier au bit
sur le noyau ; le plancher ne se prend qu'en lançant moins.
