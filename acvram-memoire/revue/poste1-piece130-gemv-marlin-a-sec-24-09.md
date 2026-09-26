# Pièce 130 — GEMV sur disposition Marlin à la vitesse du naturel : dossier à sec, aucun code — 24/09 03 h 0x (poste1)

But (chef) : une seule disposition (Marlin) sans perte à b=1, pour ne plus doubler 8,5 Gio de poids (129 (2) :
capacité KV 55 136 → 32 768 jetons). Point de départ : 129 (1), GEMV Marlin **+15,0 %** en projection b=1.

## 1. Ce que font les deux noyaux (fichier:ligne, main + poste1-mtp cce34391)
| | naturel `nvfp4_gemv_kernel` (`acvram_kernels.cu:264`) | Marlin `nvfp4_gemv_marlin_kernel` (`:2047`) |
|---|---|---|
| unité de bloc | ROWS lignes de sortie, fils répartis sur K (`:293-307`) | UNE tuile de 64 colonnes, 8 warps sur 1/8 des tuiles k chacun (`:2081-2105`) |
| lecture des poids | chaque ligne = K/2 octets CONTIGUS (2 560 o à K = 5 120), `uint4` par fil, double tampon (`:309-330`) | par tuile k : 512 o contigus, puis saut de `LN·512` o (= toute la ligne k de la disposition, 8·N o) à la tuile suivante (`:2089-2093`) |
| activation | en registres (`load_xs`, `:225-253`) | copiée en mémoire partagée par CHAQUE bloc (K floats, `:2078-2084`) + `__syncthreads` ; K ≤ 11 264 (`:2226`) → down en deux lancements |
| parallélisme à N petit | grille N/ROWS × k_splits | split-K auto `mb_splitk` (`:2208-2214`) : S doublé tant que N/64·S < 384 → **S = 8 à N = 5 120**, réduction des S partiels par le DERNIER bloc (compteur, `__threadfence`) |
| registres (ptxas sm_120a) | — | 61 (bf16, NW = 1), 0 déversement ; shared K·4 + 2 Kio → 2-3 blocs/SM à K = 8 704 |

## 2. Lecture des pertes de 129 (1) par ces mécanismes
| forme | N / 64 | S auto | perte | cause probable |
|---|---|---|---|---|
| GDN out, o_proj (5 120 × 6 144) | 80 | **8** | **+25 %** | queue : 8 partiels [8][5 120] fp32 écrits/relus, sommés par un seul bloc par tuile |
| down (5 120 × 17 408, deux moitiés) | 80 | 8 × 2 lancements | +17 % | la même queue, deux fois, + un `add` |
| gate‖up (34 816 × 5 120) | 544 | 1 | **+15 %** | accès : 512 o par saut de 278 Kio (N grand) contre des lignes contiguës ; x rechargé par 544 blocs |
| tête (248 320 × 5 120) | 3 880 | 1 | +8 % | idem, x amorti sur plus de colonnes par octet de L2 |
| q (12 288) / GDN qkv (10 240) / GDN gate (6 144) | 192 / 160 / 96 | 2 / 4 / 4 | +7 / +7 / +4 % | intermédiaire |
Deux causes, donc : **la queue du split-K à S = 8** (petits N) et **le motif d'accès à tuile unique** (grands N).
Bande atteinte sur gate‖up : naturel 1,57 To/s (88 % du pic), Marlin 1,36 To/s (76 %).

## 3. Leviers, du moins cher au plus cher
0. **Sans code** (banc 129 (1) rejoué avec `ACVRAM_GEMV_SPLITK` = 1, 2, 4, 8 par forme) : le meilleur S par forme.
   Prédit : à N = 5 120, S = 2-4 bat S = 8 (+25 % → +5 à +10 %) ; il ne change rien à gate‖up (S = 1 déjà).
1. **Plusieurs tuiles de colonnes par bloc** (2 ou 4 adjacentes : 1-2 Kio contigus par ligne k, x chargé une fois pour
   128-256 colonnes). Prédit gate‖up +15 % → **+0 à +5 %**, tête +8 → +0-3 %.
2. **x lu en global (L1)**, sans copie en mémoire partagée : la borne K ≤ 11 264 tombe (down en un lancement), plus de
   `__syncthreads` en tête. Prédit down +17 % → **+3 à +8 %**.
3. Réduction du split-K sans dernier bloc sériel (S partiels sommés dans l'ordre par un noyau d'épilogue, ou S ≤ 4 fixé)
   si 0 ne suffit pas.

## 4. Prédiction scellée (avant toute mesure et tout code)
* Après 0 + 1 + 2 : projection b=1 Marlin seul **10,13 × (1,00 à 1,04) = 10,1-10,5 ms** (129 (1) : 11,65). Critère du
  chef : ≤ +3 % → tenu dans la moitié basse de la fourchette.
* **Réfuté si** après 1 et 2, gate‖up reste **> +8 %** : le coût est alors dans le calcul par élément (décodage E2M1
  + échelle S0E5M3 par PRMT/IMAD toutes les 16 valeurs, contre une échelle E4M3 lue une fois par bloc de 16 dans le
  naturel), pas dans l'accès → contrôle ncu (dram__throughput contre sm__inst_executed) avant tout autre code.
* **Ce qui me gênerait** : que le levier 0 seul donne déjà ≤ 3 % sur les petits N, et que gate‖up résiste à 1 — la
  disposition Marlin, faite pour les tensor cores (tuiles 16 × 64), serait alors intrinsèquement moins bonne pour un
  GEMV qu'une disposition par lignes. Il faudrait alors revenir aux doubles, pour gate‖up seulement.

## 5. Ordre de travail proposé
Carte, après 06:00, < 2 min : levier 0 (aucun code) + ncu d'une forme (gate‖up) sur les deux noyaux. Puis code 1 + 2 (un
noyau `nvfp4_gemv_marlin` v2, opt-in derrière la disposition 129), tests au juge 2⁻⁷ par ligne + reproductibilité +
bras cassants, banc 129 (1) rejoué.
