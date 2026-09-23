# Lecture de la passe ncu (Laure, verdict-ncu-gemv-experts-rpw-18-09, 344ffcd) — ce que le tableau dit, ce qu'il ne dit pas, et ce que serait le pas suivant (Laurine, 18/09, aucune ligne de noyau écrite)

## 1. Ce que rpw = 4 a fait, et pourquoi (confirmé)

gate/up 95,5 → 87,9 µs (−8 %), down 81,0 → 65,2 (−20 %) ; requêtes L1
−37 % | −50 % ; `long_scoreboard` (latence mémoire) 6,5 → 4,4 | 6,3 → 4,1 ;
`barrier` 0,79 → 0,42. C'est le mécanisme annoncé : moins de mises en scène
de x (une par bloc de 32 lignes au lieu de 8), moins de blocs (vagues 9 →
2,3 | 24 → 6). Le seuil 1 300 non tenu et non rouvert : pris.

## 2. Ce qui reste, lu dans les décrochages — et ce n'est pas la DRAM

À rpw = 4, gate/up : `issue_active` **56 %** (le SM émet une instruction un
cycle sur deux : ce n'est pas le profil d'un noyau qui attend la mémoire),
`mio_throttle` **4,6** et `short_scoreboard` **4,5** par issue — tous deux
au niveau de `long_scoreboard` (4,4). MIO = file des instructions de mémoire
PARTAGÉE, short_scoreboard = attente d'un chargement partagé. Le noyau lit
l'activation depuis la shared **par flottant** (`xA[2*b]`, `xA[2*b+1]` :
32 LDS.32 par uint4 de poids) : pour 16 octets de poids lus une fois en
global, **128 octets lus en shared** — rapport 8 : 1. La bande visée est
celle du bus ; le noyau est borné par le pipeline LSU/shared. Le down
(K = 768) : `issue_active` 65 %, short_scoreboard 1,85 (moins de x par
ligne), mais 24 uint4 par ligne pour 32 voies : **8 voies sur 32 ne
chargent rien** (25 % du warp inactif sur la charge), un seul chargement en
vol par voie.

Ce que le tableau ne dit pas : la DRAM à 40 % | 27 % est lue sous ncu
(horloges verrouillées, noyaux sérialisés) — pas comparable aux 1,26 To/s
en situ ; seule la comparaison rpw4/rpw1 dans le même régime compte.
`dram__bytes_read` = n/a sur ncu 2026.3 : pas d'octets DRAM, la relecture
L2 (hit 66-77 %) reste une estimation.

## 3. Pas suivant possible, SI Sage rouvre (sinon fermé) — sans toucher à l'arithmétique

a. **Lecture vectorisée de x en shared** : `float4` (LDS.128) au lieu de
   `float` — 8 instructions par uint4 de poids au lieu de 32 ; l'indice
   décalé `i + (i >> 5)` (1 flottant par 32) casse l'alignement 16 octets :
   passer à un décalage de 4 par 32 (`XSH_PAS = 36`, LDS.128 sans conflit
   par quart de warp : voies 0-7 → bancs 0, 4, …, 28). Même arithmétique,
   même ordre : **sortie identique au bit** (le juge de v2 sert tel quel).
   Prédiction : mio_throttle et short_scoreboard ÷ 3, gate/up 87,9 →
   70-78 µs, down 65 → 58-62 ; pas 7,35 → 6,2-6,7 ms ; Coder b=12 nu
   1 262 → **1 330-1 400** (faux si < 1 300).
b. **Down (K = 768)** : deux lignes par passage de warp, voies 0-23 sur la
   ligne r, 24-31 + reprise sur r+1 — ou `uint2` (48 chargements de 8 o :
   32 voies actives, 1,5 par voie) ; sortie identique si la réduction garde
   l'ordre par ligne (à vérifier au juge). Prédiction : down 65 → 55 µs.
c. Ce que je ne propose pas : x en registres (64 par voie sur K = 2048 :
   la pression de registres — 40 aujourd'hui — ferait tomber l'occupation
   de 6 blocs à 3), ni TPB/v2 (réfuté), ni CUDA nouveau.

## 4. Défauts du script (`outils/ncu_gemv_experts_rpw_18-09.sh`), à porter

1. `sudo -n ncu` remet l'environnement à zéro : aucune `ACVRAM_*` n'atteint
   le processus — Laure a posé le régime dans un wrapper Python ; le script
   doit faire de même (`sudo -n env ACVRAM_…=… ncu …` ou wrapper).
2. Parseur en locale fr (« 91 040 », « 45,56 ») : `replace(",", "")`
   fabrique 4556 — retirer les espaces, puis virgule → point.
3. `dram__bytes_read.sum` = n/a sur 5090 / ncu 2026.3 : retirer, la bande
   par `dram__throughput` %.
Ces trois corrections ne sont utiles qu'à une passe suivante ; je les porte
si Sage rouvre, sinon le script reste avec cette note.

## 5. Dernier geste (sage-gemv-experts-dernier-geste-18-09) — livré à sec

`acvram_kernels.cu` : `nvfp4_gemv_grouped_xreg_kernel<XT, RPW, NP>` et
`_gateup_xreg_kernel` — étage x à décalage de 4 flottants par 32
(`XSH_PAS4 = 36`, alignement 16 o, bancs distincts par quart de warp) ;
chaque voie charge UNE fois sa tranche (NP ≤ 2 uint4 → `float4 v[NP][8]`,
64 registres à K = 2 048) par LDS.128 (`charger_xreg`), puis balaie ses RPW
lignes sans toucher la shared ; `nvfp4_row_dot_warp_xreg` = les MÊMES
expressions que `nvfp4_row_dot_warp` (a0/a1/c0/c1, `(…) * gscale`,
reste `part0 * s0 + part1 * s1`, réduction par shuffles), même condition de
paire et de reste → identique au bit attendu. Réservé à K ≤ 2 048
(Coder : gate/up K = 2 048 → NP = 2, down K = 768 → NP = 1) ; au-delà, le
chemin partagé. Aiguillage dans les wrappers v1 par
`ACVRAM_GROUPED_XREG=1` (régime, défaut 0 : témoin), fonctions aussi
exposées (`nvfp4_gemv_grouped_xreg`, `_gateup_xreg`) pour le juge et le
banc. `nvcc -c` contrôlé sans erreur (C++20, sm_86 + sm_120).

Juge (carte) : `test_gemv_experts_v2.py` +5 — xreg = v1 `torch.equal`
sur (2048, 768), (768, 2048), (1024, 512) × {aléatoire, un expert saturé,
fantômes} ; gate/up xreg = v1 ; bras cassant : x décalé d'un flottant →
différent. Banc : bras `xreg` (ms/pas, bit-exact par routage). Script ncu
corrigé (env par `sudo -n env …`, LC_ALL=C + parseur tolérant à la locale
fr, `dram__bytes_read` retiré, noyaux xreg dans le filtre, somme
gateup + down en ms/pas imprimée contre la porte 6,2).

Prédiction scellée (ncu, Laure) : mio_throttle 4,6 → ≤ 1,5, short_scoreboard
4,5 → ≤ 1,5, issue_active 56 → ≤ 45 % ; occupation par registres 6 → 3-4
blocs (≈ 64-80 registres/fil) ; gateup 87,9 → 68-76 µs, down 65,2 → 55-60 ;
**gateup + down 5,9-6,5 ms/pas** — la porte 6,2 au milieu de ma fourchette :
tenu ou faux à parts égales, je n'engage pas plus. Faux si > 6,2 : fermé à
1 262. Si tenu, en situ ≥ 1 300 nu (Coder b=12 : 1 262 × 7,35/6,2 ≈ 1 330
si le reste du pas ne bouge pas).

## 6. Verdict (Laure) : porte FAUSSE — chantier fermé à 1 262 t/s nu

gateup + down 7,80 ms/pas contre témoin 7,37 (> 6,2). gateup 105,2 µs
(**+19 %** ; ma prédiction 68-76 réfutée) : 96 registres par fil → 2 blocs
par SM, warps actifs 32 % contre 87 % ; le décrochage shared tombe bien
(mio_throttle 4,66 → 0,15, short_scoreboard ÷ 30) mais l'occupation le
paie — j'avais écrit « 64-80 registres, 3-4 blocs » ; c'est 96 et 2 : le
`float4 v[2][8]` plus les deux uint4 et les quatre accumulateurs, le
compilateur n'a pas replié. down **tenu** (57,2 µs, −12 %, 56 registres,
4 blocs). pytest 20/20 bit-exact : la méthode d'équivalence (mêmes
expressions) a tenu, c'est le seul acquis.

Pistes hors chantier, notées, non engagées : xreg sur down seul (K = 768,
NP = 1 : le gain mesuré, −12 % du down = −0,4 ms/pas) ; gateup borné à 64
registres par `__launch_bounds__(256, 4)` (le compilateur déverse ou
replie — à mesurer, pas à prédire). Règle de Sage appliquée : porte
fausse = pas d'in situ.
