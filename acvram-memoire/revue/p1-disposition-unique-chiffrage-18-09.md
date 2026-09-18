# P1 — disposition unique : chiffrage à sec des deux formes avant d'écrire (poste4, 18/09, ½ j ; poste7-p1-disposition-unique-18-09, ordre corrigé de chef)

## 0. Correction d'un chiffre faux : « +1,7 Gio » → **+14,5 Gio**

Ma note disait « seconde disposition ≈ 1,7 Gio sur Coder » : faux. Le Plan
ajoute Σ mlp_bytes des 48 couches MoE (`loader.py _reserve_prefill`, sous
`ACVRAM_PREFILL_GROUPED=marlin`) = 128 experts × 3 projections × 768 × 2 048 ×
(½ o de codes + ⅛ o d'échelles) × 48 = **14,5 Gio** — poste3 l'a mesuré
(30/48 couches exilées, DÉGRADÉ). Le 1,7 était le huitième (les échelles
seules). Il n'y a pas d'option « double disposition » : P1 vit ou meurt sur
UNE disposition. Le témoin GEMV du banc décodage lit ses échelles en uint8
(fa49f1a, porté).

## 1. Ce que le SASS dit (sans carte : `cuobjdump -sass`, `--dump-resource-usage` sur le .so compilé à sec)

Marlin, config préfill (256 fils, m_blocks 4, n 16, k 4, sm_120) : boucle K
**6 459 instructions pour 256 HMMA = 25,2 instr/HMMA** (LDS 237, LDSM 36,
LDGSTS 39, LOP3 495, PRMT 83, SHF 246, HMUL2 160, IMAD 746, IADD 564) ;
255 registres/fil. Budget d'émission : une HMMA.16816 bf16 (4 096 FLOP)
occupe le pipeline tenseur ~33 cycles par sous-partition à la crête de 200
TFLOPS (200e12 / 170 SM / 2,4 GHz / 4 sous-partitions ≈ 122 FLOP/clk) — à
25 instr/HMMA sur 33 slots, le noyau émet 76 % du temps : **c'est le 152/200
mesuré, le noyau est déjà au bord de l'émission**, pas seulement du
pipeline tenseur. Décodage (m_block_size_8, 256 fils) : 79 instr/HMMA —
d'où le 6,81 ms : la boucle paie la même colle pour 4× moins de MMA.

GEMV v1 (`nvfp4_gemv_grouped_gateup_kernel`, rpw=4, ncu 18/09) : par uint4
de poids (32 codes) : 1 LDG.128, 2 LDS.8 d'échelle, **32 LDS.32 de x**, ~32
extractions de quartets (LOP3/SHF/PRMT), 32 FFMA, 2 conversions E4M3 →
≈ 100 instructions par 16 octets ; mio_throttle + short_scoreboard = 60 %
des décrochages (la shared), issue_active 56 %.

## 2. Disposition Marlin, lue exactement (`gptq_marlin_repack.cu` : `tc_offsets {0,1,8,9}`, `pack_idx {0,2,4,6,1,3,5,7}`, `out[tile + th_id·4 + warp_id]`)

Tuile 16 (k) × 64 (n) = 512 o = 128 mots ; le mot `t·4 + w` (voie t, warp
w du repack) porte 8 quartets : colonne n = w·16 + t/4, k = (t%4)·2 +
{0, 1, 8, 9}, et la colonne n + 8, mêmes k ; ordre des quartets : k0(n),
k8(n), k0(n+8), k8(n+8), k1(n), k9(n), k1(n+8), k9(n+8). Échelles (groupe
16) : `marlin_permute_scales` transpose des blocs 8 × 8 de colonnes
(position p → n = (p % 8)·8 + p / 8) puis S0E5M3 entrelacé [0,2,1,3] par 4.

## 3. Forme (b) — le GEMV lit la disposition Marlin

Une voie lit **un uint4 = 4 mots consécutifs = les 4 warps du repack pour la
même voie t** : 8 colonnes (n = w·16 + t/4 et +8, w = 0..3) × 4 k
({(t%4)·2 + 0, 1, 8, 9}) ; **une tuile entière (64 n × 16 k) par lecture de
warp, parfaitement coalescée, 100 % des octets utiles** (les quartets de
n+8 servent aussi). Par uint4 : **4 valeurs de x** (deux float2 : k, k+1 et
k+8, k+9) au lieu de 32, 8 quartets à extraire, 8 FFMA dans 8 accumulateurs
(un par colonne), 8 octets d'échelle à la position permutée (8 LDS.8 depuis
une ligne d'échelles étagée par tuile), conversion E4M3 : 8 (au lieu de 2).
Par 16 octets : 1 LDG.128 + 4 LDS.32 + 8 LDS.8 + ~10 (quartets) + 8 FFMA +
8 (échelles) + 2 (adresses) ≈ **41 instructions contre ≈ 100** pour v1 —
**÷ 2,4**, et la shared 8× moins sollicitée (le poste de ncu). Réduction :
par colonne, 4 voies (t%4) se partagent les 16 k d'un bloc → 2 shuffles
par colonne à la fin (8 colonnes × 2 = 16 SHFL par voie et par ligne de
sortie, amortis sur K/16 tuiles = 128 pour K = 2 048 : négligeable).
Registres : 8 accumulateurs + 4 x + 4 mots + 8 échelles ≈ 40, comme v1.
Bit à bit contre v1 : **impossible** (v1 somme les 16 produits d'un bloc
dans une chaîne de fma par voie ; (b) en somme 4 par voie puis réduit sur
4 voies — un autre ordre fp32) ; déterministe, oui ; juge = fp32 par ligne
(le critère de P1). Prédiction : b=12 gate/up + down **5,0-5,8 ms/pas**
(GEMV 7,3 : ≥ 0,97× tenu avec marge, le poste shared tombe), b=1
**2,3-2,8 ms** (GEMV 3,0). Faux si > 0,97 × GEMV — alors la réduction 4
voies ou les 8 octets d'échelle dispersés coûtent plus que les 28 LDS
retirées, à lire au ncu.

## 4. Forme (b') — Marlin lit la disposition naturelle

Le fragment B de `mma.m16n8k16` veut, par voie c = lane % 4, les k = {2c,
2c+1, 2c+8, 2c+9} de la colonne n = lane / 4 : dans la ligne naturelle
[n, K/2] ce sont les OCTETS c et c+4 du segment de 8 octets — **2 LDS.8 par
fragment** (ou 1 LDS.64 + 2 PRMT) là où le repack donne **1 LDS.32 pour 2
fragments** ; conflits de bancs (foulée de ligne 1 024 o) à casser par un
rembourrage de 16 o par ligne en shared ; extraction des quartets par un
autre masque (bas/haut du même octet), même nombre de LOP3. Coût :
**+1,5 LDS par HMMA sur 25,2 → 26,7 (+6 %)**, sur une boucle déjà à 76 %
d'émission ⇒ 152 → **~143 TFLOPS** si l'émission reste le seul mur, et la
shared prend 4 fois plus de transactions pour B (LDS.8 : 32 o par
instruction au lieu de 128). Prédiction : **135-145 TFLOPS** — la porte
137 au milieu de la fourchette, **sans marge** ; et le décodage resterait
Marlin (6,81 ms, hors bande) : (b') ne règle que la VRAM, pas le décodage.

## 5. Décision proposée : **(b)**

(b) a la marge (÷ 2,4 d'instructions par octet, la shared soulagée, le
mécanisme même que ncu désignait) et règle le décodage ET la VRAM ; (b')
est au fil de sa porte et laisse le décodage à 6,81. Coût (b) : un noyau
CUDA `nvfp4_gemv_marlin_gateup` + `_down` (fusion act·up dans l'épilogue
comme v1, TPB sans objet : une paire par bloc comme v1), l'aligneur
inutile (v1 : paires dans l'ordre des jetons), tables d'échelles étagées
par tuile ; juge fp32 par ligne + bras cassant (échelles décalées) ; banc =
`banc-marlin-decode-18-09.py` + bras (b). Porte à sec (poste7) : (b) ≥
0,97 × GEMV à b=1 ET b=12, déterministe, ≤ 2⁻⁷ par ligne contre fp32 ;
faux ⇒ P1 fermé VRAM, verdict daté, ligne utilisateur (pas de repli).
Délai : 1 j de noyau + tests, ½ j de banc ; `-Xptxas -v` du noyau avant
la carte (REGLES § 3).
