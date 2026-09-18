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

## 6. Écrit (GO poste7 18/09) — et une correction du § 3 avant la carte

**Le § 3 comptait faux** : « 8 FFMA par uint4 » — un uint4 porte 32 poids,
donc 32 FFMA dans (b) comme dans v1 ; le « ÷ 2,4 » n'existe pas. Compte
réel, SASS sm_120 du .cu compilé à sec (`nvcc -cubin -Xptxas -v`,
scratchpad/acvram.sass, boucle principale) :

| noyau | registres | spill | boucle | instr / octet de poids | LDS / 64 o |
|---|---|---|---|---|---|
| v1 `gateup<bf16,4>` | (inchangé) | 0 | ~6,5 instr/octet + 2 LDS.32/octet | **≈ 8,5** | 128 |
| (b) `gemv_marlin<bf16,2>` gate+up | **64** | 0 | 571 instr / 64 o | **8,9** | **4** (LDS.64) |
| (b) `gemv_marlin<bf16,1>` down | **48** | 0 | 294 instr / 32 o | 9,2 | 4 |

Le plancher commun (par octet : F2FP 1, HADD2 2, PRMT/LOP3 ~1,5, FMUL/FFMA
2) est le même ; (b) paie 8 décodages d'échelle par uint4 (2 instr chacun :
PRMT + LEA, le SEL du zéro retiré — l'octet 0 vaut 2⁻²²/facteur, ≤ 2⁻²²·6·|x|
par poids) là où v1 en paie 2. **Le gain de (b) n'est donc PAS le compte
d'instructions (égal) : c'est la mémoire partagée, 32× moins de LDS**, le
poste que ncu désignait (mio_throttle + short_scoreboard 60 % des
décrochages, issue_active 56 %). Si ce poste était bien le mur, (b) monte
vers l'émission : 7,3 × 0,56/0,80 ≈ 5,1 ms ; s'il ne l'était pas, (b) = v1
et le scellé tombe.

**Prédiction scellée (remplace le § 3)** : b=12 **5,0-6,5 ms/pas** (GEMV
7,3 ; porte 0,97 × = 7,08), b=1 **2,2-2,9 ms** (GEMV 3,0 ; porte 2,91).
Faux si (b) > 0,97 × GEMV à l'un des deux → P1 fermé VRAM.

Livré (branche poste4) : `nvfp4_gemv_marlin[_gateup]` (acvram_kernels.cu :
bloc 8 warps par tuile de colonnes, chaque warp un huitième des tuiles k,
deux tuiles en vol par voie, réduction 4 voies puis 8 warps en partagée,
ordre fixe → déterministe) ; `ACVRAM_GEMV_LAYOUT=marlin|naturel` (regime.py,
toujours dans regime_ligne ; `_construire_marlin` bâtit les piles sous
GEMV_LAYOUT=marlin aussi) ; aiguillage `_forward_grouped` avec compteurs
`gemv_marlin` / `gemv_v1` ; `ACVRAM_TRACE_ROUTAGE=f.pt` capture les routages
réels hors graphe, `banc-marlin-decode-18-09.py --routages f.pt` les rejoue ;
bras (b) dans le banc avec le scellé imprimé ; tests/test_gemv_marlin.py
(à sec : place et décodage des 8 octets d'échelle contigus, vérifiés contre
permuter_echelles/traiter_echelles_nvfp4 en CPU — 2 passés ; carte : (b)
contre fp32 par ligne sur Coder gate/up, down, une tuile, fantômes, GELU,
déterminisme, bras cassant échelles ET poids décalés, bloc MoE avec
`attendre_chemin("gemv_marlin")` et témoin `gemv_v1`).

Non couvert à sec : la permutation des POIDS (gptq_marlin_repack est un
noyau CUDA) — lue dans la source, testée par la carte seulement. Pas encore
fait : retirer la pile naturelle sous GEMV_LAYOUT=marlin (le gain VRAM) —
après le scellé, pas avant.

Ordre poste3 : `CUDA_VISIBLE_DEVICES="" python -c "from acvram.kernels import
get_extension"` ne compile pas sans carte ; compiler AVANT le banc sous
carte.sh (import seul, processus séparé), puis `pytest tests/test_gemv_marlin.py
-q`, puis `banc-marlin-decode-18-09.py` (Triton cache vidé, sha du .so dans
l'en-tête).

## 7. Disposition unique dans le CHARGEUR (poste3 : 17,04 Gio réservés, 21/48 couches exilées)

Ma « disposition unique » n'avait touché que l'aiguillage du décodage ; le
chargeur réservait toujours Σ mlp_bytes pour une seconde disposition
(loader.py `_reserve_prefill`, 5a0f6c4) et `_try_build_stacks` gardait la
pile NVFP4 à côté de la pile Marlin. Corrigé : (1) la réserve double est
retirée (Σ × 1, jamais × 2 — test à sec avec bras cassant :
`test_le_plan_ne_reserve_pas_de_seconde_disposition`) ; (2)
`MoEBlock._liberer_pile_naturelle` après le repack : qw/bs de `_stacks[n]`
→ None, experts au gabarit vide (`[:0].cpu().clone()`, comme
`runner._demote_expert`), `experts_layout = "marlin"` ; tout chemin qui
relirait la pile naturelle (mma préfill, direct, MMA décodage, v1) est
coupé sous `unique` — casser plutôt que mesurer une double disposition
(test à sec `test_la_pile_naturelle_est_rendue_apres_le_repack_a_sec`) ;
(3) régime `experts_layout=marlin|naturel`, « double » n'existe plus ;
(4) `ACVRAM_GEMV_LAYOUT=marlin` ⇔ `ACVRAM_PREFILL_GROUPED=marlin`, sinon
refus à l'import. Couches exilées (pile « table ») : pas de repack, elles
restent naturelles dans le pool — hors du périmètre de cette note.

## 8. Biais du GEMV (b) in situ (+0,008 PPL) — instruments et hypothèses (poste7-p1-situ-verdict-18-09)

**(0) Critère de biais** : dans `banc-marlin-decode-18-09.py` et dans un
instrument par couche sur les piles RÉELLES, `outils/biais-gemv-marlin-18-09.py`
(v1, (b), témoin, chacun contre fp32) : `biais_signe = moy(Δ)/moy|y|` (poste7,
≤ 1e-4) **et** `gain − 1 = Σ y·ref/Σ ref² − 1` — un biais MULTIPLICATIF
(échelles systématiquement trop petites, par exemple) est invisible à la
moyenne signée sur une sortie centrée, ce qu'un MoE est ; le témoin négatif =
échelles de bloc tronquées d'un bit (toujours ≤ la vraie, gain − 1 ≈ −3 %)
doit rendre faux, sinon l'instrument est aveugle et le verdict « ne compte pas ».
Second témoin dans l'instrument par couche : accumulation fp16 séquentielle sur
down.

**(1) bf16 dans le produit ou l'accumulation — réfutée par le SASS** (§ 6,
scratchpad/acvram.sass, boucle principale de `nvfp4_gemv_marlin_kernel<bf16,2>`) :
HADD2.F32 128 = conversions half→fp32 (2 par octet, comme v1), puis FFMA 96 +
FMUL 64 + FADD 32 = 192 opérations fp32 = 128 produits + 32 FFMA d'échelle +
32 FADD ; **aucune HFMA2/HMUL2**, accumulateurs fp32 (`float acc[NW][4][2]`),
x en fp32 (`__bfloat162float`), échelles décodées en fp32 exact (test à sec
rtol 1e-6). Rien à remédier sur ce point.

**(2) tables d'échelles ≠ E4M3 — chiffrée à sec sur le Coder réel**
(Qwen3-Coder-30B-A3B-nvfp4, 48 couches, `--echelles-seulement`) : la pile
Marlin ANNULE les échelles avec s·facteur·2⁷ < 2 (`traiter_echelles_nvfp4`) ;
facteur = 1 partout. Annulées : gate 69 582 blocs (0,012 %), up 68 074
(0,011 %), down 0 — presque toutes en couche 0 (69 078 gate, 0,55 % de ses
blocs), puis 201 (c. 1), 175 (c. 2), 128 (c. 4), 0 ailleurs ; déjà nulles dans
la pile NVFP4 : 619 849 (0,10 %, couches 0-2). Ces blocs ont une échelle
< 2⁻⁶ pour un maximum de 448 (rapport 3,5·10⁻⁵) : leur poids est déjà
négligeable ; le prefill Marlin, qui lit la même pile, tient à Δ+0,0007. Le
noyau (b) décode l'octet 0 en 2⁻²²/facteur (§ 6) : ±codes·2⁻²²·g, plus petit
que tout poids vivant. Hypothèse 2 : **peu probable** ; l'instrument par couche
tranchera (gain − 1 de (b) contre v1, couche 0 en tête).

**(3) bornes M = 1** : le noyau n'a aucun chemin dépendant de M (grille N/64 ×
G, x en partagée par bloc) ; b = 1 : 0 hors 2⁻⁷ au banc. Rien à chercher là.

Ce que l'instrument par couche dira : si (b) et v1 ont le même gain − 1 et le
même biais signé contre fp32 sur toutes les couches (|·| ≤ 1e-4) et que le
témoin est vu, le biais de +0,008 n'est pas dans le GEMV mais dans ce que le
moteur lui donne ou fait de sa sortie sous disposition unique (à isoler par
bissection in situ : préfixe par Marlin + décodage v1 est impossible sous
disposition unique — alors comparer décodage (b) et v1 sur les MÊMES piles
en double disposition, hors régime, dans le seul but du diagnostic).

### GLM prefill Marlin −0,003 et GEMV (b) décodage +0,008 : même famille, à nommer ensemble (poste7)

Le scan à sec des échelles sur GLM-4.7-Flash-nvfp4 (46 couches, facteur 1) :
**0 échelle annulée, 0 déjà nulle** — la pile Marlin de GLM est
bit-équivalente à la pile E4M3. Le −0,003 du prefill GLM ne vient donc pas des
tables ; il vient de l'arithmétique du chemin : Marlin sort g, u et down en
**bf16** (`gemm_moe(…, c=)`) et l'activation silu(g)·u est prise en bf16 avant
down, là où `groupe`/`direct` gardent des intermédiaires fp32 — trois arrondis
2⁻⁸ par jeton et par expert, sans biais de signe mais pas équivalents ; le
sens (mieux) est un hasard de ce modèle, pas une propriété. Le +0,008 du GEMV
(b) au décodage est l'autre face : un chemin dont l'arithmétique par élément
est exacte (§ 8-1) mais dont la sortie in situ diffère. Aucune revendication
tant que la cause de chacun n'est pas nommée.

**Bissection in situ** (ajoutée : `ACVRAM_DOUBLE_DISPOSITION_DIAG=1`, régime
`experts_layout=double(diag)`, jamais servi) : sur les MÊMES piles, quatre
cellules ppl-decode-kv — préfixe {groupe, marlin} × décodage {naturel = v1,
marlin = (b)} (`ACVRAM_PREFILL_GROUPED` × `ACVRAM_GEMV_LAYOUT`). Si le +0,008
suit le décodage (b) quel que soit le préfixe → le noyau (instrument par
couche pour le localiser) ; s'il suit le préfixe Marlin → le cache KV écrit par
le prefill bf16 (même cause que le −0,003 GLM, signe opposé sur Coder).

### Verdict de poste3 (revue/verdict-biais-gemv-b-18-09, 9584ad7) et suite

Par couche, routages réels, 48/48 : (b) ≡ v1 (|biais| ≤ 1,8e-9, |gain−1| ≤
1,2e-7 ; témoin tronqué vu à −0,13). La 2×2 en double disposition est
impossible sur Coder (OOM ×8) ; la bissection par longueur de préfixe donne
Δ = +0,0173 (16), +0,008 (256), +0,0206 (2048) : **le biais suit le décodage
(b)** — ma prédiction (« suit le préfixe ») est RÉFUTÉE, celle de poste7 tenue.
En eager, Δ = +0,0034 : ×5 sous rejeu de graphe. Donc : le noyau est juste
en isolation, le chemin CAPTURÉ de (b) diverge du chemin eager de (b), et un
résidu eager reste à nommer.

Instrument livré pour la piste de poste7 (premier pas divergent) :
`ACVRAM_DUMP_MOE=<dossier>` (model.py, bas de fichier) — tampon statique par
couche et par forme, rempli par `copy_` dans le forward (un graphe capturé y
écrit à chaque rejeu), sauvé après chaque pas par `Engine.step`
(`_sauver_dump_moe`, pas-NNNNN.pt) ; `outils/comparer-dump-moe-18-09.py A B`
nomme le premier (pas, couche) divergent et le profil du pas. Protocole :
ppl-decode-kv 16/1024 deux fois sous disposition unique, `enable_cuda_graphs`
vrai puis faux, même dossier de dump chacun ; puis la même paire en régime
naturel (v1 capturé contre v1 eager) comme témoin : si v1 diverge aussi entre
capture et eager, la divergence n'est pas propre à (b).
