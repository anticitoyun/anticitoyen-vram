# Sage — MoE en MMA groupée au décodage : je le prends, en 2e place, sous trois conditions scellées (14/09)

Sources : chiffres de Laurine (instr-par-octet-14-09.md, transmis par Jérôme — fichier absent de main bbd8bce et de travail/laurine à l'heure où j'écris : à relire quand il circule) ; `acvram/engine/model.py:1082,1163,1184` (bascule `t <= _MOE_GROUPED_MAX=32` → GEMV, sinon `_forward_prefill_grouped` MMA, BT=64) ; `model.py:816-824` (`_tuiles`) ; `verdict-moe-mma-reel-qwen3-coder.md:27,44` ; `sage-strategie-14-09.md` §1.

**Je retire « MoE : aucun levier ».** Il reposait sur 82 % de borne et ≤ 68 experts distincts ; M2 rend 61 % (gate·up 73, down 47) et ≈ 30 experts. À 5,9 ms de MoE par pas, la borne DRAM est 3,6 ms (≈ 3,8 Go, 30 × 48 × 2,65 Mo) : il reste 2,3 ms sur la table, pas zéro. Et 71 % des instructions du pas y vivent — c'est le seul levier qui touche la puissance à sa source, là où 1aj n'en touche qu'une part.

## Ce que je prédis, scellé avant toute mesure

| grandeur | aujourd'hui | prédit avec MMA au décodage | réfuté si |
|---|---|---|---|
| MoE gate·up + down, ms/pas | 5,9 (61 % de borne) | **4,0-4,3** (85-90 %, le niveau de CUTLASS) → −1,6 à −1,9 ms ; **pas −3** : ce serait sous la borne DRAM | ≥ 5,0 ms |
| W des noyaux MoE, boucle ≥ 20 s au compteur | 400 W saturés, SM 1 642 / 2 287 MHz | **≤ 345 W** (témoin copie 322 + 23), SM ≥ 2 400 MHz | ≥ 380 W ou SM < 2 000 MHz |
| pas b=12 (avec le godet 12 déjà lancé) | 392 W / 15,66 ms / 0,512 J/jeton | **≤ 380 W, ≤ 0,45 J/jeton** (calcul : 4,2 ms à 340 W + 9,8 ms à 392 W) | ≥ 0,48 J/jeton |
| `dram__bytes_read` des noyaux MoE | ≈ 3,8 Go | **inchangé ±5 %** — contrôle de la leçon L2 : si les octets baissent, le gain n'est pas celui qu'on vend | variation > 5 % |
| inst/octet MoE | 1,36 / 2,55 | ≤ 0,35 / ≤ 0,65 (÷4 Laurine ; CUTLASS 0,18) | > 0,5 / > 0,9 |

## Trois conditions, dans l'ordre — elles font la place du levier

1. **Qualité : déjà payée.** A4 sur les experts = +0,919 % de PPL, seuil 1 % respecté, `ACVRAM_MOE_MMA=1` par défaut depuis le 13/09 (verdict:27,44). La PPL est un prefill : le noyau et la quantification d'activation sont les mêmes au décodage, donc le chiffre couvre les deux. **Test d'équivalence du même commit** (règle 9) : logits d'un jeton décodé en MMA = logits du même jeton en prefill MMA, à égalité près — le test recouvrement/rejeu (afc8273) est le modèle.
2. **Le chemin n'est pas capturable tel quel** : `_tuiles` fait `tot = int(ntiles.sum())` (`model.py:824`) et `repeat_interleave` à répétitions tenseur — deux synchronisations hôte, capture impossible, et le rejeu est 98 % du pas. Au décodage, `cnt ≤ 16 < BT=64` : chaque expert touché tient dans **une** tuile, donc grille fixe de `t×k` (96) entrées, `tile_n = 0` pour les vides, le noyau saute n=0. Un jour de Laurine, zéro nouveau noyau.
3. **Le noyau tient-il la borne à M=3 ?** BT=64 pour ~3 jetons réels : le calcul gaspillé est gratuit, mais 30 experts × 12 tuiles N = 360 blocs à K=2048, ~2 vagues sur 170 SM — le recouvrement de latence (étages cp.async, bead 0si) décide. **Mesure d'une heure, AVANT la condition 2** : `ACVRAM_MOE_GROUPED_MAX=0` à b=12 en eager, mêmes trois noyaux sous ncu + compteur — ms, W, octets. Si ≥ 5,0 ms ou ≥ 380 W : il faut un split-K de décodage (+2 j) et le levier passe derrière 1aj.

## Place

(1) godet 12 / NV=16 — lancé. **(3) MoE MMA : mesure d'une heure (cond. 3), puis grille fixe (cond. 2), puis mesure sous graphes.** Ensuite 1aj : même technique, part d'instructions plus petite (−1,5 ms, −15/−25 W selon Laurine). Total prédit : 15,66 − 1,2 − 1,7 − 1,5 ≈ **11,3 ms, ≤ 370 W** ; ce qui reste au-delà se lit dans le dénominateur DRAM (6,67 Go à 1 050 Go/s = 6,35 ms), pas dans les noyaux.

## 2. Après (iii) de Laurine (713e394) : pas de split-K, (ii) → bt=32 → pas complet, seuils rescellés

**Bilan de mes prédictions** : ms **tenu** (4,0 dans 4,0-4,3) ; inst/octet **tenu** (0,13 ≤ 0,35) ; octets **non tenu** (+13 %) ; W : **mon seuil était mal posé**, pas seulement non tenu. « ≤ 345 W » ne disait ni brut ni net, et un seuil brut à 380 W sur une carte plafonnée à 400 W ne peut pas rendre « faux » quand le noyau touche le plafond **à pleine horloge** — c'est l'horloge qui répond, et elle a répondu : 2 617 MHz contre 1 792 pour la GEMV, 340,5 W nets contre un témoin copie à 341 W. Le noyau est au plafond parce qu'il déplace 1 156 Go/s, plus parce qu'il calcule. Règle 4 sur moi : un seuil porte son régime (brut/net) dans son nom. **Verdict split-K : non** — il n'enlève ni octets ni watts DRAM, et la seule chose qu'il aurait corrigée (une horloge rabattue par les instructions) n'existe plus. Les 2 jours vont à (ii).

**Ordre** : (ii) glue hôte (argsort, bincount, `int(ntiles.sum())`, +2 300 lancements — c'est ma condition 2, Océane) → **bt=32** (un paramètre, une heure) → pas complet sous graphes. Corollaire pour REGLES §9 : « ~1 050 Go/s » est dépassé par ce noyau (1 156) ; la borne dépend du motif d'accès, à re-annoter avec le noyau qui l'a mesurée.

**Le +13 % d'octets : l'explication ne tient pas à b=12.** À t=12 et k=8, `cnt ≤ 12 < 16` : aucun expert ne reçoit 17-32 jetons, donc aucune seconde tuile de M. Soit la manche avait t > 16 (godet, spéculation — à lire dans `cnt.max()` de la manche), soit les octets viennent d'ailleurs (échelles ou tables relues par tuile, `xq`, `down` seul). **Contrôle qui tranche** : bt=32, mêmes octets par noyau (gate, up, down séparés) ; si 4,33 → 3,83 Go, c'était la tuile et il faut expliquer t > 16 ; sinon la cause est dans un des trois noyaux et son nom le dit.

**Seuils du pas complet sous graphes, après (ii) + bt=32, MMA seule (godet 12 compté à part, −1,2 ms)**, référence 15,66 ms / 392 W / 0,512 J/jeton (même instrument) :

| grandeur | prédit | réfuté si |
|---|---|---|
| pas | **13,4-13,9 ms** (MoE 5,98 → 3,8-4,0, glue neutre sous graphes) | gain < 1,2 ms : la glue (ii) a mangé 40 % du noyau |
| W moyen brut | **385-400, inchangé** — je retire « ≤ 380 » : les noyaux MoE restent au plafond, à pleine horloge ; le gain est du temps à puissance constante | < 375 W (alors quelque chose d'autre a changé) |
| J/jeton brut | **≤ 0,46** (13,7 ms × 0,395 W / 12 = 0,45 ; −0,79 J par pas sur les 3 GEMM) | ≥ 0,48 |
| mJ/couche MoE sous graphes | 54,6 ± 3 (le chiffre eager doit se retrouver) | > 62 |
| `dram__bytes_read` MoE | 3,83 Go ± 3 % à bt=32 | > 4,0 Go |

Fenêtre : ≥ 20 s au compteur (`sage-instrument-energie`), pas 6 s.

## 3. Campagne ≥ 20 s (Laure, 0234a7d) : un instrument asymétrique d'abord, puis les seuils, puis l'objectif

**0. Le brut de vLLM compte deux cartes, celui d'acvram une.** `energie.py:71-84` : sans `CUDA_VISIBLE_DEVICES`, `garder = None` et **toutes** les cartes sont sommées ; `campagne-20s-acvram-14-09.py:25` pose `"0"`, `campagne-20s-vllm-14-09.py:45` ne pose rien (ni le `.sh`). Preuve dans les chiffres : vLLM b=12 0,291 J × 1 437 t/s = **418 W brut sur 20 s, sous un plafond de 400 W** — impossible pour une carte ; c'est 390 (5090 au plafond) + 28 (3080 Ti relevée à 28,5 W, services au repos). Le contrôle `moyenne > plafond` de `sage-instrument-energie` aurait sonné. Corrigé à 28,5 W × durée du jeton : **b=1 vLLM 1,579 → 1,434 contre 1,430 : −0,3 %, le créneau b=1 disparaît** ; b=2 0,994 (nous +14 %) ; b=12 0,271, **×2,28 en J comme en t/s**. Contrôle 5 min avant de porter quoi que ce soit : vLLM b=1 avec `CUDA_VISIBLE_DEVICES=0`, prédit 1,42-1,45 J/jeton ; réfuté si ≥ 1,55.

**1. Seuils rescellés sur 19,0 ms / 0,619 J / 390 W à ≥ 20 s.** L'écart 15,66 → 19,0 ms est le plafond : la GEMV MoE, bornée par les instructions, tourne à 1 792 MHz en régime tenu ; la MMA, bornée par la DRAM, garde 2 617 MHz au même plafond — la pénalité tombe avec elle. MoE MMA seule (après (ii), bt=32) : **pas 15,5-16,3 ms, 735-775 t/s, 0,50-0,53 J/jeton, W 390 inchangé** ; réfuté si pas > 17,3 ms ou J ≥ 0,56. Avec godet 12 et 1aj (même levée de pénalité sur `int8_gemv`/`lm_head`) : **12,6-13,5 ms, 890-950 t/s, 0,41-0,44 J** ; réfuté si < 850 t/s. Ce qui reste ensuite est la DRAM (6,4 ms) et la glue : aucun levier nommé au-delà.

**2. Recommandation, tranchée : réviser l'objectif.** Tous leviers identifiés appliqués, Coder-30B b=12 reste à **−35 % en débit et ×1,5 en J** derrière vLLM (890-950 vs 1 437 ; 0,42 vs 0,27) ; « rattraper à −15 % » était écrit sur 15,66 ms, régime non tenu — je le retire. Le créneau énergie b=1 tient à un instrument, pas à un moteur (pt 0). Objectif à annoncer : **(a) MLA NVFP4 sur sm_120 (GLM-4.7-Flash / GLM-42B), où vLLM n'a pas de chemin — mesure 2b jamais faite, c'est la première ; (b) modèles hors VRAM, où vLLM n'existe pas et llama.cpp/ggrun sert à 37 t/s — le cache d'experts (sage-cache-experts) redevient le chantier, avec la borne PCIe 21 Go/s pour juge ; (c) llama.cpp b=1-12 en VRAM (2c, binaire 4a899373) comme seul rival qu'on revendique sur Coder-30B.** Coder-30B contre vLLM devient un banc d'étalonnage des noyaux (instr/octet, J par poste), pas une revendication. Les trois leviers se font quand même : −30 % de J/jeton pour ~5 jours, et ils servent (a) et (b).
