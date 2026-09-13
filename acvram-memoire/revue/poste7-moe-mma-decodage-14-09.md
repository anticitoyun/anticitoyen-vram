# poste7 — MoE en MMA groupée au décodage : je le prends, en 2e place, sous trois conditions scellées (14/09)

Sources : chiffres de poste4 (instr-par-octet-14-09.md, transmis par chef — fichier absent de main bbd8bce et de travail/poste4 à l'heure où j'écris : à relire quand il circule) ; `acvram/engine/model.py:1082,1163,1184` (bascule `t <= _MOE_GROUPED_MAX=32` → GEMV, sinon `_forward_prefill_grouped` MMA, BT=64) ; `model.py:816-824` (`_tuiles`) ; `verdict-moe-mma-reel-qwen3-coder.md:27,44` ; `poste7-strategie-14-09.md` §1.

**Je retire « MoE : aucun levier ».** Il reposait sur 82 % de borne et ≤ 68 experts distincts ; M2 rend 61 % (gate·up 73, down 47) et ≈ 30 experts. À 5,9 ms de MoE par pas, la borne DRAM est 3,6 ms (≈ 3,8 Go, 30 × 48 × 2,65 Mo) : il reste 2,3 ms sur la table, pas zéro. Et 71 % des instructions du pas y vivent — c'est le seul levier qui touche la puissance à sa source, là où 1aj n'en touche qu'une part.

## Ce que je prédis, scellé avant toute mesure

| grandeur | aujourd'hui | prédit avec MMA au décodage | réfuté si |
|---|---|---|---|
| MoE gate·up + down, ms/pas | 5,9 (61 % de borne) | **4,0-4,3** (85-90 %, le niveau de CUTLASS) → −1,6 à −1,9 ms ; **pas −3** : ce serait sous la borne DRAM | ≥ 5,0 ms |
| W des noyaux MoE, boucle ≥ 20 s au compteur | 400 W saturés, SM 1 642 / 2 287 MHz | **≤ 345 W** (témoin copie 322 + 23), SM ≥ 2 400 MHz | ≥ 380 W ou SM < 2 000 MHz |
| pas b=12 (avec le godet 12 déjà lancé) | 392 W / 15,66 ms / 0,512 J/jeton | **≤ 380 W, ≤ 0,45 J/jeton** (calcul : 4,2 ms à 340 W + 9,8 ms à 392 W) | ≥ 0,48 J/jeton |
| `dram__bytes_read` des noyaux MoE | ≈ 3,8 Go | **inchangé ±5 %** — contrôle de la leçon L2 : si les octets baissent, le gain n'est pas celui qu'on vend | variation > 5 % |
| inst/octet MoE | 1,36 / 2,55 | ≤ 0,35 / ≤ 0,65 (÷4 poste4 ; CUTLASS 0,18) | > 0,5 / > 0,9 |

## Trois conditions, dans l'ordre — elles font la place du levier

1. **Qualité : déjà payée.** A4 sur les experts = +0,919 % de PPL, seuil 1 % respecté, `ACVRAM_MOE_MMA=1` par défaut depuis le 13/09 (verdict:27,44). La PPL est un prefill : le noyau et la quantification d'activation sont les mêmes au décodage, donc le chiffre couvre les deux. **Test d'équivalence du même commit** (règle 9) : logits d'un jeton décodé en MMA = logits du même jeton en prefill MMA, à égalité près — le test recouvrement/rejeu (afc8273) est le modèle.
2. **Le chemin n'est pas capturable tel quel** : `_tuiles` fait `tot = int(ntiles.sum())` (`model.py:824`) et `repeat_interleave` à répétitions tenseur — deux synchronisations hôte, capture impossible, et le rejeu est 98 % du pas. Au décodage, `cnt ≤ 16 < BT=64` : chaque expert touché tient dans **une** tuile, donc grille fixe de `t×k` (96) entrées, `tile_n = 0` pour les vides, le noyau saute n=0. Un jour de poste4, zéro nouveau noyau.
3. **Le noyau tient-il la borne à M=3 ?** BT=64 pour ~3 jetons réels : le calcul gaspillé est gratuit, mais 30 experts × 12 tuiles N = 360 blocs à K=2048, ~2 vagues sur 170 SM — le recouvrement de latence (étages cp.async, bead 0si) décide. **Mesure d'une heure, AVANT la condition 2** : `ACVRAM_MOE_GROUPED_MAX=0` à b=12 en eager, mêmes trois noyaux sous ncu + compteur — ms, W, octets. Si ≥ 5,0 ms ou ≥ 380 W : il faut un split-K de décodage (+2 j) et le levier passe derrière 1aj.

## Place

(1) godet 12 / NV=16 — lancé. **(3) MoE MMA : mesure d'une heure (cond. 3), puis grille fixe (cond. 2), puis mesure sous graphes.** Ensuite 1aj : même technique, part d'instructions plus petite (−1,5 ms, −15/−25 W selon poste4). Total prédit : 15,66 − 1,2 − 1,7 − 1,5 ≈ **11,3 ms, ≤ 370 W** ; ce qui reste au-delà se lit dans le dénominateur DRAM (6,67 Go à 1 050 Go/s = 6,35 ms), pas dans les noyaux.

## 2. Après (iii) de poste4 (713e394) : pas de split-K, (ii) → bt=32 → pas complet, seuils rescellés

**Bilan de mes prédictions** : ms **tenu** (4,0 dans 4,0-4,3) ; inst/octet **tenu** (0,13 ≤ 0,35) ; octets **non tenu** (+13 %) ; W : **mon seuil était mal posé**, pas seulement non tenu. « ≤ 345 W » ne disait ni brut ni net, et un seuil brut à 380 W sur une carte plafonnée à 400 W ne peut pas rendre « faux » quand le noyau touche le plafond **à pleine horloge** — c'est l'horloge qui répond, et elle a répondu : 2 617 MHz contre 1 792 pour la GEMV, 340,5 W nets contre un témoin copie à 341 W. Le noyau est au plafond parce qu'il déplace 1 156 Go/s, plus parce qu'il calcule. Règle 4 sur moi : un seuil porte son régime (brut/net) dans son nom. **Verdict split-K : non** — il n'enlève ni octets ni watts DRAM, et la seule chose qu'il aurait corrigée (une horloge rabattue par les instructions) n'existe plus. Les 2 jours vont à (ii).

**Ordre** : (ii) glue hôte (argsort, bincount, `int(ntiles.sum())`, +2 300 lancements — c'est ma condition 2, poste1) → **bt=32** (un paramètre, une heure) → pas complet sous graphes. Corollaire pour REGLES §9 : « ~1 050 Go/s » est dépassé par ce noyau (1 156) ; la borne dépend du motif d'accès, à re-annoter avec le noyau qui l'a mesurée.

**Le +13 % d'octets : l'explication ne tient pas à b=12.** À t=12 et k=8, `cnt ≤ 12 < 16` : aucun expert ne reçoit 17-32 jetons, donc aucune seconde tuile de M. Soit la manche avait t > 16 (godet, spéculation — à lire dans `cnt.max()` de la manche), soit les octets viennent d'ailleurs (échelles ou tables relues par tuile, `xq`, `down` seul). **Contrôle qui tranche** : bt=32, mêmes octets par noyau (gate, up, down séparés) ; si 4,33 → 3,83 Go, c'était la tuile et il faut expliquer t > 16 ; sinon la cause est dans un des trois noyaux et son nom le dit.

**Seuils du pas complet sous graphes, après (ii) + bt=32, MMA seule (godet 12 compté à part, −1,2 ms)**, référence 15,66 ms / 392 W / 0,512 J/jeton (même instrument) :

| grandeur | prédit | réfuté si |
|---|---|---|
| pas | **13,4-13,9 ms** (MoE 5,98 → 3,8-4,0, glue neutre sous graphes) | gain < 1,2 ms : la glue (ii) a mangé 40 % du noyau |
| W moyen brut | **385-400, inchangé** — je retire « ≤ 380 » : les noyaux MoE restent au plafond, à pleine horloge ; le gain est du temps à puissance constante | < 375 W (alors quelque chose d'autre a changé) |
| J/jeton brut | **≤ 0,46** (13,7 ms × 0,395 W / 12 = 0,45 ; −0,79 J par pas sur les 3 GEMM) | ≥ 0,48 |
| mJ/couche MoE sous graphes | 54,6 ± 3 (le chiffre eager doit se retrouver) | > 62 |
| `dram__bytes_read` MoE | 3,83 Go ± 3 % à bt=32 | > 4,0 Go |

Fenêtre : ≥ 20 s au compteur (`poste7-instrument-energie`), pas 6 s.
