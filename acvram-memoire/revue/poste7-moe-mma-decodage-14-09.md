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
