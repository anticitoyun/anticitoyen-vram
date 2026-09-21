# (5) MoE en MMA groupée au décodage, pas complet sous graphes (14/09, 10h48-11h08, Laurine)

En-tête (REGLES §3) : 5090 seule (`CUDA_VISIBLE_DEVICES=0`), plafond 400 W,
horloge libre (3 135 max), 33-36 °C au départ, compteur NVML
`TotalEnergyConsumption` (energie.py), venv anticitoyen-vram, Coder-30B
NVFP4 (experts NVFP4, attention/lm_head int8), b=12, invites 128, rejeu de
graphes (godet 16), une carte, une manche par série, fenêtres 22 s.
Levier (3) de Sage (`sage-moe-mma-decodage-14-09.md` § 3, seuils rescellés
sur la référence 20 s de Laure 19,0 ms / 630,7 t/s / 0,619 J).

## Ce qui est câblé (main, `ACVRAM_MOE_DECODE_MMA=1`, coupé par défaut)

`MoEBlock._forward_grouped_mma` (`acvram/engine/model.py`) : le chemin MMA
du prefill (quant_act → 3 GEMM groupées BT=16 → moe_act → quant_act →
moe_reduce_trie) rendu CAPTURABLE : grille de tuiles fixe d'Océane
(`_tuiles(cnt, 16, t_max = ceil(t·k/16) + E)`, c8096c0), `scatter_add_`
au lieu de `bincount` (qui lit le max sur l'hôte), fantômes `topi = -1` →
expert 0 à poids 0 (contribution nulle et finie). Deux défauts trouvés en
route, corrigés dans le même commit : `bincount` incapturable (attrapé par
le test de capture) ; **les trois noyaux groupés lisaient `t0 - 1` sur
une tuile vide** (`min(r, nt-1)` avec nt=0 → accès mémoire illégal sur
Coder-30B, E=128, où la grille fixe pousse t0 hors de xq ; invisible sur le
petit test, E=16) → `if (nt <= 0) return;` (`acvram_kernels.cu`, ×3).
Tests : `tests/test_moe_decode_mma_graphe.py` (4 : = prefill-MMA bit à
bit, fantômes nuls, capture réelle + rejeu = eager, grille large t0 hors
tampon) + `tests/test_moe_mma_decodage.py` (condition (i) de Sage).

## Mesures (A = GEMV défaut, B = MMA décodage), même manche

| | A (GEMV) | B (MMA) | Δ |
|---|---:|---:|---:|
| ms/pas, banc 62 pas ×5 (ABAB) | 14,58 / 14,59 | 13,27 / 13,26 (σ 17 vs 5) | **−1,32 ms, −9,0 %** |
| jetons/s (banc) | 823 | 905 | +10 % |
| pas complet 22 s au compteur : ms/pas | 15,91 | 13,75 | **−13,6 %** |
| W brut / net | 398,6 / 356,0 (2 587 MHz, bridé) | 374,9 / 313,5 (2 932 MHz) | −24 W |
| J/pas | 6,34 | 5,15 | −19 % |
| **J/jeton brut / net** | 0,528 / 0,472 | **0,429 / 0,365** | **−19 % / −23 %** |
| repos avant la fenêtre | 42,5 W | 61,4 W (carte chaude après A) | |

Les deux instruments ne donnent pas le même Δ de temps (−9,0 % au banc
court à répétitions, −13,6 % sur 22 s) ; le banc B a une variance ×3 (σ 17
contre 5) — à comprendre avant de publier un t/s (un pas MMA plus
sensible à l'état thermique ? ou le godet/fantômes). Le J/jeton, lui,
vient d'une seule fenêtre de 22 s au compteur, sans ce doute.

Jetons : B graphes contre B eager, 12 séquences, 10 identiques ; s4 (jeton
13) et s7 (jeton 14) divergent — **la même paire que la divergence
graphes/eager du chemin GEMV (Océane, `graphes-preparer-divergence-14-09.md`,
ties bf16 < 0,4 logit)** ; à passer par son contrôle de ties avant de
conclure « bruit », pas fait ici. A contre B : s4, s7, s11 divergent (W4A4
contre W4A16, attendu — qualité +0,919 % déjà payée, A4).

## Contre les seuils rescellés (§ 3, base 19,0 ms / 630,7 t/s / 0,619 J)

| seuil | prédit | mesuré (Δ appliqué à la base) | verdict |
|---|---|---|---|
| pas 15,5-16,3 ms (réfuté > 17,3) | −14 à −18 % | −13,6 % (22 s) → **16,4 ms** ; −9,0 % (banc) → 17,3 | **à la limite** : tenu sur la fenêtre 20 s (16,4 vs 16,3), au seuil de réfutation sur le banc court |
| 735-775 t/s (réfuté < 850 concerne 1aj+godet, pas ici) | | 630,7 × 1,157 = **730** (22 s) ; × 1,10 = 694 (banc) | idem, à la limite basse |
| J/jeton 0,50-0,53 (réfuté ≥ 0,56) | −19 % | 0,619 × 0,813 = **0,503** | **tenu** |
| mJ/couche MoE 54,6 ± 3 | | 55,4 (3 GEMM seules, tampons tournants) | tenu |
| W moyen 385-400 inchangé | | 375 brut : **le pas sort du plafond** (2 932 MHz contre 2 587) | mieux que prédit — le gain n'est pas « du temps à puissance constante », c'est du temps ET des watts |
| octets MoE = routage × 2,654 Mo ± 3 % | | mesuré le 14/09 soir à 98-100 % (bras `experts`) | tenu |

Verdict : **levier réel, −19 % de J/jeton et −13,6 % de temps sur la
fenêtre de 20 s, tenu sur l'énergie, à la limite sur le temps.** Ce qui
manque pour le −18 % prédit : la glue (argsort, gather, 2 quant_act,
moe_act, reduce_trie, ~0,6 ms) et le lm_head/projections qui restent au
plafond. Défaut par défaut : `ACVRAM_MOE_DECODE_MMA=0` tant que (a) les ties
s4/s7 ne sont pas contrôlés, (b) Manon n'a pas donné la PPL de CE chemin
(même W4A4 que le prefill, +0,919 % attendu, mais au décodage la
quantification des activations touche chaque jeton généré).

## P7 (Sage) — tampons tournants, même manche

`energie_par_poste.py` tourne désormais chaque poste sur les 48 couches
(2,5 Go d'experts, 0,9 Go de projections ≫ 96 Mo de L2). Contre la
mesure du 14/09 soir (couche 5 seule, L2-résidente) : W identiques
(399-401 brut, tous au plafond), gate·up 1 590 MHz (1 642 chaud), mJ par
couche gate·up 43,5 (40,2 chaud), down 31,7 (25,6), 3 GEMM MMA 55,4 (54,6).
**Le classement ne change pas** : le L2 chaud sous-estimait de 8 à 24 %
l'énergie par couche des GEMV, pas leur puissance ni l'ordre.
