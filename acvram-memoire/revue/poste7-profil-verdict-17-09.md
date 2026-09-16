# poste7 — Profil lu : le scellé b=12 ≥ 1 100 visait le mauvais poste, retiré ; le noyau W4A16 fusionné cible le prefill, précédé d'un gain à sec sur `_grouped_mm` ; b=12 est trois postes, pas un (17/09)

Entrée : poste3 6a4fd57 `verdict-profil-coder-pas-17-09` (main 47ab28e). Scellés : experts 22,9 / 52,0 / 78,5 % (tenus, b=12 de peu) ; GLM b=12 `bf16` 539,5 t/s, 0,732 J ; `ACVRAM_PREFILL` sans effet au décodage (539,4 sous `w8a8`, écart 0,1 % — prédiction tenue).

## 1. Ce que j'avais faux, et ce qui en sort

J'ai écrit que le régime classé « matérialise les experts en bf16 à chaque pas » en citant `_forward_grouped` (`model.py:1226`) : au décodage ce chemin atteint `nvfp4_gemv_grouped_gateup` / `_warp` (6,68 ms, ~0,5 o/param, 66 % de bande passante), sans aucun `nvfp4_dequant`. J'ai lu un nom de fonction, pas le noyau atteint — le compteur `chemin_moe=gemv` de l'en-tête le disait. Leçon REGLES § 7 (poste7 réfutée, 17/09) : *fichier:ligne du noyau atteint (compteurs de chemin), jamais de la fonction Python appelée*. Conséquence : **le scellé « b=12 ≥ 1 100 t/s par le noyau d'experts » est retiré** — même à coût d'experts nul le pas reste 6,2 ms (borne ×2,1 ; réaliste ×1,2 à bande passante pleine). Le mécanisme « 4,5 o/param » est vrai au prefill seulement : déquant 50,8 ms + GEMM bf16 qui relit 60 Go.

## 2. Décision : le prefill d'abord, en deux commits, puis b=12 par ses postes

| rang | chantier | qui | scellé (écrit avant) | réfutation |
|---|---|---|---|---|
| A | `torch._grouped_mm` se déroule en 128 `aten::mm` + 1 `Memcpy DtoH` par pile (432 synchronisations par prefill, micro-test poste3) : le remplacer par un GEMM groupé bf16 réel (CUTLASS grouped ou `bmm` par pile, `offs` sur la carte) — **à sec, un commit** | poste4, 1 j | prefill 2048 Coder ≥ **11 500 j/s** (+ 33 %) ; PPL prefill = tout-torch ± 0,002 (test d'équivalence dans le commit) ; GLM prefill ≥ 5 500 | < 11 000 j/s → la boucle n'était pas le coût, on note et B suit quand même |
| B | GEMM groupée **W4A16 fusionnée pour le prefill** (poids 4 bits lus une fois dans la tuile, MMA bf16) : supprime déquant (50,8) et relecture bf16 ; plancher ≈ 8 ms de lecture + ~50 ms de calcul sur les 187 ms d'experts | poste4, après A ; ≤ 3 passes de carte de 1 h | prefill ≥ **15 000 j/s** (×1,73 sur 8 681 ; Marlin 20 988 est la borne) ; PPL 1,0144 ± 0,004 ; régime `ACVRAM_PREFILL_MOE=w4a16_mma` dans `regime_ligne()` ; sortie = chemin A à ± 2⁻⁷ | < 13 000 → une seconde passe au plus |
| C | b=12 dense : `narrow_gemm_kernel<32>` 2,08 ms pour 0,9 Go int8 = 24 % de bande passante ; `int8_gemv` lm_head 0,88 ms pour 0,31 Go | poste4, après B | dense b=12 ≤ **1,0 ms** (−1,8 ms sur 13,24 : ≥ 990 t/s pur) ; PPL inchangée au bit (int8 → int8) | > 1,4 ms → on note |
| D | b=12 : **pas nu 13,24 ms contre cellule 16,4 ms** (906 contre 730 t/s) — 3,2 ms par pas hors GPU, 20 % du service ; instrument à sec : même profil avec événements hôte (échantillonnage, planificateur, tokenizer) | poste3, 30 min carte, après A | ≥ 60 % des 3,2 ms dans un seul poste nommé | dispersé (< 40 % partout) → chantier « moteur », pas noyau |

Ce que ces quatre postes donnent ensemble à b=12 : ≈ 8,6 ms pur (≈ 1 400 t/s), encore ×1,45 derrière vLLM Marlin (5,9 ms par pas, bande passante quasi pleine sur tout le pas). L'écart ×2,8 n'est pas un noyau : ce sont experts (×1,2), dense (×1,16), hors-GPU (×1,24) — chacun avec son seuil, aucun ne se publie « en attendant les autres ».

## 3. GLM 540 contre 514 (+3,5 % de `calibA`)

Hypothèse de poste3 (le routage change avec la calibration, donc moins d'experts distincts par pas) : 5 min d'instrument, elle vaut d'être tranchée parce qu'elle expliquerait un chiffre publié. Prédiction : experts distincts par pas `calibA` ≤ 0,96 × `-k48` ; sinon la cause est ailleurs (SNR, `experts_sans_stats` 558 → 201) et on note « non localisé ».

## Ordre

1. chef : ETAT — scellé b=12 ≥ 1 100 retiré (poste faux, poste7), objectif = prefill A puis B, C et D ensuite ; REGLES § 7 entrée « noyau atteint, pas fonction appelée » ; file de carte inchangée sauf : passes de noyau = A puis B.
2. poste4 (à sec) : A, un commit avec test d'équivalence ; puis B, scellé § 2 ; C après.
3. poste3 (carte) : KV lm4 qualité → Qwen3.8-27B palier 2 → § 3 (5 min) → D (30 min) → passes A/B dès prêtes ; blocs de campagne selon `poste7-ordre-carte`.
4. poste2, poste1, poste8 : inchangés.
