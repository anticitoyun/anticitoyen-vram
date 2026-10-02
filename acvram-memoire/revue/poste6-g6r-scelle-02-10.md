# g6r — cible KV sous l'anneau : scellé à sec AVANT le code (poste6, 02/10, branche poste6-gemma-anneau ad341b8b9)

Ordre : chef 02/10 (g6r puis anneau gemma, sans carte, prédiction écrite avant). Instrument à sec : rejeu de la planification de G1
(`_plan_from_manifest` → `_poser_anneau` → `_borner_kv_avec_exil` → `_kv_blocks_per_device`) sur le VRAI manifeste
`gemma-4-31B-it-nvfp4-4sur6-vision`, `CUDA_VISIBLE_DEVICES=""`, VRAM simulée aux chiffres du journal G1 (capacité d'étage 29,5 Gio,
`mem_get_info` 30,7 / 31,84 Gio). Script : `scratchpad/poste6-g6r/sim_g6r.py` (diagnostic, pas un instrument de cellule).

## Ce que la lecture du code a trouvé (fichier:ligne, arbre ad341b8b9)

1. **Cause de l'exil 39/60 de G1** : `loader.py:2393` — `_plan_from_manifest` passe à `_reajuster_plan` un `kv_min` calculé par
   `_kv_plancher` AVANT `_poser_anneau` (`loader.py:354`, dans `load_model`) : `spec.kv_anneau` vaut encore 0, le plancher est celui du
   KV plein (30,2 Gio à 65 536), `min(kv, plancher) = kv` (`loader.py:1757`) → le KV (10,39 Gio) ne cède jamais, les MLP partent.
   Même défaut dans `_plafonner_mlp_prefill.essai` (critère « tient »).
2. **Régression que j'ai introduite hier APRÈS la carte, non mesurée** : commit 2808a8190, `config.py` (`cles = T` dès qu'il y a une
   fenêtre). À 65 536 le terme de scores vaut 32 × 1 024 × 4 × 65 536 = 8,0 Gio ; rejeu à sec de G1 sur l'arbre poussé : réserve
   11,66 Gio plafonnée (2,82 à G1), 60/60 exilés, **REFUS** « budget KV insuffisant ». L'arbre poussé ne sert plus gemma à 65 536.
   Origine : la chauffe compare le pic d'un SEUL TENANT de 16 384 (phase 1, `contexte.py:243`, 3,91 Gio) à la formule du régime PAR
   MORCEAUX (2,07) ; j'ai corrigé la formule au lieu de la comparaison.
3. `_kv_blocks_per_device` (`loader.py:1557`) : créneaux d'anneau = « un quart du budget » ; avec un budget au plancher d'une
   séquence, 3 créneaux mangeraient 0,82 Gio du pool des couches pleines → < 65 536 jetons.

## Correctif (g6r)

* A — `activations_prefill_bytes(n, seul_tenant=False)` : réserve du plan = régime par morceaux (clés ≤ S + fenêtre) ; la chauffe
  compare un tenu d'un seul tenant à la formule `seul_tenant=True` (lignes et clés en T).
* B — plancher PRÉVU = min(plein, anneau) quand l'anneau est permis (`ACVRAM_KV_ANNEAU` ≠ 0, R > 0) : `kv_min` et le critère de
  `_plafonner_mlp_prefill`. Le KV cède jusqu'au plancher d'UNE séquence sous l'anneau avant tout poids ; `_poser_anneau` (auto)
  prend l'anneau dès que le KV plein ne tient pas SANS exil.
* C — créneaux = clamp(budget ÷ plancher d'une séquence, 1, séquences planifiées) ; à budget ≥ plancher, les couches pleines
  gardent ≥ ⌈ctx/16⌉ blocs.

## Prédictions (avant le rejeu à sec du correctif, avant la carte)

Arithmétique : poids réels 20,37 Gio (attention 4,44, MLP 10,90 = 60 × 186 Mio, embed 2,62, tête 1,34, annexes 1,07) ; capacité −
marge = 29,5 − 1,59 − 4,04 (réserve 2,82 + tampons denses 1,22) = 23,87 ; reste au KV sans exil : 3,50 Gio ; plancher 5,45.

| | prédit | faux si |
|---|---|---|
| S1 à sec, 65 536, auto | anneau R=67, budget KV 5,45 Gio (± 0,1), **11 MLP exilés (10-12)**, 1 créneau, couches pleines ≥ 4 096 blocs, pas de refus | exilés ≤ 5 ou ≥ 20, refus, < 4 096 blocs |
| S2 à sec, 27 648, auto | anneau, **0 exilé** (KV 2,53 ≤ 3,50) ; avant correctif : 60/60 | ≥ 1 exilé |
| S3 à sec, 4 096, auto (témoin) | KV plein, 0 exilé, plan IDENTIQUE avant/après correctif (budget, blocs) | un octet de budget diffère |
| S4 à sec, plus grande fenêtre SANS exil sous l'anneau | entre 38 912 et 46 080 | hors intervalle |
| S5 à sec, ACVRAM_KV_ANNEAU=0, 65 536 | comme avant (refus ou exil total), l'anneau n'est pas pris | anneau posé |
| G1 carte (rejeu) | 65 536 tenus, 10-12 exilés, DÉGRADÉ, graphes off | NOMINAL |
| **G5 carte (≥ 20 j/s NOMINAL)** | **FAUX avec g6r seul** : 5-9 j/s (15,7 ms + 11 × 8,8 ms PCIe, eager) | ≥ 20 j/s |

La prédiction du bead (« 0-5 exilés, NOMINAL ≥ 20 j/s », écrite hier sans ce compte) est donc annoncée FAUSSE avant mesure : il
manque 1,95 Gio, que le KV ne peut pas rendre (il est au plancher). Si le rejeu à sec donne ≤ 5, c'est mon compte des poids qui est
faux et je le dirai.

## Levier nommé pour la suite (anneau gemma, bead 5y0), hors g6r

La table de plongements bf16 (2,62 Gio) est placée sur la carte AVANT les MLP (`tiering.py:581`) et `_reajuster_plan` n'exile que
des MLP. C'est pourtant le poids le moins cher à sortir (une ligne par jeton ; un MLP exilé coûte 8,8 ms/jeton et coupe les
graphes). 2,62 > 1,95 : table en RAM hôte, tête liée int8 gardée sur la carte → prédit **0 exilé à 65 536, NOMINAL**. Points à
lire avant : `_tete_liee` exige `embed.is_cuda` (`loader.py:1164`), la tour de vision suit `embed_tokens.device`
(`runner.py:762`), `_essai_de_chauffe` mesure sur ce même appareil (`contexte.py:103`).
