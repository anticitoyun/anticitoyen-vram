# Pièce 126 — profil nsys Qwen3.8-27B-nvfp4, b=8, décodage établi (24/09, ordre chef)

Contexte : 102 de poste2, NInfer +56 % débit / −35 % énergie vs acvram à b=8 sur ce modèle (égalité à b=1).

## Instrument

- `outils/carte.sh` (ACVRAM_NOM=poste3-piece126-nsys-qwen38, ACVRAM_DUREE_MAX=300), prise 66 s d'attente
  + profil, rc 0, cpu-safe (load 2,06 avant prise).
- Modèle : `/mnt/AI_GENERATOR/models_acvram/Qwen3.8-27B-nvfp4` (config : `Qwen3_5ForConditionalGeneration`,
  64 couches, 48 `linear_attention` (GDN, `full_attention_interval=4`) + 16 `full_attention`).
- `outils/gpu/mesure/frontiere-pas.py`, B=8, 20 pas, sous nsys (`-t cuda --cuda-graph-trace=node`).
- Commit worktree : 8f9210b2.

## Familles

`_dense_etroit_kernel` (gemm_dense_etroit nvfp4) est le chemin **PARTAGÉ** attention (q/k/v/o,
`attention.py:_multi_projection`) + GDN (`loader.py` : les `lin()` de KimiDeltaAttention/GDN passent par
le même `QuantLinear`) + MLP (gate/up/down, même mécanisme) + tête (`model.py:296-302`, DENSE_NVFP4_MIN_M
≤ b ≤ 32) — **pas séparable par nom de noyau** sans instrumenter les sites d'appel (lu au bit dans
`attention.py`, `model.py`, `loader.py`). Rapporté comme un seul poste plutôt qu'un éclatement inventé.
Les GEMM cutlass_80_tensorop bf16 vus dans la trace brute (26 k+ lancements) tombent TOUS hors des
fenêtres de décodage établi (prefill/chauffe uniquement) : 0 dans le régime mesuré, confirmant que le
chemin triton nvfp4 est bien le défaut actif à b=8 (`gemv_marlin_temoin` aussi à 0, cohérent avec
`regime.py:66`).

n=66 fenêtres jugées (sur 68, tête/queue exclues), mur (borne GDN à borne GDN) 26 184,5 µs/pas.

| famille | µs/pas | lancements/pas | % plafond |
|---|---|---|---|
| gemm_dense_etroit_nvfp4 (attn+GDN-proj+MLP+tête, partagé) | 21 072,3 | 401 | 80,5 % |
| auxiliaires (elementwise/reduce/copy torch) | 1 618,7 | 1 374 | 6,2 % |
| gdn_recurrence (fused_recurrent + chunk_* + conv1d + l2norm) | 1 155,6 | 96 | 4,4 % |
| normes (rmsnorm) | 492,1 | 129 | 1,9 % |
| copies (memcpy/memset CUDA) | 444,4 | 61 | 1,7 % |
| attention_pleine (paged_attn + rope + kv_write, 16 couches) | 397,7 | 64 | 1,5 % |
| mlp_activation (swiglu) | 70,6 | 64 | 0,3 % |
| gemv_marlin_temoin | 0,0 | 0 | 0,0 % |
| gemm_bf16_cutlass | 0,0 | 0 | 0,0 % |
| noyaux classés | 25 251,4 | — | 96,4 % |
| trou (hors noyaux) | 933,1 | — | 3,6 % |

## Verdict

Le poste dominant est de très loin le GEMM dense nvfp4 partagé (80,5 % du pas, 401 lancements) — c'est
lui qui porte l'écart avec NInfer à b=8, pas GDN (4,4 %) ni l'attention pleine (1,5 %). Reste non
distingué : quelle part de ces 401 lancements/401 relit les poids par godet vs par pas (comme signalé
`gemm_dense_etroit.py:3`, nvfp4_gemv relit par séquence à b>1 — vérifier si `_dense_etroit_kernel` a le
même défaut à b=8, ou s'il tient déjà la promesse « poids lus une fois par pas »). Piste suivante si
demandée : `--detail gemm_dense_etroit_nvfp4` par forme (Grd) pour situer combien de ces 401 sont
QKVO/tête vs MLP par TAILLE de sortie (dims connues : hidden 5120, intermediate 17408 → MLP largement
plus gros, distinguable par shape même sans instrumentation).

## Suite (24/09, même trace, ordre chef) — éclatement par forme de grille

`GrdX` du noyau = `tuiles_n = ceil(N/64)` (`gemm_dense_etroit.py:236,241`) : lu directement, pas déduit.
N attendus (config `text_config`) : q_proj-attn 6144, k/v_proj-attn 1024, o_proj-attn 5120, GDN q/k_proj
2048, GDN v_proj 6144, GDN out_proj 5120, MLP gate/up 17408 (fusionnés 34816, `attention.py:597`), MLP
down_proj 5120, tête vocab 248320 — 16 couches attn, 48 couches GDN, comptes de lancements/pas croisés
contre ces layer-counts pour lever l'ambiguïté quand plusieurs projections partagent le même N.

| GrdX (N≈) | µs/pas | lanc/pas | % plafond | identification |
|---|---|---|---|---|
| 544 (34816) | 9 079,5 | 64 | 34,7 % | **MLP gate+up fusionné** — exact, seul candidat à ce N |
| 80 (5120) | 6 440,3 | 128 | 24,6 % | o_proj-attn(16) + GDN-out_proj(48) + MLP-down_proj(64) = 128 lanc., triangulé par comptage de couches, **mélangé, non isolable plus finement** |
| 160 (10240) | 1 836,2 | 48 | 7,0 % | non identifié avec certitude (48 lanc. ~ un poste par couche GDN, N ne correspond à aucune somme simple des projections connues) |
| 96 (6144) | 1 323,2 | 48 | 5,1 % | **GDN v_proj**, probable — 48 lanc./pas = exactement le compte de couches GDN (q_proj-attn donnerait 16, écarté) |
| 3880 (248320) | 883,7 | 1 | 3,4 % | **tête** — exact, seul candidat à ce N |
| 224 (14336) | 867,1 | 16 | 3,3 % | non identifié avec certitude (16 lanc. ~ compte de couches attn, N ne correspond à aucune somme simple) |
| 1 (64) | 637,8 | 96 | 2,4 % | non identifié (probable gating GDN à petit N — f_a/f_b/g_a/g_b/beta, 96 = 48×2, non vérifié au bit) |

Total gemm_dense_etroit_nvfp4 sur ces 7 formes : 21 067,8 µs/pas (80,5 % du mur), 401 lanc./pas — cohérent
au bit avec le poste classer.py.

**Verdict pour poste1** : le seul sous-poste identifié sans ambiguïté et isolable est **MLP gate+up
fusionné, 34,7 % du pas** — la plus grosse cible unique. Le bucket 5120 (24,6 %) mélange MLP-down_proj
avec deux projections d'attention/GDN non-MLP et n'est PAS isolable par forme seule (les trois partagent
N=5120) : si W4A4/Marlin cible spécifiquement le MLP, gate+up (34,7 %) est la cible sûre ; down_proj
(inclus dans les 24,6 %, poids non isolé) reste à confirmer par instrumentation des sites d'appel si le
gain mesuré sur gate+up seul ne suffit pas à expliquer l'écart NInfer. 12,8 % du pas (160+224+1) reste
non identifié avec certitude — pas de conjecture au-delà de ce que corrobore le compte de lancements.
