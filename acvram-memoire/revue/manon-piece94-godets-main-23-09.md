# Pièce 94 — capture-godets rejouée sur main HEAD, 4 alias (ordre Jérôme)

instrument : `outils/gpu/mesure/capture-godets.py` (corrigé `0a5fc353`, refuse un acvram hors de l'arbre), godets {1,2,8,16} (REGLES § 3)
commit : main `ff264256` (fusion locale de `manon` sur `origin/main`, worktree `manon-w-21-09`, `git worktree list` confirme l'arbre)
régime : `ACVRAM_GEMV_LAYOUT=marlin`, `ACVRAM_GDN=fla`, `glue=compact(8)`, horloge libre ; canal en plus `ACVRAM_KV_INT8_CANAL=1` (kv=int8-canal16)
scellé : graphes==eager (seuil de l'outil), capture complète, 0 repli — 4/4 alias, 16/16 lignes godet
mesuré : `acvram.__file__` relevé en tête de chaque prise = l'arbre `manon-w-21-09` (jamais l'installation) aux 4 prises
verdict : **TENU 4/4 (16/16)** — aucun repli eager, `graphes_on=True` et `ok=True` à chaque godet
durée : 4 prises 13:34-14:24 (~50 min, dont chargement des 4 modèles), `carte.sh` journal cohérent (rendue à chaque tour)

| alias | godet 1 | godet 2 | godet 8 | godet 16 |
|---|---|---|---|---|
| Coder (Qwen3-Coder-30B-A3B-nvfp4) | 2,979 | 4,102 | 5,054 | 6,480 |
| GLM MLA (GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA) | 4,876 | 6,735 | 10,663 | 14,159 |
| Gemma (gemma-4-26B-A4B-heretic-APEX-I-Quality-nvfp4) | 4,219 | 5,356 | 6,659 | 7,695 |
| Coder + KV canal (ACVRAM_KV_INT8_CANAL=1) | 3,955 | 5,299 | 6,223 | 8,130 |

(ms/pas ; `ok=True`, `replis_eager=[]`, `n_replis_eager=0` partout)

Les 8 verdicts du 19/09 rejoués par famille (MLA/GLM, Coder, Gemma, KV canal) sur le code réellement fusionné dans main : aucun n'échoue, rien à nommer comme pièce distincte.

suite : Jérôme — les 8 verdicts du 19/09 peuvent être requalifiés « tenus sur main » ; ma file : cellule (a) attention 4 vs 8 warps, puis (b) rejeu vLLM b=12 (89) aux nouveaux défauts.
