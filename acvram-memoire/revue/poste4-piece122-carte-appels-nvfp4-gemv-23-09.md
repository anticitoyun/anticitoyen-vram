# Pièce 122 — carte des appels de nvfp4_gemv_kernel et narrow_gemm sur les chemins SERVIS, à sec

Ordre chef : localiser les sites d'appel réels, la valeur de `NV` (ou `M`) selon le godet b, et l'alias
concerné. Lecture de code seule, aucun code écrit, aucune mesure carte.

## `nvfp4_gemv_kernel` (`acvram_kernels.cu:264`) — un seul site d'appel

`acvram/kernels/__init__.py:624`, dans `nvfp4_matmul` (`:569`) — le backend générique
`{"nvfp4": nvfp4_matmul, ...}` (`:1180`) de **toute** projection dense NVFP4 (q/k/v/o, gate/up/down non-MoE,
tête). Aucun autre appelant trouvé (`grep -rn "\.nvfp4_gemv("` sur tout `acvram/` : une seule occurrence).

**Mécanique de `NV`** (`acvram_kernels.cu:1315-1373`, fonction C++ `nvfp4_gemv`) : `N = xc.size(0)` (le nombre
de lignes d'activation = le lot de jetons présentés à CET appel, un par séquence au décodage) est traité par
tranches de 8 ; `NV = min(8, N - base)` sélectionne l'instanciation du gabarit. **`NV` suit donc directement le
godet b présenté à ce noyau précis** — mais seulement si l'appel atteint réellement `nvfp4_gemv` (voir
ci-dessous, ce n'est pas automatique).

### Ce qui intercepte AVANT nvfp4_gemv, par défaut

`nvfp4_matmul` (`__init__.py:569-625`) essaie, dans l'ordre, pour `n` = lot de jetons :
1. **Marlin dense** (`_PROJ_MARLIN`, `:908`) — **`ACVRAM_PROJ_MARLIN=0` par défaut** : jamais pris sans
   activation explicite.
2. **GEMM dense étroit Triton** (`gemm_dense_etroit`, `_DENSE_NVFP4`, `:945-946`) — **`ACVRAM_DENSE_NVFP4=
   "triton"` par défaut, `_DENSE_NVFP4_MIN_M=4`** : pris pour **`4 ≤ n ≤ 32`**, dès lors que
   `gemm_dense_etroit.disponible()` (Triton chargé + CUDA visible) et `xf.dtype == bf16` — les deux vrais en
   service normal.
3. **`narrow_gemm`** (tensor cores) — `_NARROW_GEMM=0` **et** `_NARROW_NVFP4=0` par défaut (`:554,558`) :
   jamais pris pour cette voie sans double activation.
4. **Repli : `nvfp4_gemv`** — atteint seulement si aucune des trois voies ci-dessus n'a intercepté.

### Par godet, en régime servi par défaut

| Godet b | n présenté | Voie prise (défaut) | `NV` dans nvfp4_gemv | Atteint ? |
|---|---|---|---|---|
| 1 | 1 | aucune des trois n'accepte n=1 (Marlin/étroit exigent n≥2/4) | **NV=1** | oui, `nvfp4_gemv` |
| 2 | 2 | Marlin off, étroit exige n≥4 | **NV=2** | oui, `nvfp4_gemv` |
| 4 | 4 | **GEMM dense étroit Triton** (`4≤n≤32`) | — | **non**, intercepté |
| 8 | 8 | **GEMM dense étroit Triton** | — | **non**, intercepté |
| 12 | 12 | **GEMM dense étroit Triton** | — | **non**, intercepté |

**`NV≥4` — la zone de déversement sévère de la pièce 120 (255 registres, jusqu'à 456 `LDL`/`STL` à NV=8) —
n'est PAS atteinte en service par défaut, pour aucun des cinq godets demandés.** Elle ne serait exercée que si
`ACVRAM_DENSE_NVFP4` était mis à autre chose que `"triton"`, ou si `gemm_dense_etroit.disponible()` échouait en
service (Triton absent/indisponible sur ce processus) — ni l'un ni l'autre n'est le régime par défaut, et aucun
des deux n'a été vérifié en service ici (à sec, aucune mesure).

### Alias concerné

* **i8c** (servi, témoin) et **S1b** (`poste6-piece107ter-octets-joules-23-09.md` : « projections ET tête de
  i8c ») ont des projections **int8**, pas NVFP4 dense — elles passent par `int8_matmul`/`int8_gemv_kernel`
  (`__init__.py:1063` et suivants), **jamais par `nvfp4_gemv_kernel`**.
* **A** (`nvfp4-qkv-alphaqkv-23-09`, `poste6-piece107-dossier-alias-prose-23-09.md`) a des projections **NVFP4
  denses** — **seul alias nommé qui puisse exercer `nvfp4_matmul`/`nvfp4_gemv`**, aux godets 1 et 2 seulement
  (§ ci-dessus) par défaut.
* **« tête nvfp4 »** : le format de la tête (lm_head) est indépendant de celui des projections (pièce 103 :
  `_tete()`, `model.py:262-303`) et suit le MÊME seuil `_DENSE_NVFP4_MIN_M` pour le chemin étroit — même
  conclusion (interception Triton dès n≥4). Quel alias sert précisément une tête NVFP4 (i8c ? A ? un troisième ?)
  n'a pas été tranché ici — `fiche-service.tsv`/`notes-modeles.tsv` non consultés, à faire séparément si utile.
* **« experts » (MoE)** : passent par `nvfp4_gemv_grouped_kernel`/`nvfp4_gemv_marlin_kernel` — des gabarits
  **différents**, hors du périmètre de cette pièce (le déversement de la 120 concernait spécifiquement
  `nvfp4_gemv_kernel`, pas les variantes `_grouped`/`_marlin`).
* **S8** : alias non retrouvé avec certitude dans le temps imparti (la file S1-S7 est documentée dans
  `poste6-piece107-dossier-alias-prose-23-09.md`, S8 n'y apparaît pas explicitement) — à demander à poste6.

## `narrow_gemm` (gemm_etroit) — hors du chemin servi par défaut, sauf MLA

Deux sites d'appel, NVFP4 (`__init__.py:615-621`) et int8 (`:1063-1067`), tous deux gardés par
`_NARROW_GEMM`/`_NARROW_NVFP4` — **`0` par défaut dans les deux cas**. Une exception : `int8_matmul` accepte
aussi un drapeau **par tenseur**, `t.etroit`, qui contourne la variable globale — **posé uniquement dans
`engine/mla.py:641`** (poids d'attention MLA, donc potentiellement des alias à couches MLA comme GLM), jamais
pour les projections q/k/v/o standard de Coder (i8c/A/S1b). **Conclusion : `narrow_gemm_kernel` n'est pas sur
le chemin servi par défaut pour i8c, A ou S1b** — les 48 déversements constants relevés en pièce 120 ne
s'exercent donc pas en service normal pour ces trois alias, sauf activation explicite ou modèle MLA.

## Bilan pour chef

**NV≥4 n'est atteint en service par défaut pour aucun godet {1,2,4,8,12} d'aucun alias nommé** — le régime
Triton (`gemm_dense_etroit`, actif par défaut dès n≥4) intercepte avant `nvfp4_gemv`. L'hypothèse « le
déversement de `nvfp4_gemv` explique le nvfp4 étroit à 0,45 To/s / la lenteur de A à b=12 » **ne tient pas telle
quelle** dans la configuration par défaut — soit la lenteur a une autre cause (le chemin réellement pris à b=12
pour A est `gemm_dense_etroit`/Triton, pas `nvfp4_gemv`), soit le service de A tourne avec `ACVRAM_DENSE_NVFP4`
changé de son défaut ou Triton indisponible — **à vérifier en service (ligne de régime, pas ici) avant de
retenir ou d'écarter la piste**.
