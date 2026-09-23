# acvram_rust prise 1 : relevé des noyaux servis, Qwen3-4B-srcgguf-nvfp4, b=1 (poste5, 23/09)

instrument : nsys `--cuda-graph-trace=node` sur `acvram serve` (`scratchpad/poste5-rust-p1-23-09/prise1.sh`, `analyse.py`) ; noms, grilles et ordre seulement (temps hôte sous nsys faux, poste5-p69)
commit : 6f8828a5 (branche poste5) ; extension `kernels-86bf52a9f2f9` ; cache Triton neuf (9 cubins)
régime : eco=2700(2692), graphes=on(hybrides≤4), `--max-batch 1 --speculative none`, 5 invites × 128 jetons gloutons
scellé : aucun chiffre ; porte de l'étape 1 inchangée (sha des ids 5/5)
mesuré : 277 236 lancements, 82 noyaux distincts ; 5/5 réponses ; `prompt_tokens` 18/30/32/28/41 = fixture Rust (tokeniseur au bit, déjà testé)
verdict : décodage transcriptible par NOS noyaux ; préfill NON (cuBLAS/cuBLASLt + flash de libtorch) → décision du chef (§ 3)
durée : prévu ≤ 300 s ; tenu 10 s (1er essai, OOM) + prise 2 (attente carte 368 s, tenue ≈ 25 s)

## 1. Pas de décodage b=1 : 432 lancements (période exacte, 8 tours, trouvée sur la fin du relevé)

| famille | /pas | origine |
|---|---|---|
| `nvfp4_gemv_kernel<4,1,bf16,bf16>` | 140 | .cu (espace anonyme) |
| `int8_gemv_kernel<4,1,bf16,bf16>` / `<…,float>` (tête, grille 37 984 = 151 936/4) | 60 / 1 | .cu |
| `rmsnorm_bf16_kernel` | 73 | .cu |
| `rope_inplace_kernel`, `kv_write_int8_kernel` | 36 + 36 | .cu |
| `_partiel_reduit_kernel` (attention) | 36 | Triton (cache) |
| `swiglu_bf16_kernel` / `swiglu2_bf16_kernel` | 20 / 16 | .cu |
| ATen : argmax, `indexSelect` (plongement du jeton suivant, SUR la carte), copie | 3 | à écrire en Rust |
| ATen : scatter, max, abs, eq, masked_fill, add, exp, sum, log, add ×2 (logsumexp du log-prob renvoyé) | 11 | hors justesse du jeton ; non porté |

* Tête liée : **int8, quantifiée au chargement** (branche `loader.py:1087-1089`), sortie fp32 → Rust la reçoit
  d'un vidage Python à l'étape 1 (sha au journal), réimplantation plus tard avec test au bit.
* Couches NON uniformes (grilles de GEMV 4 864 vs 2 432 : gate·up fusionnés ou non ; swiglu ou swiglu2 ; q/k
  int8 ou nvfp4 selon la couche) : la table par couche se dérive du manifeste et se vérifie contre ce relevé.

## 2. Préfill (invites de 18 à 41 jetons)

`cutlass::Kernel2<…bf16_s16816gemm…>` et `cublasLt::splitKreduce_kernel` (matmul torch → cuBLAS/cuBLASLt),
`pytorch_flash::flash_fwd_kernel` (SDPA de libtorch), `fmha_cutlassF…` (mem-efficient), `_dense_etroit_kernel`
(Triton, T ≤ 32), `vectorized_gather`. Pour un préfill au bit, Rust devrait charger la MÊME libcublasLt que
torch (venv, cu130) avec la même taille d'espace de travail et les mêmes heuristiques, et prendre le noyau flash
dans le fatbin de `libtorch_cuda.so` : faisable, mais lourd et fragile.

## 3. Question au chef (une ligne, deux options)

**A** — porte de l'étape 1 = sha 5/5 du DÉCODAGE à préfill injecté : le Python vide sa KV et son premier jeton
après le préfill, Rust les charge puis décode 128 jetons avec nos noyaux. Isole la question (a) (débit b=1 =
décodage) ; préfill Rust approché (boucle q_len=1), jugé par KL. **B** — préfill au bit (cuBLASLt du venv + flash
de libtorch) avant toute prise de débit : + 1 à 2 jours, risque d'heuristique cuBLAS. Recommandation : **A**.

Incident de la prise 1 : le verrou est revenu (journal : `ANOMALIE 2884083 env … .qui déjà repris`) pendant que le
serveur du détenteur précédent tenait encore 28,5 Gio → OOM. Garde ajoutée (carte vide hors 8081-8083 avant de
charger, sinon refus 75). Le défaut est dans la libération du verrou, pas dans cette prise.
