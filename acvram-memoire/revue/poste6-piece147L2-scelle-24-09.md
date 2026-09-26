# Scellé — pièce 147 L2 : GEMM de préfill Marlin sur la disposition unique, sans dépaquetage (poste6, 24/09, AVANT code et mesure)

* **Mécanisme** : au préfill (M > NVFP4_GEMV_MAX = 32) la disposition unique dépaquette TOUT le poids en bf16 (147 L3',
  `depaqueter_marlin`, ≈ 38-47 ms et ≈ 70 Go de trafic par passe) puis cuBLAS. Le chemin `gemm_dense` (Marlin W4A16, fp32
  déterministe) existe déjà pour M ≤ 32 (`_marlin_seul`, branche else). **Opt-in `ACVRAM_PREFILL=marlin`** : `gemm_dense` aussi
  au préfill, jusqu'à `ACVRAM_PREFILL_MARLIN_MAX_M` lignes (défaut 8 192 ; au-delà, dépaquetage + cuBLAS comme aujourd'hui).
  Défaut inchangé au bit (`bf16`). Jamais au défaut sans décision de l'utilisateur : la sortie change (± 1 ulp, 134).
* **ERRATUM de ma ligne du verdict 160 (« −0,15 à −0,20 s/passe »)** : elle supposait une GEMM au débit de lecture des poids
  fp4 ; à M = 624 la GEMM est bornée par le CALCUL (33,7 TFLOP, cutlass sm80 à 207 TFLOPS mesurés), pas par les octets. Le
  gain sûr est le dépaquetage (38 ms/passe) ; le reste dépend du rendement Marlin r = t(Marlin)/t(cuBLAS) à M ≥ 512, inconnu.

## Prédictions (8 × 78 = M 624, Qwen3.8 défaut ; TTFT servi b=1 à 512 / 2 048 / 4 096 ; J du préfill)
| grandeur | A (bf16) | prédit B | seuil « tenu » |
|---|---|---|---|
| passe 8 × 78 Qwen3.8 | 0,276 s | **−0,03 à −0,07 s** (r 0,9-1,2) | ≤ −0,03 s/passe (5 passes, médiane, mêmes lots) |
| TTFT 512 Qwen / gemma | 229 / 262 ms | **−20 à −50 ms** | B/A ≤ 0,92 sur 3 lots B contre 3 A |
| TTFT 2 048 | 756 / 951 | −40 ms à +150 ms (r 0,9-1,2) | tenu si ≤ 0 ; sinon MAX_M ≤ 1 024 recommandé |
| TTFT 4 096 | 1 440 / 2 025 | −40 ms à +300 ms | idem ; le seuil MAX_M se lit ici, pas de reformulation |
| J / préfill 2 048 Qwen | 242 J | **−8 à −20 J** (poids lus une fois en fp4, pas de bf16 écrit-relu) | J_B/J_A ≤ 0,97 |
| KL (C1 8 × 78, C2 mêlées) | témoins T1 seul/lot, T2 ordre | KL_max(A‖B) ≈ 0,003-0,006 (134 : 0,00545 au pas 0) | ≤ 2 × max(T1, T2) ; argmax A/B ≥ argmax T1/A − 0,5 pt |
| PPL par fenêtre (ppl-decode-kv, préfixe 2 048, 512 notés, 3 tranches, 2 modèles) | A | B/A dans [0,995 ; 1,005] par tranche | aucune tranche hors [0,99 ; 1,01] |

## Issues nommées, écrites avant
(a) r < 0,9 : Marlin bat cutlass sm80 sur la 5090 → gain au-delà du dépaquetage, MAX_M grand ; (b) 0,9 ≤ r ≤ 1,2 : gain = le
dépaquetage seul (−30 à −50 ms), MAX_M à poser où B/A repasse > 1 ; (c) r > 1,2 : Marlin perd à grand M, l'opt-in ne vaut
que pour M ≤ ~512 (invites courtes, chat) — l'énergie peut tenir quand même (moins d'octets) ; (d) KL ou PPL hors seuil →
« hors défaut », l'opt-in reste diagnostic ; (e) bilan Marlin en repli (port non compilé, .so partagé) → prise NULLE, dite telle
quelle, et la 160 reçoit son erratum (profil pris sur la disposition naturelle). Bras cassant : une tuile Marlin corrompue en
mode marlin doit changer la sortie (le chemin lit bien les tuiles) ; `ACVRAM_PREFILL=inconnu` doit refuser (ValueError).
## Instruments (fichiers suivis dans `scratchpad/poste6-p147l2-24-09/`)
`passes-kl.py` (passes 8 × 78 / mêlées, A→B à chaud, compteurs CHEMINS_NVFP4 et `proj_marlin_bilan` imprimés = preuve du
chemin), `ppl-decode-kv.py` (outils, --prefixe 2048), `prise-ttft.sh` (copie de la 145, bras A/B = ACVRAM_PREFILL), tests
`tests/test_prefill_marlin_l2.py` sous carte.sh avant toute prise. Régime : -lgc 2700, ECO=off, cpu-safe 100, worktree figé.
