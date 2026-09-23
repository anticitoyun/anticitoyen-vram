# Ce qui reste avant la fin du projet — 22/09 15 h 3x (Maîtresse)

Objectif du projet : acvram plus rapide et plus économe (J/jeton) que llama.cpp, vLLM et TensorRT-LLM sur RTX 5090, à qualité prouvée (KL contre bf16), et livré (.deb, GitHub).

## Où en sont les portes (mesuré, pas promis)
| Porte | État | Chiffre |
|---|---|---|
| b=1 décode | TENUE | 380,8 t/s (vLLM 290,6 ; TRT-LLM 46, ordonnanceur sérialise) |
| Prefill | TENUE | 22 707 j/s (vLLM 21 054) |
| b=12 décode | NON TENUE | acvram 1 634 · vLLM 1 782 (−8 %) · TRT-LLM 1 998 (−18 %) |
| J/jeton b=12 | NON TENUE (= débit au plafond 400 W) | acvram 0,196 · TRT-LLM 0,156 · llama.cpp 0,206-0,216 ; vLLM à mesurer (5e chaîne en cours) |
| Qualité W4A4 (Coder) | TENUE 5/5 | KL max 0,519 vs bf16 (seuil 1,0) ; TRT-LLM 0,965 sur 1 invite (à compléter) |
| Qualité 31B 4sur6 | NON JUGEABLE | eval PPL cassée (53 202) — diag-eval-nll à exécuter (pièce 37) |
| 30B-VL qualité | NON TENUE | P3(3) 12,6 % (repli mediane_couche réfuté) — reste la pièce 27 |
| Livraison | 0.6.35 publiée + erratum | 0.6.36 à publier (feu) |

## Reste, dans l'ordre où je le tiens
A. Mesures en cours (Manon, aujourd'hui) : 5e chaîne énergie 4 moteurs (vLLM et TRT-LLM manquent encore), rejeu invite4 du KL, diag-eval-nll 31B (2 min) ; Laure : KL TRT-LLM 4 invites (≤ 10 min) ; preuve du hang de capture (≤ 5 min).
B. Combler b=12 (Océane, une pièce par heure en régime mesuré) : 35 occupation des étroites (ptxas -v, balayage BLOCK_M/num_warps/num_stages au bit ; prédit ≥ 10 % sur `o`) → 39 fusion tête + échantillonnage (nsys d'abord, ≤ 3 %, fermée si < 2 %) → 30 cellule kv-fp8 (Laure, jugée par débit + KL). Fermés sans gain : split-K (33), PDL (34), Marlin isolé.
C. Qualité : 37 eval 31B (bras eval/serve/hf, P1/P2/P3) → PPL+KL 31B 4sur6 ; 27 experts froids 30B-VL (trace de routage par couche/expert/modalité, corpus photo/OCR/schéma, min 512 obs) ; 32 collecte AWQ à 512 obs ; 36 logprobs dans `acvram serve` (KL par API pour tous les moteurs).
D. Livraison 0.6.36 (feu utilisateur) : README 32 langues (TRT-LLM, courbe débit(b), énergie 4 moteurs, KL, replis non muets, garde de capture), .deb, release ; FEUILLE-DE-ROUTE déjà alignée sur l'erratum.
E. Reportés après B-D : C9 119B (porte : h_static(C abordable) ≥ 0,72 depuis l'histogramme de routage de la calibration, sinon S1 fermée sans code), bf16 30B témoin (20), modularisation model.py 4-5 (13, 15), parc/dpkg 0.6.34 rejeu (1, 16), THP/EPP (17), noyau AVX-512 (22), profil 4sur6 (23), prédicteur 4sur6 (26), énergie deux périmètres (28), tests levier 1 (31).
F. Régime : hebdo à 78 % (réserve) jusqu'à samedi 15 h ; Océane et Laure n'avancent B que par exceptions ≤ 15 min ; le gros de B se fera après la remise hebdo.

## Ce qui décide de la fin
Fin honnête possible dès que : (1) les 4 moteurs sont mesurés en J/jeton par la même chaîne ; (2) KL TRT-LLM complet ; (3) 35 et 39 tranchés (tenus ou réfutés) — alors l'écart b=12 est soit comblé, soit nommé avec sa cause (occupation des étroites) ; (4) eval 31B réparée ou déclarée hors périmètre ; (5) 0.6.36 publiée. C9, bf16 témoin, modularisation et parc sont des suites, pas des conditions.
