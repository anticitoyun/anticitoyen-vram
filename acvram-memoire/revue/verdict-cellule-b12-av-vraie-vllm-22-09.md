# Cellule b=12 officielle — vraie alternance acvram/vLLM (A/V/A/V/A/V) — TENU, 1 634,0 t/s — 22/09 (poste2)

* instrument : `scratchpad/poste4-b12-21-09/chaine-cellule-b12-vraie-vllm.sh` (harnais ea9baa0a, sidecar atomique + lecteur cassant), checkpoint `models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4-a16`, sous carte.sh (verrou interne par fenêtre)
* commit : main à jour (5d541f8a)
* régime : A1 V1 A2 V2 A3 V3, vLLM encadre chaque paire acvram, b=12, 1024 jetons/séq
* scellé (poste4, repris de `poste4-scelle-cellule-b12-22-09.md`) : médiane acvram 1 590-1 660 t/s, réfuté < 1 596 t/s ; écart horloge par paire ≤ 3 %
* mesuré : **acvram** A1=1 604,3 · A2=1 641,2 · A3=1 634,0 t/s → **médiane 1 634,0 t/s**. **vLLM** V1=1 895,6 · V2=1 773,7 · V3=1 782,0 t/s → médiane 1 782,0 t/s. Horloges médianes : A1/V1 2550/2640 (3,5 %), A2/V2 2542/2655 (4,4 %), A3/V3 2587/2647 (2,3 %). Aucune ligne `ÉCHEC LECTURE` (sidecar OK sur 3/3 fenêtres vLLM).
* verdict : **TENU** — médiane acvram 1 634,0 t/s dans la fourchette prédite (1 590-1 660), au-dessus du plancher de réfutation (1 596). **Tension nommée sans trancher** : l'écart d'horloge par paire dépasse 3 % sur 2/3 paires (3,5 % et 4,4 %) — bridage puissance actif sur les deux fenêtres vLLM (`"bridages": "puissance"`), pas un défaut du protocole d'alternance mais un signal que la carte n'atteint pas son horloge de croisière stable en 20 s de fenêtre sous ce régime. vLLM reste au-dessus d'acvram sur cette cellule (1 782,0 vs 1 634,0 t/s).
* durée : 07:37-07:47 (10 min, 3 chargements vLLM complets comme annoncé)

## Suite
Carte rendue. Prochaine pièce : nsys sur trtllm-serve b=12 (`poste1-ecart-trtllm-22-09.md` § 3), puis poste3 (courbe débit(b) trtllm ≤10min), puis PPL A/B 31B après correctif BOS ae519990.
