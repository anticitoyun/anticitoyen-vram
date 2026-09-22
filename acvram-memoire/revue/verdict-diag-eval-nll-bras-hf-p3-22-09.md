# diag-eval-nll, bras hf sur Gemma 4 31B (correctif VRAM tenu) — P3 CONFIRMÉ — 22/09 (poste2)

* instrument : `outils/carte.sh env HF_PYTHON=... MAXMEM=26GiB,80GiB .venv/bin/python outils/gpu/mesure/diag-eval-nll.py gemma-4-31B-it-nvfp4-vision --jetons 300 --source .../gemma-4-31B-it-bf16`, sha `9a1c06d9` (libération VRAM avant hf, tenue : « modèle acvram libéré, 30,6 Gio libres avant le bras hf »)
* commit : main à jour
* mesuré : `PPL eval 227232,509 · serve 227215,62 · hf 108039,472` — **première divergence eval/hf : position 1** (dès le premier jeton généré), **`|Δ| eval−hf médian = 2,0254` nats** (contre `|Δ| max eval/serve = 0,0062`, quasi nul).
* verdict : **P3 CONFIRMÉ** — eval ≈ serve (chemins acvram internes cohérents entre eux) mais **eval/serve ≠ hf dès le début**, écart médian massif (2,03 nats). C'est un défaut du moteur acvram sur Gemma 4 (forward divergent de la référence transformers dès la première position), pas un bug d'instrument, pas un problème de fenêtrage. **Point non résolu, nommé sans trancher** : la PPL hf elle-même (108 039) est aussi très au-dessus du plausible (8-20) — soit le texte connu utilisé par l'instrument n'est pas un bon texte de test (trop rare/inhabituel pour ce corpus), soit la construction des `ids` partagée par les trois bras (BOS, gabarit) porte un défaut commun en amont des trois calculs — à distinguer avant de conclure définitivement que le défaut est SEULEMENT dans le forward acvram.
* durée : ~4 min de carte (correctif de libération VRAM tenu, plus d'OOM)

## Suite
P3 confirmé : le forward acvram diverge de hf dès la position 1 sur Gemma 4 — défaut moteur à nommer par couche/position (poste1). Séparément : vérifier que le texte de test et la construction des ids ne biaisent pas la PPL hf elle-même (108 039, déjà anormale) avant d'attribuer tout l'écart au moteur acvram seul.
