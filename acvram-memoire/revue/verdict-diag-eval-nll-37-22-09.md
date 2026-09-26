# diag-eval-nll (pièce 37), bras eval/serve sans hf — eval≈serve, tous deux FAUX — 22/09 (poste2)

* instrument : `outils/carte.sh .venv/bin/python outils/gpu/mesure/diag-eval-nll.py gemma-4-31B-it-nvfp4-vision --jetons 300`, sans `--source` (2 bras, ≤2 min)
* commit : main à jour
* contrôle qui rend faux : PPL(ctx≥32) bras eval ≤ 30 sur ce texte connu — **FAUX** (227 232,5, très loin de 30)
* mesuré : `PPL eval 227232,509 · serve 227215,62` — quasi identiques (0,008 % d'écart) ; **première divergence eval/serve : aucune** (`|Δ| max 0,0062` nat sur les 300 positions) — les deux chemins produisent des logits quasi identiques.
* verdict : **eval ≈ serve, contrôle FAUX sur les deux** — ni P1 (exigerait PPL 8-20) ni P2 (exigerait une divergence eval/serve, absente ici). Compatible avec P3 (eval≈serve≠hf, défaut moteur commun aux deux chemins) mais **non tranché** : le bras hf n'a pas été lancé (2 min demandées, hf ≤10 min de plus, hors budget de cette prise). Le fait que serve (le chemin du service réel, déjà validé par KL/scellé E à 0,87 sur un autre corpus) donne la MÊME PPL astronomique que eval affaiblit l'hypothèse P1 (bug de fenêtrage propre à `perplexity`) : si le service lui-même produit ces logits sur CE texte précis, le défaut n'est pas spécifique à l'instrument d'évaluation.
* durée : ~2 min de carte

## Suite
Bras hf (`--source <HF_DIR bf16>`, ≤10 min de plus) nécessaire pour trancher P1/P3 définitivement — à budgéter par le groupe. Carte rendue à poste3 (débit kv-fp8).
