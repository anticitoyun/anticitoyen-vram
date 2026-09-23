# diag-eval-nll, épreuve de la référence HF (correctif) — RÉFÉRENCE INVALIDE, P3 retiré — 22/09 (Manon)

* instrument : `outils/carte.sh env HF_PYTHON=... MAXMEM=26GiB,80GiB .venv/bin/python outils/gpu/mesure/diag-eval-nll.py gemma-4-31B-it-nvfp4-vision --jetons 300 --source .../gemma-4-31B-it-bf16`, sha `1fe613a6` (épreuve de la référence ajoutée)
* mesuré : **`RÉFÉRENCE INVALIDE (ids ou chargement)`** — PPL 189 203,7 sur SON PROPRE encodage (hf jugé contre lui-même), suite dégénérée : **1 seul jeton distinct sur 20** — la suite produite est littéralement `' own own own own own own own own own own own own own own own own own own own own'`. Message de l'instrument : « Le moteur n'est PAS jugé : corriger la référence (gabarit, BOS, tokeniseur, offload) avant toute conclusion. »
* verdict : **RÉFÉRENCE INVALIDE — je retire le « P3 CONFIRMÉ » de mon verdict précédent** (`verdict-diag-eval-nll-bras-hf-p3-22-09.md`). La divergence eval/hf mesurée alors (Δ médian 2,03 nats) ne peut PAS être attribuée au moteur acvram : la référence hf elle-même produit une sortie dégénérée sur ce texte/gabarit, confirmant le doute que j'avais nommé (« la PPL hf elle-même est aussi anormale »). Ni P1 ni P2 ni P3 ne sont tranchés — l'instrument entier attend une référence saine avant de pouvoir juger Gemma 4.
* durée : ~4 min de carte

## Suite
Corriger le montage de la référence (gabarit de conversation, BOS, ou offload HF) avant tout nouveau rejeu sur Gemma 4 — à Océane. Aucune conclusion sur le moteur acvram n'est valide tant que cette étape échoue.
