# dut — repli eager du premier serveur à froid (poste1, 27/09)

* instrument : `scratchpad/poste1-dut-27-09/prise-dut.sh` (charge de la paire 1 de la 284 c : `client-carte-274.py`, b=1 et b=12)
* commit : T = 9b641ebc6 (acvram/ = v0.7.7), B = c91e0fdf5 (v0.7.7 + `_garde_capture` de t5e, 3acfc6006) ; arbre de prise : poste1-t5e
* régime : carte 0, -lgc 2700, Qwen3.8-27B-unsloth-mixte-i8c, contexte 4 096, max-batch 16, sans spéculation ; chaque bras : worktree
  jetable neuf + TRITON_CACHE_DIR vide (froid garanti) ; ordre T B B T
* scellé (écrit avant la prise) : voir ci-dessous
* mesuré : À VENIR
* verdict : À VENIR
* durée : prévu ≤ 25 min

## Lecture à sec (avant la prise)
Journal de A1 (`~/.cache/acvram/dumps-poste1/p284c-rejeu-serve-1.log`) : avant capture « libre 4,15 Gio, cache non rendu 0,78 » ;
EN SERVICE, deux « repli eager — mémoire libre 998/996 Mio < 1024 : capture refusée » (nouvelles clés de lot pendant b=12).
B2 (même photo au départ, à chaud) : aucun repli. La garde ne regardait que la mémoire libre du pilote ; le préfill de b=12 laisse
ses activations dans le cache de l'allocateur — le mécanisme que t5e a corrigé. À froid, l'autotune Triton y ajoute ses tampons.

## Scellé
* Prédiction : T ≥ 1 repli eager (« capture refusée ») sur au moins un de ses deux bras ; B 0 repli sur ses deux bras.
* Issues : T 0 et B 0 → froid non reproduit, NON CONCLU (le correctif n'est ni prouvé ni réfuté pour dut) ; T ≥ 1 et B 0 →
  même cause, fermée par la 0.7.8 ; B ≥ 1 → FAUX : autre cause (lire `replis_eager_raisons` et la photo après l'échec).
* Alarme : un PID hors verrou au début d'un bras → REFUS (le script s'arrête).
