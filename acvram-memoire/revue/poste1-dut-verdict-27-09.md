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

## Prise 1 (16 h 56) — interrompue par mon script au premier bras
T à froid : chauffe 38,4 s, « 1 607 Mio libres après la passe » (à chaud : 4 969), photo avant capture « libre 1,57 Gio,
réservé 26,41, alloué 25,79, cache non rendu 0,62 », deux « capture refusée » (920/916 Mio), puis « contexte non tenu » (3 072) :
serveur MORT. Le script s'arrêtait sur « SERVEUR MORT » au lieu de noter le bras et de continuer → corrigé, prise rejouée.
Lecture : à froid, ~3,3 Gio manquent HORS de l'allocateur PyTorch (réservé identique à chaud) — modules Triton compilés par
l'autotune (code + mémoire locale réservée). La garde de t5e ne rend que 0,62 Gio : prédiction inchangée, mais B peut
échouer pour cette autre part (issue « FAUX » du scellé).

## Prise 2 (17 h 10-17 h 24, T B B T à froid) — scellé FAUX dans sa forme « B 0 »
* T ×2 : 2 « capture refusée », « non tenu » (3 072), serveur mort. B ×2 : 0 capture refusée (la garde de t5e rend les 0,62 Gio),
  mais « non tenu » quand même, serveur mort. Photo identique aux 4 bras : libre 1,57, réservé 26,41, cache non rendu 0,62.
* Hors allocateur : 31,36 − 26,41 − 1,57 = 3,38 Gio à froid contre 0,64 à chaud (serve-1 de la 284 c). La garde de t5e ne
  ferme donc PAS dut : la cause principale est ailleurs.
## Sonde (hypothèse 1, scellée avant) : mémoire locale gardée par le pilote
* Prédiction : à froid, `cuCtxSetLimit(pile, même valeur)` fait tomber « hors allocateur » de ~3,4 à ≤ 1 Gio. FAUX si < 0,5 Gio
  rendus → hypothèse 2 (modules chargés : code des configurations d'autotune).
* Sonde 1 (17 h 28) : pile **12 256 o/fil** (défaut 1 024) ; reposée à la même valeur, rien rendu (3,38 → 3,38) — attendu après
  coup : le pilote ne réduit que si la limite baisse. 12 256 × 170 SM × 1 536 fils ≈ 3,2 Go : l'ordre de grandeur de l'écart.
* Sonde 2 (scellée avant) : limite ramenée à 1 024 avant la première capture. VRAI si ≥ 2 Gio rendus et serveur vivant
  (b=1 et b=12 servis sans repli) ; FAUX si < 0,5 Gio rendus.
* Sonde 2 (17 h 32) : **VRAI** — pile 12 256 → 1 024 o/fil, hors allocateur 3,38 → 0,64 Gio (l'état à chaud), libre 1,57 →
  4,30 Gio ; serveur VIVANT, b=1 47,6 t/s, b=12 206,5 t/s, repli_eager 0, refus 0.
## Correctif (graphs.py `_rendre_pile`, défaut ; témoin ACVRAM_PILE_RENDUE=0)
Avant la première capture, et sous le seuil de la garde (autotune d'une forme nouvelle en service) : limite de pile ramenée à
ACVRAM_PILE_OCTETS (1 024). Tests à sec `tests/test_pile_rendue_dut.py` : 3 rouges sur v0.7.7, verts ; voisins 150 passés.
## Scellé de la preuve à froid (défaut, sans sonde) — écrit avant
* T (9b641ebc6) contre B (ce commit), T B B T, froid garanti. Prédiction : T meurt 2/2 ; B vit 2/2, repli_eager 0, ligne
  « pile locale rendue » présente. FAUX si un B meurt ou replie.
