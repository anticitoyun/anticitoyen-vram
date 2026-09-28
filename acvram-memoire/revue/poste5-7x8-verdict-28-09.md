# 7x8 — OOM Marlin au chargement de Qwen3-Coder-30B (qkvo-i8c) depuis l'arbre : verdict (poste5, 28/09)

* instrument : `scratchpad/poste5-7x8-28-09/diag.py` (Engine de `acvram serve` en processus, relevés allocateur) + serve réel, `prise.sh`
* commit : B = poste5-7x8 64eaa917e ; A = témoin 54ab21f1c (origin/main) ; venv du dépôt, torch 2.14 cu130
* régime : Coder-30B srcQ4_K_M-nvfp4, `--max-model-len 32768 --speculative ngram` ; llama-server 8081 (5 606 Mio) seul hors verrou, avant = après
* scellé : `scratchpad/poste5-7x8-28-09/scelle.md` (écrit et poussé avant la prise)
* mesuré : A1 A2 OOM 288 Mio au 41e facteur_nvfp4 (alloué 24,00 Gio, cache libre 6,51 Gio, plus grand bloc libre 192 Mio, pilote 220 Mio) ; B1 B2 Engine construit, Marlin 42/48 (6 refus sous-normales), graphes actifs, pilote libre 6 520 Mio ; Bs HTTP 200, 16 jetons
* verdict : VRAI (2/2 A, 2/2 B, serve 200 : toutes les issues prédites)
* durée : prévue 12 min / tenue 425 s (carte.sh 23:36:04-23:43:09)

## Cause
1. `acvram/kernels/marlin_port/__init__.py:213` (main) `mx = ws[nz].max()` : sur CUDA, l'index par masque passe par
   nonzero → [nnz, 3] int64 = 24 o/élément. Pile Coder (128, 768, 128) : 301 989 888 o = 288 Mio, soit 6 × la copie
   flottante qu'elle indexe. Il est alloué dans `GraphRunner._eligible` (graphs.py:476 → moe.py:346 → moe.py:513 →
   preparer_pile:294), APRÈS le KV. Le cache (6,51 Gio) est morcelé en blocs ≤ 192 Mio. C'est une fragmentation plus
   un temporaire démesuré, pas un manque : alloué + 288 Mio < capacité.
2. Paquet contre arbre : `pyproject.toml:55` n'avait pas `*.hpp`. Le venv du paquet n'a donc pas
   `marlin_port/libtorch_stable/core/math.hpp` ni `core/scalar_type.hpp`, et la compilation Marlin échoue
   (« fatal error: libtorch_stable/core/math.hpp », cache marlin_port-2187da99ba88). La pile naturelle est gardée et
   rien n'est repacké : **le .deb 0.7.11 sert tous les MoE nvfp4 SANS Marlin** (repli nommé sur la ligne de régime
   seulement). C'est pour cela que « le paquet sert » et que l'arbre meurt. `test_le_deb_contient_les_sources_marlin_port`
   ne voyait rien : il lit le .deb (/usr/share/acvram), pas le venv installé depuis celui-ci.

## Correctif (poste5-7x8)
* `facteur_nvfp4` : `masked_fill_(~(ws > 0), 0)` puis `amax`, même facteur au bit (NaN et négatifs écartés comme par
  le masque), aucun indice.
* `pyproject.toml` : `kernels/**/*.hpp` en package-data.
* `tests/test_marlin_chargement_7x8.py` : 10 tests, dont 8 d'égalité avec l'écriture d'avant (zéros, négatifs, 448,
  sous-normales, tout nul, NaN, 2D). Deux cassent sans le correctif (vérifié) : le test d'index par masque et celui de
  package-data qui couvre toutes les sources des noyaux.

## Conséquences et restes
* Le prochain .deb compile Marlin au premier lancement (~25 s, une fois par version). Les MoE du paquet changent alors
  de régime (Marlin au lieu de la pile naturelle) : à signaler dans le CHANGELOG. Le test des menus sur ce .deb doit
  être rejoué.
* Non traité : le Plan ne réserve pas le transitoire du repack Marlin (moe.py:450, « rien de plus à réserver »). Avec
  le correctif, le pic tient (6,5 Gio libres en fin de chargement), mais un modèle plus serré pourrait retomber ailleurs.
