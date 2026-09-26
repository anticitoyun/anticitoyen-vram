instrument : lecture de code (grep/lecture de tous les `outils/gpu/mesure/*.py` important `acvram`/`energie`) + `tests/test_arbre_outils_mesure.py` (nouveau), rejoué sous `outils/carte.sh`
commit : origin/main (be837ca1 + 208/210), branche poste2-p211
régime : mesure, plein — lecture à sec, seule la suite pytest a pris la carte (2 prises courtes, < 1 min chacune)
scellé : aucun — correctif ponctuel demandé par chef (constat poste5), même défaut que la 168
mesuré : 13 fichiers défaillants trouvés et corrigés sur 22 candidats dans `outils/gpu/mesure/` ; test dédié 2/2 vert
verdict : TENU — les 13 fichiers importent maintenant l'arbre de leur propre worktree, jamais celui de l'installation editable ; garde `a86fa1dd` non contournée
durée : ~45 min (lecture + correctifs + tests), 2 prises carte de moins d'une minute chacune

## Le défaut (constat poste5, même mécanisme que la 168)

`outils/gpu/mesure/energie.py:_regime_noyaux()` fait `import acvram; return acvram.regime_ligne()`, sans jamais
toucher `sys.path` lui-même — c'est à l'appelant (le script de banc/TTFT) de poser la racine AVANT cet import. La
168 (24/09) avait corrigé `scratchpad/banc-llamacpp-16-09.py` pour dériver sa racine de `__file__` (deux `dirname()`
depuis `scratchpad/`) ; le même défaut traînait, non corrigé, dans une partie de `outils/gpu/mesure/` : script qui
n'insère que son propre dossier (`ttft-service-p145.py`, `banc-4moteurs.py`), racine calculée avec un `dirname()`
en moins qu'il n'en faut (`banc-marlin-dense-e1.py`), racine dérivée du `cwd` plutôt que de `__file__`
(`banc-moe-w13.py`, `kl-lot-mele-p100(.py|-confirmation.py)`), ou aucun `sys.path` du tout, un `import acvram` nu qui
ne tient que par ce qui traîne déjà sur `sys.path` (`mesure-gemv-nvfp4.py`, `passage-tranche.py`, `qualite-tranche.py`,
`sortie-tranche.py`, `temoin-tampon.py`, `tranche-au-lot.py`, `vagues-ou-blocs.py`).

## Ce qui a été corrigé (13 fichiers, tous dans `outils/gpu/mesure/`)

Tous reçoivent la même racine, dérivée de `__file__` (quatre `dirname()`/trois `..` depuis
`outils/gpu/mesure/<fichier>.py`), posée AVANT le premier `from acvram`/`from energie import` :

* `banc-4moteurs.py`, `ttft-service-p145.py` — n'inséraient que leur propre dossier ; racine ajoutée en plus.
* `banc-marlin-dense-e1.py` — `R` n'atteignait que `outils` (2 `..`) au lieu de la racine (3 `..`) : corrigé.
* `banc-moe-w13.py`, `kl-lot-mele-p100.py`, `kl-lot-mele-p100-confirmation.py` — racine dérivée du `cwd`
  (`os.getcwd()`, `"."`, `"tests"` en dur) plutôt que de `__file__` : corrigés.
* `mesure-gemv-nvfp4.py`, `passage-tranche.py`, `qualite-tranche.py`, `sortie-tranche.py`, `temoin-tampon.py`,
  `tranche-au-lot.py`, `vagues-ou-blocs.py` — aucun `sys.path` du tout, `import acvram` nu : racine ajoutée.

## Ce qui n'a PAS été touché (déjà correct, vérifié à la lecture)

`banc-attn-decoupe.py`, `banc-etroites-latence.py`, `banc-gemv-creneaux.py`, `banc-horloge-decodage.py`,
`banc-routage-b1-p68.py`, `banc-sampler.py`, `energie-familles-p99.py`, `c9-m-hote.py`, `frontiere-pas.py`
(celui-ci utilise `git rev-parse --show-toplevel` depuis son propre dossier — robuste, différent mais correct) —
tous dérivent déjà leur racine de `__file__` avec le bon nombre de niveaux. `non-regression-parc.py` utilise
`sys.path.append` (pas `insert`) avec un commentaire explicite (« APRÈS PYTHONPATH : le paquet mesuré passe avant
l'arbre ») — DESIGN INTENTIONNEL (il compare l'arbre au paquet installé), laissé intact.

## Test de non-régression (`tests/test_arbre_outils_mesure.py`)

Scanne tous les `.py` de `outils/gpu/mesure/` qui importent `acvram`/`energie`, rejoue le code de CHAQUE fichier
qui précède son premier import concerné (avec `__file__` pointé sur le fichier réel, `sys.path`/`argv` réels
sauvegardés-restaurés), et vérifie qu'au moins un chemin inséré contient `acvram/__init__.py`. Capable de rendre
faux : `test_le_controle_peut_rendre_faux` fabrique un faux outil qui n'insère que son propre dossier (le défaut
d'origine) et vérifie que le contrôle le détecte. 2/2 verts (`/tmp/p211-tests2.log`), rejoué isolément sous carte.

## Observation hors périmètre (non traitée ici)

Le run combiné avec `tests/test_sonde_regime_arbre.py` a fait TIMEOUT (60 s) sur
`test_sonde_ne_leve_jamais_sans_arbre_acvram` (`/tmp/p211-tests.log`) — l'arbre factice de ce test n'a ni `.git` ni
`acvram/`, donc la garde n'a rien à comparer et l'import retombe sur l'installation editable réelle, qui a
déclenché une compilation CUDA (« verrou de compilation orphelin retiré », `[acvram] ...`) au lieu de répondre en
quelques secondes. **Ce fichier n'a pas été touché par la 211** (pas dans le périmètre `outils/gpu/mesure/`,
c'est un test) — probablement un aléa d'environnement (compilation JIT sous forte contention de carte ce tour),
signalé mais non creusé, hors périmètre de cette pièce.

## Périmètre non couvert (à la demande de chef, `outils/gpu/mesure/…`)

`outils/` à la racine (hors `gpu/mesure/`) contient d'autres scripts avec le même risque (`banc-marlin-p1-18-09.py`,
`banc_1aj_*`, `attn-isole.py`, etc. — une soixantaine de candidats au grep initial), non revus ici : chef a nommé
`outils/gpu/mesure/…`, `ttft-service-p145` et « banc-chat » comme le périmètre de cette pièce. À reprendre en pièce
séparée si le même défaut y est confirmé.
