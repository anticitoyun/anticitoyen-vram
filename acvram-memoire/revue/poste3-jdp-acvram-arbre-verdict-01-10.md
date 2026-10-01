# bd jdp — ACVRAM_ARBRE exportée partout où un script de mesure lance python (poste3, 01/10)

instrument : pytest ciblé, processeur seul, `CUDA_VISIBLE_DEVICES=""`. commit : `HEAD` de
`poste3-jdp`. régime : aucun (pas de carte — tous les tests tournent sous un `ACVRAM_VERROU` de
test, jamais le verrou réel). scellé : aucun. mesuré : oui (cassant vérifié par `git stash` du
correctif `carte.sh`). verdict : CORRIGÉ, contrôle cassant ajouté. durée : ~35 min.

Avant de lancer un test : `/tmp/acvram-carte-0.lock.qui` montrait la mesure 275 de poste2 tenue
tout du long — chaque test de cette pièce utilise son propre `ACVRAM_VERROU` (tmp_path) et
`CUDA_VISIBLE_DEVICES=""`, jamais le verrou réel ; aucun croisement avec sa prise.

## Précision de chef (corrige mon hypothèse de départ)

Il existe déjà une garde (`acvram/__init__.py:45-80`, `_garde_arbre`, pièce 276 f) : elle refuse
l'import si le cwd est dans un AUTRE arbre acvram que celui réellement importé, et
`ACVRAM_ARBRE=<arbre>` impose l'arbre voulu. **Le trou n'est pas dans cette garde** — un cwd HORS
de tout arbre acvram (scratchpad, `/tmp`, le cas réel des scripts de `outils/gpu/mesure` qui ne
`cd` jamais dans le dépôt), sans `ACVRAM_ARBRE` posée, passe en silence : rien à ajouter côté
garde, seulement la poser partout où elle manque.

## Correctif

`outils/carte.sh` : `DEPOT` calculé depuis `${BASH_SOURCE[0]}`, `ACVRAM_ARBRE="${ACVRAM_ARBRE:-$DEPOT}"`
ajoutée aux six points où le script lance la commande (`env CUDA_VISIBLE_DEVICES=... "$@"`),
jamais écrasée si déjà posée (ABBA à deux arbres, chaque bras fixe la sienne avant d'appeler
carte.sh). Onze scripts de `outils/gpu/mesure/*.sh` qui lancent python l'exportent désormais,
chacun avec SA propre racine déjà calculée (`$DEPOT`/`$ARBRE`/`$RACINE`/`$S` selon le fichier) :
`ppl-balayage-11e.sh` (défense en profondeur, déjà protégée par son propre contrôle
`acvram.__file__`), `prise-reserve-plafonnee-kv31b.sh`, `prise-s1-morceaux-kv31b.sh`,
`ncu-etroites-p48.sh`, `nsys-kda-p81.sh`, `protocole-49us.sh`, `gabarit-chaine.sh` (sourcé — pose
la sienne sauf si la chaîne appelante l'a déjà fixée), `banc-horloge-decodage.sh`,
`banc-horloge-memoire-decodage.sh`, `duel-moteurs.sh`, `nsys-tete-sampler.sh` (la lisait déjà
mais ne la réexportait jamais pour ses enfants). `serveur-bras.sh` EXCLU à dessein : harnais
générique sourcé qui exec la commande du CALLEUR, jamais la sienne — poser `ACVRAM_ARBRE` ici
masquerait l'arbre réel du script appelant.

## Contrôle cassant

`tests/test_mesure_exporte_acvram_arbre.py` : (1) `test_carte_sh_refuse_un_acvram_dune_autre_racine`
rejoue le VRAI piège à travers `outils/carte.sh` — cwd hors de tout arbre (`tmp_path`),
interpréteur du venv PRINCIPAL réel de ce poste (import `acvram` différent du dépôt mesuré), exige
le refus ; (2) `test_le_controle_peut_rendre_faux` — le même import, SANS passer par carte.sh, doit
passer en silence (prouve que le contrôle teste bien l'absence du correctif, pas une coïncidence
d'environnement) ; (3) `test_chaque_script_de_mesure_qui_lance_python_exporte_acvram_arbre` — balaie
tous les `.sh` de `outils/gpu/mesure/` qui lancent python (hors commentaires) et exige la présence
de `ACVRAM_ARBRE=`.

**Preuve cassante** : `git stash` du correctif `carte.sh` seul — `test_carte_sh_refuse_un_acvram_dune_autre_racine`
échoue EXACTEMENT comme prédit (« import accepté en silence : 'IMPORT OK' ») ; correctif restauré,
16/16 verts (ce fichier + `test_garde_arbre.py`, `test_garde_arbre_276f.py`,
`test_arbre_outils_mesure.py`, `test_carte_253_qui_absent_sans_bruit.py`, `test_carte_ticket_fifo.py`).

## Correctif v2 (chef, après relecture) : l'arbre du cwd prime sur $DEPOT

Fixer `ACVRAM_ARBRE=$DEPOT` (l'arbre qui PORTE le script) cassait un usage réel : des scripts
appellent le `carte.sh` d'un AUTRE arbre depuis leur propre worktree
(`scratchpad/poste1-p221-25-09/chaine.sh:5`, `~/.config/acvram/chef/fenetre-g2c.sh`) — avec
`$DEPOT` fixe, le cwd de l'appelant (son propre worktree, un arbre acvram VALIDE) ne correspond
plus à l'arbre imposé, et l'import est refusé À TORT (refus bruyant, jamais un faux positif
silencieux, mais l'usage légitime casse).

`outils/arbre-defaut.sh` (nouveau, sourcé par `carte.sh` et les 11 scripts) : fonction
`_acvram_arbre_defaut <repli>` qui reproduit la remontée de `_garde_arbre` (même règle :
`acvram/__init__.py` + `.git`) — l'arbre qui CONTIENT le cwd prime, `<repli>` (l'arbre qui porte
le script) seulement si le cwd n'est dans aucun arbre acvram. `carte.sh` calcule
`ACVRAM_ARBRE` une seule fois (évite de remonter le cwd à chaque site de lancement).

**Tests ajoutés** : `test_carte_sh_dun_arbre_importe_lacvram_du_cwd_dans_un_autre_arbre` — le
`carte.sh` DE CE DÉPÔT, cwd dans l'arbre PRINCIPAL réel, doit importer le PRINCIPAL (pas de
refus). **Cassant vérifié** : `ACVRAM_ARBRE="${ACVRAM_ARBRE:-$DEPOT}"` (v1) remis temporairement
dans `carte.sh` → ce test échoue EXACTEMENT comme prédit (« acvram importé depuis
.../anticitoyen-vram alors que ACVRAM_ARBRE demande .../poste3-jdp ») ; v2 restauré, 17/17 verts
(ce fichier + `test_garde_arbre.py`, `test_garde_arbre_276f.py`, `test_arbre_outils_mesure.py`,
`test_carte_253_qui_absent_sans_bruit.py`, `test_carte_ticket_fifo.py`). Le test `/tmp` d'origine
(piège v1) reste vérifié rouge sans aucun correctif, inchangé.

verdict: acvram-memoire/revue/poste3-jdp-acvram-arbre-verdict-01-10.md — cause (précisée par chef) : la garde d'import 276 f existe déjà mais un cwd hors de tout arbre sans ACVRAM_ARBRE passe en silence ; corrigé en exportant ACVRAM_ARBRE dans carte.sh (6 points) et 11 scripts de outils/gpu/mesure/*.sh (serveur-bras.sh exclu, c'est un harnais générique) ; test cassant vérifié par git stash ; 16/16 verts sous CUDA_VISIBLE_DEVICES=""
