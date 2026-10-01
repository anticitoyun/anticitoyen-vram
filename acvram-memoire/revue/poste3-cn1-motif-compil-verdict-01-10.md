# bd cn1 — carte-libre.sh : pgrep de compilation non isolé, cause trouvée et corrigée (poste3, 01/10)

instrument : pytest ciblé, processeur seul, `CUDA_VISIBLE_DEVICES=""`. commit : `HEAD` de
`poste3-cn1`. régime : aucun (pas de carte, pas de prise). scellé : aucun. mesuré : oui (cassant
vérifié par `git stash` du correctif). verdict : CAUSE CONFIRMÉE, corrigée, contrôle de fuite
ajouté. durée : ~40 min.

## Cause

`outils/carte-libre.sh` a TROIS critères ; le 3e (détection de compilation, `nvcc`/`ninja`/…,
code de sortie 2) est au bout du script, après le pgrep `MOTIF` (déjà isolable depuis g2c/73c).
Il était câblé en DUR (`pgrep -af '(^|/)(nvcc|cicc|ptxas|cudafe\+\+|ninja)( |$)'`, ligne 197),
**sans aucune variable d'isolation**, contrairement à `MOTIF`. `test_carte_libre_service_declare_poste2.py`
neutralisait déjà `MOTIF` (commentaire « non hermétique, même piège que la pièce 73c ») mais
pas ce pgrep-là ; `test_carte_libre_pause_sans_allocation_g2c.py` ne neutralisait rien du tout
pour lui. Pendant qu'un vrai poste compile des noyaux (nvcc/ninja, typique d'un chargement de
modèle — ce que chef a vu sur le vrai dépôt), ce pgrep attrape le VRAI processus, `carte-libre.sh`
rend 2 (« carte libre, mais N processus de compilation en cours ») au lieu de 0 pour
`test_pause_sans_verrou_ni_gpu_nest_plus_un_refus` et `test_service_declare_dans_qui_carte_nest_plus_intrus`
— les deux seuls tests qui REJOIGNENT la fin du script (les tests `rc=1` sortent avant
d'atteindre ce 3e critère, donc jamais touchés).

## Correctif

`MOTIF_COMPIL=${MOTIF_COMPIL:-'(^|/)(nvcc|cicc|ptxas|cudafe\+\+|ninja)( |$)'}` (même défaut,
paramétrable comme `MOTIF`) ; le pgrep de la ligne 197 le lit. Les deux fichiers de test le
neutralisent (`MOTIF_COMPIL=__jamais_aucune_correspondance__`) dans l'env des cas qui
atteignent la fin du script.

## Contrôle de fuite (demandé par chef)

Un leurre nommé **littéralement `ninja`** (motif par défaut de `MOTIF_COMPIL`) tourne PENDANT
les deux tests corrigés — si quelqu'un retire `MOTIF_COMPIL` de leur env plus tard, ce leurre
les refait échouer immédiatement. `test_isolation_compil_casse_sans_motif_compil` (un par
fichier) rejoue le MÊME leurre SANS la variable et exige `rc=2` — preuve directe que le leurre
matche le motif réel, pas une coïncidence de nommage.

## Preuve cassante

`git stash` du correctif (`outils/carte-libre.sh` seul, tests déjà modifiés avec les leurres) :
2 échecs, EXACTEMENT le symptôme rapporté — `assert 2 == 0`, stderr « carte 0 libre, mais 1
processus de compilation en cours ». Correctif restauré (`git stash pop`) : 8/8 verts.

## Tests

`tests/test_carte_libre_pause_sans_allocation_g2c.py` (3, +1 neuf), `test_carte_libre_service_declare_poste2.py`
(3, +1 neuf) : 8/8. Suite élargie (`-k "carte_libre or carte_ticket or verrou"`, 58 cas) sous
`CUDA_VISIBLE_DEVICES=""` : 58 passed, 1 skipped — aucune régression.

verdict: acvram-memoire/revue/poste3-cn1-motif-compil-verdict-01-10.md — cause : le pgrep de détection de compilation de carte-libre.sh (3e critère, rc=2) n'était isolé par aucune variable ; MOTIF_COMPIL ajoutée et neutralisée dans les deux tests concernés ; contrôle de fuite (leurre « ninja » tournant pendant les tests + test dédié sans l'isolation, rc=2 exigé) ; cassant vérifié par git stash ; 58/58 sous CUDA_VISIBLE_DEVICES=""
