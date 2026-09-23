# Pièce 108 — un `.qui` absent ne prouve pas que la carte est libre (poste3, 23/09)

`outils/surveillance-groupe.sh` et `openwebui:47` lisaient l'existence du `.qui` seule
pour dire « libre ». Or REGLES § 2 : « le verrou `outils/carte.sh` est la seule source de
vérité » — le `.qui` n'est qu'une annonce, et le ticket jxm a montré qu'il peut manquer
(gardien de service, trou entre la mort réelle et la mise à jour). Sans interroger flock
lui-même, les deux outils pouvaient dire « libre » pendant qu'une mesure ou un service
tenait réellement la carte.

## `outils/surveillance-groupe.sh` (dans le dépôt)

Si `.qui` absent : `flock -n <verrou> true` avant de conclure « libre ». Échec du flock
(non bloquant) → `TENU sans .qui`. Verrou paramétrable par `ACVRAM_VERROU` (n'existait pas
avant, ajouté pour isoler les tests des vrais `/tmp/acvram-carte-*.lock`).
Test : `tests/test_carte_service.py` … non, `tests/test_surveillance_groupe_carte.py` (3
cas : libre sans rien, tenu avec `.qui`, **tenu sans `.qui`** — le cassant). `| head -2`
coupe la sortie avant le `git fetch` (réseau) du reste du script. Cassé une fois (garde
retirée, « libre » revenu à tort pour la bonne raison), restauré.

## `~/.local/bin/openwebui` (HORS DÉPÔT — à signaler)

Ce lanceur n'est PAS suivi par git : ni dans `anticitoyen-vram`, ni symlink, ni copie
d'un fichier du dépôt (`find` sur le dépôt : aucune occurrence). Édité directement sur ce
poste, donc **aucun commit, aucune revue possible ici** — seule cette note en garde la
trace. Même correctif que ci-dessus : `VERROU_LOCK=/tmp/acvram-carte-0.lock` séparé de
`VERROU=$VERROU_LOCK.qui`, et `cmd_comfyui` refuse (rc 65, même message que le refus
existant) si `flock -n "$VERROU_LOCK" true` échoue, même sans `.qui`.
Vérifié à sec (copie temporaire du script, `VERROU_LOCK` et `COMFY_SH` substitués vers
un dossier jetable, jamais le vrai verrou) : trois cas — rien ne tient (tente de
lancer, comportement inchangé), `.qui` présent (REFUS 65, inchangé), **flock tenu sans
`.qui`** (REFUS 65, nouveau). `bash -n` : syntaxe correcte.
Pas de test automatisé en dépôt pour ce fichier : y écrire un test aurait mis un chemin
de machine personnel (`/home/<nom>/.local/bin/openwebui`) dans la suite versionnée, ce
que `test_paquet_parc.py` interdit explicitement pour tout le reste du dépôt. Si ce
lanceur doit rester couvert par une garde durable, il faudrait le faire vivre DANS le
dépôt (comme `parc/bin/acvram-serveur`) — pas décidé ici, remonté à chef.
