# Crochets versionnés

Ces fichiers **s'exécutent depuis `~/.claude/hooks/`**, hors du dépôt. La copie
ici n'est pas active : elle existe pour que le travail survive à la machine.

    installer :  cp outils/crochets/13_rtk_comptage_guard.sh ~/.claude/hooks/

`13_rtk_comptage_guard.sh` refuse `rtk git <listage> | <compteur>` sans `-n`
explicite. Le 10/09, `rtk git log` valait `git log --no-merges -n 50` : il
plafonnait à 50 **et** retirait les fusions, donc il enlevait aussi des lignes
du milieu — les 50 rendues couvraient les rangs 1 à 60, sans que rien signale
les trous. Trois sessions ont publié des comptes faux en croyant se vérifier.

Motif étendu le 10/09 au soir à `shortlog|branch|tag|ls-files|rev-list --all` :
le nom du fichier promet « comptage rtk » alors que le motif ne visait que
`git log`. Éprouvé sur sept cas.

**Pourquoi cette copie existe** : le crochet vit hors du dépôt, donc son
extension serait morte avec la machine, et le travail aurait été refait sans
que personne sache qu'il l'avait déjà été.
