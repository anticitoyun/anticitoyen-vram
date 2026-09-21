# Crochets — copie de référence

**Ces fichiers s'exécutent depuis `~/.claude/hooks/`, hors du dépôt.** Ils n'y
survivent ni à un redémarrage de machine ni à un changement de poste.

**Signalé le 10/09/2026** : le motif du crochet de comptage venait d'être étendu
(`shortlog|branch|tag|ls-files`) et corrigé (il bloquait l'écriture d'un fichier
qui le *cite*, pas seulement la commande qui l'exécute). **Ce travail aurait
disparu avec la machine, et personne n'aurait su qu'il avait été fait** — il
aurait été refait à l'identique, ou pas du tout.

## Installer

```bash
cp outils/crochets/*.sh ~/.claude/hooks/ && chmod +x ~/.claude/hooks/*.sh
```

Le crochet doit ensuite être déclaré dans `~/.claude/settings.json`.

## Vérifier qu'une copie n'a pas dérivé

```bash
diff outils/crochets/13_rtk_comptage_guard.sh ~/.claude/hooks/13_rtk_comptage_guard.sh
```

**Une différence n'est pas une erreur — c'est une question :** laquelle des deux
a été corrigée en dernier, et pourquoi personne ne l'a reportée.
