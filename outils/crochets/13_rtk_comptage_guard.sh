#!/usr/bin/env bash
# PreToolUse: Bash — empêche de COMPTER une sortie rtk.
#
# Mesuré le 10/09/2026 sur le dépôt anticitoyen-vram :
#
#   rtk git log --format='%H' | wc -l          ->      50   (rtk injecte -n 50)
#   rtk git log --format='%H' -n 200 | wc -l   ->     200   (il obeit si on demande)
#   git log --format='%H' | wc -l              ->     637
#
# `rtk find` est un cas DIFFERENT et n'est pas bloque ici : il ne perd rien, il
# reformate — il groupe par repertoire et annonce le total exact en premiere
# ligne. Compter ses lignes ne compte pas ses elements ; il faut lire son total.
# Confondre les deux ferait se mefier de l'outil la ou il est correct, et une
# regle qui interdit a tort finit par etre ignoree la ou elle avait raison.
#
# Trois sessions ont produit 50, 41 et 130 pour le même comptage en croyant se
# corriger l'une l'autre. Une troncature ment TOUJOURS dans le sens rassurant :
# elle rend moins, jamais plus, et sa sortie reste bien formée — rien ne la
# signale. Un comptage se fait sur la source, pas sur un résumé.
#
# Ce garde-fou existe parce qu'un contrôle qu'on doit penser à faire est un
# contrôle qu'on oubliera le jour où il compte.
INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.command // .tool_input.command // ""' 2>/dev/null)

[ -z "$CMD" ] && exit 0

# Un listage rtk suivi d'un compteur, dans le même pipeline.
# Le nom du fichier promet « comptage rtk » ; le motif ne visait que
# `git log`. Eprouve sur sept cas : `rtk git shortlog | wc -l` passait,
# comme branch, tag et ls-files. Un nom qui declare un perimetre plus
# large que le code est une garde a moitie presente.
LISTAGE='rtk[[:space:]]+git[[:space:]]+(log|shortlog|branch|tag|ls-files|rev-list[[:space:]]+--all)'
COMPTEUR='\|[[:space:]]*(wc([[:space:]]|$)|grep[[:space:]]+-[a-zA-Z]*c|sort[[:space:]]*\||uniq[[:space:]]+-c)'

# UNE GARDE QU'ON CONTOURNE POUR LA TESTER EST UNE GARDE QU'ON CONTOURNERA
# POUR TRAVAILLER. Signale par 0a le 10/09 : ce crochet bloquait aussi
# l'ECRITURE d'un fichier qui MENTIONNE `rtk git log | wc` — documentation,
# regles, message de commit. Elle a du le contourner deux fois pour l'eprouver.
#
# Le crochet doit s'appliquer a une commande EXECUTEE, pas a un contenu ECRIT.
# Un heredoc ou une redirection vers un fichier transporte du texte ; il ne
# lance rien. Ecrire la regle n'est pas l'enfreindre.
if echo "$CMD" | grep -qE '<<-?[[:space:]]*.?[A-Za-z_]|>[[:space:]]*[^|&[:space:]]+\.(md|txt|sh|py|json)'; then
    exit 0
fi

if echo "$CMD" | grep -qE "$LISTAGE" && echo "$CMD" | grep -qE "$COMPTEUR"; then
    # `-n N` ou `--max-count=N` explicite : rtk obéit, le comptage est sûr.
    if echo "$CMD" | grep -qE '(-n[[:space:]]+[0-9]+|--max-count[=[:space:]][0-9]+)'; then
        exit 0
    fi
    cat <<'RAISON'
{"decision":"block","reason":"COMPTAGE SUR UNE SORTIE `rtk git log` — le chiffre serait faux, et faux dans le sens rassurant.\n\nMesure du 10/09 : `rtk git log` vaut `git log --no-merges -n 50`. Il plafonne a 50 ET retire les fusions, donc il enleve aussi des lignes DU MILIEU — les 50 lignes rendues couvraient les rangs 1 a 60. Rien ne signale les trous.\n\nTrois voies, au choix :\n  - la commande NUE : `git log --format=%H | wc -l`\n  - un compte que rtk calcule lui-meme : `rtk git rev-list --count HEAD`\n  - un `-n` explicite plus grand que le total attendu : `rtk git log -n 5000 | wc -l`\n\nVoir acvram-memoire/REGLES.md section 6."}
RAISON
    exit 0
fi

exit 0
