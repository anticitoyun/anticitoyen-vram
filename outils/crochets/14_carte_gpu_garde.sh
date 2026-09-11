#!/bin/bash
# CROCHET PreToolUse — refuse un script GPU lancé sans sa déclaration.
#
# CE QU'IL COUVRE, exactement :
#   * les commandes Bash lancées par Claude Code (les seules qui traversent ce
#     crochet). Un `systemctl start`, un `acvram serve` tapé dans un terminal,
#     ou une commande d'un autre outil ne passent pas ici — le crochet protège
#     les sessions les unes des autres, pas de l'utilisateur.
#   * `outils/gpu/mesure/*` : exige que la commande passe par `outils/carte.sh`
#     (verrou). Refus nommé, code 2.
#   * `outils/gpu/service/*` : exige une entrée dans `outils/gpu/journal-services.tsv`
#     nommant le script. Refus nommé, code 2.
#   * tout le reste : passe, y compris les scripts hors `outils/gpu/`.
#
# POURQUOI CE PARTAGE : la discipline du verrou s'attache à ce qui ressemble à
# une mesure. Un outil qui produit de l'occupation au lieu d'en consommer
# (charge, service) échappe à la catégorie, et c'est celui qui nuit le plus.
# Le rangement pose la question à la place de la mémoire : un script dans
# `mesure/` demande le verrou par son emplacement, dans `service/` par
# journalisation. Une convention qui n'est pas mécanisée se contourne un soir
# de fatigue — c'est ce que `charge-gpu.py` a coûté le 10/09 (six manches).
#
# ENTREE : JSON sur stdin, {"tool_input":{"command":"..."}}.
# SORTIE : rien = laisse passer ; JSON avec decision:block = refuse.

set -u
CMD=$(jq -r '.tool_input.command // empty' 2>/dev/null)
[ -z "$CMD" ] && exit 0

# Est-ce une commande qui EXÉCUTE le script, ou un outil qui le manipule comme
# argument ? La première approximation par regex de position échouait sur
# `git log -- outils/gpu/mesure/x` — `--` se lit comme un argument, pas comme
# une exécution. On regarde donc le premier mot de la ligne : seuls quelques
# lanceurs exécutent réellement leur argument, le reste (git, sed, ls, cat,
# grep, diff, cp, mv, chmod, echo…) le manipule sans le lancer.
premier_mot=$(echo "$CMD" | sed 's/^[[:space:]]*//' | awk '{print $1}')
case "$premier_mot" in
  bash|sh|python|python3|python3.*|zsh|/*|./*|outils/*)
    ;;    # lanceur, on continue vers la classification
  *)
    exit 0 ;;  # git, sed, ls, ... : le chemin est un argument textuel
esac

SCRIPT=$(echo "$CMD" | grep -oE 'outils/gpu/(mesure|service)/[A-Za-z0-9_.-]+' | head -1)
[ -z "$SCRIPT" ] && exit 0

CATEGORIE=$(echo "$SCRIPT" | cut -d/ -f3)

if [ "$CATEGORIE" = mesure ]; then
    # `carte.sh` doit précéder ce script dans la commande. Un `outils/carte.sh`
    # quelque part avant le script suffit ; l'ordre est ce que le lecteur voit.
    if echo "$CMD" | grep -qE 'outils/carte\.sh[[:space:]].*'"$(echo "$SCRIPT" | sed 's|/|\\/|g')"; then
        exit 0
    fi
    cat <<RAISON
{"decision":"block","reason":"outils/gpu/mesure/*.* exige le verrou. Une mesure lancée sans lui peut tourner pendant celle d'une autre session, et aucun des deux résultats ne vaudra — c'est ce que 'charge-gpu.py' a coûté le 10/09 (six manches perdues).\n\nCorrection :\n  outils/carte.sh <votre commande>\n\nPour prendre le verrou en son nom :\n  ACVRAM_NOM=<intitulé> outils/carte.sh <votre commande>"}
RAISON
    exit 0
fi

if [ "$CATEGORIE" = service ]; then
    # Le script doit apparaître dans le journal des services. Le journal est un
    # TSV : PID<TAB>UNITE<TAB>DEPUIS<TAB>PORTS. On cherche le NOM du script en
    # colonne 2 (unité) — les PID varient, le nom ne varie pas.
    JOURNAL=${ACVRAM_JOURNAL_SERVICES:-outils/gpu/journal-services.tsv}
    NOM=$(basename "$SCRIPT")
    if [ -r "$JOURNAL" ] && awk -F'\t' -v n="$NOM" '$2 ~ n { found=1 } END { exit !found }' "$JOURNAL"; then
        exit 0
    fi
    cat <<RAISON
{"decision":"block","reason":"outils/gpu/service/*.* exige une entrée dans $JOURNAL nommant le script. Un service détaché sort de l'enveloppe carte.sh — le verrou est incompatible avec le détachement, seul le journal peut répondre de l'occupation.\n\nCorrection :\n  printf '%s\\t%s\\t%s\\t%s\\n' \$\$ '$NOM' \"\$(date +%s)\" '<ports>' >> $JOURNAL\n  (à retirer à l'arrêt du service, sinon 'carte-libre.sh' verra une entrée morte)"}
RAISON
    exit 0
fi

exit 0
