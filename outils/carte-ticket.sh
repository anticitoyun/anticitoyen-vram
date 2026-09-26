#!/bin/bash
# Pièce anticitoyen-vram-6it (24/09, ordre chef) : file FIFO pour l'attente de
# `outils/carte.sh` (fd 9). `flock()` ne garantit PAS d'ordre entre plusieurs
# processus bloqués sur le même descripteur — un flux de prises courtes qui
# arrivent APRÈS une attente longue peut l'affamer indéfiniment (bead
# anticitoyen-vram-6it, poste4 > 540 s pendant qu'poste6/poste5 enchaînent).
#
# Principe : un ticket (compteur monotone, mutex COURT fd 12) donne un numéro
# à chaque arrivant, dans l'ordre d'arrivée. `$VERROU.servi` est le numéro
# actuellement autorisé à TENTER le verrou réel (fd 9) — un seul prétendant en
# vol à la fois, donc l'ordre d'obtention suit l'ordre des tickets, quel que
# soit l'arbitrage interne du noyau sur fd 9. `servi` avance à L'OBTENTION du
# verrou réel (jamais avant la tentative — sinon deux tickets consécutifs
# tenteraient fd 9 en même temps, retour au problème initial) et se répare
# tout seul si le détenteur du tour meurt avant de tenter (même idiome que
# `_purger_verrou` / « verrou de compilation orphelin retiré » plus haut dans
# `carte.sh` : auto-guérison, jamais un blocage silencieux).
#
# Fournit : _ticket_prendre, _ticket_attendre_son_tour, _ticket_avancer.
# Appelant : carte.sh, dans la boucle d'attente (après un `flock -n 9` raté).
set -u

_TICKET_DELAI_ORPHELIN=${ACVRAM_TICKET_DELAI_ORPHELIN:-5}   # s avant de suspecter un tour mort

# Prend un ticket. Écrit son numéro sur stdout. Section critique de l'ordre de
# la microseconde (lecture+incrément+écriture d'un entier) : une éventuelle
# iniquité résiduelle ICI est sans effet pratique face aux minutes d'attente
# qu'elle corrige plus loin.
_ticket_prendre() {
    local verrou="$1" tfd="$2"
    eval "exec $tfd>\"\$verrou.ticket_lock\""
    flock "$tfd"
    local n
    n=$(cat "$verrou.ticket" 2>/dev/null || echo 0)
    [ -n "$n" ] || n=0
    echo $((n + 1)) > "$verrou.ticket"
    [ -s "$verrou.servi" ] || echo 0 > "$verrou.servi"
    flock -u "$tfd"
    eval "exec $tfd>&-"
    echo "$n"
}

# Attend (poll borné) que ce soit le tour de `$mon_ticket`. Rend 0 dès que
# c'est son tour, 1 si `$attente_max` est dépassé sans que ce soit son tour
# (l'appelant décide alors : abandon, ou — comme aujourd'hui pour le verrou
# principal — signaler et continuer d'attendre).
_ticket_attendre_son_tour() {
    local verrou="$1" mon_ticket="$2" attente_max="$3" tfd="$4"
    local debut
    debut=$(date +%s)
    while :; do
        local servi
        servi=$(cat "$verrou.servi" 2>/dev/null || echo 0)
        [ -n "$servi" ] || servi=0
        [ "$servi" -ge "$mon_ticket" ] && return 0
        local ecoule=$(( $(date +%s) - debut ))
        [ "$ecoule" -lt "$attente_max" ] || return 1
        # Auto-guérison : le tour en cours (`servi`) est-il mort sans avoir
        # tenté fd 9 ? `servi_pid` est écrit par CE tour au moment de sa
        # propre attente (voir carte.sh) ; absent ou PID mort après le délai
        # d'orphelin -> on avance `servi` nous-mêmes (protégé par le même
        # mutex fd 12, jamais un `servi` qui régresse).
        if [ "$ecoule" -ge "$_TICKET_DELAI_ORPHELIN" ]; then
            local pid=""
            [ -r "$verrou.servi_pid" ] && read -r pid < "$verrou.servi_pid" 2>/dev/null
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                eval "exec $tfd>\"\$verrou.ticket_lock\""
                flock "$tfd"
                local servi2
                servi2=$(cat "$verrou.servi" 2>/dev/null || echo 0)
                [ -n "$servi2" ] || servi2=0
                if [ "$servi2" -lt "$mon_ticket" ]; then
                    echo "$servi2" > "$verrou.servi.orphelin_a" 2>/dev/null || true
                    echo $((servi2 + 1)) > "$verrou.servi"
                fi
                flock -u "$tfd"
                eval "exec $tfd>&-"
            fi
        fi
        sleep 0.2
    done
}

# À appeler DÈS l'obtention du verrou réel (fd 9), jamais avant. Avance
# `servi` à `mon_ticket + 1` SEULEMENT si `servi` vaut encore `mon_ticket`
# (idempotent : un appelant qui double-avance par erreur ne fait rien).
_ticket_avancer() {
    local verrou="$1" mon_ticket="$2" tfd="$3"
    eval "exec $tfd>\"\$verrou.ticket_lock\""
    flock "$tfd"
    local servi
    servi=$(cat "$verrou.servi" 2>/dev/null || echo 0)
    [ -n "$servi" ] || servi=0
    if [ "$servi" -eq "$mon_ticket" ]; then
        echo $((mon_ticket + 1)) > "$verrou.servi"
    fi
    flock -u "$tfd"
    eval "exec $tfd>&-"
}
