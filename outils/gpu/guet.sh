#!/bin/bash
# UNE CARTE INVISIBLE PAR DEFAUT NE SE SURVEILLE PLUS PAR LE VERROU SEUL.
#
# Depuis le 14/09 (Sage, revue/sage-carte-invisible-sans-verrou-14-09.md),
# les lanceurs de session exportent CUDA_VISIBLE_DEVICES="" — plus aucun
# processus ne voit la carte SANS passer par `carte.sh`, qui seul la rend
# visible (ACVRAM_CARTE, voir carte.sh:33+168). Mais un contournement reste
# possible : quelqu'un peut poser CUDA_VISIBLE_DEVICES=0 a la main, hors de
# tout verrou. `guet.sh` ne PREVIENT pas ce contournement — rien ne le peut,
# une variable d'environnement ne se verrouille pas — il le VOIT et le NOMME,
# ce que rien ne faisait avant.
#
# TROIS QUESTIONS PAR PID GPU (nvidia-smi --query-compute-apps) :
#   1. Est-ce un service permanent (8081-8083, ports-reserves-services-ia) ?
#      -> ignore, silencieux.
#   2. Est-ce un DESCENDANT du process qui tient actuellement le verrou d'UNE
#      carte (le PID ecrit dans carte.sh:$INFO, "$VERROU.qui") ?
#      -> autorise, silencieux : `carte.sh` execute la commande en enfant
#         direct (jamais `exec`, carte.sh:168), donc chaque PID sous verrou
#         remonte a ce PID par la chaine PPid de /proc.
#   3. Ni l'un ni l'autre -> HORS-VERROU. On lit ACVRAM_SESSION dans
#      /proc/<pid>/environ (pose par les lanceurs de session, Sage 14/09)
#      pour nommer QUI, et on l'ecrit dans hors-verrou.log — jamais efface,
#      comme le journal de carte.sh (meme raison : une ligne CONSTATEE apres
#      coup vaut mieux qu'une garde qui bloque en silence).
set -u
JOURNAL="${ACVRAM_GUET_JOURNAL:-$(dirname "${BASH_SOURCE[0]}")/hors-verrou.log}"
PERIODE="${ACVRAM_GUET_PERIODE:-5}"
PORTS_SERVICE="8081 8082 8083"

# PIDs des trois services permanents, relevés à CHAQUE tour (un service peut
# redémarrer, changer de PID) — jamais mis en cache d'un tour à l'autre.
pids_services() {
    for p in $PORTS_SERVICE; do
        ss -tlnpH 2>/dev/null | awk -v port=":$p" '$4 ~ port"$" {print $NF}' \
            | grep -oE 'pid=[0-9]+' | cut -d= -f2
    done
}

# PID du detenteur de CHAQUE carte actuellement verrouillee — un par fichier
# .qui present ; absent = carte libre, aucun descendant possible.
pids_detenteurs() {
    for qui in /tmp/acvram-carte-*.lock.qui; do
        [ -r "$qui" ] || continue
        read -r p _ < "$qui" 2>/dev/null && [ -n "${p:-}" ] && echo "$p"
    done
}

# `$1` descend-il de `$2` (egal compris) par la chaine PPid de /proc ?
# Bornee a 32 sauts : une chaine plus longue est un cycle ou un /proc mort,
# jamais un cas reel — mieux vaut rendre faux que boucler sans fin.
descend_de() {
    local pid="$1" ancetre="$2" sauts=0
    while [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$sauts" -lt 32 ]; do
        [ "$pid" = "$ancetre" ] && return 0
        pid=$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null)
        sauts=$((sauts + 1))
    done
    return 1
}

nom_session_de() {
    local pid="$1"
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
        | sed -n 's/^ACVRAM_SESSION=//p' | head -1
}

echo "[guet] surveillance GPU, période ${PERIODE}s, journal $JOURNAL" >&2

while true; do
    services=$(pids_services)
    detenteurs=$(pids_detenteurs)
    while IFS=, read -r pid nom; do
        [ -z "${pid:-}" ] && continue
        pid=$(echo "$pid" | tr -d ' '); nom=$(echo "$nom" | tr -d ' ')
        autorise=0
        for s in $services; do [ "$pid" = "$s" ] && autorise=1 && break; done
        if [ "$autorise" = 0 ]; then
            for d in $detenteurs; do
                descend_de "$pid" "$d" && autorise=1 && break
            done
        fi
        if [ "$autorise" = 0 ]; then
            session=$(nom_session_de "$pid")
            printf '%s HORS-VERROU pid=%-8s process=%-24s session=%s\n' \
                "$(date +%FT%T)" "$pid" "$nom" "${session:-inconnue}" \
                >> "$JOURNAL"
        fi
    done <<< "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)"
    sleep "$PERIODE"
done
