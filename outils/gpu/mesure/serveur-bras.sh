# Harnais commun de lancement et d'arrêt d'un serveur par bras de mesure — à SOURCER, jamais exécuter.
# Ordre de chef (01/10), après trois prises perdues par des scripts de prise neufs, dont poste1-5v7-bit :
# `( cd "$HOME" && setsid serveur … & echo $! )` met la LISTE en fond, `$!` est le sous-shell et jamais le serveur ;
# l'arrêt visait un PID mort, le serveur du bras 1 a servi les requêtes du bras 0 (qui avait refusé faute de VRAM).
#
#   . outils/gpu/mesure/serveur-bras.sh
#   bras_servir 8095 "$S/bras-a.log" env ACVRAM_X=1 python -m acvram serve … --port 8095 || exit 1
#   bras_pret 8095 coder "$BRAS_PID" || { bras_arreter "$BRAS_PID" 8095; exit 1; }
#   … requêtes …
#   bras_arreter "$BRAS_PID" 8095 || exit 1
#
# Garanties, chacune éprouvée contre un faux serveur par tests/test_serveur_bras.py :
# * le PID est écrit par le processus qui devient le serveur (`echo $$` puis `exec`), dans `<journal>.pid` : juste
#   que `setsid` forke ou non, jamais une substitution de commande (elle attendrait le serveur) ;
# * un bras refuse de démarrer si le port répond déjà (un serveur d'un bras précédent répondrait à sa place) ;
# * « prêt » exige que le port soit écouté par CE PID (ss), pas seulement qu'un /v1/models réponde avec le bon nom ;
# * l'arrêt tue le groupe de session entier, puis vérifie le port libre et, si BRAS_VRAM_MAX_MIO est posé, la VRAM
#   rendue ; sinon rc 1 : le bras suivant ne part pas sur une carte occupée.
# Variables : BRAS_CWD (défaut $HOME : un serveur ne lit pas l'arbre de travail par son cwd), BRAS_VRAM_MAX_MIO,
# BRAS_CARTE (0), BRAS_NVIDIA_SMI (nvidia-smi ; remplacé par les tests), BRAS_ARRET_S (30), BRAS_VRAM_S (60).

bras_port_libre() {   # PORT → 0 si rien ne répond sur 127.0.0.1:PORT
    ! curl -s -m 2 -o /dev/null "http://127.0.0.1:$1/v1/models"
}

bras_ecoute_par() {   # PORT PID → 0 si PID écoute PORT (ss -ltnp, mêmes droits que le serveur)
    ss -ltnpH "sport = :$1" 2>/dev/null | grep -q "pid=$2,"
}

bras_servir() {   # PORT JOURNAL CMD… → BRAS_PID ; rc 1 si le port répond déjà ou si le PID n'est pas écrit
    local port=$1 journal=$2 _
    shift 2
    BRAS_PID=
    if ! bras_port_libre "$port"; then
        echo "== port $port déjà servi : bras refusé" >&2
        return 1
    fi
    rm -f "$journal.pid"
    setsid bash -c 'echo $$ > "$1"; cd "$2" || exit 1; shift 2; exec "$@"' _ "$journal.pid" "${BRAS_CWD:-$HOME}" "$@" \
        > "$journal" 2>&1 < /dev/null &
    for _ in $(seq 1 50); do [ -s "$journal.pid" ] && break; sleep 0.1; done
    BRAS_PID=$(cat "$journal.pid" 2>/dev/null)
    if [ -z "$BRAS_PID" ]; then
        echo "== PID du serveur non écrit ($journal.pid)" >&2
        return 1
    fi
}

bras_pret() {   # PORT NOM PID [DÉLAI_S=480] → 0 quand PID écoute PORT et que /v1/models sert NOM ; 1 si PID meurt
    local port=$1 nom=$2 pid=$3 fin=$((SECONDS + ${4:-480}))
    while [ "$SECONDS" -lt "$fin" ]; do
        kill -0 "$pid" 2>/dev/null || { echo "== serveur $pid mort avant d'être prêt" >&2; return 1; }
        if curl -s -m 2 "http://127.0.0.1:$port/v1/models" | grep -q "\"$nom\""; then
            bras_ecoute_par "$port" "$pid" && return 0
            echo "== le port $port sert « $nom » mais n'est pas écouté par $pid : autre serveur" >&2
            return 1
        fi
        sleep 1
    done
    echo "== serveur $pid pas prêt en ${4:-480} s" >&2
    return 1
}

bras_arreter() {   # PID PORT → 0 si le groupe est mort, le port libre et (option) la VRAM rendue ; sinon 1
    local pid=$1 port=$2 _ mio
    kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 "${BRAS_ARRET_S:-30}"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- -"$pid" 2>/dev/null
    for _ in $(seq 1 10); do bras_port_libre "$port" && break; sleep 1; done
    if ! bras_port_libre "$port"; then
        echo "== port $port toujours servi après l'arrêt de $pid" >&2
        return 1
    fi
    [ -n "${BRAS_VRAM_MAX_MIO:-}" ] || return 0
    for _ in $(seq 1 "${BRAS_VRAM_S:-60}"); do
        mio=$(${BRAS_NVIDIA_SMI:-nvidia-smi} -i "${BRAS_CARTE:-0}" --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "${mio:-999999}" -le "$BRAS_VRAM_MAX_MIO" ] && return 0
        sleep 1
    done
    echo "== VRAM non rendue après l'arrêt de $pid : ${mio:-?} Mio > $BRAS_VRAM_MAX_MIO" >&2
    return 1
}
