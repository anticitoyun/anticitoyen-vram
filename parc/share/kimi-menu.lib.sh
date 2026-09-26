# kimi-menu.lib.sh — menu commun des lanceurs kimi (jan / tabby / yals).
# L'appelant définit : lister_alias (7 colonnes TSV : alias, modèle, ctx,
# refus, tok/s, qualité, usage), les couleurs c_t/c_d/c_v/c_r/c_0, err().
# Usage : choisir_alias "Titre du menu" "modèle-actuellement-chargé"

vram_libre() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{u+=$1; t+=$2} END{if (t) printf "VRAM libre %.0f/%.0f Go", (t-u)/1024, t/1024}'
}

choisir_alias() {
  local titre="$1" charge="${2:-}"
  local -a alias=() modeles=() ctx=() refus=() tps=() qual=() usage=()
  local a m c r t q u
  while IFS=$'\t' read -r a m c r t q u; do
    alias+=("$a"); modeles+=("$m"); ctx+=("$c"); refus+=("$r"); tps+=("$t"); qual+=("$q"); usage+=("$u")
  done < <(lister_alias)
  # parc vide (premier lancement, aucun moteur/modèle dans parc.toml/config.toml) :
  # ce n'est pas une erreur, rc 0, le message le nomme pour que l'appelant sache
  # s'arrêter proprement (parc-installer n'a pas encore tourné).
  [ "${#alias[@]}" -gt 0 ] || { err "parc vide : aucun modèle dans $CONFIG (lancez parc-installer)"; return 0; }

  local filtre="" choix i n point bas
  while :; do
    local -a vus=()
    for i in "${!alias[@]}"; do
      if [ -z "$filtre" ]; then vus+=("$i"); continue; fi
      bas="${alias[$i],,} ${usage[$i],,} ${modeles[$i],,} ${qual[$i],,}"
      [[ "$bas" == *"${filtre,,}"* ]] && vus+=("$i")
    done
    if [ "${#vus[@]}" -eq 0 ]; then
      err "Aucun modèle ne correspond à « $filtre »."
      filtre=""
      continue
    fi
    # un filtre qui ne laisse qu'un candidat le sélectionne directement
    if [ -n "$filtre" ] && [ "${#vus[@]}" -eq 1 ]; then
      echo "${alias[${vus[0]}]}"
      return 0
    fi

    printf '\n%s%s%s   %s%s%s\n' "$c_t" "$titre" "$c_0" "$c_d" "$(vram_libre)" "$c_0" >&2
    printf '%sMédias : les modèles savent générer images/vidéos/GIF via « generer-media » (aussi depuis une image : --image) · navigateur : « bcode »%s\n' "$c_d" "$c_0" >&2
    [ -n "$filtre" ] && printf '%sFiltre actif : « %s » (%d/%d modèles)%s\n' \
      "$c_d" "$filtre" "${#vus[@]}" "${#alias[@]}" "$c_0" >&2
    printf '\n%s   n°   %-26s %-10s %10s  refus %-11s %-12s modèle · contexte%s\n' \
      "$c_d" "alias" "qualité" "tok/s" "" "usage" "$c_0" >&2
    n=0
    for i in "${vus[@]}"; do
      n=$((n + 1))
      point='  '
      if [ -n "$charge" ] && { [ "${modeles[$i]}" = "$charge" ] \
          || [ "$(basename "${modeles[$i]}")" = "$charge" ] \
          || [ "$(basename "${modeles[$i]%.gguf}")" = "${charge%.gguf}" ]; }; then
        point="${c_v}●${c_0} "
      fi
      printf '  %s%2d%s %s%-26s %s%-10s %4s tok/s  refus %-11s %-12s %s%s · %s tokens%s\n' \
        "$c_t" "$n" "$c_0" "$point" "${alias[$i]}" "$c_v" "${qual[$i]}" "${tps[$i]}" \
        "${refus[$i]}" "${usage[$i]}" "$c_d" "${modeles[$i]:0:34}" "${ctx[$i]}" "$c_0" >&2
    done
    [ -n "$charge" ] && printf '  %s● = modèle actuellement chargé%s\n' "$c_d" "$c_0" >&2
    printf '  %s* après un débit = mesuré avant le 20/09 (poste précédent), sans mise à l’échelle%s\n' "$c_d" "$c_0" >&2

    printf '\n%sNuméro, texte pour filtrer (code, créatif, nsfw…), entrée = %s > %s' \
      "$c_t" "${alias[${vus[0]}]}" "$c_0" >&2
    read -r choix || { echo >&2; return 1; }
    if [ -z "$choix" ]; then
      echo "${alias[${vus[0]}]}"
      return 0
    fi
    if [[ "$choix" =~ ^[0-9]+$ ]]; then
      if [ "$choix" -ge 1 ] && [ "$choix" -le "${#vus[@]}" ]; then
        echo "${alias[${vus[$((choix - 1))]}]}"
        return 0
      fi
      err "Entre un nombre entre 1 et ${#vus[@]}."
      continue
    fi
    # correspondance exacte d'alias → sélection directe
    for i in "${!alias[@]}"; do
      if [ "${alias[$i]}" = "$choix" ]; then
        echo "$choix"
        return 0
      fi
    done
    filtre="$choix"
  done
}
