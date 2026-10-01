# À sourcer : fonction `_acvram_arbre_defaut <repli>` — l'arbre acvram qui CONTIENT le cwd
# (remonte jusqu'à un dossier portant `acvram/__init__.py` ET `.git`, même règle que
# `acvram/__init__.py:_garde_arbre`), ou `<repli>` si le cwd n'est dans AUCUN arbre.
#
# bd jdp (chef, 01/10) : poser ACVRAM_ARBRE au DEPOT du script est faux pour un appelant qui
# invoque le carte.sh d'un AUTRE arbre depuis son propre worktree (ex. scratchpad/poste1-p221/
# chaine.sh:5, `../../anticitoyen-vram/outils/carte.sh` depuis un worktree, ou
# ~/.config/acvram/chef/fenetre-g2c.sh) : avec `ACVRAM_ARBRE=$DEPOT` fixe, le cwd (le worktree de
# l'appelant) ne correspond plus à l'arbre importé et l'import est refusé à tort — un refus
# bruyant, donc jamais un faux positif silencieux, mais l'usage légitime casse. La racine suit le
# CWD en priorité, `$DEPOT` (l'arbre qui PORTE le script) seulement en dernier recours.
_acvram_arbre_defaut() {
  local repli=${1:?_acvram_arbre_defaut <repli>} d
  d=$(pwd -P) || { echo "$repli"; return; }
  while :; do
    if [ -f "$d/acvram/__init__.py" ] && [ -e "$d/.git" ]; then
      echo "$d"; return
    fi
    [ "$d" = "/" ] && break
    d=$(dirname "$d")
  done
  echo "$repli"
}
