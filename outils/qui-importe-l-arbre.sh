#!/bin/bash
# Une fusion dans `main` reecrit les .py que d'autres processus IMPORTENT.
#
# Le 10/09/2026 : le venv de anticitoyen-vram installe acvram en mode EDITABLE
# et pointe /home/.../anticitoyen-vram/acvram. Une mesure lancee avec ce venv
# relit donc l'arbre a chaque nouveau processus — et une campagne a quinze tours
# en lance quinze. Fusionner pendant qu'elle tourne change le code SOUS elle,
# tour par tour, sans que rien n'echoue.
#
# `git merge` ne previent pas : il n'a aucune raison de savoir qu'un processus
# lit ses fichiers. Le controle est donc EXTERIEUR a git, et c'est celui-ci.
#
# Code 0 : personne n'importe l'arbre, la fusion est sans risque.
# Code 1 : au moins un processus l'importe — attendre, ou lui demander.
set -u
ARBRE=${1:-${ACVRAM_ARBRE:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)/../../anticitoyen-vram}}
ARBRE=$(readlink -f "$ARBRE")
# LE DETECTEUR SE VOYAIT LUI-MEME. Premiere version : le shell qui invoque ce
# script porte le TEXTE du script dans sa ligne de commande, donc le chemin de
# l'arbre aussi — il se comptait, et le compte etait gonfle de ses propres
# ancetres. Meme faute que `pgrep` qui se trouve dans sa propre sortie.
# On exclut donc soi-meme ET toute la chaine d'ancetres.
MOI=""
p=$$
while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ]; do
    MOI="$MOI $p"
    p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
done

trouves=0
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    case " $MOI " in *" $pid "*) continue;; esac
    # un PID peut disparaitre entre le listage et la lecture : ce n'est pas une
    # erreur, c'est la vie normale d'un /proc. On se taît.
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    [ -z "$cmd" ] && continue
    # trois voies d'import : le venv de l'arbre, un chemin dans l'arbre, ou un
    # cwd dans l'arbre avec un interpreteur python
    cw=$(readlink "/proc/$pid/cwd" 2>/dev/null)
    if [[ "$cmd" == *"$ARBRE"* ]] || [[ "${cw:-}" == "$ARBRE"* && "$cmd" == *python* ]]; then
        # les worktrees ne comptent pas : ils ont leur propre copie des .py
        [[ "$cmd" == *"/tmp/"*"-acvram"* && "$cmd" != *"$ARBRE"* ]] && continue
        age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
        printf 'PID %-8s %5ss  %s\n' "$pid" "${age:-?}" "$(echo "$cmd" | cut -c1-110)"
        trouves=$((trouves + 1))
    fi
done
if [ "$trouves" -eq 0 ]; then
    echo "aucun processus n'importe $ARBRE — fusion sans risque de ce cote."
    exit 0
fi
echo
echo "$trouves processus importent $ARBRE."
echo "Une fusion changerait leur code au prochain processus qu'ils lancent."
exit 1
