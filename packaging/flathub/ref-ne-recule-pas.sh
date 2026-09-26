#!/usr/bin/env bash
# 266 l : le ref master du dépôt Flatpak servi par gh-pages ne recule jamais.
#   ref-ne-recule-pas.sh <version du tag> <fichier VERSION du dépôt servi>
# code 0 : construire et publier (dépôt vide, même version rejouée, ou version plus récente) ;
# code 3 : NE PAS toucher au dépôt — il sert déjà une version plus récente (release ancienne relancée après une plus
#          récente : le 26/09 les jobs des v0.7.1 et v0.7.2 couraient en même temps). Le motif est écrit sur stdout.
set -u
V=${1:?version}; F=${2:?fichier VERSION}
SERVIE=$(cat "$F" 2>/dev/null | tr -d '[:space:]')
if [ -z "$SERVIE" ]; then echo "dépôt vide ou sans VERSION : construire la $V"; exit 0; fi
case "$SERVIE" in *[!0-9.]*) echo "VERSION illisible (« $SERVIE ») : construire la $V"; exit 0 ;; esac
PLUS_RECENTE=$(printf '%s\n%s\n' "$SERVIE" "$V" | sort -V | tail -n1)
if [ "$PLUS_RECENTE" = "$V" ]; then echo "dépôt à la $SERVIE, tag $V : construire"; exit 0; fi
echo "dépôt à la $SERVIE, plus récente que la $V : le ref master ne recule pas, dépôt non touché"; exit 3
