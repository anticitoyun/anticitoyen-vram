#!/bin/bash
# GABARIT DE CHAÎNE À COÛT CROISSANT (poste7-tests-30min-20-09 § 3.4) — sortie au premier « faux ».
# Une chaîne = des étapes `etape N "nom" "scellé" cmd…` de coût croissant : (0) test au bit / ulp sur carte
# (< 1 min) → (1) nsys ou compteur (2-5 min) → (2) PPL avec SE (3-4 min) → (3) capture 4/4 (2 min) → (4) ABAB /
# cellule (3 min). Chaque étape porte SON scellé (écrit avant, REGLES § 3) ; la commande rend 0 = tenu, autre = faux
# (ou « PARTIEL » : code 2 = mesuré mais non tranché : la chaîne continue et le dit) ; la chaîne s'arrête au premier
# scellé réfuté et rend PARTIEL (code 1). Une chaîne qui rend « tenu » partout sans avoir pu rendre faux (scellé
# absent) n'entre pas dans INDEX : ici une étape sans scellé est REFUSÉE (code 64).
# L'étape PPL apparie ses tranches et s'arrête par outils/gpu/mesure/geo-sequentiel.py (3.3 : NMIN 5, SE plancher
# 1,97 %/√n, codes 10/11/12/13 ; l'étape rend 0 si 10, 1 si 11, 2 = partiel si 13).
# Chaque étape prend elle-même outils/carte.sh (plafond ACVRAM_DUREE_MAX, 3.1) ; jamais lancer la chaîne sous un
# verrou tenu (ACVRAM_CARTE_TENUE → refus).
# Usage dans une chaîne :
#   . "$(dirname "$0")/../../outils/gpu/mesure/gabarit-chaine.sh"   # ou le chemin relatif au dépôt
#   chaine_debut "C15-prefill d17a719d" "$O"
#   etape 0 "tests carte au bit" "warp = bloc au bit, 0 failed" outils/carte.sh "$PYA" -m pytest tests/test_x.py -q
#   etape 1 "nsys A/B" "noyaux B <= 83 ms" bash "$ICI/nsys.sh"
#   etape 2 "PPL 3 tranches" "au bit" "$PYA" "$ICI/ppl-au-bit.py" "$O"
#   chaine_fin
set -u
[ -n "${ACVRAM_CARTE_TENUE:-}" ] && { echo "ECHEC : chaîne lancée sous un verrou tenu (chaque étape prend carte.sh)"; exit 3; }
# Racine du parc (M3, poste2 11:59) : les chaînes effacent ACVRAM_* avant de sourcer ce gabarit, et les
# instruments (juge-2a.py, ppl-decode-kv.py…) relisent ACVRAM_MODELES dont le repli 980PRO n'existe plus ;
# la racine du moment vient de outils/racine_modeles.py (variable → ~/.config/acvram/modeles → littéral).
if [ -z "${ACVRAM_MODELES:-}" ]; then
  ACVRAM_MODELES=$("$(dirname "${BASH_SOURCE[0]}")/../../racine_modeles.py") || { echo "ECHEC : racine_modeles.py"; exit 3; }
fi
export ACVRAM_MODELES
_CH_NOM=; _CH_O=; _CH_ETAT=tenu; _CH_JOURNAL=
chaine_debut() {   # chaine_debut <nom> <dossier de sortie>
  _CH_NOM=$1; _CH_O=$2; mkdir -p "$_CH_O"; _CH_JOURNAL=$_CH_O/chaine.tsv
  printf 'etape\tnom\tscelle\tdebut\tfin\tcode\tverdict\n' > "$_CH_JOURNAL"
  echo "=== chaîne $_CH_NOM — $(date +%FT%T) HEAD $(/usr/bin/git rev-parse --short HEAD 2>/dev/null) ; sortie $_CH_O"
}
etape() {          # etape <N> <nom> <scellé> <cmd…>  → continue si 0 ou 2 (PARTIEL), s'arrête sinon
  local n=$1 nom=$2 scelle=$3; shift 3
  [ -n "$scelle" ] || { echo "REFUS étape $n « $nom » : scellé absent (un contrôle qui ne peut rendre faux n'est pas un contrôle)"; exit 64; }
  local deb; deb=$(date +%T); echo "=== ($n) $nom — scellé : $scelle — $deb"
  "$@" > "$_CH_O/etape-$n.log" 2>&1; local code=$?
  # Un juge qui IMPRIME « FAUX » et rend 0 est un défaut d'instrument (verdict-n3-piece2a-19-09 : juge-2a.py
  # « VERDICT 2a FAUX » puis « (0) tenu (code 0) ») : l'étape est fausse par ce qu'elle a écrit, code 65, quel
  # que soit son rc. Convention : une ligne VERDICT/RESULTAT/JUGE tenue n'écrit jamais le mot FAUX en capitales.
  if [ "$code" = 0 ] && grep -aqE '^(VERDICT|RESULTAT|JUGE)\b.*\bFAUX\b|"verdict": *"FAUX"' "$_CH_O/etape-$n.log"; then
    code=65; echo "DEFAUT D'INSTRUMENT : l'étape a imprimé FAUX et rendu 0 — comptée fausse (code 65)"
  fi
  local verdict
  case $code in 0) verdict=tenu ;; 2) verdict=partiel ;; *) verdict=faux ;; esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$nom" "$scelle" "$deb" "$(date +%T)" "$code" "$verdict" >> "$_CH_JOURNAL"
  grep -aE '^(RESULTAT|VERDICT|JUGE|EXPERTS|GÉO|GEO|PREDIT|ECHEC)' "$_CH_O/etape-$n.log" | cut -c1-240 | head -12
  echo "--- ($n) $verdict (code $code) $(date +%T)"
  case $code in
    0) ;;
    2) _CH_ETAT=partiel ;;
    *) _CH_ETAT=faux; echo "=== ARRÊT au premier scellé réfuté : étape $n « $nom » — la chaîne rend PARTIEL (les étapes suivantes ne sont pas mesurées)"; chaine_fin; exit 1 ;;
  esac
}
chaine_fin() {
  echo "=== chaîne $_CH_NOM : $_CH_ETAT — $(date +%FT%T) ; journal $_CH_JOURNAL"
  column -t -s $'\t' "$_CH_JOURNAL" 2>/dev/null || cat "$_CH_JOURNAL"
}
