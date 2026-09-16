#!/bin/bash
# Six bras contre tests/test_menus.py — verdict-menus-tests-relecture-poste4-17-09.
# À lancer depuis la racine du dépôt ; modifie puis restaure les menus et le TSV par git
# checkout, crée puis retire un dossier vide dans une racine indexée. Ne touche à rien d'autre.
#
# Contre la version de poste8 (63ace69), prédictions scellées puis mesurées :
#  E1 entrée fabriquée au menu           → 4 passed (aucune direction menu→disque)   confirmé
#  E2 dossier non listé ajouté sur disque → 4 passed (dénominateur = TSV)             confirmé
#  E3 taille menu/TSV d'un modèle ×12     → 4 passed (c : `deviations` jamais rempli)  confirmé
#  E4 modèle retiré du menu               → 1 failed (b)                              confirmé
#  E5 E4 mais nom laissé en prose `…`     → 4 passed (regex backticks)                confirmé
#  E6 chemin TSV inexistant               → 1 failed (a)                              confirmé
# Contre la réécriture (dénominateur = scandir des racines) : attendu 6/6 rouges —
#  E1 b (+a), E2 b (+TSV), E3 c, E4 b, E5 b, E6 fixture (racine absente) → erreurs.
set -u
T=$(mktemp); trap 'rm -f "$T"' EXIT
P=../../anticitoyen-vram/.venv/bin/python; R=acvram-memoire/revue
run(){ CUDA_VISIBLE_DEVICES="" $P -m pytest tests/test_menus.py -q -p no:cacheprovider 2>&1 | tail -1; }
restore(){ git checkout -q -- $R/claude-modeles.md $R/kimi-modeles.md $R/inventaire-disque-brut.tsv; }
M=$(sed -n '2p' $R/inventaire-disque-brut.tsv | cut -f2); echo "modèle témoin: $M"
echo "E1:"; printf -- '- `Modele-Fictif-Zzzzzzzz` (9.9G)\n' >> $R/claude-modeles.md; run; restore
D=$(dirname "$(sed -n '2p' $R/inventaire-disque-brut.tsv | cut -f4)")/Dossier-Non-Liste-Zz
echo "E2:"; mkdir "$D" && touch "$D/acvram_manifest.json"; run; rm -r "$D"
echo "E3:"; sed -i "s/^\(- \`$M\` (\)[0-9.]*\([KMG]\?)\)/\199\2/" $R/claude-modeles.md; grep -o "^- \`$M\` ([^)]*)" $R/claude-modeles.md; run; restore
echo "E4:"; sed -i "/^- \`$M\`/d" $R/claude-modeles.md $R/kimi-modeles.md; run; restore
echo "E5:"; sed -i "/^- \`$M\`/d" $R/claude-modeles.md $R/kimi-modeles.md; printf 'Note : le modèle `%s` a été retiré.\n' "$M" >> $R/claude-modeles.md; run; restore
echo "E6:"; sed -i "2s#/mnt/#/mnt/INEXISTANT/#" $R/inventaire-disque-brut.tsv; run; restore
git status --short $R | head -3
