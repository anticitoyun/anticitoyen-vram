#!/bin/bash
# Six bras contre tests/test_menus.py (poste8, 63ace69) — verdict-menus-tests-17-09.
# À lancer depuis la racine du dépôt ; modifie puis restaure les menus et le TSV par git checkout,
# crée puis retire un dossier vide sur le disque 4TO. Ne touche à rien d'autre.
# Prédictions scellées avant exécution (verdict-menus-tests-17-09) :
#  E1 entrée fabriquée au menu           → prédit 4 passed (aucune direction menu→disque)
#  E2 dossier non listé ajouté sur disque → prédit 4 passed (dénominateur = TSV, jamais le disque)
#  E3 taille TSV d'un modèle ×3           → prédit 4 passed (c : `deviations` jamais rempli)
#  E4 modèle retiré du menu               → prédit 1 failed (b) — seul bras qui doit casser
#  E5 E4 mais nom laissé en prose `…`     → prédit 4 passed (regex backticks)
#  E6 chemin TSV inexistant               → prédit 1 failed (a)
set -u
T=$(mktemp); trap 'rm -f "$T"' EXIT
P=../../anticitoyen-vram/.venv/bin/python; R=acvram-memoire/revue
run(){ CUDA_VISIBLE_DEVICES="" $P -m pytest tests/test_menus.py -q -p no:cacheprovider 2>&1 | tail -1; }
restore(){ git checkout -q -- $R/claude-modeles.md $R/kimi-modeles.md $R/inventaire-disque-brut.tsv; }
M=$(sed -n '2p' $R/inventaire-disque-brut.tsv | cut -f2); echo "modèle témoin: $M"
echo "E1:"; printf -- '- `Modele-Fictif-Zzzzzzzz` (9.9G)\n' >> $R/claude-modeles.md; run; restore
D=/mnt/4TO_SATACMR_2022/Modeles/Dossier-Non-Liste-Zz; echo "E2:"; mkdir "$D" && touch "$D/acvram_manifest.json"; run; rm -r "$D"
echo "E3:"; awk -F'\t' -v OFS='\t' -v m="$M" '$2==m{sub(/[0-9.]+/, "99", $3)} 1' $R/inventaire-disque-brut.tsv > "$T" && mv "$T" $R/inventaire-disque-brut.tsv; grep -P "\t$M\t" $R/inventaire-disque-brut.tsv | cut -f3; run; restore
echo "E4:"; sed -i "/\`$M\`/d" $R/claude-modeles.md $R/kimi-modeles.md; grep -c "$M" $R/claude-modeles.md $R/kimi-modeles.md; run; restore
echo "E5:"; sed -i "/\`$M\`/d" $R/claude-modeles.md $R/kimi-modeles.md; printf 'Note : le modèle `%s` a été retiré.\n' "$M" >> $R/claude-modeles.md; run; restore
echo "E6:"; sed -i "2s#/mnt/#/mnt/INEXISTANT/#" $R/inventaire-disque-brut.tsv; run; restore
git status --short $R | head -3
