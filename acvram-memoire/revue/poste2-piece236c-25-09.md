instrument : gabarit animematrix (docs/TRADUIRE.md, locale/_source.json, .github/workflows/release.yml:117-127) ; test direct (python3, sans pytest/carte — JSON pur)
commit : poste2-236 depuis origin/main 025ed4b3b
régime : à sec (demande utilisateur, via chef)
scellé : `_source.json` = identité exacte de `_cles.json`, ni clé manquante ni orpheline
mesuré : test vérifié vert en exécution directe (hors pytest, pas de carte nécessaire — JSON pur)
verdict : livré (_source.json, test, section Weblate) ; composant README écarté, motif ci-dessous
durée : —

## Livré

- `packaging/langues/_source.json` : identité `{clé: clé}` des 83 clés de `_cles.json` (`{k: k for k in cles}`,
  même format que `locale/_source.json` d'animematrix). Formatage aligné sur `_cles.json` (indent=1,
  ensure_ascii=False, retour à la ligne final).
- `tests/test_langues_source_weblate_236.py` : garde `_source.json` synchronisé avec `_cles.json` (déjà
  synchronisé avec le script par `test_langues_gui.py`) — clé manquante, clé orpheline, ou valeur qui diverge
  de son propre nom, tout rend rouge. Bras cassant inclus. Vérifié vert en direct (`python3 -c "..."`, sans
  pytest — JSON seul, aucune dépendance GTK/torch, pas besoin de la carte).
- `docs/TRADUIRE.md` : nouvelle section « Avec Weblate », sur le modèle d'animematrix — composant
  **Interface** (`packaging/langues/*.json`, base `_source.json`, filtre de langues `^[a-z]{2}(-[A-Z]{2})?$`),
  marche à suivre pour qui ouvre le projet sur Hosted Weblate. Renommé « Traduire acvram » (portait avant
  uniquement le README).

## Composant README (Markdown) — écarté, pas construit

chef : « si Weblate le permet proprement ». Vérifié : Weblate a un support Markdown, mais pas de découpage
par ancre/section dans ce format — un fichier `docs/README.<code>.md` entier deviendrait UNE unité de
traduction, sans clé par phrase. Cela perd exactement ce que `docs/TRADUIRE.md` § « Ce qui NE se traduit PAS »
protège (chiffres et noms de champs HTTP à l'intérieur du même fichier, pas au niveau fichier) — aucune garde
Weblate ne pourrait vérifier qu'un chiffre n'a pas été retraduit par erreur, contrairement au JSON par clé.
Non construit, motif documenté dans `docs/TRADUIRE.md`. Un échec est un résultat (REGLES point 10) : je ne
force pas un composant qui retirerait une garantie existante.

## Pour poste3 (release.yml)

Gabarit animematrix `.github/workflows/release.yml:117-127` (job `translations`) :
```yaml
translations:
  runs-on: ubuntu-24.04
  steps:
    - uses: actions/checkout@v4
      with: { ref: ${{ env.TAG }} }
    - name: Translation files (Weblate)
      run: |
        V="${TAG#v}"
        zip -q "translations-$V.zip" packaging/langues/*.json docs/TRADUIRE.md
        gh release upload "$TAG" "translations-$V.zip" --clobber --repo "$GITHUB_REPOSITORY"
```
Chemin acvram : `packaging/langues/*.json` (pas `locale/*.json`) ; le reste est identique au gabarit. À toi de
brancher dans `.github/workflows/release.yml` d'acvram.
