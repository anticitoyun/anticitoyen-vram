# Traduire acvram

Le README existe en français (source) et en anglais (`docs/README.en.md`). Vingt-neuf autres
langues sont listées dans la barre du README mais pas encore traduites — toute contribution est
bienvenue, faite au fil de l'eau par qui veut s'en charger.

## Avec Weblate (le plus simple, pour l'interface)

Les textes de l'interface graphique (`packaging/acvram-gui`, boutons et étiquettes) se traduisent en
ligne sur **Hosted Weblate**, sans outil ni compte GitHub — le lien apparaîtra ici et dans le README
dès l'ouverture du projet. Weblate propose ensuite les changements au dépôt.

Directement dans le dépôt, sans Weblate :

- `packaging/langues/<code>.json` : un fichier par langue, `"texte source": "traduction"`.
- `packaging/langues/_source.json` : le texte français de chaque clé (fichier de base pour
  Weblate — identité `{clé: clé}`, gardée par `tests/test_langues_source_weblate_236.py`) ;
  `packaging/langues/_cles.json` : la liste des clés, extraite du script par
  `tests/test_langues_gui.py`. Ces deux fichiers ne se traduisent pas.
- Les accolades (`{nom}`, `{port}`, `{ports}`…) et les balises HTML (`<code>…</code>`) restent
  telles quelles.
- Vérification : `.venv/bin/python -m pytest -q tests/test_langues_gui.py tests/test_langues_source_weblate_236.py`.

Nouvelle langue : copier `packaging/langues/en.json`, traduire, ouvrir une demande de fusion — les
tests ci-dessus rendent rouge toute clé manquante ou en trop.

### Pour la personne qui ouvre le projet Weblate

Sur <https://hosted.weblate.org> (offre gratuite « Libre » pour les projets sous licence libre) :

1. Créer le projet **acvram**, puis un composant **Interface** :
   - dépôt : le dépôt acvram, branche `main` ;
   - format : **JSON file** ; masque des fichiers : `packaging/langues/*.json` ;
   - fichier de base monolingue : `packaging/langues/_source.json` ; langue source : **français** ;
   - filtre des langues : `^[a-z]{2}(-[A-Z]{2})?$` (écarte `_source` et `_cles`).
2. Envoi des traductions : « Demandes de fusion GitHub », ou clé SSH de Weblate ajoutée comme clé
   de déploiement en écriture du dépôt.
3. Ajouter le lien du projet et son badge dans le README.

**README (`docs/README.<code>.md`) : pas de composant Weblate pour l'instant.** Un fichier Markdown
entier comme unité de traduction se prête mal au format JSON monolingue attendu ci-dessus (pas de
clé par phrase, tout le fichier serait « une » traduction) ; Weblate propose un format Markdown
mais sans le découpage par ancre/section dont ce README a besoin (« Ce qui NE se traduit PAS »
ci-dessous — chiffres et noms de champs HTTP, à l'intérieur d'un même fichier). À revoir si Weblate
change son support Markdown, ou si les README passent à un format par blocs (JSON/YAML) plus tard.

## Ce qui se traduit

Chaque fichier `docs/README.<code>.md` reprend la MÊME structure que `README.md` (source) et
`docs/README.en.md` (référence de traduction déjà faite, pour le format des liens relatifs et des
ancres) :

- logo, badges, phrase d'accroche ;
- barre des langues (mettre SA langue en gras, garder les autres inchangées, mêmes drapeaux) ;
- capture `docs/captures/resultats-22-09.png`, inchangée ;
- « Sommaire » avec les mêmes ancres `<a id="...">` (ne pas les traduire : les liens du sommaire et
  des autres langues les visent par leur id anglais actuel — `#idees`, `#demarrage`, `#plan`,
  `#optimisations`, `#http`, `#chiffres`, `#documentation`, `#resultats`, `#etat`, `#credits`,
  `#licence`, `#soutien`) ;
- mêmes sections, séparées par `---`, mêmes tableaux (traduire les en-têtes et le texte des
  cellules, jamais les noms de champs HTTP — `chat/completions`, `prompt_tokens`… — ce sont ceux du
  protocole OpenAI) ;
- Crédits, Licence, Soutenir en fin.

## Ce qui NE se traduit PAS

- **Aucun chiffre.** Chaque valeur du README (débits, joules, SNR, bits par poids, dates, versions)
  vient d'une mesure de ce dépôt (`acvram-memoire/revue/`, fichier:ligne ou pièce citée en note) —
  une traduction ne remesure rien, elle recopie le chiffre exact de la source.
- Les noms de commandes (`acvram plan`, `acvram serve`…), les chemins de fichiers, les blocs de
  code et leur sortie affichée (voir `docs/README.en.md` : la sortie de `acvram plan` est traduite
  mot à mot dans l'exemple, mais reste alignée en colonnes).
- Les noms de champs des réponses HTTP (protocole OpenAI, section « Points d'entrée HTTP » /
  « HTTP endpoints ») — la section elle-même le précise, à garder dans chaque langue.
- Les liens vers `acvram-memoire/revue/*.md` (notes internes, jamais traduites, jamais publiques).

## Vérification avant d'ouvrir une demande de fusion

- Chaque lien relatif fonctionne depuis `docs/` (comme dans `docs/README.en.md` : `../LICENSE`,
  `../README.md`, `ARCHITECTURE.md` sans `../` puisqu'il est dans le même dossier).
- La barre des langues du nouveau fichier est identique à celle du README source, langue courante
  en gras.
- Aucun chiffre ne diffère de `README.md` (comparaison ligne à ligne des tableaux).

## Nouvelle langue

Copier `docs/README.en.md`, traduire, ajouter l'entrée dans la barre des langues du `README.md`
racine ET de tous les `docs/README.*.md` déjà traduits, ouvrir une demande de fusion.
