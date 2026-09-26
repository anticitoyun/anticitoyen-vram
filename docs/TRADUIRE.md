# Traduire le README

Le README existe en français (source) et en anglais (`docs/README.en.md`). Vingt-neuf autres
langues sont listées dans la barre du README mais pas encore traduites — toute contribution est
bienvenue, faite au fil de l'eau par qui veut s'en charger.

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
