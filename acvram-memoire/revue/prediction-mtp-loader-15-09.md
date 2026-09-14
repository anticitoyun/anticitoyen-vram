# Prédiction scellée — le loader ne doit pas exécuter le bloc MTP comme couche 47

poste1, 15/09/2026, avant mesure. poste2 a trouvé la cause réelle du
régime dégradé (2 944 nvfp4 + 64 int4_awq) : les 64 experts int4_awq
sont le bloc MTP de GLM-4.7-Flash (`model.layers.47.*`,
`num_nextn_predict_layers=1`), pas une promotion par expert — elle
corrige le convertisseur. chef (poste7) : vérifier que le LOADER,
indépendamment de ce correctif, n'exécute jamais `layers.47` comme une
48e couche de décodage.

## Lu (pas encore mesuré)

- `config.json` source de GLM-4.7-Flash : `num_hidden_layers=47`,
  `num_nextn_predict_layers=1` — 47 EXCLUT déjà le bloc MTP (couches
  0-46 réelles, 47 = MTP, hors compte).
- `ModelSpec.num_layers = cfg.get("num_hidden_layers", 32)` (config.py) :
  lit ce champ tel quel, donc 47, correct pour ce modèle précis.
- `convert.py:904` : `manifest["model"] = spec.to_dict()` — le nombre de
  couches écrit dans le manifeste converti est CELUI DU SPEC SOURCE
  (47), jamais recompté depuis les tenseurs traités pendant la
  conversion.
- `loader.py:190-199` : `load_model` reconstruit `spec` DEPUIS LE
  MANIFESTE (`ModelSpec(**manifest["model"])`), donc `spec.num_layers`
  reste 47 au chargement aussi.
- La boucle principale de construction des couches (`loader.py:324`,
  `for lp in plan.layers:`) est bornée par `plan.layers`, lui-même
  construit à partir de `spec.layers` (`config.py:162`,
  `[self._build_layer(i) for i in range(self.num_layers)]`) — donc
  exactement 47 entrées, jamais 48, quels que soient les tenseurs
  PRÉSENTS dans le manifeste.
- `_charger_mtp` (loader.py:919) lit un préfixe DIFFÉRENT
  (`model.mtp.{n}.`), pas `model.layers.47.` — une convention de
  renommage à la conversion, distincte du bug de poste2.

## Prédiction

Un manifeste converti qui contient EN PLUS des tenseurs
`model.layers.{spec.num_layers}.*` (simulant le bogue actuel du
convertisseur, où le bloc MTP est écrit sous la même clé qu'une couche
MoE normale) ne change RIEN au nombre de couches construites par
`load_model` : `len(loaded.model.layers) == spec.num_layers` toujours,
et aucun tenseur de cette couche fantôme n'est lu (donc jamais résident,
ni CPU ni GPU).

**Seuil de réfutation** : si `len(loaded.model.layers)` vaut
`spec.num_layers + 1` avec ce manifeste forcé, le loader EXÉCUTE bien le
bloc MTP comme une 48e couche — il faut alors un correctif, pas
seulement le rapport de poste2 sur le convertisseur.

## MESURÉ : CONFIRMÉ — le loader est sain, le bogue est bien au convertisseur seul

`tests/test_loader_ignore_bloc_mtp.py`, 2 tests, reproduisant le VRAI
bogue (pas un cas d'école) :
1. Point de contrôle à 2 couches déclarées (`num_hidden_layers=2`) +
   des tenseurs `model.layers.2.*` présents sur le disque (bloc fantôme,
   même forme que le bloc MTP de GLM écrit à tort par le
   convertisseur). `convert_checkpoint` ÉCRIT bien ces 9 tenseurs
   fantômes dans le manifeste — le bogue de conversion est reproduit,
   ce test n'est pas vide.
2. `manifest["model"]["num_layers"] == 2` (pas 3) : le nombre de couches
   du manifeste vient de `spec.to_dict()` (convert.py:904), jamais d'un
   recomptage des tenseurs présents.
3. `load_model` sur ce converti : `len(loaded.model.layers) == 2`. Le
   bloc fantôme n'est **jamais lu** — la boucle principale
   (`loader.py:324`, bornée par `plan.layers`/`spec.layers`, eux-mêmes
   dimensionnés à `spec.num_layers`) ne demande jamais
   `model.layers.2.*`. `_charger_mtp` (loader.py:919) lit de toute façon
   un préfixe différent (`model.mtp.{n}.`), sans rapport avec ce bogue.

**Conséquence sur la mesure VRAM demandée** : puisque le bloc fantôme
n'est JAMAIS l'objet d'un `get()`/lecture de tenseur par le loader (pas
un cas particulier de CE modèle, une propriété de la boucle elle-même),
`memory_allocated` avant/après ne peut pas monter à cause de lui — la
preuve structurelle suffit, la mesure GPU sur le vrai converti de GLM ne
changerait pas ce verdict. Proposée mais pas faite : si chef la veut
malgré tout comme confirmation finale sur le modèle réel (pas
seulement le point de contrôle synthétique), elle prend quelques
secondes de carte dès que le converti homogène de poste2 existe.

**Verdict** : le loader n'exécute PAS `layers.47` comme 48e couche. Le
régime dégradé observé venait entièrement du convertisseur (formats
mélangés dans le manifeste, corrigé par poste2), pas d'un défaut du
chargement.
