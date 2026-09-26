# Remède aux écritures inventées — la classe 3, et pourquoi « Optional » seul ne suffit pas

Décision de forme datée du 10/09 (« avant une campagne, pas pendant »), traitée
le 11/09 avant `base_croissant`.

## Les trois demandes de `1c`, mesurées

### 1. Les champs qui passent en `Optional`, par classe de gravité

Seule la **classe 3 (devinable et fausse)** est obligatoire, et elle tient en
**deux champs** :

```
champ               invente /126   la source varie sur           consommé au runtime ?
hidden_activation        61        silu 52 · gelu 8 · relu² 5     OUI (MLP, 5 sites loader)
torch_dtype              22        bfloat16 92 · float16 12       NON — nulle part
```

Un `silu` deviné sur un modèle gelu ou relu² **corrompt le MLP** : c'est le seul
des 48 champs à défaut qui produit une sortie fausse, pas seulement un inventaire
faux. `torch_dtype` ne fabrique qu'un chiffre d'inventaire — il n'est lu par
aucun calcul (le dtype du KV cache est une **propriété calculée de son propre
format**, `KVCacheConfig.torch_dtype`, sans lien avec `ModelSpec.torch_dtype`).

**Correction d'un chiffre d'hier** : l'audit annonçait `hidden_activation` inventé
sur **96** manifestes. Faux — la table d'alias de l'outil ne couvrait pas
`hidden_act`, la clé HF standard. Le vrai compte est **61**. Même défaut
d'instrument que ceux traqués la veille, sur mon propre audit.

Les classes 1 (convention documentée : `rope_theta`, `head_dim`) et 2 (champ
inapplicable : `mamba_*`, `kv_lora_rank`…) ne sont **pas** touchées : garder leur
défaut est acceptable si l'énoncé le dit. Les toucher modifierait leur sémantique
sans corriger un mensonge.

### 2. Ce que `load_model_spec` fait d'un `None` — vérifié, et ça inverse la prémisse

**« Optional dans `ModelSpec` » ne fait rien à lui seul.** Mesuré :
**62 des 63 champs** du dataclass sont écrits explicitement par
`load_model_spec` (seul `layers` ne l'est pas). Le défaut du dataclass est donc
**du code mort** pour ces champs — `load_model_spec` l'écrase avec
`cfg.get(cle, DEFAUT)`.

Le mensonge est écrit à l'**extraction**, pas au défaut. Le remède est donc :
1. retirer le défaut substitué de l'extraction (`or "silu"`, `, "bfloat16"`) →
   `None` quand la source est muette ;
2. porter le défaut au **point d'usage**, jamais à l'enregistrement.

Appliqué : `hidden_activation` → propriété `ModelSpec.mlp_activation` (`or "silu"`)
lue par les 5 sites de `loader.py` ; `torch_dtype` → rien (non consommé). Le
comportement runtime est **identique à l'octet** — seul le manifeste change.

### 3. Le test qui casse

`tests/test_provenance_champs.py` : une config muette sur les deux clés doit
donner `hidden_activation is None` et `torch_dtype is None` (et non `silu`/
`bfloat16`). Il échouait avant le correctif, il passe après. Un second cas vérifie
qu'une valeur **déclarée**, alias `hidden_act` compris, survit — sinon le remède
casserait les 65 manifestes honnêtes.

## Ce qui n'est pas fait

Les 126 manifestes existants ne sont pas réécrits. Ils gardent leur valeur
devinée ; seuls les manifestes produits **à partir de maintenant** enregistrent
`None`. Un ré-inventaire du parc devra distinguer « manifeste d'avant le
11/09 » — l'horodatage du champ, pas sa valeur.

## Un piège attrapé au passage

La propriété `mlp_activation` a d'abord atterri dans `LayerSpec` au lieu de
`ModelSpec` — l'ancre `activation_ratio` existe dans les deux classes. La syntaxe
passait, l'import passait ; seul l'appel runtime `spec.mlp_activation` levait.
Trouvé en exécutant, pas en relisant. Une édition se vérifie sur l'objet qu'elle
produit, pas sur sa compilation.
