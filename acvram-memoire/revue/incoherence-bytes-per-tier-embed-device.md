# Incohérence dans le manifeste : `bytes_per_tier` contredit `embed_device`

Trouvée en passant le 10/09/2026 en creusant l'exil de `Qwen2.5-Coder-14B-bf16-pur`
sur `_reajuster_plan`. **Classée cosmétique pour ce cas précis** (elle n'a pas
causé l'exil observé — vérifié : le compte réel utilise `embed_device`, pas
`bytes_per_tier`) mais pas anodine en soi : deux champs du même fichier qui
se contredisent finiront par tromper quelqu'un qui lit l'un sans l'autre.
Note seule, pas de correctif — chantier pour un autre jour.

## Le fait

Dans `acvram_manifest.json` (exemple : `Qwen2.5-Coder-14B-bf16-pur`) :

    plan.bytes_per_tier = {"cuda:0": 27982921728, "cpu": 1557135360}
    plan.embed_device   = "cuda:0"

`1557135360` octets est exactement la table `embed_tokens`
(`152064 × 5120 × 2`, vérifié). **`bytes_per_tier` affirme donc que cette
table est stockée sur `cpu`**, pendant qu'`embed_device` affirme `cuda:0`.

## Lequel fait foi à l'exécution

`acvram/engine/loader.py:265` (au moment de ce constat) place réellement le
tenseur d'embedding selon `plan.embed_device`, pas selon `bytes_per_tier` :

    embed_dev = dev(plan.embed_device) if plan.embed_device != "cpu" else ...

Et `_octets_reels`/`_reajuster_plan` (`loader.py:991,1100` au moment du
constat) comptent `embed` dans le tally d'un tier **quand `embed_device`
correspond**, jamais en lisant `bytes_per_tier`. **`bytes_per_tier` n'est
donc consulté nulle part pour les décisions de placement ou d'exil** — c'est
un champ d'affichage/résumé, potentiellement calculé à une étape différente
de la conversion que `embed_device`, jamais revérifié contre lui.

## Pourquoi ça n'a pas trompé cette enquête

Le calcul du "poids réels" utilisé par `_reajuster_plan` (celui qui décide
de l'exil) suit `embed_device`, la valeur qui correspond au comportement
réel de chargement (`loader.py:265`). `bytes_per_tier` n'entre dans aucun
calcul consulté ce soir-là. L'incohérence est donc restée sans effet
observable — mais un futur outil qui lirait `bytes_per_tier` pour estimer
"ce qui vit où" (un rapport, un autre balayage, une doc générée) obtiendrait
une réponse fausse pour ce champ précis, sans qu'aucune erreur ne se
déclenche.

## Ce qui reste à établir, pour qui reprend ce chantier

* **D'où vient `bytes_per_tier`** — quelle fonction du convertisseur
  l'écrit, et écrit-elle avant ou après la décision finale d'`embed_device` ?
  Si c'est avant, c'est un instantané périmé par une décision ultérieure ;
  c'est le genre de défaut qu'on a retiré plusieurs fois ce soir sous
  d'autres formes (une estimation figée mutée après sa propre mesure).
* **Combien de modèles du parc portent la même contradiction** — un
  balayage identique à celui de ce soir (`bytes_per_tier[cpu]` contenant
  exactement la taille d'un tenseur que `embed_device`/`lm_head_device`
  place ailleurs) donnerait le compte, sans GPU.
* **Le correctif, s'il s'impose** : soit recalculer `bytes_per_tier` à partir
  des placements réels (`embed_device`, `lm_head_device`, `attn_storage`,
  `mlp_storage`) au moment de l'écrire dans le manifeste, soit le documenter
  explicitement comme "estimation de conversion, non contractuelle" pour que
  personne ne le prenne pour la source de vérité.
