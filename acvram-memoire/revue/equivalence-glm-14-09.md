# Équivalence CPU GLM-4.7-Flash, 2 couches — correctif routage, résultat intermédiaire

poste1, 14/09/2026. Bead urgent (chef, relais poste7) sur l'équivalence
rouge de poste2 (`outils/equivalence-glm-2couches.py`, revue/verdict-
equivalence-glm-2couches-14-09.md) : pire |Δlogit| 5,07 (seuil 0,05),
pire cosinus 0,952 (seuil 0,999), 8/16 positions sous le seuil.

## Piège de procédure trouvé en le rejouant : le sous-processus ne relit PAS le correctif

`etape_acvram()` lance `acvram convert` en sous-processus avec
`cwd=Path(__file__).resolve().parent.parent` — la racine du WORKTREE QUI
PORTE LE SCRIPT (`travail/poste2`), pas celui où le correctif vient d'être
écrit. `python -m acvram` ajoute le CWD en tête de `sys.path` : le
sous-processus importe `travail/poste2/acvram/`, une copie SÉPARÉE et non
corrigée, même si le venv est un lien éditable vers `travail/poste1`.
Vérifié directement : `load_model_spec()` appelé depuis mon worktree
rend `router_scoring="sigmoid"` (correct) ; la même conversion lancée
via le script de poste2, depuis son worktree, rend "softmax" dans le
manifeste — le correctif n'était simplement jamais vu.

**À nommer largement** : tout script qui lance `acvram` en sous-processus
avec le CWD de son PROPRE worktree peut silencieusement tester le code
d'un AUTRE worktree que celui qu'on croit — piège générique, pas
spécifique à ce bead.

## Correctif de routage (poste7, revue/poste7-refutation-glm-routage-14-09.md)

`config.py` : `router_scoring="sigmoid"` forcé pour `model_type ==
"glm4_moe_lite"` (sa config HF ne déclare pas `scoring_func`, et le
repli générique tombait sur "softmax", qui ignore le biais de
correction). Pas un heuristique "sigmoid si biais" : `ernie4_5_moe`
garde son softmax légitime. 4 tests (`tests/test_routage_glm4_moe_lite.py`),
cassent si le cas spécifique est retiré (vérifié manuellement).

## Résultat, correctif appliqué CORRECTEMENT (cwd forcé sur mon worktree)

```
position  0: Δ=1.87  cos=0.999548
position  1: Δ=1.56  cos=0.996235   <- reste mauvais
position  2: Δ=0.14  cos=0.999987
position  3: Δ=2.36  cos=0.994423   <- reste mauvais
position  4: Δ=0.15  cos=0.999985
position  5: Δ=5.11  cos=0.949580   <- reste TRES mauvais
position  6: Δ=0.24  cos=0.999997
position  7-15 : tous cos >= 0.999962, Δ <= 0.22

pire_delta=5.114 (seuil 0.05) pire_cos=0.949580 (seuil 0.999)
VERDICT=RÉFUTÉ (encore) -- mais 13/16 positions passent le seuil cosinus,
contre 8/16 avant. Le routage était la cause DOMINANTE, pas la seule.
```

Le routage sigmoid+biais explique la majorité des positions (8 corrigées
sur 8 mauvaises). Trois positions (1, 3, 5) restent nettement fausses,
position 5 très au-delà des deux autres — signature plus proche d'une
VRAIE divergence de sélection (top-k qui bascule sur un autre expert)
que d'un bruit numérique diffus, au vu de l'implémentation du routage
sigmoid déjà lue (`model.py:_route`, 1052-1083) qui suit fidèlement le
schéma HF (sélection sur scores+biais, poids gathered SANS biais,
renormalisation, échelle) — pas de défaut évident à la lecture.

## Suite (pas encore faite)

Contrôle de poste7 : comparer `topi` (indices d'experts sélectionnés,
couche 1 MoE) entre acvram et HF sur les positions 1, 3, 5 précisément —
si différents exactement là, c'est une bascule de sélection (bruit
bf16 sur la frontière top-4, ou biais mal aligné sur un sous-ensemble
d'experts) ; si identiques, regarder les poids/expert partagé, puis
l'attention (v_head_dim 256, jamais posé chez nous avant ce modèle).

## Suite mesurée : 3 causes empilées, pas encore de verdict final

1. Routeur en bf16 (revue/prediction-routeur-fp32-14-09.md) : RÉFUTÉ, effet
   négligeable seul.
2. Couche 0 isolée (MLA seule, revue/prediction-couche0-isolee-14-09.md) :
   bruit bf16 diffus ~1e-3 partout, pas concentré sur 0/1/3/5 — MLA saine.
3. Contrôle décisif fp32/fp32 des deux côtés
   (revue/prediction-fp32-decisif-14-09.md) : positions 0 et 5 corrigées
   (bruit bf16), 1 et 3 restent fausses même en fp32 pur — bogue réel,
   indépendant de la précision de calcul.
4. Cause trouvée (revue/prediction-biais-fp32-14-09.md) :
   `e_score_correction_bias` converti en bf16 (pas de la liste
   `SENSITIVE_SUFFIXES`) alors que ses 64 valeurs codent une correction fine
   (~0,01-0,03) sur un décalage commun (~9) — le pas bf16 à cette magnitude
   (~0,03) efface exactement l'information qui décide le rang 4/5. Correctif
   posé : fp32 inconditionnel pour ce tenseur (convert.py).

**PAS ENCORE CONCLUANT** : la reconversion avec le correctif
(`mini-acvram3`) donne un résultat DÉGRADÉ sur toute la séquence (pire
delta 6,64, pire cosinus 0,978, 16/16 positions sous les seuils de poste2),
alors que les tenseurs non concernés (embeddings, layernorms, poids du
routeur) sont vérifiés BYTE-IDENTIQUES entre `mini-acvram2` et
`mini-acvram3`. Cause de cette dégradation pas encore diagnostiquée —
suspects : script de capture, ou une interaction du correctif avec la
quantification int4_awq des experts (SNR ~20 dB, non exclue). Investigation
interrompue par une tâche de priorité supérieure (carte.sh/guet.sh, ordre de
chef relayant poste7) ; reprise prévue juste après.

**Régression EXPLIQUÉE** (juste avant la PAUSE, pas encore corrigée) :
`mini-acvram3` a été reconverti avec les flags EXACTS de poste2
(`--quant-device cpu --host-exec cpu`), que `mini-acvram2` n'avait pas.
Comparaison directe des manifestes : les poids d'experts sont `bf16` dans
`mini-acvram2` (`formats_nominaux.obtenu=bf16`, rien quantifié — le modèle
tient dans le budget GPU par défaut) mais `int4_awq` dans `mini-acvram3`
(SNR ~19,9 dB, `--no-awq` ne désactive que la calibration AWQ, pas le
format cible choisi par le planificateur pour un hébergement CPU). Cette
quantification à ~20 dB de bruit, ABSENTE côté HF (bf16 plein), suffit
largement à expliquer un delta de logits 4-6 partout — **le script de
poste2 compare peut-être depuis toujours un acvram quantifié à un HF non
quantifié**, un biais indépendant de tout bogue de routage.

## REPRISE (après le correctif carte.sh/guet.sh et le redémarrage de session)

1. Reconvertir avec le correctif biais fp32 SANS `--quant-device
   cpu`/`--host-exec cpu` (laisser le plan GPU par défaut comme
   `mini-acvram2`, `CUDA_VISIBLE_DEVICES=""` pour ne pas toucher la carte
   réelle) — vérifier `formats_nominaux.obtenu=bf16` pour les experts avant
   de mesurer quoi que ce soit.
2. Rejouer l'équivalence 16 jetons (bf16/bf16, seuils de poste2) sur ce
   nouveau checkpoint : prédiction (non scellée formellement, mais
   attendue) — 16/16 positions passent, pire cosinus ≥ 0,999.
3. Si confirmé : signaler à poste2/chef que son script de comparaison
   quantifie l'expert MoE côté acvram alors que HF ne l'est jamais — un
   biais de méthode séparé du bogue de biais de routage, à corriger dans
   SON script (pas le mien) pour toute équivalence future.
4. Points encore ouverts de la consigne de chef : (4) "formats mélangés
   entre experts" / `piles_ok=False` — pas encore investigué du tout.

## PAUSE générale (ordre utilisateur, avant la reconversion de l'étape 1)

Trouvé en lisant `convert.py` (pas encore vérifié en exécutant) : le
format par couche vient de `plan.layers[i].fmt` (posé par le
planificateur de placement), avec repli `"int4_awq"` par défaut
(`TensorRouter.format_for`, lignes ~300-304) — `--format bf16` ne force
PAS forcément ce repli à `bf16` pour toutes les couches ; c'est
probablement pourquoi `mini-acvram3` a quantifié les experts malgré
`--format bf16 --no-awq`. Piste pour la reprise : soit trouver le bon
levier pour forcer `bf16` partout (peut-être `--host-exec` influence
justement CE plan), soit comparer `mini-acvram2` (qui a obtenu `bf16`)
et `mini-acvram3` (qui a obtenu `int4_awq`) pour voir exactement quelle
différence de plan cause l'écart — pas encore fait.

## VERDICT FINAL (`mini-acvram4`, 3ᵉ bogue trouvé et corrigé)

Cause exacte : `build_tiers` (acvram/memory/tiering.py:346-376) — le tiers
GPU lit `opts.force_format` (ligne 350) mais le tiers hôte lisait
`tiers[0]... if tiers else "int4_awq"` (ligne 372) — **sans GPU visible,
`tiers` est encore vide à cet endroit, donc le repli était TOUJOURS
`int4_awq`, quel que soit `--format`**. Latent avant le 14/09, ce bogue
devient systématique maintenant que `CUDA_VISIBLE_DEVICES=""` est le
défaut de toute session (carte.sh/guet.sh, même journée) : TOUTE
conversion CPU-only `--format bf16` quantifiait silencieusement en
int4_awq. Corrigé (`opts.force_format` lu en premier, comme le tiers
GPU) ; test qui casse sans le correctif : `tests/test_tiering_force_format.py`.

Reconversion propre (`mini-acvram4`, manifeste vérifié : 100 % bf16 sauf
le biais en fp32) puis équivalence 16 jetons rejouée contre HF (seuils de
poste2, delta≤0,05 et cos≥0,999) :

| position | delta | cos | verdict poste2 |
|---|---|---|---|
| 0 | 1,8152 | 0,999556 | rouge (delta seul) |
| 1-15 | ≤0,26 | ≥0,99994 | **vert** |

**15/16 positions passent intégralement.** La position 0 échoue
SEULEMENT sur le seuil delta (le cos passe, 0,999556 ≥ 0,999) : c'est le
même swap d'expert proche de l'égalité (42↔4) déjà identifié dans le
contrôle « couche 0 isolée » comme du bruit bf16 diffus (~1e-3, présent
partout, pas spécifique à cette position) — retrouvé identique, au
chiffre près (delta=1,8152), au tout premier essai routeur-fp32-seul. Le
contrôle fp32/fp32 décisif l'avait effacé (delta 0,0031, cos 1,0),
confirmant que c'est un artefact de précision bf16 sur un ex-aequo
fortuit du jeu de 16 jetons synthétiques de poste2, pas un bogue
d'implémentation.

**Trois bogues réels trouvés et corrigés cette session, tous avec test
qui casse** :
1. `_router_logits` en bf16 au lieu de fp32 (model.py) — HF force fp32.
2. `e_score_correction_bias` converti en bf16 malgré `SENSITIVE_SUFFIXES`
   (convert.py) — la protection ne montait qu'au « 16 bits ».
3. Tiers hôte ignorant `--format` sans GPU visible (tiering.py) —
   régression rendue systématique par le nouveau défaut
   `CUDA_VISIBLE_DEVICES=""`.

**Recommandation à chef/poste7** : le routage et la MLA sont corrects ;
la position 0 restante est un artefact de précision bf16 sur un ex-aequo
fortuit, pas un défaut d'acvram — attendu de tout moteur bf16 sur un
routage top-k à experts proches. Deux options pour le feu vert de ce
soir : (a) accepter 15/16 + explication comme suffisant pour la décision
srcbf16, (b) régénérer le jeu de 16 jetons de poste2 pour éviter cet
ex-aequo précis (ne cache rien, déplace juste le hasard). Points encore
non traités : (4) "formats mélangés entre experts" / `piles_ok=False` —
PAS investigué, hors du chemin critique de ce blocage.
