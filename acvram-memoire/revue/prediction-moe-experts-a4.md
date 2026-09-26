# Prédiction — A4 restreint aux projections MLP/experts (recadrage de chef, 13/09)

## Mesure 1 — Llama-2-7B, gate+up+down (A4), q/k/v/o en A16

Régime `moe-proj-seul` de `outils/smoothquant-a4-sweep.py`, sur
`Llama-2-7b-nvfp4` (dense, même dossier que toutes les mesures A4/A8
précédentes). Réutilise `installer_hooks_genres` déjà testé (2 tests dans
`tests/test_hooks_activations_a4.py`, dont la couverture exacte
`3 × len(model.layers)` vérifiée sur le modèle jouet).

Seuil inchangé : PPL ≤ 5,6663 (étalon A16 = 5,6102).

**Prédiction chiffrée** : down_proj seul coûte +0,557 % (mesuré). gate et
up n'ont jamais été isolés — mais leur activation d'entrée est la MÊME
que celle de down_proj (`x`, l'entrée du bloc MLP), alors que
l'activation d'entrée de down_proj est `act = SiLU(gate(x)) * up(x)`,
déjà transformée et potentiellement plus difficile. Je prédis donc que
gate+up ajoutent MOINS que down_proj seul individuellement, mais que les
trois ENSEMBLE dépassent le seuil : **PPL entre 5,68 et 5,80** (écart
+1,0 % à +3,4 %), **échec probable du seuil**, mais nettement moins
sévère que le A4 complet à 7 genres (+2,58 %).

**Issue qui me gênerait** : que ce régime RESPECTE le seuil — cela
signifierait que la quasi-totalité du coût du A4 nu à 7 genres provient
de l'ATTENTION (q/k/v/o), une hypothèse jamais posée jusqu'ici et qui
changerait la priorité des chantiers suivants (il faudrait alors chercher
un lissage ciblé sur l'attention, pas sur le MLP).

## Mesure 2 — cible réelle : A4 restreint aux projections d'experts (Qwen3-Coder-30B-A3B)

poste2, 13/09/2026, avant mesure. Recadrage : le noyau MMA de poste4
(`nvfp4_gemm_grouped_mma`) ne sert QUE le chemin MoE groupé (gate/up/down
des experts) — jamais q/k/v/o. Deux mesures posées par chef.

## Ce qui est livré, à sec

* `outils/hooks_activations_a4.py::installer_hooks_moe_experts` :
  monkeypatch `MoEBlock._grouped` (les experts n'appellent JAMAIS
  `QuantLinear.forward()` — vérifié en lisant `_forward_grouped`,
  `acvram/engine/model.py:873-913` : le chemin groupé appelle
  `self._grouped(x32, pile, eid, tok)` directement sur les poids
  empilés). Masque temporairement `nvfp4_gemv_grouped_gateup` sur
  l'extension CUDA pour forcer le repli (gate et up séparés, chacun
  passant par `_grouped`) — sans ce masquage, le noyau fusionné
  bypasserait `_grouped` pour gate+up et ne serait jamais fake-quantifié.
* `tests/test_hooks_moe_experts.py`, 5 tests sur des objets FACTICES
  (aucun modèle MoE jouet n'existe dans ce dépôt — `conftest.py` ne
  construit qu'un Llama dense) : l'entrée reçue par `_grouped` est bien
  fake-quantifiée pendant le hook, le noyau fusionné est bien masqué
  (les autres attributs de l'extension restent accessibles), la
  restauration remet tout en place.
* `outils/moe-experts-a4-qwen3-coder.py` : campagne à 2 régimes (a16
  témoin, moe-a4) sur `Qwen3-Coder-30B-A3B-nvfp4`, gardée derrière
  `--pour-de-vrai`, vérifiée à sec (compile, plan sans carte).

## Régime nommé

**Aucun étalon PPL n'est archivé pour `Qwen3-Coder-30B-A3B-nvfp4`** — ni
dans les campagnes quota/compte-égal (qui portent sur Llama-2-7B), ni
ailleurs dans `revue/`. Le témoin a16 est donc mesuré DANS LA MÊME
campagne, corpus wiki-gptq.txt (sha vérifié), fenêtre 2048/2048,
min_context 0 — même protocole que les campagnes précédentes. Seuil : PPL
moe-a4 ≤ PPL a16 mesuré × 1,01.

**Limite assumée** : mesure en un seul passage par régime, pas de
jumelles (le temps de carte est partagé — poste3, poste4, poste1 en
attente). Si le résultat est proche du seuil, une seconde passe est
recommandée avant de trancher `ACVRAM_MOE_MMA`.

## Prédiction chiffrée

**Réussite probable, avec plus de confiance que les mesures précédentes.**
Contrairement au A4 nu sur Llama-2-7B (7 genres, y compris l'attention,
qui n'a pas les mêmes statistiques d'activation que le MLP), ce régime
isole EXACTEMENT le sous-ensemble déjà mesuré comme le moins coûteux sur
Llama-2-7B : le MLP (down_proj seul = +0,557 %, un tiers du chemin ;
gate+up ajoutent une part inconnue mais le MLP entier reste, d'après la
mesure isolée de down_proj, structurellement moins fragile que
l'attention). Je prédis un écart entre **+0,4 % et +1,0 %** — proche du
seuil, sans certitude du côté où il tombera.

**Deuxième source d'incertitude, propre à ce modèle** : Qwen3-Coder-30B-
A3B est un MoE (30B total, ~3B actifs), architecture et régime
d'activation différents de Llama-2-7B dense — la prédiction ci-dessus
extrapole depuis un modèle dense et pourrait se tromper d'un facteur
significatif si les experts MoE ont un profil d'outliers différent du
MLP dense de Llama-2-7B (aucune donnée Bridging Gap ou DuQuant publiée
spécifiquement sur des experts MoE routés, à ma connaissance).

**Issue qui me gênerait** (règle 4) : un écart largement supérieur à
1,5 % — cela contredirait l'idée que le MLP (gate/up/down) est
structurellement le sous-ensemble le moins fragile, remise en cause déjà
partielle par le fait que down_proj seul ne représente que ~22 % du coût
A4 complet sur Llama-2-7B (donc gate+up pourraient porter une part non
négligeable, jamais isolée jusqu'ici).

## Addendum, 13/09 : mesure du noyau RÉEL, pas d'un fake-quant

`installer_hooks_moe_experts` s'est révélé inerte à `window=2048` (le
prefill de cette taille prend `_forward_prefill_grouped`, pas
`_forward_grouped`/`_grouped` — seuil `ACVRAM_MOE_GROUPED_MAX=32`,
model.py:980-983 — vérifié : PPL identique bit à bit entre témoin et
« moe-a4 », donc non publié comme mesure). poste4 signale mieux :
`ACVRAM_MOE_MMA=1` (main ≥ 6c99fa5) fait passer CE chemin de prefill par
`nvfp4_gemm_grouped_mma` — le VRAI noyau W4A4, pas une simulation :
active/désactivée par un flag process-level (`_MOE_MMA` figé à l'import
de `model.py`), donc deux processus séparés, pas un hook.

**Nouvelle prédiction, remplace celle du fake-quant ci-dessus** : même
fourchette qu'avant sur le fond (+0,4 % à +1,0 %), mais la confiance
augmente — c'est le noyau qui tournera en production, pas une
approximation. Vérification de mécanisme : débit mesuré doit différer
nettement entre les deux régimes (~10 500 j/s à L=2048 avec MMA contre
~8 600 sans, d'après poste4) — sinon le chemin MMA n'est pas pris et la
mesure ne vaut rien, comme pour le fake-quant.

## Ce qui reste, sur carte

`python outils/moe-mma-reel-qwen3-coder.py --pour-de-vrai --regime a16`
puis `--regime mma-reel` (deux processus séparés, `ACVRAM_MOE_MMA` figé
à l'import).
