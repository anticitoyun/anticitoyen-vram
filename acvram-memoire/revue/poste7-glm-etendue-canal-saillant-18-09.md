# poste7 — Étendues GLM : 9/14 = échantillonnage (tenu), 5/14 = autre chose ; hypothèse canal massif isolé ; en observation, pas de chantier, avec un déclencheur écrit (18/09)

Entrée : chef/poste2 — corpus six phrases rejoué : 9/14 tenseurs de k48 à `n_samples` < 8, **5/14 à ≥ 8** (pire : `layers.46.experts.42.down_proj`, n = 54, étendue 16 457-81 391 sur plusieurs convertis). Scellé 14/14 **réfuté**. Classé `k48-calibA` : 0, aucun chiffre publié touché.

## 1. Ce que je retire, ce qui reste

Réfuté : « tout vient de l'échantillonnage ». Tenu pour 9/14 (le seuil 8 les couvre depuis 2a68aa6). Pour 5/14, avec 54 jetons, `mean_abs` estime quelque chose de réel : le tenseur récurrent est le même sur quatre convertis et deux corpus — c'est une propriété du **modèle**, pas de la calibration. Je garde « classé propre » mais je retire la conclusion « rien à comprendre ».

## 2. Hypothèse : un canal d'activation massif à l'entrée de `down_proj`

Le motif est connu et documenté : activations massives sur quelques canaux fixes à l'entrée de `down_proj`, sur des couches précises, valeurs 10²-10³× la médiane (Massive Activations, arXiv 2402.17762 ; DuQuant 2406.01721, cité par les trois modèles web dans `duck-poste2-12-09` : couches 1, 30, 31 de Llama-2-7B, > 1 000). Sur un MoE, l'expert (46, 42) est la version routée du même phénomène. AWQ fait alors exactement ce pour quoi il est conçu — `s_k ∝ mean_abs_k^α` — et une seule valeur à 10³× donne une étendue 10³-10⁵ selon α, bornée seulement par le clamp **par valeur** (`calibrate.py:354, 450` : [1e-4, 1e4]), jamais par l'étendue. Ce n'est pas le mécanisme Nemotron (médiane ≈ 0 par ReLU²) : ici le **maximum** est anormal, pas le plancher.

Conséquence sur la quantification, à écrire pour savoir quoi mesurer : `W[:, k] · s_k` avec `s_k` énorme domine chaque groupe NVFP4 de 16 le long de K qui contient k — les 15 autres poids du groupe tombent à ~0, sur toutes les lignes. Le repli identité (garde 4096, `convert.py:1434`) évite ça mais laisse le canal massif multiplier un poids quantifié sans protection. Entre les deux, **plafonner** `s ≤ 4096 · s_min` garde la protection sur les autres canaux. Aucune des trois options n'a de différence mesurée : k48 (14 tenseurs) ≈ calibA (0) à ≤ 0,002 — sur GLM, ce tenseur ne pèse pas sur le privé.

## 3. Contrôle à sec (poste2, 15 min, sur ses rapports existants), scellé avant

Pour les 5 tenseurs : rapport `s_max / s_(2)` (deuxième plus grande échelle) et nombre de canaux à `s > 100 · médiane(s)`. **Prédiction : 5/5 ont ≤ 3 canaux au-dessus de 100× la médiane** (canal massif isolé). Faux si < 5 : l'étendue est étalée sur beaucoup de canaux, ce n'est pas une activation massive, et là on ouvre (lecture de `mean_abs` brut du tenseur (46, 42)). Issue qui me gênerait : 5/5 mais le canal n'est pas le même d'un converti à l'autre — alors ce n'est pas une propriété du modèle et l'hypothèse tombe malgré le compte.

## 4. Décision : observation, déclencheur écrit

Pas de chantier : gain mesurable nul sur le seul modèle où le motif existe. Déclencheur qui rouvre, sans discussion : **un converti futur avec `tenseurs_replies` > 0 non classé (> 1,020)** ⇒ un bras `-plafonne4096` (cap au lieu de repli, 20 lignes dans `convert.py:1436`) avant de fermer ce modèle. Sans ce cas, on n'y touche pas. Le manifeste doit porter la liste des repliés (déjà demandé, `poste7-glm-etendue-historiques-18-09` § 2).

## 5. Point séparé, .deb 0.6.10 (chef, message précédent)

Un `pip install` dans le `.venv` partagé pendant une campagne est un changement de régime invisible, même classe que fusionner main pendant une passe (REGLES § 1). Deux lignes : (a) installation dans le venv **hors fenêtre de carte** seulement ; (b) `regime_ligne()` (`regime.py`) porte `torch/triton/fla` versions (`2.14.0+cu130 / 3.8.0 / 0.5.2` aujourd'hui) — elle ne les porte pas, vérifié par grep ; une mesure dont le JSON ne les a pas ne distingue pas un noyau Triton d'une autre version.

## Ordre

* poste2 : § 3 sur les rapports existants → `verdict: revue/verdict-glm-canal-massif-18-09.md — n/5, même canal oui/non`.
* chef : § 5 (b) dans `regime.py`, à sec, test qui lit les trois versions dans la ligne ; (a) est une règle, à ajouter à REGLES § 2 en une ligne datée. Chantier GLM étendue : **fermé en observation**, déclencheur § 4.
