# Sage — Étendues > 4096 sur les GLM historiques : pas un chantier, un contrôle de 30 min ; hypothèse = experts à 1-7 jetons sur le corpus six phrases, avant `MIN_ECHANTILLONS_AWQ` (18/09)

Entrée : Jérôme — scan Manon 118/118 ; GLM (SiLU à porte, sans ReLU²) : `srcbf16-nvfp4` 1 tenseur 81 391 ; `avant-echelle-fix`/`avant-mtp-fix`/`avant-noawq-experts`/`k48` 14 tenseurs chacun, max 16 457, même tenseur récurrent couche 46 expert 42 ; `k48-w4a4` 52 (17 875) ; `k48-calibA-verif17-09`/`k48-calibB` 3 chacun (5 188-16 723). **`k48-calibA` classé : 0.** Manon sans hypothèse.

## 1. Hypothèse, et pourquoi elle n'est pas « le corpus » en général

Le motif n'a pas besoin de ReLU² : `n_samples` est un compte de **jetons** (`calibrate.py:156`, `flat.shape[0]`), et `MIN_ECHANTILLONS_AWQ = 8` n'existe que depuis 2a68aa6 (17/09 18:13, `convert.py:1364`). Tout converti antérieur a reçu une AWQ sur les experts vus par **1 à 7 jetons** — `mean_abs` sur trois jetons d'un produit SiLU(g)·u est creux par construction, et l'étendue des `s` explose sans aucune non-linéarité pathologique. Le corpus six phrases (~230 jetons × 16 répétitions, les répétitions n'ajoutant rien) routait 558 experts à zéro (`verdict-calib-k48-A`) : la tranche 1-7 juste au-dessus est mécaniquement peuplée, et **déterministe** — même corpus, même routage, même expert (46, 42) dans les quatre convertis. La récurrence est la signature du corpus, pas du modèle.

Ce que l'hypothèse explique aussi : `verif17-09` et `calibB` (bras A/B, 3 tenseurs à ≤ 1,7e4) sont **après** le seuil 8 ou juste à sa frontière — 8-15 jetons restent une statistique bruitée, le seuil est lâche, et c'est la garde d'étendue (f9b6190, `convert.py:1434-1437`, repli identité) qui couvre cette tranche désormais. `k48-w4a4` (52) : même corpus, activations quantifiées avant la statistique — plus de zéros, plus d'étendue ; pas de contrôle à faire, le converti est réfuté par ailleurs (`verdict-glm-k48-w4a4-prefill-16-09`).

## 2. Pourquoi ça ne touche aucun chiffre publié, et pourquoi 4096 reste

Dose-effet observé : Nemotron 3-6 tenseurs à 2,7e5-1,7e6 → +0,26 à +0,43 PPL ; GLM `k48` 14 tenseurs à 1,6e4 → PPL indiscernable de `calibA` à 0 tenseur (« calibration pas le levier sur GLM », ETAT), `verif` 3 tenseurs → Δ −0,0017. L'écart entre casse qui se voit et casse qui ne se voit pas est de **un à deux ordres de grandeur d'étendue**, et il faut que l'expert soit routé sur le corpus privé. 4096 est donc conservateur de ×4 sous la première étendue GLM et de ×60 sous la première étendue nocive ; son coût est un repli identité, que le résultat GLM dit gratuit. On ne le bouge pas. Note : dans le code la garde **replie** (`tenseurs_replies`), elle ne refuse pas comme je l'avais écrit — acceptable, à condition que le manifeste porte la liste (à vérifier par Manon, une ligne).

## 3. Contrôle (Manon, à sec, ≤ 30 min), scellé avant

Statistiques du corpus six phrases sur GLM, à sec (`collect.py` par défaut, CPU) ; pour les 14 tenseurs de `k48`, lire `n_samples` de l'expert. **Prédiction : 14/14 ont `n_samples` < 8. Faux si < 14** : alors un expert bien échantillonné produit une étendue > 4096 sur SiLU, le mécanisme n'est pas l'échantillonnage, et on ouvre le chantier (lecture de `search_channel_scales` sur ce tenseur précis). Issue qui me gênerait : 14/14 mais `verif17-09` ou `calibB` a un tenseur > 4096 avec `n_samples` ≥ 32 — le seuil 8 est faux d'un facteur 4 et la garde d'étendue porte tout ; à noter, pas à corriger aujourd'hui. Manifestes : dater chaque converti historique par son commit de conversion (manifeste obligatoire seulement depuis abe108c ; sinon mtime du dossier) — 8/8 doivent être antérieurs à 2a68aa6, sinon même conclusion que « faux ».

Convertis historiques à étendue > 4096 : non servis, hors menus, listés avec sha256 dans le verdict de Manon ; suppression du disque = décision utilisateur.

## Ordre

* Manon : contrôle § 3 (n_samples des 14 tenseurs de k48 sur le corpus six phrases + date de conversion des 8 convertis) → `verdict: revue/verdict-glm-etendue-historiques-18-09.md — <14/14 ou n/14>` ; une ligne sur le manifeste des repliés.
* Jérôme : rien à republier ; chantier non ouvert sauf verdict < 14/14. Le scan du parc est clos.
