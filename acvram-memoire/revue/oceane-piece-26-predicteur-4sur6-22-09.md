# Pièce 26 — prédicteur « Four Over Six » hors ligne : G est positif par construction, la classe gagnante est l inverse de ce qu on dit, et le prédicteur MSE ne prédit pas la qualité (22/09, Océane, à sec)

Instrument : `outils/gpu/mesure/predire-4sur6.py` (test `tests/test_predire_4sur6.py`, 3 verts, à sec).
Méthode : poids bf16 de la source, blocs de 16 sur l axe K, **fonctions du dépôt** (`_coder`, `_levels_tensor` de `acvram/quant/nvfp4.py`) — mêmes blocs, mêmes arrondis E4M3 (échelle) et E2M1 (quartets) qu à la conversion ; critère du 21/09 : G = (Σ MSE6 − Σ MSE4) / Σ MSE6 par tenseur, par couche et global, pondéré par les blocs.

## 1. Ce que le prédicteur MSE ne peut pas faire, dit avant la mesure
`quantize_nvfp4` retient par bloc le **meilleur des deux au sens de la MSE** (`nvfp4.py:302`, comparaison stricte, égalité → max6). Donc **G ≥ 0 par construction** : un G positif ne prédit rien, il récite la règle de sélection. Le prédicteur n a de contenu que par ce que la MSE ne voit pas : l erreur **maximale** par bloc (L∞) et la **saturation** de l échelle amax/4 en E4M3.

## 2. Mesure (6 tenseurs du 31B bf16-vision, 26,2 M blocs, à sec, indicatif — le passage complet est une commande)
| grandeur | valeur |
|---|---|
| G global (MSE) | **+0,157** |
| G global (L∞) | +0,150 |
| blocs où amax/4 est retenu | 43,5 % |
| blocs où l échelle amax/4 sature (≥ 448) | 0,00 % |
| blocs où l erreur **maximale** augmente | **2,69 %** |
Stable d un tenseur à l autre (k_proj, o_proj, gate, up, projection vision : G 0,158-0,169, L∞ pire 2,8-2,9 %).

## 3. Classe des blocs gagnants — MESURÉE, et à l envers de l intuition courante
amax/4 gagne sur les blocs **PLATS** (seize valeurs du même ordre : platitude moyenne(|w|)/amax élevée) ; sur un bloc **PIQUÉ** (une valeur écrase les quinze autres) il ne gagne **jamais** — 0,0 % sur le tenseur synthétique du test, contre 68 % sur le tenseur plat.
Raison, vérifiable à la main : les niveaux E2M1 hauts sont espacés (…, 2, 3, 4, 6). À l échelle amax/6, une valeur à 0,8·amax tombe entre 4 et 6 → erreur ≤ 0,167·amax ; à amax/4 elle tombe entre 3 et 4 → ≤ 0,125·amax. Sur un bloc piqué au contraire, amax/6 donne le pas fin (0,5·s = amax/12 contre amax/8) là où sont les quinze petites valeurs, et le pic reste exact (6·s = amax).
**Les modèles extérieurs qui annoncent « amax/4 gagne sur les queues lourdes » ont le signe à l envers** ; c est la contradiction que la Maîtresse demandait de trancher. `platitude_gagnants` / `platitude_perdants` publient la mesure par tenseur.

## 4. Verdict, jugé contre le scellé E (KL 4sur6 **1,39** > max6 recalibré **0,87** : 4sur6 PERD)
Règle écrite avant : « réfuté s il prédit 4sur6 gagnant globalement ».
Le prédicteur rend : G = +0,157 > 0 **par construction**, et une dégradation réelle sur 2,69 % des blocs (L∞), sans saturation. Il est donc **D ACCORD** avec la mesure au sens strict (il nomme le canal par lequel 4sur6 perd), mais le chiffre utile n est pas G : **la MSE par bloc n est pas un prédicteur de la qualité**, et un G de +15,7 % lu seul aurait annoncé un gain là où la KL mesure une perte. Ce qui reste utilisable hors ligne : la part de blocs à L∞ dégradée (2,69 % ici) et la saturation (0 ici — le clamp n est pas la cause).
Ce qui rendrait « faux » l instrument : G < 0 (sélection différente de celle du convertisseur → `INVALIDE`) ; L∞ pire nulle partout avec G > 0 → `RÉFUTÉ` (il prédirait 4sur6 gagnant).

## 5. Reste
Passage complet des 31B (≈ 10 min processeur, `--limite 0`) pour G par couche et la platitude agrégée ; lien quantitatif entre « 2,69 % de blocs à L∞ dégradée » et « KL 1,39 contre 0,87 » **non établi** — corrélation nommée, pas causalité chiffrée.
