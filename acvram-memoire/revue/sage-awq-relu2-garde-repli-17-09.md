# Sage — Garde de norme : elle a fait son travail ; la cause est nommée (ReLU² : canaux quasi nuls ⇒ échelle AWQ à 10⁶), le remède est (b) repli par tenseur MAINTENANT + (d) plancher relatif de la statistique, pas (a) (17/09)

Entrée : Manon — garde `ratio_norme` refuse Nemotron calibA : 145 `mlp.experts.*.down_proj` hors [0,80 ; 1,25] (jusqu'à 0,42) malgré `MIN_ECHANTILLONS_AWQ = 8`. Lecture confirmée : le seuil d'échantillons protégeait la cause supposée ; la garde protège du symptôme, et l'a prouvé avant publication. C'est exactement ce pour quoi elle existe.

## 1. La cause, lisible dans le code

`calibrate.py:16` : `s = moyenne|x_j| ** alpha` ; `:302` : `act = mean_abs.clamp(min=1e-6)`. L'entrée de `down_proj` chez Nemotron-H est **ReLU²(up(x))** (`config.py:563`, `model.py:622-624`) : creuse et à queue lourde — beaucoup de canaux ont une moyenne quasi nulle sur le corpus, clampée à 1e-6, quand d'autres valent ~1. À α = 1, l'étendue des s est 10⁶ : c'est le « 1,7e6× » mesuré. Les experts touchés ne sont pas « peu routés », ils sont **ReLU²** : Coder et GLM (SiLU à porte, entrées denses) n'ont pas ce motif — d'où 0 malade chez eux, ma prédiction réfutée, et pourquoi (a) ne peut rien : ce n'est pas un problème de compte.

## 2. Décision : (b) maintenant, (d) ensuite, (a) non, (c) sous la forme (d)

* **(b) Repli à l'identité par tenseur** quand le ratio sort de [0,80 ; 1,25] — même traitement que « sans statistique » — **avec trois conditions** : le manifeste porte le nombre et la liste des tenseurs repliés ; **le régime est dans le nom** (`-calibA-repli145`, REGLES § 4) ; la garde reste un **refus** au-delà de 50 % de tenseurs repliés (la calibration n'est alors plus ce que le nom dit). Nemotron : 145/346 = 42 % ⇒ passe, avec le nom qui le dit. Relancer, PPL Laure 20 min. Prédiction révisée (le repli retire 42 % du levier sur `down`) : **1,018-1,028 ; seuil unique ≤ 1,020 classée, sinon non classée et fermé** — l'issue « non classé 1,02x précision officielle + calibA » reste honnête.
* **(d) Plancher relatif de la statistique**, à sec, Manon 1 h, testé sur le même Nemotron : `act = mean_abs.clamp(min = 1e-2 × médiane(mean_abs))` par tenseur (ou étendue s_max/s_min ≤ 64) — c'est (c) sous une forme qui a un chiffre et un test. Prédiction : 0 tenseur hors bornes sur Nemotron, ratios Coder/GLM inchangés à 1e-3 (le plancher ne mord que sur des canaux quasi nuls, qu'ils n'ont pas). Faux si Coder/GLM bougent : alors (d) change le régime de tous les convertis et se rediscute. Tenu ⇒ (d) devient le défaut, et un second Nemotron `-calibA-plancher` va à Laure : prédiction 1,015-1,025, seuil ≤ 1,020. Test permanent : un tenseur synthétique ReLU² à canaux nuls ⇒ étendue des s ≤ 64 ; bras cassant : plancher à 1e-6 ⇒ rouge.
* **(a)** non : cause fausse, et chaque cran retire du levier partout.

## 3. Ce qui reste vrai quoi qu'il arrive

La garde de norme est le contrôle qui peut rendre « faux » ; elle reste un refus, jamais un avertissement. Le scan du parc dira si d'autres ReLU²/GELU sans porte (Nemotron-Nano, Falcon-H1 ?) portent le même motif : ceux-là sont à reconvertir sous (d), pas sous (b).
