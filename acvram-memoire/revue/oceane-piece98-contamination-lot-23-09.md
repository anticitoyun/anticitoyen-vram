# Pièce 98 — KL 0,81 à b=4 en lot mêlé : BRUIT D'ORDRE du MoE groupé au préfill, pas une contamination — 23/09 (Océane)

* **instrument** : `scratchpad/oceane-p98-23-09/composition.py`, dérivé de kl-b.py (p81/p82/p97) : Coder qkvo-i8c, défaut du jour, eager, teacher forcing 8 pas, 5 dumps HF.
  * La séquence 0 est fixe, en ligne 0, dans six compositions : C1 seule ; C2 quatre copies ; C3 lot mêlé de la p97 (préfixes L−7/14/21) ; C4 C3 permuté ; C5 et C6 C3 aux mêmes longueurs mais au contenu étranger.
  * Écart = max |Δlogit| en ulp bf16 de max|logit| (critère du 15/09).
  * Prises 2 et 3 : chauffe, rejeu de chaque composition, bisection par sous-module (crochets, ligne 0 au décodage, lignes de la séquence 0 au préfill).
* **commit** : prises de1cb141 (1), 669466c7 (2), 1d9a629a (3). **régime** : éco 2 700 ; au début et à la fin, seul llama-server 4627 tourne (le relevé de fin de la prise 3 est manuel : ligne d'affichage de prise.sh cassée par ma substitution, mesure rc=0).
* **scellés** : `scelle.md`, `scelle2.md`, `scelle3.md`, chacun commité avant sa prise.
* **mesuré** :
  1. **Prise 1 invalide au scellé 1** : le rejeu de C1 n'est pas au bit sur la première invite jouée (sélection au premier appel). Avec la chauffe, le **rejeu vaut 0 ulp dans les 30 compositions** (prise 2) et les chiffres de la prise 1 sont reproduits au bit.
  2. **Écarts de la séquence 0** (max sur 5 invites × 8 pas) :
     * « forme seule » (C1–C2, C2–C3) : 37,5 ulp ;
     * « contenu seul » (C3–C5, C5–C6) : 34,1 ulp ;
     * permutation (C3–C4) : 10,6 ulp.

     Ces écarts sont du même ordre. Un seul argmax change (invite 3, C5–C6, pas 5, 7/8 contre HF des deux côtés).
  3. **Bisection au décodage** (C3–C5, invite 2) : premier écart à `layers.2.self_attn.o_proj`, 0,25 ulp. C'est la classe « contamination » du scellé 2 **au sens littéral**, mais ce scellé ne voyait que le décodage : `o_proj` (GEMM étroit W8A16, ligne à ligne) hérite de l'attention, qui lit le cache KV de la séquence 0 écrit au préfill. C'est dit au scellé 3 avant la prise 3.
  4. **Bisection au préfill** (prise 3) : C3–C5 et C3–C6 divergent d'abord à **`layers.0.mlp` (MoE), de 0,01 ulp**. `input_layernorm`, `qkv_proj`, `o_proj` et `self_attn` de la couche 0 restent égaux ; C3–C3 ne montre aucun écart.
  5. **KL contre HF par invite, min–max sur les 6 compositions** : 0,007–0,037 · 0,096–0,119 · **0,519–0,810** · 0,054–0,092 · 0,233–0,319. Sur l'invite 2, le contenu seul donne à lui seul C3 0,810 · C5 0,684 · C6 0,710.
* **verdict** : **BRUIT D'ORDRE, pas de contamination** (scellé 3, prédiction tenue : couche MoE, ≤ 1 ulp).
  * Aucune donnée d'une autre séquence n'entre dans le calcul de la séquence 0 : normes, projections et attention sont égales au bit jusqu'au MoE.
  * Au préfill, le MoE groupé forme ses groupes d'experts avec les jetons de TOUT le lot. La taille des groupes dépend donc du contenu des voisins, et avec elle l'ordre flottant de leur somme, soit 0,01 ulp à la couche 0.
  * Ce bruit traverse 48 couches, le cache KV et 8 pas, et devient 1-34 ulp sur les logits. Il ne change qu'un argmax sur 280 comparaisons contenu seul.
  * La KL de l'invite 2 est la plus sensible, déjà dans la 82 (0,41 à b=12) : elle erre de 0,52 à 0,81 selon le lot, avec le même défaut.
  * Le chemin exact (fichier:ligne) de la somme dépendante des groupes n'est **pas** établi. Les candidats lus sont la combinaison des experts par `atomicAdd` fp32 (`acvram_kernels.cu:5193`, `nvfp4_moe_fused_kernel`) et les sommes split-K des GEMV groupés (`:780`, `:852`), avec l'ordre des atomiques fixé par l'ordonnancement, qui dépend lui-même du routage. Ce serait à trancher par une capture à l'intérieur de `layers.0.mlp` si l'on voulait un mode « invariant au lot ». Je ne l'ai pas fait : ce n'est pas un défaut de justesse.
* **Critère de KL à lot mêlé juste pour l'avenir** (proposition, **non appliquée au passé**) :
  1. **Seuil absolu (0,74) à b=1 seulement.** C'est la seule forme sans composition. À lot mêlé, la KL d'une invite n'est pas une grandeur mais une distribution : 0,52-0,81 pour l'invite 2 avec le même moteur.
  2. **À lot mêlé, porte RELATIVE et APPARIÉE** : pour chaque invite et chacune d'au moins 3 compositions « contenu seul » (C3/C5/C6), KL(candidat, Cx) ≤ KL(témoin, Cx) + max(0,05 ; étendue du témoin sur ces compositions), avec le témoin joué dans la même prise. Invite 2 : étendue 0,126, soit une marge de 0,126 ; les autres invites gardent 0,05.
  3. **Rejeu obligatoire** : une composition jouée deux fois doit donner 0 ulp, sinon la prise est invalide ; chauffe jetée avant la première invite.
* **Conséquence** : l'échec « 4/5 » de la KL à b=4 dans la p97 n'accusait pas le défaut ; il accusait le seuil absolu appliqué à une seule composition.
* **durée** : ≈ 50 min ; carte 3 prises (22 s, 31 s, 24 s).
