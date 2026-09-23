# familles-noyaux sur l'alias qkv-alpha-22-09 — marlin presque complet (47/48), pas encore comparable — 22/09 (Manon)

* instrument : `nsys profile --cuda-graph-trace=node` + `familles-noyaux.py --detail proj_etroites_int8`, `ACVRAM_MODELE_MESURE=Qwen3-Coder-30B-A3B-nvfp4-qkv-alpha-22-09`, sha `cc0887aa`
* mesuré : **`experts_layout=marlin(47/48)`** — 47 couches sur 48 ont bien basculé en Marlin (nette amélioration vs `naturel` total de l'alias sans alpha commun), `chemin_moe=mma-a4(atteint=gemv_marlin+decode_mma)`. Mais **1 couche reste hors Marlin** : `glue_torch` explose à **0,956 ms/pas (583 lancements)** contre 0,043-0,044 ms/17 lancements en régime pleinement marlin — cette seule couche traîne un surcoût disproportionné. Pas complet : **8,083 ms/pas (mur 8,531)**, contre ~6,7-6,8 ms en référence -qkvo-i8c — **plus LENT de +1,3-1,4 ms**, à l'opposé de la prédiction (B −0,38 à −0,46 ms).
* verdict : **NON COMPARABLE tel quel** — 47/48 n'est pas 48/48 ; la couche restante en régime naturel domine le coût par sa glue torch (583 lancements pour 1 couche = beaucoup de petits noyaux élémentaires, signature d'un chemin non-graphé/non-fusionné). L'ABBA donnerait un résultat trompeur (B plus lent) qui refléterait cette couche isolée, pas le format qkv+marlin en général. Je n'ai pas lancé l'ABBA sur cette base.
* durée : ~2 min de carte

## Suite
Identifier la couche restée hors Marlin (probablement encore un expert sous-calibré malgré `--obs-min 0`, ou un cas limite de la fusion gate/up) avant l'ABBA — à Océane, ou je peux chercher si le groupe préfère. Sans cette dernière couche réglée, la pièce 42 reste ouverte.
