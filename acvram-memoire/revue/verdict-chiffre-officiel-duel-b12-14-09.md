# Verdict — chiffre officiel du duel b=12 : 712,6 t/s, la traîne n'explique PAS l'écart

poste3, 14/09/2026. Suite à
[`protocole-chiffre-officiel-duel-b12-14-09.md`](protocole-chiffre-officiel-duel-b12-14-09.md).
Les deux prédictions écrites d'avance sont **réfutées** — rapporté tel quel.

## Montage

Qwen3-Coder-30B-A3B-nvfp4, `outils/gpu/mesure/energie.py` (identique octet
pour octet entre `travail/poste3` et `anticitoyen-vram`, vérifié par
`diff`), carte exclusive, `ACVRAM_TYPE=mesure`, repos 8 s (`banc-horloge-
decodage.py` d'origine), **acvram importé depuis `anticitoyen-vram`
(main), PAS `travail/poste3`** — vérifié par `acvram.__file__` dans le
journal.

## Chiffres

    bras                              tok/s    J/jeton net   bridage
    A — EOS actif (montage historique)  712,62   0,4807        puissance
    B — EOS neutralisé (comme
        banc_decodage_moe.py:34)        745,62   0,4487        puissance

    ratio B/A = 1,0463 (+4,63 %)

## Prédiction 1 : RÉFUTÉE

J'attendais un chiffre proche de 568,6 (le régime décodage n'aurait pas
changé). **Il vaut 712,62 t/s — +25,3 % au-dessus de 568,6**, avec le
MÊME script, le MÊME modèle, la MÊME carte, la seule différence étant le
code chargé (`main` d'aujourd'hui contre le commit qui a produit 568,6).
**`main` a bien changé le chemin décodage b=12** depuis cette mesure —
malgré ma lecture initiale (aucun des correctifs cités n'était censé être
actif par défaut). Je n'ai pas isolé LEQUEL des correctifs récents (masquage
des créneaux fantômes d'poste1, +11 % annoncé et cohérent en ordre de
grandeur ; MoE MMA par défaut de poste2 ; autre) explique ce gain — hors du
périmètre de cette mesure, qui visait le chiffre agrégé, pas son
décomposition.

## Prédiction 2 : RÉFUTÉE

J'attendais un facteur ≥ 1,3 entre EOS actif et neutralisé si la traîne de
fin de lot expliquait l'écart avec les 826 t/s de `banc_decodage_moe.py`.
**Le facteur mesuré est 1,046** — la neutralisation de l'EOS gagne 4,6 %,
pas les ~15-45 % qu'il faudrait pour combler l'écart entre 712,6/745,6 et
826. **La traîne de fin de lot n'est PAS l'explication principale de
l'écart avec `banc_decodage_moe.py`.**

## Ce qui reste ouvert

`banc_decodage_moe.py` diffère de `banc-horloge-decodage.py` sur PLUSIEURS
axes à la fois, pas seulement l'EOS : `enable_cuda_graphs` **désactivé par
défaut** (`BANC_GRAPHES=0`, contre-intuitif qu'un chemin sans graphe soit
plus rapide — à vérifier), `enable_prefix_cache=False`, `BANC_JETONS=64`
(contre 200 ici — une fenêtre 3× plus courte est plus sensible à un effet
de bord de démarrage/chauffe), et un lot construit différemment. Je n'ai
isolé qu'UNE variable (l'EOS) et elle ne suffit pas — la source du 826
reste à départager entre les autres axes, non mesurés ici.

## Le chiffre officiel

**712,62 t/s / 0,4807 J/jeton net, sur `main` d'aujourd'hui, protocole
`banc-horloge-decodage.py` cellule « défaut », EOS actif** — c'est le
chiffre qui remplace 568,6 pour toute comparaison future (duel vLLM inclus :
le ratio vLLM/acvram passe de ×2,11 à ×1,68 avec ce chiffre, sans nouvelle
mesure vLLM — à confirmer si le duel est republié).

## bd

Chiffre officiel produit. Écart 568→826 PAS refermé : la traîne EOS
explique 4,6 points sur ~45, le reste (main vs commit historique, ~25 pts ;
graphes/N_JETONS/prefix-cache dans banc_decodage_moe, non mesuré) reste à
départager. Ne pas répéter la prédiction traîne=explication-principale sans
nouvelle mesure — celle-ci l'a réfutée une fois.
