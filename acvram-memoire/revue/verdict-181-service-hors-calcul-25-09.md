# Verdict — 181 : le temps hors calcul d'un lot au banc chat b=8, alias mixte — 25/09 (poste1)

* **instrument** : `scratchpad/poste1-p181-25-09/serveur-trace.py` (acvram serve INCHANGÉ, horodaté de l'extérieur : pas du
  moteur, submit, middleware HTTP) + `banc-trace.py` (copie horodatée du banc de la 102) ; perf_counter commun ; passes N
  (sans trace) et T (avec) alternées ; `analyse.py`. Aucun nsys (p69 : il rend synchrone le lancement de notre graphe).
* **commit** : prise poste1-p181 (05:1x-05:28), scellé `scratchpad/poste1-p181-25-09/scelle.md` écrit avant.
* **instrument neutre** : N 326,3 t/s (et N 1re passe), T 329,1 et 325,6 t/s (le banc de la 173 donnait 326,0).
* **mesuré** (médianes des 4 lots de mesure de chaque passe T ; 257 pas par lot) :

| poste | ms par lot |
|---|---|
| **durée du lot** | 6 230-6 293 |
| somme des pas (préfill ET décodage) | 6 219-6 284 |
| **hors pas, total** | **9,8** (0,16 % du lot) |
| … boucle moteur entre les pas (livraison, sommeils) | 5,1-5,5 |
| … HTTP → dernier submit (gabarit, tokenisation des 8) | 3,1 |
| … dernier submit → 1er pas | 1,1-1,6 |
| … dernière fin → réponse HTTP | 0,6 |
| … réponse → client ; client entre lots ; envoi → HTTP | 1,0 ; 0,25 ; 0,4 |

* **verdict** : prédiction du total **FAUSSE** (9,8 ms contre 0,10-0,30 s prédits). La boucle entre les pas est bien le
  premier poste hors pas (52-56 %, prédit ≥ 40 %), mais sur 10 ms. **La prémisse des « 0,26 s hors calcul » était fausse** :
  elles ne sont pas hors calcul, elles sont DANS les pas de préfill.
* **où sont-elles** : chaque lot a 2 ou 3 pas lourds : 435 ms (2 séquences, 6 en attente) puis 939 ms (les 6) ; ou 717 +
  650 ms (4 + 4) ; parfois un pas de ≈ 90 ms en plus. Les 8 requêtes du banc arrivent en ≈ 3 ms ; le moteur, réveillé par les
  premières, préfille 2 ou 4 séquences, et les autres attendent le pas suivant. Préfill servi 1,30-1,46 s par lot, contre
  0,907 s pour 8 × 92 groupées en un pas (177, isolé) : **≈ 0,4-0,5 s par lot** tiennent à l'admission en DEUX pas, plus que
  les 0,10 s estimées par la 177 sur le cas 7 + 1. C'est le terrain de poste5 (179, admission en un pas) : rien touché ici.
* **reste nommé** : un pas de ≈ 90 ms sans séquence en attente (run 8 ou 0, att 0) dans 2 lots sur 4 ; non attribué.
* **contre NInfer** : banc − noyaux ≈ 1,3 ms/pas chez eux ; chez nous, le vrai service hors pas vaut 9,8 ms / 257 pas
  ≈ **0,04 ms/pas**. L'écart de « service » tient au préfill et à son admission, pas à HTTP.
* **dumps** : traces hors git (dossier de session).
