# Scellé — pièce 177 : le coût des préfills au banc chat servi, alias mixte b=8 (poste5, 25/09 03 h, AVANT la prise)

Ordre de chef, suite de la 173 d'poste1. Au banc chat (lots de 8 requêtes simultanées, 256 jetons, invite fixe), le
débit servi est de 326 t/s, contre 402 en décodage pur (19,90 ms/pas). Un lot dure 2 048 / 326 = 6,28 s, dont
256 × 19,9 ms = 5,09 s de décodage : il reste **≈ 1,19 s par lot hors décodage**, soit les ≈ 4,6 ms/pas de la 173.

**Lecture du code** (`runner.py:_step`, l. 1550-1630) : les séquences admises au même pas sont préfillées EN UN LOT si
aucune n'a de frontière d'instantané (invites courtes : aucune), sinon une par une. Pendant un préfill, aucun décodage
ne tourne dans le pas. Au banc, les 8 requêtes d'un lot partent ensemble et finissent ensemble : la 150 mesurait
1,7 pas d'admission par lot. `prefill_seconds` est un temps HÔTE sans synchronisation, sauf sous
`ACVRAM_CHRONO_SYNC=1`.

**Instrument** (`scratchpad/poste5-p177-25-09/prise.sh`) : (1) serveur mixte, banc de la 102, 3 passes (N1, N2 et
SYNC), `/metrics` avant et après : pas, pas avec préfill, jetons de préfill, secondes de préfill et de décodage, mur ;
(2) `iso177.py` : préfill isolé en processus à la longueur relevée, 8 × L groupé, 1 × L, 7 × L puis 1 × L, LOT=1, et
le compteur de B' ; temps hôte contre temps carte.

**Prédictions** :
* invite L ≈ 30-45 jetons (gabarit de chat + texte) ; 1 à 2 pas de préfill par lot ; ≈ 0,16 lot/s (6,3 s par lot),
  soit 0,16 à 0,32 pas de préfill par seconde et 1,3 requête préfillée par seconde ;
* coût réel du préfill (SYNC) : **0,6 à 1,0 s par lot**, soit 50 à 85 % des 1,19 s ; le reste est l'aller-retour HTTP
  et l'admission entre deux lots (0,1-0,4 s) ;
* préfill isolé 8 × L : 0,5 à 0,9 s ; dans les ±20 % du préfill servi ; 7 × L puis 1 × L ≈ 8 × L + 30 à 60 % ;
* arrêt des décodages : quasi nul au banc (les lots sont synchrones) ; sauf admission en deux pas, où 7 séquences
  attendent le préfill de la 8e (≤ 10 % des 1,19 s) ;
* leviers : **B' ne s'applique PAS au mixte** (projections GDN en int8 par canal, sans déquantification ; compteur
  prédit : 0) ; LOT=1 (pas au bit) −55 à −65 % du préfill (150 bis : 1,36 → 0,55 s par lot) ; graphe de préfill : la
  part hôte du préfill (hôte / carte) dit s'il est borné par les lancements ; préfill par tranches intercalé : sans
  objet pour des invites de 40 jetons.
* Issue qui me gênerait : un préfill servi (SYNC) bien plus court que 0,6 s ; les 1,19 s seraient alors surtout HTTP ou
  admission, et aucun levier moteur de préfill ne paierait.
