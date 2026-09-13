# Prédiction scellée : masquage des créneaux fantômes (bead pds, 14/09)

poste1, 14/09/2026, AVANT toute mesure. Correctif : `valid` (booléen [t])
masque `topi` à -1 pour les créneaux fantômes du remplissage godet
(`bucket_batch`) dans `decode_fixed`/`decode_fixed_res` — ni comptés
(`_compter_routage`), ni dispatchés (les 4 noyaux groupés NVFP4, pile et
table, rendent zéro sans lire aucun poids pour `e < 0`).

## Prédictions, chiffrées, avant mesure

**Résident complet Coder-30B b=12** (protocole poste3, témoin 568,6 t/s /
0,601 J/jeton, `banc-horloge-decodage.py`) : correctif SANS effet sur le
PCIe (tous les experts sont déjà résidents, fantôme ou pas) — seul du
CALCUL est économisé, 4/16 du travail MoE (25 %) sur UNE PARTIE du pas
(le MoE, pas l'attention ni le reste). Le MoE n'est pas la totalité du
pas de décodage (mesuré par poste4 le 13/09 : glue+GEMM+attention se
partagent le reste). **Prédiction : effet neutre à faible, < 3 % de
variation de débit**, dans un sens ou dans l'autre — sous ce seuil,
aucune conclusion n'en est tirée ; au-dessus, ce serait une surprise à
expliquer avant de continuer.

**Exil par expert b=12, graphes activés (`ACVRAM_GRAPHES_TABLE=1`) +
masquage** : la cause confirmée le 14/09 (1 à 7 lectures PCIe inutiles
par couche, 48 couches, à CHAQUE pas) est directement adressée.
**Prédiction : le facteur repasse sous celui obtenu SANS graphes ni
masquage (20,2) et SANS graphes AVEC AUTOPIN (9,1 à b=1, valeur de
référence)** — sans m'engager sur le chiffre exact, faute de savoir
combien du facteur 31,9 (graphes seuls, sans masquage) venait
STRICTEMENT du remplissage contre d'autres effets non isolés ici (par
exemple l'hypothèse du double déréférencement, ni confirmée ni infirmée,
seulement rendue inutile pour EXPLIQUER la régression observée).
**Issue nommée qui gênerait cette conclusion** : si le facteur b=12 reste
au-dessus de celui de b=1 (9,1) même après le correctif, cela dirait que
le remplissage n'explique pas TOUT le facteur 31,9 — une part viendrait
d'ailleurs (double déréférencement, ou autre chose de non identifié).

## Résultat (après mesure)

**Résident complet b=12** : 612,7 t/s / 0,549 J/jeton (vs témoin 568,6 /
0,601) — +7,8 % débit, -8,6 % énergie. **Fenêtre invalidée par l'outil
lui-même** (`banc-horloge-decodage.py`) : bridage de puissance pendant la
mesure, `repos()` à 8 s (< 30 s recommandés, duck.ai 14/09). Chiffre
directionnellement positif, PAS confirmé — une refonte propre (carte
calme, `repos()` complet) reste à faire avant d'en tirer une conclusion.
Contention réelle au moment de la mesure : plusieurs sessions actives sur
la carte (nsys de poste4, banc de poste2).

**Exil par expert b=12, graphes + masquage (`ACVRAM_GRAPHES_TABLE=1`)** :
29,58 j/s, **facteur 23,0**. Mieux que graphes-seuls-sans-masquage (31,9)
mais **PIRE que la prédiction** (« sous 20,2 ») et pire que sans graphes du
tout (20,2). **Le masquage aide (31,9→23,0) sans suffire à rendre le
chemin table+graphes rentable à b=12** : soit l'hypothèse initiale (latence
de double déréférencement), reléguée à « inutile pour expliquer » plutôt
que réfutée, joue réellement un rôle résiduel ; soit une part du signal
vient de la contention de carte au moment de cette mesure (mêmes sessions
concurrentes que ci-dessus) et n'a pas été isolée. Non tranché ici.

## Conséquence

`ACVRAM_GRAPHES_TABLE` reste à son défaut sûr (garde active, hors) : le
masquage est un vrai gain (validé par test, et le résident complet en
profite directionnellement), mais graphes+table pour b=12 n'est
toujours pas un gain net démontré contre l'eager. Remesure propre (carte
calme) et éventuellement ncu restent à faire avant de changer le défaut.
