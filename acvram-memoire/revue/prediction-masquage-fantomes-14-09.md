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
