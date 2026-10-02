# La sortie ne doit plus dépendre du découpage du préfill : réduction bf16 exacte de cuBLAS — scellé AVANT le code et la carte (poste6, 02/10 22 h 4x, ordre chef)

Décision de chef : « une sortie qui change avec le découpage est le bogue, pas le réglage ». Cause mesurée
(`poste6-s1-dense-h4-verdict-carte-02-10.md`) : `k_proj` et `v_proj` (N = 1 024, hors Marlin, `kernels/__init__.py:1270`)
passent par `F.linear` (`kernels/__init__.py:932`) ; cuBLAS bf16 y prend, pour certaines formes (M, N ≤ 1 024), une
réduction à précision réduite — 15 % et 30 % des éléments diffèrent entre 7 865 lignes d'un coup et 4 096 + 3 769.
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` : 0 élément différent, sortie à M = 7 865 inchangée.

## Correctif (à écrire après ce scellé)

1. `ACVRAM_BF16_REDUCTION` = `exacte` (pose le drapeau à `False` au chargement du moteur, drapeau global du processus) |
   `reduite` (retrait : laisse torch comme avant, pour la mesure) ; déclaré au régime, dit sur la ligne de régime.
   Défaut du premier commit : `exacte` (ordre de chef) — il ne reste le défaut de la version QUE si le seuil de coût et la
   garde de qualité ci-dessous sont tenus ; sinon le défaut revient à `reduite` avant fusion, et je le dirai.
2. Test d'équivalence dans le même commit : sur carte, un produit de la forme mesurée (N = 1 024, K = 5 120) d'un seul tenant
   contre découpé, identique au bit sous `exacte`, avec son témoin (sous `reduite` il doit différer, sinon le test ne prouve
   rien et s'ignore en le disant) ; sans carte : le chargement pose bien le drapeau (casse si l'on retire la pose).

## Carte : une fenêtre, verrou tenu d'un bout à l'autre (garde de chaîne du même poste — jamais jouée, donc premier point jugé)

| # | grandeur | prédit | faux si / seuil de décision |
|---|---|---|---|
| C0 | garde de chaîne | prise acceptée, aucun autre poste entre deux bras, rendue à la fin | refus ou trou → dit, prise reprise bras par bras avec la carte seule |
| C1 | chaîne dense S1 (Devstral, 7 865 jetons, morceaux de 4 096 contre seul tenant), drapeau POSÉ : B / A1 | **≤ 0,008** (2 × le témoin de 15:19) ; attendu 0 à 0,004 | > 0,008 : k/v n'étaient pas (toute) la cause |
| C2 | même chaîne, drapeau RETIRÉ, même séance | 0,03-0,06 (0,0428 à 15:19) | < 0,01 : le rejeu ne reproduit pas l'écart, la comparaison ne vaut rien |
| C3 | seul tenant A1 posé contre A1 retiré (7 865 jetons) | 32 ids égaux, Δ logprob = 0 | Δ ≠ 0 : le drapeau change aussi la sortie du seul tenant à cette longueur |
| C4 | témoin reprise sous drapeau posé | 0,002-0,01 (0,00403) | — |
| C5 | débit du préfill moteur à M = 512, 1 024, 2 048, 4 096, posé contre retiré (ABBA dans un processus, 5 passes par bras, médiane) | **coût ≤ 2 % à chaque M** (attendu ≤ 1 % : k/v sont 2 produits étroits sur 7) | **> 2 % à un M : opt-in seulement, le défaut repasse à `reduite`** |
| C6 | formes « réduites » : `F.linear` nu N = 1 024 et 512, M de 33 à 10 240 — part des éléments qui changent entre posé et retiré | des formes changent (au moins M = 3 769 ou 4 096), d'autres non (7 865) | aucune forme ne change : la cause mesurée à 22:19 ne se reproduit pas |
| C7 | garde de qualité au noyau, sur les formes réduites : part des éléments égaux à l'arrondi exact du produit fp32, posé contre retiré | posé ≥ retiré à chaque forme (posé ≥ 99 %) | posé < retiré quelque part : le drapeau ne rapproche pas de l'exact |

Décision, fixée ici : défaut `exacte` gardé si C1, C5 (≤ 2 %) et C7 tenus. C7 n'est qu'une garde au NOYAU : la garde de
qualité au modèle (KL / PPL contre la référence, instrument de poste2) sur des invites aux longueurs réduites reste à jouer
avant une activation par défaut dans une version — je la demande à chef plutôt que de la remplacer par C7.
Issues qui me gêneraient : C1 faux (l'écart reste) ; C5 > 2 % ; la garde de chaîne refusée.
