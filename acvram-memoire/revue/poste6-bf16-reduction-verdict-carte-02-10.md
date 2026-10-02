# Réduction bf16 exacte de cuBLAS, carte : morceaux = seul tenant AU BIT sous le réglage, mais +2,40 % de préfill à M = 4 096 (seuil 2 %) et la sortie du seul tenant change aussi — le réglage reste OPT-IN, défaut `reduite`

instrument : `scratchpad/poste6-bf16/carte-bf16.sh` (garde de chaîne : une prise `service` du même poste tenue d'un bout à l'autre, chaque bras prend `carte.sh` dessous) — test `tests/test_reduction_bf16.py` sur carte, `prise-bf16.py` (débit ABBA dans un processus, formes, qualité au noyau), chaîne dense S1 (`poste6-s1-bis/carte-s1-bis.sh`) jouée deux fois (`ACVRAM_BF16_REDUCTION=exacte` puis `reduite`) ; journal `scratchpad/poste6-bf16/carte.log` (non suivi)
commit : poste6-gemma-anneau 12a13dde2 (moteur b42cd7c62), arbre propre sous `acvram/` et `outils/`
régime : RTX 5090 seule, 15 Mio occupés avant et après, aucun autre poste entre deux bras (journal du verrou) ; un `llama-server` étranger au projet sur la carte 1 (5,6 Gio), présent dans les deux bras ; Devstral-24B `srcawq-nvfp4`, NOMINAL, graphes actifs, 0 exil, invite de 7 865 jetons, 32 jetons décodés, `reduction_bf16=exacte` / `reduite` lus sur la ligne de régime de chaque serveur
scellé : `poste6-bf16-reduction-scelle-02-10.md` (C0-C7, seuil de coût 2 %, décision écrite avant)
mesuré : 02/10 22:39:08 → 22:42:43, une fenêtre
verdict : **le réglage fait ce qu'il doit (C1 : morceaux contre seul tenant, 352 valeurs sur 352 au bit) et il coûte trop pour le défaut d'après le seuil écrit avant (C5 : +2,40 % à M = 4 096, étendues disjointes).** Décision scellée appliquée : défaut `reduite`, `ACVRAM_BF16_REDUCTION=exacte` en opt-in. **C3 FAUX** : le réglage change aussi la sortie servie du seul tenant (premier jeton basculé) — ce n'est pas un réglage neutre ; une activation par défaut demande la garde de qualité au modèle (KL / PPL), que je n'ai pas et que C7 ne remplace pas.
durée : 3 min 35 s de carte (prévu ≈ 8 min)

## Prédit / mesuré

| # | prédit | mesuré | |
|---|---|---|---|
| C0 garde de chaîne | prise acceptée, aucun trou | verrou `poste6-bf16-chaine service` de 22:39:08 à 22:42:43, 8 bras dessous, aucun autre poste au journal | tenu (première fois jouée) |
| test sur carte | au bit sous `exacte`, différent sous `reduite` | 3 passés ; N = 1 024, K = 5 120, 7 865 contre 4 096 + 3 769 : `reduite` 2 532 511 éléments différents sur 8 053 760 (31,4 %), `exacte` 0 | tenu |
| C1 dense, `exacte`, B / A1 | ≤ 0,008 (attendu 0-0,004) | **Δ 0 sur 352 valeurs**, 32 ids égaux (A1 = A2 = B) | tenu |
| C2 dense, `reduite`, B / A1 | 0,03-0,06 | 0,0428, ids DIFFÉRENTS dès le premier jeton (A1 = A2 ≠ B) | tenu : l'écart de 15:19 se reproduit |
| C3 A1 `exacte` contre A1 `reduite` | 32 ids égaux, Δ = 0 | **ids différents dès la position 0** ; même candidat du premier jeton : 0,039 (point 1) | **FAUX** |
| C4 témoin reprise sous `exacte` | 0,002-0,01 | **0,0299** affiché ; la requête rejouée bascule le premier jeton — même candidat : 0,041 (point 1) | hors intervalle |
| C5 débit du préfill, posé contre retiré (6 passes par bras, ABBA, médiane) | coût ≤ 2 % à chaque M (attendu ≤ 1 %) | 512 : +0,22 % (2 357 / 2 363 j/s) · 1 024 : +0,35 % (2 940 / 2 950) · 2 048 : +0,81 % (3 282 / 3 308) · **4 096 : +2,40 %** (3 534 / 3 619 ; 1 158,1-1 160,7 ms contre 1 130,1-1 132,8) | **FAUX à 4 096** |
| C6 formes réduites, `F.linear` nu, K = 5 120 | des formes changent (3 769 ou 4 096), d'autres non (7 865) | N = 1 024 : 41-42 % à M = 33, 128, 256, 512 ; 27 % à 2 048, 3 769, 6 000 ; 36 % à 4 096 ; **0 %** à 64, 1 024, 7 865, 8 192, 10 240. N = 512 : 41 % de 33 à 1 024 ; 27 % à 3 769 et 6 000 ; 36 % à 8 192 ; 0 % à 2 048, 4 096, 7 865, 10 240 | tenu |
| C7 qualité au noyau : part des éléments égaux à l'arrondi bf16 du produit fp32 | posé ≥ retiré partout, posé ≥ 99 % | posé 99,71-99,97 % à toutes les formes ; retiré 57,8-73,2 % aux formes réduites, égal au posé ailleurs | tenu |

## Ce que la fenêtre apprend, au-delà du tableau

1. **Cette invite est une quasi-égalité au premier jeton, et l'instrument la juge mal.** Deux candidats X et Y s'y disputent
   la tête ; logprob de chacun, par sortie (lus dans `top_logprobs`, position 0) :

   | sortie | X | Y | en tête | suite d'ids |
   |---|---|---|---|---|
   | `exacte` A1 = A2 = B (au bit) | −3,1604 | −3,1981 | X | a |
   | `exacte` A1 rejouée (cache de préfixe) | −3,2010 | −3,1903 | Y | b |
   | `reduite` A1 = A2 (au bit) | −3,1997 | −3,1904 | Y | b |
   | `reduite` A1 rejouée | −3,1945 | −3,2045 | X | a jusqu'au jeton 17, puis autre |
   | `reduite` B (morceaux) | −3,1476 | −3,2255 | X | a |

   Écart du MÊME candidat (le plus grand de X et Y) : découpage sous `exacte` **0** ; découpage sous `reduite` **0,052** ;
   reprise sous `exacte` 0,041 ; reprise sous `reduite` 0,014 ; `exacte` contre `reduite` au seul tenant **0,039**.
   Le comparateur de la chaîne (`carte-s1-bis.sh`) soustrait, quand le premier jeton bascule, les logprobs de deux jetons
   DIFFÉRENTS : ses « 0,0428 », « 0,0299 » et « 0,00403 » ne sont pas des distances, et son témoin reprise bascule lui-même
   (sous les deux réglages) — le seuil « 2 × témoin reprise » de S1 dense reposait sur ce chiffre, y compris à 15:19.
   Défaut d'instrument, dit ; il ne touche pas C1 (0 sur 352 valeurs, ids égaux) ni le test du produit nu.
2. **C3, cause NON mesurée.** k_proj / v_proj à M = 7 865 ne changent pas avec le drapeau (C6 : 0 %) ; le seul tenant change
   quand même. Le drapeau est global : il touche tout produit bf16 de cuBLAS, pas les deux projections que j'ai nommées.
   Candidats non départagés : les produits de l'attention (`attention.py`, SDPA), la tête de sortie à M = 1, le décodage
   (M = 1 n'est pas dans C6). Ma phrase du verdict H4 « sans changer la sortie du seul tenant » ne valait que pour le
   produit nu de k / v à cette forme ; écrite comme une propriété du moteur, elle était fausse.
3. **C4** : sous `exacte` la sortie ne dépend plus du découpage, mais elle dépend toujours de la reprise sur cache de préfixe
   (REGLES § 4, longueur des clés du SDPA) — ici jusqu'à basculer le premier jeton. Le réglage ne rend pas le moteur
   déterministe au bit entre deux chemins d'attention ; il retire UNE cause.
4. **C6 : la réduction réduite est le cas général aux petits M**, pas une exception des morceaux : à N = 1 024, toute
   requête dont le préfill fait 33 à 512 lignes y passe (41-42 % des éléments de k / v à 1 ulp de l'arrondi exact). Le
   motif en M est irrégulier (64 et 1 024 : 0 %) — aucune règle en M ne le prédit, seul le drapeau le retire.
5. **C7** : au noyau, `exacte` est l'arithmétique la plus proche du produit fp32 (≥ 99,7 % d'éléments à l'arrondi exact contre
   58-73 %). Ce n'est pas une mesure de qualité du modèle.

## Décision appliquée (dans ce commit) et ce qui reste au chef

* Défaut `reduite` : `loader._BF16_REDUCTION`, `regime.py`, `tests/test_defaut_servi.py` (0.7.18), CHANGELOG ; `exacte` reste
  disponible, dit sur la ligne de régime, couvert par `tests/test_reduction_bf16.py`.
* Le seuil de 2 % est franchi à UN M sur quatre, de 0,40 point ; les trois autres sont sous 1 %. Le seuil est celui du chef :
  s'il le relève, la bascule est une ligne — mais C3 exige alors la garde de qualité au modèle AVANT (KL / PPL, instrument
  de poste2, `exacte` contre `reduite`), parce que les ids servis changent pour tout le monde, pas seulement sous morceaux.
* Piste non mesurée, moins chère que le drapeau global : sortir k_proj / v_proj du `F.linear` bf16 (produit fp32 étroit, ou
  seuil `_PROJ_MARLIN_MIN_N` abaissé) — ne toucherait que les deux projections, donc ni le coût à 4 096 ni C3 s'ils viennent
  d'ailleurs. À sceller avant d'y toucher.
* Les morceaux forcés restent opt-in (déjà tranché) : sous le défaut `reduite`, un préfill découpé ne rend pas le seul tenant.
