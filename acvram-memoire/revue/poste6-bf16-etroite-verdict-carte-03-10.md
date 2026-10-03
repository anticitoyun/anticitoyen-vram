# Réduction exacte étroite (k_proj / v_proj seules), carte : au bit au noyau et gratuite (E8 : 0,0 % à tout M), mais les morceaux ne rendent PAS le seul tenant (E4 FAUX, Δ 0,034 dès le premier jeton) — un autre produit cuBLAS bf16 du préfill dépend du découpage ; `etroite` reste opt-in, défaut inchangé

instrument : `scratchpad/poste6-bf16/carte-etroite.sh` (garde de chaîne `poste6-etroite-chaine`, 8 bras `service` dessous) — `tests/test_reduction_bf16.py` sur carte (E0-E3), `prise-etroite.py` (E7 chrono au noyau ; E8 / E9 débit moteur ABBA, 6 passes par bras, médiane, modes basculés dans le processus), chaîne dense S1 (`prise-s1-morceaux-kv31b.sh`) sous `ACVRAM_BF16_REDUCTION=etroite` puis `reduite`, comparateur corrigé (même candidat) ; journal `scratchpad/poste6-bf16/carte-etroite.log` (non suivi)
commit : poste6-gemma-anneau 902c7b5cf (main 4f39deccf fusionné avant), arbre propre sous `acvram/` et `outils/`
régime : RTX 5090 seule, 15 Mio avant et après, aucun autre poste entre deux bras (journal du verrou) ; `llama-server` étranger au projet sur la carte 1 (5,6 Gio), présent d'un bout à l'autre ; Devstral-24B `srcawq-nvfp4`, NOMINAL, 0 exil, invite de 7 865 jetons, 32 jetons décodés, torch 2.14.0+cu130
scellé : `poste6-bf16-etroite-scelle-02-10.md` (E0-E9, décision écrite avant)
mesuré : 03/10 01:46:58 → 01:50:58, une fenêtre
verdict : **la portée étroite fait exactement ce que le scellé lui demandait au noyau (E0-E3 : 0 élément différent partout, au bit du drapeau global) et ne coûte rien au moteur (E8 : −0,04 à +0,02 %), mais le but n'est pas atteint : sous `etroite`, un préfill découpé ne rend pas le seul tenant (E4 : Δ 0,034 à la position 0, ids divergents au jeton 22, contre 0 sur 352 sous le drapeau global).** Donc k / v ne sont pas toute la cause : un autre produit bf16 de cuBLAS, hors du chemin NVFP4 naturel, dépend du découpage — et c'est lui qui portait les +2,40 % du drapeau global à M = 4 096, puisque k / v exactes coûtent 0. Décision scellée appliquée : aucun bras retenu, `etroite` / `etroite-tranches` opt-in, défaut `reduite`. Mon H4 « attention : 0 » (couche 0, 02/10) est contredit au niveau du moteur.
durée : 4 min 00 s de carte (prévu ≈ 5 min)

## Prédit / mesuré

| # | prédit | mesuré | |
|---|---|---|---|
| C0 garde de chaîne | tenue | verrou `poste6-etroite-chaine` 01:46:58 → 01:50:58, 8 bras, aucun autre poste | tenu |
| E0 drapeau rendu | True après l'appel, régime `etroite` | True après chaque appel (test et E7) ; ligne de régime `reduction_bf16=etroite` / `reduite` lue sur chaque serveur | tenu |
| E1 produit nu 7 865 contre 4 096 + 3 769, `etroite` | 0 ; témoin `reduite` 15-31 % | **0** ; témoin 2 532 511 sur 8 053 760 (31,4 %) | tenu |
| E2 `etroite` contre drapeau global, M = 33 … 7 865 | au bit | 0 élément à chacun des 7 M | tenu |
| E3 `etroite-tranches` contre `etroite` | au bit | 0 élément à chacun des 7 M | tenu |
| **E4 dense S1, `etroite`, B / A1** | **Δ 0, ids égaux** | **Δ 0,301 sur 228 valeurs, ids divergents à la position 22** ; dès la position 0 : 0,034 (seuil 0,0533, FAUX) ; profil par position 0,02-0,15 puis 0,30 au jeton 14 | **FAUX** |
| E5 seul tenant A1 `etroite` contre A1 `reduite` | ids égaux, Δ 0 | ids égaux (sha a95ede67…), Δ 0 sur 320 valeurs | tenu |
| E6 témoin reprise sous `etroite` | 0,01-0,05, premier jeton possiblement basculé | 0,0267 sur 9 valeurs, premier jeton basculé (identique sous `reduite`) | tenu |
| E7 chrono au noyau N = 1 024, K = 5 120, exacte / réduite | 512 : × 1,0-1,6 · 1 024 : × 1,00 ± 0,03 · 2 048 : × 1,1-1,8 · 4 096 : × 1,5-2,6 ; T à 4 096 × 1,0-1,4 ; fp32 × 2-5 | 512 : **× 1,08** (40,5 / 37,6 µs) · 1 024 : × 1,01 · 2 048 : × 0,99 · 4 096 : **× 1,13** (250,8 / 221,6 µs) ; tranches × 1,07 / 1,01 / 1,04 / 1,15 ; fp32 × 4,0 / 3,6 / 3,5 / 4,2 | 2 048 et 4 096 sous l'intervalle : la réduction exacte coûte bien moins que je ne le croyais à ces formes |
| **E8 débit moteur, `etroite` (P) contre `reduite`** | 512 : +0,0 à +0,3 % · 1 024 : 0 ± 0,3 · 2 048 : +0,2 à +0,8 · **4 096 : +1,4 à +2,4 %** | 512 : −0,01 % (2 358 / 2 358 j/s) · 1 024 : −0,04 % · 2 048 : +0,01 % · **4 096 : +0,02 %** (3 619 / 3 620 j/s ; 1 131,8 contre 1 131,6 ms, étendues confondues) | seuil tenu partout ; ma fourchette à 4 096 était fausse |
| E9 débit moteur, `etroite-tranches` (T) | 2 048 : +0,0 à +0,5 · 4 096 : +0,2 à +1,2 % | +0,04 / +0,02 / −0,00 / +0,05 % | tenu ; T n'a rien à gagner puisque P ne coûte rien |

## Ce que la fenêtre apprend

1. **Le surcoût du drapeau global (+2,40 % à 4 096) ne vient pas de k / v** : leur réduction exacte, posée seule, coûte 0,02 %
   au moteur (E8) et × 1,13 au noyau sur 80 appels de 250 µs — 2 ms sur 1 132, dans l'étendue de mesure. L'issue (i) du
   scellé s'est réalisée : le coût est ailleurs, dans le ou les produits que le drapeau global touche aussi.
2. **Le découpage change encore la sortie quand k / v sont exactes** (E4), et il ne la change plus quand TOUT cuBLAS est
   exact (C1, 02/10). Le produit restant est donc un GEMM bf16 cuBLAS hors du chemin NVFP4 naturel. Candidat nommé, non
   mesuré : **l'attention du préfill**, `F.scaled_dot_product_attention` avec masque explicite `causal_lower_right`
   (`acvram/engine/layers.py:1164`, `:1170`, `:1174`) — un masque dense exclut le noyau flash ; si le repli est le chemin
   « math », scores et sortie sont des `bmm` cuBLAS bf16 dont M (lignes de requêtes : 7 865 contre 4 096 puis 3 769) change
   avec le découpage, exactement le mécanisme de k / v. H4 avait mesuré « attention : 0 » sur la couche 0 avec les K / V du
   seul tenant ; au moteur, les morceaux changent aussi la longueur des clés vues par chaque morceau. À vérifier d'abord à
   sec : quel backend SDPA le préfill prend (masque + `enable_gqa`), puis sur carte le produit nu de ces formes.
3. **E5 tenu** : la portée étroite ne change rien au seul tenant de 7 865 lignes, comme prévu (k / v y sont déjà exactes).
   Le réglage est neutre là où cuBLAS l'était déjà, et ne touche que les M « réduits » (C6).
4. **Sous `etroite`, B colle à A1 sur 22 jetons** (contre une bascule dès le premier sous `reduite`) : k / v exactes
   réduisent l'écart des morceaux sans l'annuler ; 0,034 à la position 0 contre 0,061 sous `reduite`. Ce n'est pas une
   distance comparable (le comparateur s'arrête à la divergence), c'est dit.
5. **E7** : la réduction exacte à N = 1 024 coûte au plus × 1,13 au noyau ; le bras tranches n'a aucun sens (× 1,15 à
   4 096, aucun gain) : à retirer du code si `etroite` survit, ou à laisser mort — au chef.

## Décision appliquée et ce qui reste au chef

* Aucun bras retenu (E4 faux) ; `ACVRAM_BF16_REDUCTION=etroite` et `etroite-tranches` restent opt-in, défaut `reduite`,
  comme `exacte`. Rien d'autre ne change dans ce commit que le verdict et les commentaires qui annonçaient le jugement.
* La suite qui découle des chiffres, à sceller avant d'y toucher : (a) à sec, nommer le backend SDPA du préfill ; (b) sur
  carte, produit nu de l'attention aux formes du découpage, posé contre retiré ; (c) si c'est bien lui, une portée
  `etroite+attention` donnerait B = A1 au bit (comme C1) au prix du drapeau global (≈ +2,4 % à 4 096, puisque k / v
  coûtent 0) — le seuil de 2 % ne tiendrait qu'avec un autre noyau d'attention (flash ou mémoire-efficace : réduction sur
  les clés par blocs fixes, indépendante du découpage des requêtes par construction, à mesurer), pas avec le drapeau.
* La garde de qualité au modèle reste due avant toute bascule du défaut, quel que soit le bras (les M « réduits » changent
  de sortie).
