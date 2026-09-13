# Prédiction — item A7 (audit poste7) : alpha AWQ commun gate/up

poste2, 14/09/2026, avant mesure. Suite d'A6 (RÉFUTÉ, voir
`revue/verdict-a6-int8-snrfloor0-14-09.md`) — A7 redevient prioritaire :
gain plus petit (+1,7-1,9 pt contre +10,6 %) mais coût plus petit aussi
(0 octet, pas de reconversion du parc), et surtout **pas le même
mécanisme** que celui qui a fait échouer A6 (ici gate/up d'UN tenseur,
pas 128 experts d'une pile groupée).

## Ce qui existe déjà (lu, pas re-mesuré)

- `acvram/quant/calibrate.py:search_channel_scales` garde déjà les 21
  erreurs de la grille par tenseur (`journal["erreurs_grille"]`,
  `journal["alpha_retenu"]`) — le prix d'un alpha commun est donc
  calculable SANS carte, directement depuis ces valeurs.
- **Phase 1 déjà mesurée le 10/09** (`revue/courbe-du-quota-deux-
  points.md:62`, `revue/courbe-du-quota-six-points.md:6,113`) : à budget
  4,50 Gio, **45 groupes gate/up récupérables sur 64, sous 2 % de
  surcoût relatif d'erreur**. C'est exactement le seuil que la note
  source (`alpha-partage-recuperer-59-fusions.md`) posait comme
  condition pour proposer la fusion.
- `acvram/engine/layers.py:_scaler_commun` : mécanisme d'ACCEPTATION de
  la fusion déjà en place côté moteur (compare les scalers de plusieurs
  `QuantLinear`, fusionne si identiques) — c'est LA MÊME fonction qui,
  pour les MoE, a son analogue dans `model.py::_try_build_stacks` (la
  cause du regret d'A6). Rien à changer ici : le moteur sait déjà porter
  un scaler commun, il ne le REÇOIT simplement jamais parce que le
  convertisseur cherche un alpha par tenseur, indépendamment.

## Ce qui manque

Une recherche d'alpha **conjointe** dans `calibrate.py` : au lieu
d'appeler `search_channel_scales` séparément pour `gate_proj` et
`up_proj` (chacun choisit son propre optimum sur la grille de 21
valeurs), chercher l'alpha qui minimise la SOMME des deux erreurs — pour
les groupes où ce coût conjoint reste sous le seuil (2 % au budget
4,50 Gio, déjà mesuré). Câblage dans `convert.py` pour appeler cette
nouvelle recherche sur les paires gate/up d'un même bloc MLP, à la
place des deux appels indépendants actuels.

## Seuil scellé

Reprend le seuil déjà validé le 10/09 (2 % de surcoût d'erreur relatif
à budget 4,50 Gio) — **je ne le rescelle pas différemment**, ce serait
refaire un jugement déjà porté sur la même donnée. Nouveau seuil pour
la SUITE (débit) : gain de débit décodage ≥ +1,0 pt (moitié basse de la
fourchette +1,7-1,9 pt déjà notée, marge pour la variance) et PPL
mesurée ≤ PPL étalon (5,4141 sur Llama-2-7b, régime nommé) × 1,005 (0,5 %,
plus strict que le 2 % d'erreur de quantification relative — l'écart
PPL final devrait être bien plus petit que l'écart d'erreur de
quantification d'un sous-ensemble de tenseurs).

## Prédiction chiffrée

Je prédis que l'implémentation de la recherche conjointe reproduit le
45/64 déjà mesuré (même mécanisme, mêmes données) à ±2 groupes près
(erreur d'arrondi/graine possible dans la calibration). Sur le débit,
je prédis un gain **proche de la borne basse** (+1,0 à +1,5 pt) plutôt
que +1,7-1,9 pt : le chiffre existant est déclaré NON MESURÉ par sa
propre note source (« Le gain de débit de cette récupération est
inconnu », `alpha-partage-recuperer-59-fusions.md`) — extrapolé depuis
un +2,60 % mesuré en bf16 pur (où TOUS les empilements aboutissent
déjà), pas depuis un passage 7,8 %→100 % sur int8 comme ici. Je m'attends
à ce que la fraction réellement gagnée (45/64 ≈ 70 % de couverture
supplémentaire, pas 100 %) réduise le gain proportionnellement.

**Issue qui me gênerait** : que le gain de débit soit négligeable
(<+0,5 pt) malgré 45 groupes fusionnés — cela signifierait que la
fusion gate/up elle-même n'était jamais le facteur limitant mesuré
(contredirait le +2,60 % bf16 pur cité comme référence), et que le
travail d'implémentation n'aurait servi à rien mesurable.

## Plan

1. **Fait maintenant, à sec** : `search_channel_scales_commun()` dans
   `calibrate.py` — recherche jointe sur la grille de 21 valeurs pour N
   tenseurs partageant la même entrée (généralisation directe de
   `search_channel_scales`, réutilise `_quant_dequant`). Tests CPU
   uniquement (pas de carte requise pour tester la recherche elle-même).
2. Câblage dans `convert.py` pour les paires gate/up, dernière ce
   pendant `--alpha-commun-gate-up` (flag explicite, défaut inchangé —
   ne pas changer le comportement de conversion par défaut sans mesure).
3. **Sur carte** : convertir Llama-2-7b avec le flag, `acvram eval`
   contre l'étalon 5,4141, puis débit décodage contre le témoin.
