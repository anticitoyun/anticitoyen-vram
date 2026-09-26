# Prédiction scellée — `piles_ok=False` sur un GLM 100% bf16

poste1, 15/09/2026, avant mesure. Signalé par chef relayant le journal
de poste2 : régime DÉGRADÉ, `piles_ok=False`, message « formats mélangés
entre experts » sur `GLM-4.7-Flash-srcbf16-nvfp4` — MAIS ce converti n'a
QUE des tenseurs "clairs" pour les experts (pas de quantification NVFP4/
INT4 par construction du format `--format bf16`).

## Deux hypothèses de chef

1. Faux déclenchement : bf16 n'est ni NVFP4 ni INT4 → le contrôle
   d'homogénéité refuse à tort un format uniforme mais non reconnu.
2. Vrai : sans piles, pas de graphes ni de MoE groupé → duel MLA faussé
   (comparaison injuste avec un modèle réellement quantifié).

## Lu (pas mesuré) : `MoEBlock._try_build_stacks` / `one()` (model.py:718-756)

`one()` ne reconnaît que `NVFP4Tensor` et `INT4Tensor` (lignes 726, 741) ;
tout AUTRE format (bf16/`PlainTensor` compris) tombe sur le message
générique « formats de quantification mélangés entre experts (ni tout
NVFP4, ni tout INT4) » (ligne 755) — **même quand les experts sont
UNIFORMÉMENT dans ce troisième format**, la condition ne teste que
« pas tout NVFP4 ET pas tout INT4 », jamais une vraie hétérogénéité.

En aval, `_forward_prefill_grouped` (ligne 894) et le chemin de décodage
groupé exigent explicitement `pile[0] == "nvfp4"` : même si `one()`
construisait une pile bf16, RIEN ne la consommerait — ce sont des noyaux
CUDA NVFP4-only (`nvfp4_gemm_grouped_mma`), pas un GEMM groupé générique.

## Prédiction (les deux hypothèses sont VRAIES, pas exclusives)

1. C'est un abus de langage (le message dit « mélangés » alors que le cas
   réel est « uniforme mais non pris en charge ») — un test synthétique
   avec des experts bf16 UNIFORMES doit reproduire `piles_ok=False` et
   `_raison_repli` contenant le mot « mélangés », alors qu'aucun mélange
   n'existe (`len({type(w) for w in ws}) == 1`).
2. C'est aussi une vraie limite : AUCUNE construction de pile pour bf16 ne
   changerait `piles_ok` en `True` de façon UTILE, puisque le seul
   consommateur (`_forward_prefill_grouped`) refuse tout pile dont
   `pile[0] != "nvfp4"`. Un correctif complet (nouveau noyau bf16 groupé)
   dépasse le cadre d'une heure sans carte — seul le message et la
   visibilité (nommer la cause dans `regime_ligne()`) sont corrigés ici.

**Seuil de réfutation** : si un test avec des experts bf16 UNIFORMES ne
déclenche PAS le message « mélangés » (c'est-à-dire si `one()` a déjà un
cas générique que je n'ai pas vu), l'hypothèse 1 est fausse — chercher
ailleurs.

## MESURÉ : CONFIRMÉ (les deux hypothèses)

`tests/test_piles_format_uniforme.py`, 2 tests, vérifié cassant sans le
correctif (stash temporaire, remis à l'identique) :
- 4 experts bf16 (`PlainTensor`) uniformes : `piles_ok` reste `False`
  (attendu, aucun noyau groupé bf16 n'existe — hypothèse 2 de chef,
  VRAIE) mais le message dit maintenant « format PlainTensor uniforme
  mais non pris en charge », plus « mélangés » (hypothèse 1, VRAIE aussi
  — c'était bien un abus de langage, pas une vraie hétérogénéité).
- Un mélange RÉEL (un expert NVFP4 parmi des bf16) est toujours nommé
  « réellement mélangés », avec la liste des formats en cause.

Correctif : message distingue les deux cas (`model.py`, `one()`) ;
`Engine.regime()`/`regime_ligne()` remontent maintenant `piles_raison`
(le pourquoi, pas seulement le booléen) — `acvram serve --regime` sur le
converti de poste2 dira désormais directement si c'est le cas attendu
(bf16 non quantifié) ou une vraie anomalie, sans relire le code.

**Portée du correctif** : message et diagnostic seulement. Aucun noyau
bf16 groupé n'a été ajouté (`_forward_prefill_grouped` refuse tout pile
`pile[0] != "nvfp4"`, ligne 894) — hors de portée d'une heure sans carte,
et pas ce qui a été demandé (chef voulait la CAUSE, pas forcément une
accélération). Le duel MLA reste donc légitimement plus lent pour un
GLM converti en bf16 pur : c'est une limite réelle, pas un artefact de
mesure — comparer un bf16 non quantifié à un modèle NVFP4/INT4 quantifié
compare deux régimes différents, à nommer dans le rapport du duel plutôt
qu'à corriger ici.
