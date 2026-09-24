# Verdict — pièce 167 : relecture à sec de 153 (`--gdn-int8-canal`), branche poste4 a93b634d — poste3, 24/09

Diff examiné : `git show a93b634d` (commit isolé, 27 lignes ajoutées dans `acvram/cli.py` +
`acvram/quant/convert.py`, 54 lignes de test), pas le diff bruit contre `origin/main` (la branche
`poste4` porte des retards de fusion sans rapport — .pt déjà retirés d'ailleurs, fichiers de test
déplacés — qui gonflaient le diff brut à 128 fichiers).

## (1) La promotion int8 touche-t-elle seulement l'attention et le GDN ?

**Oui, par construction — pas par un manifeste réel (coût jugé disproportionné pour une relecture à
sec sur un 27B ; à refaire si tu veux la preuve empirique).**

`acvram/quant/convert.py:1773` : `fmt = router.format_for(name)` décide seul si un tenseur devient
int8 — ce routage vient du plan (`tiering.py`), calculé AVANT la boucle, et ne lit ni
`opts.attn_qkvo_int8_canal` ni `opts.gdn_int8_canal`. Ces deux drapeaux n'entrent en jeu qu'APRÈS
cette décision (`:1871-1874` et `:2031-2033`), pour choisir `group_size_tenseur` (largeur du tenseur
= par canal, ou `opts.group_size` = groupe de 128) sur un tenseur DÉJÀ classé `fmt=="int8"`. Aucun
chemin ne peut donc faire passer MLP, experts ou `lm_head` en int8 par le seul effet de
`--gdn-int8-canal` — ni changer leur granularité, puisque `_est_projection_gdn`
(`convert.py:420-427`) exclut explicitement tout nom qui n'est pas `.linear_attn.*.weight` (testé
en négatif par `test_projection_gdn_couvre_les_cinq_poids_pas_le_reste`, qui vérifie nommément
`mlp.gate_proj.weight` et `model.norm.weight`).

## (2) Les tests 3/3 peuvent-ils rendre FAUX ?

**NON — trouvaille principale de cette relecture.** Faute réintroduite (retrait de la clause
`or (opts.gdn_int8_canal and fmt == "int8" and _est_projection_gdn(name))` aux deux points
d'intégration, `convert.py:1871-1874` et `:2031-2033` — exactement la faute nommée par chef : le
drapeau existe, ne fait plus rien, retombe sur le groupe de 128 au lieu du canal) : **3 passed en
0,08 s, identique à avant la faute.** Les trois tests (`tests/test_gdn_int8_canal_153.py`) ne testent
QUE `_est_projection_gdn`/`_est_projection_attn` en isolation et les valeurs par défaut de
`ConversionOptions` — aucun n'appelle `convert_checkpoint` ni n'inspecte `attn_canal`/
`group_size_tenseur`. La fonction de classification est bien gardée ; son INTÉGRATION dans la
boucle de conversion ne l'est pas. Un bras cassant sur les deux points d'intégration cités manque.

## (3) Déclaration et `test_regime_noyaux`

`--gdn-int8-canal` est une option CLI (`argparse`, pas une variable `ACVRAM_*` lue par
`os.environ`) : `VARIABLES_LUES` (drift-guard de `cli.py`) ne s'applique pas, l'aide CLI suffit et
elle est complète (`acvram/cli.py:1148-1152`, description citant le régime partagé avec
`--attn-qkvo-int8-canal` et leur cumul). `tests/test_regime_noyaux.py` : **11/11 verts**, sans
rapport avec cette pièce (sous-système d'exécution, pas de conversion) — confirmé par exécution,
pas seulement par lecture.

## Verdict

**TENU sur (1) et (3)** (le premier par analyse statique du code, pas par mesure — à noter comme
limite) ; **FAUX sur (2)** : les tests ne protègent pas contre la régression qu'ils sont censés
garder rouge. Rien corrigé (relecture seule, sur ordre). Proposition : ajouter à
`test_gdn_int8_canal_153.py` un test qui appelle réellement `convert_checkpoint` (ou au moins
reproduit le calcul de `group_size_tenseur`) sur un tenseur `linear_attn.*` fictif sous
`gdn_int8_canal=True`, et vérifie `group_size == tensor.shape[1]` — le bras cassant qui manque.
