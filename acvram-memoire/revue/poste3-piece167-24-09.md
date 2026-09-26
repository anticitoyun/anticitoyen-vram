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

**Addendum (preuve réelle, ci-dessous) : confirmé par manifeste sur un mini-checkpoint réel**, plus
seulement par lecture — voir § Addendum.

## (2) Les tests 3/3 peuvent-ils rendre FAUX ?

**NON — trouvaille principale de cette relecture.** Faute réintroduite (retrait de la clause
`or (opts.gdn_int8_canal and fmt == "int8" and _est_projection_gdn(name))` aux deux points
d'intégration, `convert.py:1871-1874` et `:2031-2033` — exactement la faute nommée par chef : le
drapeau existe, ne fait plus rien, retombe sur le groupe de 128 au lieu du canal) : **3 passed en
0,08 s, identique à avant la faute.** Les trois tests (`tests/test_gdn_int8_canal_153.py`) ne testent
QUE `_est_projection_gdn`/`_est_projection_attn` en isolation et les valeurs par défaut de
`ConversionOptions` — aucun n'appelle `convert_checkpoint` ni n'inspecte `attn_canal`/
`group_size_tenseur`. La fonction de classification est bien gardée ; son INTÉGRATION dans la
boucle de conversion ne l'est pas. **Corrigé ci-dessous** (bras cassant ajouté).

## (3) Déclaration et `test_regime_noyaux`

`--gdn-int8-canal` est une option CLI (`argparse`, pas une variable `ACVRAM_*` lue par
`os.environ`) : `VARIABLES_LUES` (drift-guard de `cli.py`) ne s'applique pas, l'aide CLI suffit et
elle est complète (`acvram/cli.py:1148-1152`, description citant le régime partagé avec
`--attn-qkvo-int8-canal` et leur cumul). `tests/test_regime_noyaux.py` : **11/11 verts**, sans
rapport avec cette pièce (sous-système d'exécution, pas de conversion) — confirmé par exécution,
pas seulement par lecture.

## Verdict initial

**TENU sur (1) et (3)** (le premier par analyse statique du code au moment de la relecture — voir
l'addendum pour la preuve par manifeste) ; **FAUX sur (2)** : les tests ne protégeaient pas contre
la régression qu'ils sont censés garder rouge.

## Addendum : `tests/test_gdn_int8_canal_conversion_reelle_167.py` (feu de chef)

Nouveau fichier de test, deux tests, qui appellent RÉELLEMENT `convert_checkpoint` sur un
mini-checkpoint synthétique (1 couche GDN `linear_attention`, 1 couche `full_attention`, MLP,
`lm_head` — gabarit de `test_collect_qwen35_gdn.py`, dimensions ×8 pour H=256 : voir la note dans
le fichier, H=32 de l'original fait toujours collapser `group_size=128` en `ng=1` qu'on soit par
canal ou par groupe, ce qui aurait rendu le bras cassant aveugle).

* `test_conversion_reelle_int8_canal_sur_attn_et_gdn_mlp_et_lm_head_intacts` : promotion int8
  forcée (`snr_floor=999`, `promotion_classes` restreint aux suffixes attn+GDN, `max_promotions=1.0`
  — le défaut, une fraction de 0,15 du nombre de tenseurs, plafonnait à 4 promotions sur 9 voulues
  sur ce gabarit minuscule, faute d'instrument corrigée en route). Vérifié sur les OCTETS ÉCRITS
  (pas le manifeste, qui n'a pas de bilan GDN comme `_bilan_attn_int8` pour l'attention) : les 5
  projections GDN et les 4 projections d'attention sont `format="int8"` ET `scales.shape[1] == 1`
  (par canal) ; les 6 tenseurs MLP et `lm_head` restent `!= "int8"` (jamais candidats,
  `promotion_classes` ne les cite pas). **Prouve (1) par la mesure, pas seulement par lecture.**
* `test_bras_casse_retirer_gdn_int8_canal_laisse_gdn_en_groupe` : `_est_projection_gdn` neutralisée
  par `monkeypatch.setattr` dans le module `convert` (équivalent fonctionnel exact de retirer la
  clause aux deux points d'intégration — ce sont les deux seuls appelants). **Cassé une fois avant
  correction du test** (voir ci-dessous), puis vert avec l'assertion correcte : la projection GDN
  `qkv` retombe à `ng=2` (groupe de 128 sur K=256) au lieu de `ng=1`.

**Preuve de cassage, dans l'ordre où elle a eu lieu** (deux fausses pistes corrigées en route,
consignées parce qu'elles auraient pu masquer un vrai résultat) :
1. Bras cassant lancé une première fois avec le gabarit H=32 d'origine : PASSE À TORT (`ng=1` des
   deux côtés) — le collapse de `group_size=128` sur une largeur de 32 rendait le test aveugle,
   pas la faute absente. Diagnostiqué, gabarit agrandi (H=256).
2. Avec H=256, seconde fausse alerte : `ConversionOptions()` sans `out_dir` levait une
   `TypeError` DANS le message d'assertion (évalué parce que la condition avait déjà échoué) avant
   que l'assertion elle-même ne s'affiche — corrigé (`out_dir=""`).
3. Avec les deux corrections : le bras cassant montre l'échec attendu, `ng=1` (pas cassé) puis,
   correction du message, confirmation propre `ng=1 > 1` FAUX — la faute ne cassait PAS avec le
   test tel qu'écrit à l'étape 1. Une fois le gabarit H=256 en place, rejoué : `ng=2`, bras cassant
   VERT (la faute casse bien la propriété visée).

**Suite finale** (`tests/test_gdn_int8_canal_153.py` + `tests/test_gdn_int8_canal_conversion_reelle_167.py`,
sous mon verrou, carte visible) : **5 passed en 0,48 s**.

## Verdict final

**TENU sur (1), (2) et (3)** — (2) corrigé par les deux tests ajoutés, dont un bras cassant vérifié
cassant sur la faute nommée par chef. Rien d'autre touché dans `acvram/`.
