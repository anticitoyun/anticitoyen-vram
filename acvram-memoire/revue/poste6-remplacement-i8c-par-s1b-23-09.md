# Remplacer i8c par S1b comme alias servi du Coder — dossier à sec, 23/09 22 h 5x (poste6, ordre chef)

Condition : la 123 d'poste1 tenue sur carte (échelle AWQ des experts portée dans le chemin tensor, au bit), puis la
cellule S1b rejouée (prédit 1 950-2 000 t/s, 0,134-0,140 J/jeton ; **réfuté ≤ 1 850**, `poste6.md` 22 h 5x).
Avant la 123, S1b sert à 1 646 t/s (118) : rien de ce qui suit ne s'applique.

## 0. Ce que S1b est aujourd'hui, et ce qui l'empêche d'être servi tel quel
* `…/models_acvram/Qwen3-Coder-30B-A3B-assemble-S1b-proj-tete-i8-23-09` : 5 fragments et 7 fichiers **en liens
  symboliques** vers l'alias A (`…-nvfp4-qkv-alphaqkv-23-09`, 17 G) + `acvram-assemble.safetensors` (1,23 Go : 193
  tenseurs q/k/v/o + tête, pris à i8c) + manifeste propre (19,8 Mo, sha256 `64b6caeed4433c29…`, bloc `assemblage`
  base/donneur/motifs). Effacer A casse S1b. `du -L` : 18 G ; 654 G libres sur `/mnt/AI_GENERATOR`.
* Formats par famille, lus au manifeste : **identiques à i8c** (192 projections int8 par canal, tête int8, 18 432
  experts nvfp4). Seule différence : les experts sont calibrés AWQ (`experts_sans_stats` 5 235 contre 18 432, alpha
  commun gate/up et experts). Conséquence : ni `regime_ligne()` (processus) ni les formats ne distinguent S1b d'i8c —
  seuls `chemin_moe=` et `echelle_awq=` de la ligne moteur (§ 3) et le manifeste le font.
* Anomalie de manifeste à connaître : `options.attn_qkvo_int8_canal = false` et `lm_head_format = null` (bloc copié
  de A par `assembler-alias.py`) alors que les tenseurs sont int8 (`attn_int8: canal`, formats par tenseur). Le
  chargeur lit les formats par tenseur, pas ces options (107 : S1b = i8c à 0,044 sur l'invite 11). Je propose de **ne
  pas retoucher** le manifeste : l'artefact qualifié (trois portes, 107 bis § 6) est celui-là, au sha256 près ; la
  fiche du parc porte l'anomalie.

## 1. Le parc
1. **Matérialiser**, 0 min de carte : `cp -rL <S1b> <nouveau>` puis `sha256sum acvram_manifest.json` = `64b6caee…`
   (le manifeste ne change pas : `weight_map` nomme des fichiers relatifs). Nom proposé, régime porté par le nom :
   **`Qwen3-Coder-30B-A3B-nvfp4-calibexp-qkvo-i8c`** (= i8c + experts calibrés). Ne pas renommer S1b ni effacer A/i8c.
2. `outils/poste/alias-servis-20-09.txt` : le nouvel alias entre en tête (« cellules du comparatif ») ; i8c descend
   en « témoins » (il reste le témoin des instruments : `kl-lot-mele-p100.py`, `energie-familles-p99.py`, 107/118).
   `outils/poste/verifier.sh` ne contrôle que la présence du manifeste : rien d'autre à changer là.
3. `parc-installer` (rebalayage) génère seul : `~/TSV/acvram-chemins.tsv` (ligne `acvram-qwen3-coder-30b-a3b-nvfp4-
   calibexp-qkvo-i8c-nvfp4 <dossier> 15360` — même contexte que la ligne 159 d'i8c ; l'alias exact sort de
   `alias_de`, `parc/bin/parc-installer:254`, à lire dans le TSV produit, pas à deviner) et le bloc
   `[models.<alias>]` de `~/.kimi-code/config.toml` (i8c : lignes 1703-1706). `~/TSV/notes-modeles.tsv` reçoit sa
   note (`menu_modeles/parc.py:205 ecrire_note`) : `faible · <t/s b=1> · ★★★★ · code · chat`.
   **Au passage** : la ligne 142 (i8c) porte encore **380,8**, le chiffre AVEC spéculation (erratum ¹ du README) ;
   à corriger en 312,3 (note ³) dans la même passe.
4. Chaînes de mesure : `scratchpad/poste2-p89-23-09/bloc.sh:9` (`MA=…-qkvo-i8c`) et ses copies (96) pointent i8c en
   dur ; `certifie-b12.py:28` a pour défaut `…-nvfp4` (ni i8c ni S1b). La chaîne de la cellule S1b nomme le nouvel
   alias ET écrit son manifeste (sha256) dans l'en-tête.
5. Tests touchés : `tests/test_moe_tensor_defaut.py:52` exige `ACCEPTÉ … -qkvo-i8c` de `controle-moe-tensor-alias.py`
   — après la 123 la règle statique (`forme_tensor_refus`, `moe.py:1910`) doit accepter les tables AWQ portées par
   l'aligneur : ajouter `ACCEPTÉ … -calibexp-qkvo-i8c`, garder la ligne i8c (i8c reste accepté).

## 2. Le TSV des cellules
`cellule-b12.tsv` (96, 118) : `bras moteur b commit jetons_s j_par_jeton_net watts horloge_moy bridages` — **aucune
colonne d'alias** : une cellule S1b et une cellule i8c y sont indiscernables. Ajouter `alias` et `manifeste`
(sha256 court) entre `moteur` et `b`, écrits par `bloc.sh` depuis `$MA` (et `$MV` pour vLLM), jamais à la main.
README : la ligne « décodage 12 séquences » reçoit le nouveau chiffre avec une note ⁵ (§ 5) ; l'ancienne cellule
(1 995,1, note ²) reste écrite, étiquetée i8c.

## 3. Ligne de régime attendue (moteur, `acvram serve --regime`, b=12, après la 123)
* Identique à i8c sauf deux champs : `chemin_moe=…+tensor(b≥8)` **sans** `repli:tables AWQ d activation par expert
  non unité` (118 : S1b le portait sur 12/12 lignes), et `echelle_awq=<porteur tensor>(48/48)` (nom donné par la 123 ;
  `_regime_echelle_awq`, `runner.py:409`) là où i8c dit `aucune`. Tout le reste : `NOMINAL graphes=on … experts_layout=
  marlin … kv=int8 pipeline=1 sampler=graphe`.
* Ligne processus (`regime_ligne()`, `regime.py:531`) : `défaut ACVRAM_GEMV_LAYOUT=marlin gemv_splitk=S(auto) …
  eco=2700(…)` — inchangée, elle ne voit pas le modèle.
* Alarmes écrites d'avance : `echelle_awq=gemv(…)` ou `repli:tables AWQ` → la 123 n'a pas pris, on mesure la 118
  (1 646) ; `echelle_awq=aucune` → on sert i8c, pas S1b ; `experts_layout≠marlin` ou `DÉGRADÉ` → cellule non publiable.

## 4. Le test qui prouve que l'alias servi a changé (`tests/test_alias_servi_coder.py`, même commit que la bascule)
Contrat par version, comme `test_defaut_servi.py` : `ALIAS_PAR_VERSION = {"0.6.38": "…-nvfp4-qkvo-i8c", "0.6.39":
"…-nvfp4-calibexp-qkvo-i8c"}` ; jamais modifier une ligne existante.
1. à sec, toujours : la première ligne non commentée d'`alias-servis-20-09.txt` == alias de `acvram.__version__` ;
   `forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, awq_unite=False) == ""` (les tables AWQ ne renvoient plus
   au GEMV — c'est la 123 ; **rouge avant elle, c'est voulu**).
2. si le parc est là (`ACVRAM_PARC`, sinon skip) : manifeste de l'alias servi — sha256 == `64b6caeed4433c29…`,
   `attn_int8 == "canal"`, 192 `self_attn.*_proj` int8, `lm_head` int8, 18 432 experts nvfp4, `experts_sans_stats
   < 18432`, `assemblage.donneur == …-qkvo-i8c` ; et `controle-moe-tensor-alias.py --parc` rend `ACCEPTÉ` pour lui.
3. sur carte, dans la chaîne (pas en pytest) : la ligne de régime du bras contient `echelle_awq=` ≠ `aucune` et ne
   contient pas `repli:tables AWQ` ; sinon `bloc.sh` **refuse d'écrire la cellule** (contrôle impossible à sauter,
   REGLES § 3), comme `certifie-b12` refuse un lot < N.

## 5. Texte du README pour le chiffre nouveau (à remplir depuis la cellule, jamais avant)
* Chapeau du tableau : « Qwen3-Coder-30B-A3B en NVFP4 (experts **calibrés AWQ, alpha commun gate/up**) + INT8
  (attention, tête) — alias `…-nvfp4-calibexp-qkvo-i8c` depuis 0.6.39 ; jusqu'à 0.6.38, experts non calibrés (i8c) ».
* Ligne : `| décodage 12 séquences | **X t/s** ⁵ | 2 027,0 t/s ² | — |`.
* Note ⁵ : « <date>, même protocole que ² (même séance, même client HTTP, `-lgc 2700` posé pour les deux, A V V A,
  ≥ 5 lots, écart au-delà de 2 σ ; revue/<note de la cellule>). Alias à experts calibrés AWQ (revue/poste6-piece107-
  dossier-alias-prose-23-09.md § 6 : KL b=1 16/16, lot mêlé 61/64, PPL 1,011 ± 0,010 contre i8c) servi par le chemin
  tensor avec l'échelle AWQ fondue au rassemblement (revue/poste1-piece123-…). Prédit 1 950-2 000 t/s et 0,134-0,140
  J/jeton, scellé avant la mesure ; mesuré X t/s (σ), Y J/jeton : <égalité | devant | derrière> vLLM en débit
  (Z %, 2 σ = …) et <…> en J/jeton (W %). i8c, même séance : X' t/s (témoin, note ²). »
* Paragraphe « Où acvram est devant » : « … à égalité de débit à b=12 (1 995,1 contre 2 027,0, 0.6.38, note ²) … » →
  ajouter « puis X t/s avec les experts calibrés (0.6.39, note ⁵) ; acvram y a progressé de 1 540 (0.6.34) à X ».
  Si Y ≤ 0,152 J/jeton (vLLM 0,1628 × 0,93), la phrase « vLLM garde 7,0 % de J/jeton de moins » tombe ; sinon elle reste.

## Coût et ordre
Matérialisation 0 min de carte (copie 18 G) ; test § 4 à sec ; TSV/parc à sec ; la seule carte est la cellule
(≈ 25 min, 2 bras × 5 lots + chargements). Ordre : 123 tenue → cellule → si tenue, § 1-2-4-5 dans un commit, README
avec le chiffre de la cellule, .deb 0.6.39 (poste3).
