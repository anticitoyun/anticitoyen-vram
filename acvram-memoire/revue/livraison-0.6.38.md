# Livraison 0.6.38 (poste3, 23/09, pièce 95)

Paquet construit sur main à jour (fusion de poste3 dans main jusqu'aux pièces citées ci-dessous).
Aucune installation lancée (`sudo dpkg -i` reste à l'utilisateur, feu explicite requis en plus
tant que la pièce 94 de poste2 n'est pas tenue). Aucune publication GitHub.

## 1. Journal des changements depuis 0.6.37

### Pièce 85 — effondrement du service à lots successifs (poste1)
`MAX_GRAPHS=16` sans éviction : au-delà de seize formes, une clé nouvelle refusait la capture
**sans le compter ni le nommer** (`repli_eager` restait à 0). Sur un serveur qui enchaîne des
paliers de lot (2→4→8→12), le palier 12 finissait par ne plus avoir de place et servait en
eager muet. Deux causes trouvées et corrigées :
1. Plafond relevé à 64 (au lieu de 16), refus désormais compté et nommé.
2. `/metrics` (donc toute supervision — tableau de bord, GUI) invalidait les captures en cours en
   relisant l'état éco SOUS CHARGE dans le processus de service ; corrigé (relecture au premier
   `regime()` seulement).
Mesuré (`poste1-piece85-effondrement-23-09.md`, ABBA, alias Coder-30B-A3B-nvfp4-qkvo-i8c,
-lgc 2700, régime servi 400 W) : b=12 — A (plafond 16) 536,3 t/s, 135 W ; B (plafond 64) 1 802,4 ;
D (défaut 64, sans variable) 1 742,0 t/s, 0,145 J/jeton net.

### Pièce 86 — b=2 sous b=1 en service (poste1)
À b=2, la vérification spéculative n-gramme à longueurs de proposition mêlées tombait en eager
**sans le compter** (même défaut de fond que la 85, sur un autre chemin). Corrigé : un lot dense
à longueurs mêlées décode sans spéculer, compté (`spec_longueurs_melees` dans `/metrics`).
Mesuré : b=2 avant correctif 293,8 t/s → après correctif 407,7 t/s (**+38,8 %**), replays = pas,
`repli_eager=0`.

### Pièce 87 bis — le −6,72 % de la pièce 87 n'est pas un effet du service à lots successifs (poste2)
Rejeu ABBA à 5 lots/bras (10 mesures indépendantes par bras) au lieu d'une seule fenêtre : écart
des moyennes L (lots successifs) vs N (serveur neuf) = **−1,34 %**, dans la bande ±3 % — TENU.
σ mesuré 4,01 %/3,06 % (CV 3-4 %) confirme que le −6,72 % de la pièce 87 (un seul tirage) était
compatible avec le bruit intra-serveur. Le reste (a) de la pièce 85 (lots successifs sous serveur
neuf) n'est donc plus établi.

### Pièce 88 — capture échouée : justesse des hybrides, puis tri des échecs (poste1)
Deux défauts trouvés en lisant le tri des échecs de capture :
1. **Justesse** : une capture ratée sur un modèle hybride (Qwen3-Next, Kimi-Linear…) laissait
   l'état récurrent GDN avancé de deux pas d'échauffement — le pas eager de repli calculait donc
   FAUX. Corrigé : échauffement et capture dans un `try`, restauration dans le `finally`.
2. **Tri des échecs** : un incident transitoire (capture invalidée par un autre fil) coupait TOUS
   les graphes pour la vie du serveur, y compris les sains. Corrigé : transitoire → la clé attend
   puis se retente (3 essais) ; déterministe (OOM…) → refus de cette seule clé ; contexte CUDA en
   erreur → coupure complète, comme avant.
Mesuré : état GDN après capture ratée, avant/après correctif — 7,0 contre 5,0 (deux pas non
restaurés) ; 6/6 tests du tri verts après correctif (3/3 rouges avant).

### Pièce 89 — comparatif vLLM à ≥5 lots/bras, seuil statistique explicite (poste2)
Même méthode que 87 bis appliquée au comparatif vLLM (ABBA, n=10 par moteur, seuil déclaré
2σ_diff avant la prise) :
- b=12 : acvram 1 931,2 t/s contre vLLM 2 026,5 (**−4,70 %**, au-delà du seuil 2σ — confirmé) ;
  énergie acvram 0,1376 J/jeton contre vLLM 0,1288 (**+6,82 %**, au-delà du seuil — confirmé).
- b=1 : acvram 312,3 t/s contre vLLM 284,8 (**+9,67 %**, au-delà du seuil — confirmé) ; énergie
  0,6035 contre 0,6032 (+0,04 %, **sous le seuil : égalité**, pas un avantage acvram).
Cohérent en signe et en grandeur avec la pièce 78 (b=12 −4,87 %/+6,4 % ; b=1 +9,39 %) — confirmé
ici avec un seuil statistique explicite, pas seulement une lecture brute.

### Pièce 82 ter — w13 au décodage seul, préfill séparé, PAR DÉFAUT (poste1)
`ACVRAM_MOE_W13=1` devient le défaut. La pièce 82 perdait de la qualité (KL) sur deux chemins non
séparés : le préfill groupé (corrigé par un paramètre `ldn` dans le port Marlin, empreinte au bit
identique) et le préfill court, T ≤ 32 (corrigé par un drapeau de phase `moe._EN_PREFILL`). Seul le
décodage (godets ≥ 8) garde la GEMM 2N du chemin tensor. Empreinte du défaut identique au bit,
préfill par vues w13 au bit du séparé (8/8), KL b=12 +0,025 nat au plus par invite (seuil 0,74),
KL b=1 identique au témoin (5/5), capture 28/28 godets {1,2,8,16} sur 7 alias MoE sans repli eager.
Gain repris de la pièce 82 : −0,177 ms/pas servi à b=12.

### Pièce 92 — réduction d'attention déroulée, PAR DÉFAUT (poste1)
`ACVRAM_ATTN_REDUC_DEROULEE=1` devient le défaut : le découpage de la tuile d'attention (plus de
programmes) a été RÉFUTÉ au banc (+19 à +33 %, plus lent) ; la réduction série sur le chemin
critique gagne 1,9 % déroulée, au bit sur 11/11 cellules et au test (5 formes). Gain : −0,014
ms/pas à b=12, −8,7 % à b=1 ctx 2048, −7 % à b=4. Piste « 4 warps » (−5,1 % au banc) NON retenue :
contredite par un ABAB SERVI antérieur (poste2, 20/09) — tranchée en service seulement.

### Pièce 90 — la ligne de régime des refus de graphe (poste3)
Reste (c) de la pièce 88 : une clé de graphe refusée (OOM, opération non capturable) laisse les
autres clés vivantes — `graphes=on` continuait de s'afficher sans le dire. `regime_ligne()` et
`/metrics` portent désormais `graphes=on(refus=N:raison principale)` dès qu'une clé est refusée
pendant que d'autres restent en graphe.

## 2. Ligne gelée (`tests/test_defaut_servi.py`)

Entrée `"0.6.38"` ajoutée à `DEFAUTS_PAR_VERSION`, copiée de `"0.6.37"` plus trois défauts qui ont
changé de valeur dans cette livraison : `MOE_W13=1` (pièce 82 ter), `MAX_GRAPHS=64` (pièce 85,
16→64), `ATTN_REDUC_DEROULEE=1` (pièce 92). La FIN de la ligne à sec (`mla_core=…
prefill_glue=compact`) est inchangée — vérifié en rejouant `regime_ligne()` à sec (sous-processus,
sans variable `ACVRAM_*`, `CUDA_VISIBLE_DEVICES=""`) avant d'écrire l'entrée, pas supposé : les
trois nouveaux défauts sont lus via un module importé (`lu_a`) et n'apparaissent dans la ligne que
lorsqu'ils DIFFÈRENT de leur défaut, ce qui n'est pas le cas ici. `test_la_version_servie_a_sa_ligne_gelee`,
`test_les_defauts_nommes_sont_ceux_de_la_version`, `test_la_ligne_de_regime_du_defaut_nu_a_sec` : verts.

## 3. Construction et vérification du .deb 0.6.38

- `acvram/__init__.py` et `pyproject.toml` : 0.6.37 → 0.6.38.
- Source : main à jour (worktree `poste3`, fusionné jusqu'aux pièces 82 ter/85/86/87 bis/88/89/90/92).
- `bash tools/construire-deb.sh` : `acvram_0.6.38_amd64.deb`, sha256 `<jeton-masqué>`.
- `dpkg-deb -f … Version` = `0.6.38`, cohérent avec `pyproject.toml` et `acvram/__init__.py`.
- Aucun `anticitoyenpartage` dans le paquet extrait.

## 4. Tests joués avant de pousser (carte libre, vérifiée avant ET après)

- `tests/test_defaut_servi.py` : 3 passed.
- `tests/test_piece85_service_lots_successifs.py`, `test_piece86_spec_longueurs_melees.py`,
  `test_piece88_capture_echouee.py`, `test_moe_w13.py`, `test_glue_compact.py`,
  `test_regime_graphes_vivants.py` : 34 passed, 26 skipped (tests GPU, carte non prise ici).
- Aucune exécution pendant une fenêtre de mesure d'un pair (vérifié `/tmp/acvram-carte-0.lock.qui`
  absent avant chaque commande).

## 5. Ce qui n'est pas fait

- Installation (`sudo dpkg -i`) : à l'utilisateur, et **pas d'annonce « prêt à installer » avant
  que la pièce 94 de poste2 (capture-godets sur main) soit tenue** — feu vert de chef attendu.
- Publication GitLab/GitHub + release : feu explicite requis, non demandé ici.
