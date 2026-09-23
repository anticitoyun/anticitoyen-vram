# Pièce 82 ter — w13 au décodage seul, préfill séparé : critère TENU, w13 PAR DÉFAUT — 23/09 (Océane)

* **instrument** : `tests/test_moe_w13.py` (+ 2 tests : préfill par vues w13 au bit du séparé, T 1/16/256/3072 ; préfill court par le chemin tensor au bit, T 8/12/32, avec témoin négatif) ; `scratchpad/oceane-p82ter-23-09/empreinte-marlin.py` (sha256 de 12 GEMM Marlin sur entrées fixes, arbre d'avant contre arbre d'après) ; KL `kl-b.py` de la p81/p82 inchangé, ABBA dans une prise (W0 b12, W13 b12, W13 b1, W0 b1) ; `outils/gpu/mesure/capture-godets.py` (godets 1,2,8,16, warm_graphs) sur 7 alias MoE
* **commit** : code ec48987a + defea741 (drapeau de phase), défaut 16e97b44, outils 0a5fc353 ; prises : tests f57a4032→ec48987a, KL c008950b, capture 0a5fc353 ; scellé f57a4032
* **régime** : carte seule (llama-server 4627 au début et à la fin de chaque prise), horloge libre (aucune vitesse publiée ici) ; KL en eager, comme la p81
* **scellé** (f57a4032, avant le code) : défaut au bit ; préfill W13 au bit de W0 ; décodage tensor ≤ 2⁻⁷ ; KL ≤ 0,74 et ≤ témoin + 0,05 par invite, b=12 et b=1 ; prédit : pas 0 identique, invite 2 de b=12 à 0,41
* **mesuré** :

| contrôle | critère | mesuré | |
|---|---|---|---|
| empreinte du défaut (12 GEMM, 3 formes × T 1/12/256/3072) | identique avant/après | 4c63cde7… = 4c63cde7… (deux fois : après `ldn`, après le drapeau) | tenu |
| préfill par vues w13 contre piles séparées | au bit | 8/8 (gate et up, T 1/16/256/3072) | tenu |
| préfill court (T ≤ 32) par le chemin tensor | au bit | 3/3 ; témoin GEMM 2N ≠ (le test peut rendre faux) | tenu |
| KL b=12, kl_max par invite | ≤ 0,74 et ≤ témoin + 0,05 | témoin 0,020 · 0,073 · 0,410 · 0,036 · 0,096 ; **w13 0,045 · 0,073 · 0,410 · 0,051 · 0,118** (+0,025 au plus) ; pas 0 identique 5/5 | tenu |
| KL b=1 | idem | **identique au témoin, pas par pas, 5/5** | tenu |
| passe de capture, godets 1/2/8/16 | ok, sans repli eager | **28/28** : Coder qkvo-i8c, gemma-4-26B-A4B, Qwen3.5-35B-A3B, Kimi-Linear-35B, LFM2.5-8B-A1B, Jan-v2-VL (`marlin-w13`), Nemotron-3.5 (`naturel`, w13 sans objet) | tenu |
| tests | verts | carte : 95 passés, 4 ignorés (w13, tensor, marlin, c10, c17, aligneur) ; à sec : 1 926 passés, 1 échec, le mien (chemin de session dans un script de prise, corrigé) | tenu |

* **verdict** : **critère TENU. w13 devient le défaut** (`ACVRAM_MOE_W13=1` ; témoin `=0`), sur ordre de Jerome. La 82 perdait la KL sur deux chemins que je n'avais pas séparés :
  1. le **préfill groupé**, désormais servi par deux GEMM de largeur N lues dans w13 : paramètre `ldn` (largeur stockée) ajouté au port Marlin (`marlin_template.h`, lignes des adresses de B et des échelles ; `ops.cu` le déduit de `stride(1)` ; un tenseur contigu donne ldn = N, d'où l'empreinte identique) ;
  2. le **préfill court** : T ≤ `_MOE_GROUPED_MAX` = 32 passe par `_forward_grouped`, donc par la GEMM 2N du chemin tensor dès T ≥ 8. C'est lui qui donnait +0,098 sur l'invite 3 à b=1 (1re prise KL, gardée : defea741). Le correctif est un drapeau de phase `moe._EN_PREFILL`, posé par `model.forward` (`batch.is_prefill`) et par `decode_fixed` (False).
  Seul le décodage tensor (godets ≥ 8) garde la GEMM 2N : 2⁻⁷ par ligne, KL b=12 +0,025 au plus. Le gain de vitesse est celui de la 82 : −0,177 ms/pas servi b=12, le décodage n'ayant pas changé. Le préfill revient à la vitesse du chemin séparé. Indicatif seulement (Engine direct, deux arbres, horloge libre) : godet 8 de Coder 5,30 → 4,85 ms entre la prise invalide et la 3 bis.
* **incident** : `capture-godets.py` et quatre autres outils de `outils/gpu/mesure/` calculaient la racine du dépôt à deux niveaux (`outils/gpu`) depuis leur déplacement du 19/09. `acvram` venait alors de l'**installation**, c'est-à-dire l'arbre principal, et non de l'arbre mesuré, sans le signaler. Ma prise 3 a ainsi mesuré main (`experts_layout=marlin`) ; elle est jetée et rejouée en 3 bis. Correctif 0a5fc353 : racine à quatre niveaux, et `capture-godets` refuse un `acvram` importé hors de l'arbre. **Toute capture ou tout seuil pris avec ces cinq outils depuis un worktree, depuis le 19/09, a mesuré l'arbre principal** (`capture-godets`, `qualite-seuil-gemv`, `seuil-gemv-bout-en-bout`, `tensorcores-contre-gemv`, `defauts-ecrits-comme-mesures`). À relire par le chef.
* **durée** : prévue ≤ 15 min de carte ; tenue ≈ 16 min en cinq prises (tests 1 min 15 ; KL 1 min ; tests + KL 2 min 08 ; capture invalide 6 min 21 ; capture 3 bis 5 min 13)
