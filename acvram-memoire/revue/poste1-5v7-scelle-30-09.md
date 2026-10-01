# 5v7 — coût net des piles MoE (4,58 Gio sur Coder-30B) : hypothèse et prédiction scellées AVANT la mesure (poste1, 30/09)

Mesuré (fenêtre du 30/09, `poste1-aym-verdict-30-09.md`) : après les piles construites au chargement, 10,62 → 6,04 Gio
libres sur Coder-30B qkvo-i8c (48 couches) ; le plan leur compte 0 (`loader.py`, `_reserve_prefill` : Σ × 1).

## Lecture du code (à sec)
* La pile naturelle est une copie (`torch.stack`) puis les experts deviennent des VUES (`moe.py`, `one()`) : les tenseurs
  d'origine se libèrent. La Marlin remplace la naturelle, rendue par `_liberer_pile_naturelle`. Il n'y a pas de seconde
  copie survivante au code, et `NVFP4Tensor` / `QuantLinear` ne gardent aucun cache dérivé. Le coût net attendu est ≈ 0.
* Un expert de Coder-30B pèse 768 × 2 048 / 2 = 0,75 Mio (+ 96 Kio d'échelles) : il vit dans le **bassin des petits blocs** de
  l'allocateur (< 1 Mio, segments de 2 Mio), aux côtés de petits tenseurs qui SURVIVENT à la pile :
  * les `global_scale` des couches refusées en Marlin (sous-normales, Coder) ;
  * les `ChannelScaler` AWQ par expert.
* Un segment n'est rendu au pilote que vide. Hypothèse : les segments restent **réservés, non alloués** ; `mem_get_info` les
  voit occupés. C'est la même famille que gemma (commentaire de `_rendre_le_cache_apres_la_pile`) : pas un défaut du plan,
  une fragmentation.

## Prédiction (bras de 1 min : chargement Coder-30B à 29 096, ligne « piles … au chargement » de 286c08485)
* **H vraie** :
  * réservé − alloué ≥ 3,5 Gio après les piles ;
  * dans le bassin des petits blocs, réservé − alloué ≥ 3,5 Gio.
  * Correctif alors : regrouper les petits survivants d'une couche dans UN tampon (grand bassin), par vues, AVANT de rendre
    les originaux. Les segments se vident et `empty_cache` les rend. Le plan reste juste (coût ≈ 0).
* **H fausse** :
  * réservé − alloué < 1 Gio : les piles allouent vraiment 4,5 Gio de plus ;
  * suite : un instantané mémoire (`torch.cuda.memory_snapshot`) pour trouver QUI, puis le plan compte ce coût réel (demande
    de chef, test cassant sur `_reserve_prefill`).
* **Entre les deux** (1-3,5 Gio) : les deux causes à la fois, nommées séparément.

## Pièce (2), vrm
Les poids de la tour de vision sont déjà comptés sous la borne du KV (`loader.py`, `_octets_annexes`, pièce 201). L'OOM de
qwen3-vl-30b à `vision.py:415` suit donc très probablement le coût non compté des piles. La charger avant le KV la ferait
échouer plus tôt, nommément (même geste que aym), sans la faire servir : à décider après la mesure de 5v7.

## Ajout avant la mesure : correctif et bras B (même prise)
Correctif écrit à sec (`moe.py`, `_compacter_survivants`, témoin `ACVRAM_PILES_COMPACTER=0`, 5 tests dont 5 rouges sur
main). Prise : `BRAS="A B" diag.sh` — A = compaction 0 (confirme ou réfute H), B = compaction 1.
* Prédit si H vraie : B libère ≥ 3,5 Gio de plus que A après les piles (libres B ≥ 9,5 Gio), plus de refus « 6,04 pour 6,90 ».
* Si A montre réservé − alloué < 1 Gio : H fausse, B sans effet attendu, le correctif ne part pas.

## Mesure (30/09 16:36-16:38, carte.sh `poste1-5v7-diag`, tenue 132 s ; début et fin : 5090 vide, seul 8081 ; arbre 4be289f8f)
Coder-30B qkvo-i8c, 29 096, piles au chargement ; ligne du chargeur (`diag-A.log`, `diag-B.log`) :

| bras | libres avant → après piles | alloué | réservé | petits blocs réservé/alloué | suite |
|---|---|---|---|---|---|
| A (compaction 0) | 10,62 → 6,04 Gio | 16,78 | 24,69 | **7,42 / 0,14 Gio** | refus « 6,04 pour 6,90 Gio de KV » |
| B (compaction 1) | 10,62 → **13,15 Gio** | 16,78 | 17,58 | 0,31 / 0,02 Gio | chauffe **29 096/29 096** (26,7 s, 3 962 Mio libres), Uvicorn : **SERT** |

**Verdict : H VRAIE.** Alloué identique (16,78 Gio) : les piles ne coûtent rien de réel. Les 4,58 Gio, et même 7,3 Gio de
réservé non alloué, étaient des segments de 2 Mio épinglés par les petits survivants. Le regroupement les rend tous :
13,15 Gio libres, contre 10,62 avant les piles, car les originaux des experts sont rendus eux aussi. La prémisse du plan
(Σ × 1) est JUSTE : rien à compter de plus. Le correctif ne touche pas le plan.

## Scellé AVANT la fenêtre au bit (01/10, script `bit-5v7.sh`)
Bras : Coder-30B qkvo-i8c à 16 384, serveur neuf, `--no-prefix-cache`, T=0, logprobs 5, 3 invites × 128 jetons ;
`ACVRAM_PILES_COMPACTER=1` puis `0`. La compaction ne fait que recopier des octets (aucun calcul) : prédit **AU BIT**
(3/3 réponses, texte et logprobs identiques). Issues nommées :
* DIFFÉRENT → la compaction change un tenseur servi (copie fautive, mauvaise vue, ordre de pile) : PILES_COMPACTER
  ne part PAS en défaut 0.7.17 tant que la cause n'est pas nommée ; bogue, pas bruit (T=0, même serveur, mêmes invites).
* Un bras PAS SERVI ou une réponse vide → script arrêté (exit 1), aucun verdict au bit ; le témoin 0 à 16 384 devait tenir.
* vrm (qwen3-vl-30b vision, 32 768, compaction 1) : prédit **SERT** (chauffe 32 768 tenue) si l'OOM de `vision.py:415`
  venait des segments épinglés (5v7) ; OOM de nouveau → cause distincte, vrm reste ouverte, à nommer.
