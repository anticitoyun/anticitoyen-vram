# Diagnostic 153 — VRAM eager + tranchage PPL (25/09, à sec, branche poste4-153b post-purge)

instrument : lecture de code seule (`acvram/engine/loader.py`, `acvram/evaluate.py`), aucune prise carte
commit : 0c54811f (origin/main post-purge)
régime : —
scellé : scratchpad/poste4-piece153-attn-gdn-int8-mlp-nvfp4-24-09/, revue/poste4-piece153-scelle-25-09.md
mesuré : —
verdict : deux causes distinctes, aucune n'est un bogue du planificateur
durée : 12 min

## 1. Repli eager (580 Mio libres après chargement)

Pas un défaut de `_replanifier` (KV déjà dimensionné pour 1 séquence depuis le correctif du 22/09,
confirmé par `ppl.json` : « budget : 1 séquence × 4112 jetons »). Cause réelle : le format mixte
promeut 308 tenseurs attention/GDN de nvfp4 (4 bit) à int8 (8 bit) — poids logiques 6,8 Gio int8 +
9,8 Gio nvfp4 + 3,3 Gio bf16 ≈ 19,9 Gio, CONTRE ~13 Gio pour un nvfp4 pur de même taille (scellé 102).
La promotion coûte réellement ses octets (2× les tenseurs promus), pas seulement en théorie SNR/PPL —
c'est cette marge perdue qui manque aux 1024 Mio exigés par la capture de graphes. Rien à corriger côté
loader ; c'est le coût attendu de `--attn-qkvo-int8-canal`/`--gdn-int8-canal`, à chiffrer dans le scellé
suivant (Gio supplémentaires par tenseur promu, pas juste le SNR).

## 2. Tranchage PPL : NON contourné (hypothèse du carnet réfutée à la lecture)

`_pertes_par_tranches` (evaluate.py:167-182) tranche bien AVANT `_tete`/`_logits_finaux` — `h` est
rendu une fois par `model(batch, return_hidden=True)` (états cachés normalisés, pas les logits), chaque
tranche de `_PPL_TRANCHE` (256 par défaut) applique la tête séparément (commentaire ligne ~145,
confirmé par le motif déjà couvert par `tests/test_ppl_tranches.py`, écrit pour un OOM antérieur du
même genre sur un i8c). Un tranche de 256 = 256×151936×4o ≈ 155 Mio, pas 2,49 Gio. L'allocation qui
échoue (2,49 puis 4,74 Gio dans `ppl-153.err`) n'est pas expliquée par le tranchage — probablement le
même déficit de marge que le point 1 (mixte int8/nvfp4/bf16 laisse ~0,4-0,6 Gio libres avant même
l'évaluation) plutôt qu'un bogue de `_pertes_par_tranches`.

## 25/09 11 h — prises sous feu chef (deux conditions), poste4-p153-*

**Condition 1 (PPL/KL fenêtre 2048/2048, wiki-gptq) : OOM identique, ARRÊTÉ (pas de nouvel essai).**
Même échec exact qu'à window=4096 : tentative d'allouer 5 085 593 600 o (4,74 Gio), 1,32 Gio libres sur
31,36 Gio, 30,02 Gio déjà utilisés. **Octets IDENTIQUES entre 4096 et 2048** — l'allocation qui échoue
ne dépend donc PAS de `window` : ni un effet de tranchage PPL, ni un effet de taille de fenêtre. Cause
encore non identifiée, mais le champ des hypothèses est réduit à quelque chose de fixe (chargement du
modèle mixte lui-même, ou une allocation à taille constante indépendante de l'éval). Ligne de régime
complète (avant l'échec) :
`[régime] ACVRAM_CPUS=0-15 ACVRAM_GEMV_LAYOUT=marlin gemv_splitk=S(auto) ACVRAM_GDN=fla ab=auto(0:format×48) extension=oui torch=2.14.0+cu130 triton=3.8.0 fla=0.5.2 hote=thp,omp8,cpus0-15 mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact eco=2700(2670) dense=triton≥2|cuda+marlin(doubles=0,seuls=129,0.00Go,kv=2064,exclus=0,replis=0) marlin_port_so=732b95dea8675d33`
KL T1/T2 non tentés (bloqués par le même OOM, comme prévu par le scellé).

**Condition 2 (ABBA b=1/b=8, graphes=on) : NON MESURÉE — carte jamais obtenue.** Trafic circuit dense
(poste2-p190, poste5-191, poste1-p185c, poste6-p193 en file continue) : attente 1801 s (30 min, plafond
`carte.sh`), ABANDON sans avoir chargé le modèle une seule fois. Pas de deuxième essai identique lancé
(règle « on ne relance pas à l'identique » + plafond de prise) — en attente d'un créneau ou d'un arbitrage
de priorité de chef.

## 25/09 12 h — cause trouvée et corrigée (piste chiffrée de chef)

chef a factorisé 5 085 593 600 = 248 320 × 5 120 × 4 = vocab × hidden × fp32 (config.json de ce modèle,
vocab étendu vision : 248 320, pas les 151 936 habituels). Trouvé : **`acvram/kernels/__init__.py:747`**
(`nvfp4_matmul`, repli « naturel » du mode prefill bf16 par défaut, pris quand `n=256 > gemv_threshold`
— jamais le cas en service où `n≤12`, toujours le cas en PPL par tranches de 256). Ce repli déquantifiait
la matrice ENTIÈRE en bf16 (`nvfp4_dequant(t, bf16)`, 2,49 Gio) **puis** la recopiait entière en fp32
(`w.to(x.dtype)`, 4,74 Gio) — deux allocations plein tenseur d'affilée, jamais tranchées (contrairement au
repli int8 voisin, déjà corrigé pour Gemma-4-31B, poste3 0cf7fe6, jamais porté ici).

**Corrigé** : tranchage par lignes de sortie borné par `_DEQUANT_TRANCHE_MAX` (même motif que le repli
int8). **Test** : `tests/test_nvfp4_matmul_tranches.py` (3 tests, CUDA réelle sous `carte.sh`, skip sans
GPU). **Écart trouvé en testant** : PAS au bit sur GPU — cuBLAS choisit un ordre de réduction K différent
selon la largeur N du GEMM (40×1000 en un appel vs 8× 40×≤128) ; écart absolu mesuré 1,53e-5 sur des
logits ~O(1-10) (`scratchpad/poste4-p153-25-09/test-tranches-cuda.py`), homogène sur les 1000 colonnes
(signature de non-associativité flottante, pas un bug localisé). Négligeable devant tout seuil KL du
protocole (0,5-1,4), mais ce n'est PAS la promesse « au bit » de REGLES §1 pour le défaut — je le signale
tel quel plutôt que de le cacher sous `torch.equal`. **À ton arbitrage** : ce chemin n'est de toute façon
jamais servi (n≤12 en service, seuil GEMV jamais franchi) — accepter l'écart nommé pour PPL/KL seulement,
ou une autre piste (algo cuBLAS déterministe, buffer combiné) si tu préfères le bit strict même ici.

## 25/09 13 h — mesure du scellé après correctif (arbitrage chef accepté)

**PPL fenêtre 2048/2048, wiki-gptq réel (correction : chemin corpus faux dans ma 1ère prise, corrigé
`/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt`, PAS `/mnt/AI_GENERATOR/...`) : RÉUSSIE, plus d'OOM.**
`perplexity: 7.2157`, 8188 jetons, 4 fenêtres, régime nominal. KL T1/T2 contre HF pas encore tentée
(pas d'instrument prêt dans ce dépôt pour ce protocole précis — à écrire si tu veux ce chiffre).

**ABBA b=1/b=8, graphes=on : régime confirmé `graphes=on` aux DEUX bras (hybrides≤4 et ≤8), mais les
deux CRASHENT avant toute mesure de débit** — pas un repli eager, un OOM différent, à chaque bras :
- b=1 : `acvram/kernels/__init__.py:868` (`_i8c_poids`, projections attention int8 partagées),
  120 Mio demandés, 37,94 Mio libres.
- b=8 : `acvram/kernels/__init__.py:1121` (mon propre correctif, `_marlin_seul` prefill) — cette fois
  c'est `MP.depaqueter_marlin` lui-même (son buffer initial, PAS le cast que j'ai tranché) qui échoue :
  340 Mio demandés, 265,94 Mio libres.

Les deux OOM confirment le diagnostic du 25/09 11h (pas de nouveauté) : à ce point de la mesure, 31,08-
31,30 Gio sur 31,36 sont déjà pris — quelques centaines de Mio de marge, insuffisants pour presque
n'importe quelle allocation transitoire. Mes deux correctifs (naturel + marlin_seul) réparent la
matérialisation EN DOUBLE d'un tenseur DÉJÀ TROP GROS ; ils ne créent pas de marge. J'arrête ici comme
demandé — pas de nouvel essai, pas de mesure de débit à donner.

## 25/09 13 h 4x — pièce 153c (ordre chef) : cause trouvée et chiffrée, correctif écrit,
## MAIS SANS EFFET sur le crash — plancher KV, pas la réserve, qui décide ici

**fichier:ligne** (comme demandé) : `config.py:395` `activations_prefill_bytes` (+ nouvelle méthode
`octets_transitoires_i8c_bytes`, `config.py`) et `loader.py:1889` `_reserve_prefill` (+ nouvelle fonction
`_plus_grosse_nvfp4_marlin_bytes`, `loader.py`) → `loader.py:1173` `_marge_carte`. La réserve ne comptait
QUE les tenseurs nvfp4 « doubles » (`_octets_marlin`, `loader.py:1869`, filtre explicite
`format != "nvfp4"`) : les 308 tenseurs promus int8 (pièce 153) et un gate+up nvfp4 fusionné en un seul
tenseur n'y existaient pas. **Chiffré (config Qwen3.8, branche poste4-153c)** : pic i8c = 284,1 Mio
(GDN qkv+gate+alpha+beta+out domine sur attn q+k+v+o, 83,9 Mio) — indépendant de b (poids, pas
activations) ; pic gate+up fusionné = 356,5 Mio (lu au manifeste, `_plus_grosse_nvfp4_marlin_bytes`, pas
à l'architecture seule — sinon le 70B de poste3, jamais fusionné, aurait été sur-réservé de 448 Mio).

**Correctif écrit, testé (22 tests, dont 2 nouveaux fichiers, tous verts, `test_reserve_depaq_172.py`
adapté avec raison documentée puisque mon nouveau terme domine désormais sur ces dimensions), poussé
(poste4-153c)** : la réserve inclut maintenant `max(plus_grosse, poids_bf16_couche_lineaire, i8c)` et
`+ plus_grosse_nvfp4_marlin_manifeste`.

**ABBA rejoué (même modèle, même script) : CRASH IDENTIQUE AU BIT** (mêmes deux sites, mêmes deux
tailles de tentative, 120 Mio/37,94 libres et 340 Mio/265,94 libres). `kv_budget=2560/1` **inchangé**
avant/après le correctif — la cause : `_kv_plancher` (`loader.py:1229`) impose un PLANCHER (KV minimal
pour UNE séquence à `max_model_len`) que la réserve ne peut pas faire descendre plus bas ; ici le budget
EST déjà à ce plancher. Ma réserve plus grosse ne réduit donc RIEN (il n'y a plus de KV « au-dessus du
plancher » à retirer), et rien dans le chemin de chargement ne déclenche d'EXIL de couches en réponse à
une réserve trop grosse pour la marge restante (0/64 exilées, identique avant/après) : le couplage
réserve → exil, qui existe pour d'autres cas (poids seuls dépassant la capacité), ne semble pas réagir à
CE type de dépassement (poids + plancher KV + réserve > capacité, poids seuls < capacité). C'est un
mécanisme différent de celui que j'ai corrigé, plus profond, que je n'ai pas touché.

J'arrête ici comme la dernière fois : mon correctif est correct et testé pour ce qu'il fait (compter
ces tampons), mais insuffisant seul à empêcher ce crash précis — il faudrait soit forcer un exil quand
poids + plancher + réserve dépasse la marge, soit réduire le plancher lui-même pour ce cas, et je n'ai
pas d'ordre pour toucher à ce mécanisme.

## Suite proposée

Le format mixte lui-même est plus lourd, pas le planificateur. Options avant nouvelle mesure : (a)
réduire `--max-model-len`/window pour l'éval PPL sur ce format précis, (b) `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` (suggéré par le message CUDA), (c) mesurer sur une carte à plus de marge, ou
(d) accepter que ce format mixte n'a pas assez de VRAM libre pour CUDA-graphes/PPL fenêtre pleine sur
cette carte et l'écrire comme contrainte du format, pas comme régression à corriger.
