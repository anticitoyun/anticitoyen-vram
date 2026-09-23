# Bead 1aj, volet prefill (W4A4 projections) — prédiction scellée A (vitesse)

Océane, 14/09/2026 soir, AVANT mesure. Décision de Jérôme après lecture de
deux précédents directement pertinents (message du soir, voir aussi
`acvram-memoire/revue/duel-mla-glm-14-09.md` pour le style du garde-fou
"regarde ce qui existe avant d'écrire") : découpler VITESSE et QUALITÉ
avant d'écrire quoi que ce soit dans le chemin chaud du modèle.

## Ce qu'on veut démontrer, au final (les deux volets réunis)

« W4A4 fusé sur les projections en prefill donne le temps de vLLM
(3,3 ms sur pp2048) à ≤ +0,5 % PPL. »

## Volet A — vitesse seule, qualité ignorée

**Prédiction scellée (Jérôme) :** brancher `nvfp4_gemm_grouped_mma`/`mma2`
(noyau écrit main, PAS `torch._scaled_mm` — ce chemin a déjà perdu le
10/09, `acvram/kernels/__init__.py:719-755`, coût de quantification
d'activation proportionnel non amorti) avec **E=1** (un seul "expert" =
la projection dense entière) et quantification d'activation fusée
(`nvfp4_quant_act`, bloc 16) sur les projections q/k/v/o + `lm_head` en
prefill, Coder-30B, pp2048.

- Départ mesuré : 30 ms (Laurine, laurine.md:4346 — `int8_gemv` dense,
  192 appels = 48 couches × 4 projections + lm_head compté à part).
- **Seuil de preuve : ≤ 8 ms.**
- **Seuil de réfutation : ≥ 20 ms.**
- Entre les deux : ni preuve ni réfutation, à documenter tel quel.

Si réfuté : on s'arrête là, note honnête, bead fermé côté vitesse — pas
de volet B à lancer (la qualité ne rachète pas une perte de vitesse).

## Volet B — qualité seule, vitesse ignorée (après A si A tient)

D'où viennent les +2,58 % PPL de Manon (13/09, côté MoE, même geste
W4A4, sans lissage) ? Comparer sur des activations réelles (quelques
centaines de jetons, 4 couches) l'erreur de trois schémas :
1. le nôtre (échelle globale dynamique, amax par lot ?) ;
2. le schéma ModelOpt de vLLM : échelle globale STATIQUE calibrée
   (`input_scale`) + blocs 16 e4m3 — vLLM tient sa PPL avec ça ;
3. (2) + rotation de Hadamard sur l'ACTIVATION (QuaRot/SpinQuant — la
   rotation réfutée le 12/09 portait sur les POIDS `w2`, pas sur les
   activations : régime différent, pas le même contrôle).

Rendre SNR / erreur relative par schéma. Si (2) ou (3) descend sous
l'erreur de l'int8 actuel, le seuil 0,5 % devient plausible et on écrit
le chemin qualité ; sinon, note et arrêt.

## Méthode du volet A

Micro-banc sur les VRAIS poids du modèle chargé (pas de tenseurs
synthétiques) : pour chaque couche, extraire q/k/v/o (+ `lm_head` une
fois) déjà au format `nvfp4`, construire une activation bf16 réaliste
[2048, hidden], chronométrer :
- chemin ACTUEL : `kernels.matmul(x, w)` (registre, backend résolu
  aujourd'hui pour nvfp4+prefill) ;
- chemin NOUVEAU : `nvfp4_quant_act(x)` puis `nvfp4_gemm_grouped_mma`
  avec E=1 (table d'adresse à une entrée, `tile_e/tile_t0/tile_n` via
  `MoEBlock._tuiles([2048], bt)`).

Synchronisé aux deux bouts, plusieurs répétitions, médian. Somme sur les
48 couches (× 4 projections) + lm_head une fois, comparée aux 30 ms de
départ.

Les poids INT8 PROMUS (sensibles, gardés en int8 par la conversion)
restent sur leur chemin actuel dans ce micro-banc aussi — le volet A ne
touche que les projections réellement `nvfp4`.

## Correction avant mesure : q/k/v/o sont TOUS int8 sur ce modèle, pas nvfp4

Vérifié directement sur le modèle chargé (14/09 soir) : `q_proj`,
`k_proj`, `v_proj`, `o_proj` ET `lm_head` sont TOUS au format `int8` sur
Coder-30B — aucun n'est `nvfp4`. Pas une promotion ad hoc : `convert.py`
leur donne un plancher `int8` délibéré (même raisonnement que la tête,
151 936 classes). Décision de Jérôme après ce constat : mesurer quand
même, en quantifiant CES poids int8 vers nvfp4 **en mémoire** (pas le
fichier converti, rien touché à la conversion), pour ce banc vitesse
seule — `lm_head` exclu (reste int8 dans tous les cas, décision
distincte). Le script et les seuils ci-dessus restent inchangés,
seulement la source du poids nvfp4 testé.

## Résultat (14/09 soir, carte, `outils/banc_1aj_prefill_vitesse.py`)

192 projections (48 couches × q/k/v/o), pp2048 :

```
SOMME actuel  (int8, chemin reel)                 : 30,67 ms
SOMME nouveau (nvfp4 en memoire + MMA E=1)         : 43,02 ms
Seuil de preuve      : <= 8 ms
Seuil de refutation   : >= 20 ms
VERDICT : RÉFUTATION
```

Le chemin actuel (30,67 ms) confirme le départ mesuré par Laurine
(30 ms) — méthode validée. Le nouveau chemin est **+40 % plus lent**,
pas plus rapide, uniformément sur toutes les couches et toutes les
tailles (q_proj 4096 sorties : +41 % ; k/v_proj 512 sorties : +45-55 % ;
o_proj 2048 sorties : +38 %) — un surcoût à peu près PROPORTIONNEL, pas
concentré sur une taille particulière, cohérent avec un coût fixe par
appel (quantification d'activation + ordonnancement de tuiles) qui ne
s'amortit pas mieux ici que dans le chemin `torch._scaled_mm` rejeté le
10/09 pour la même raison générale (voir section précédente). Hypothèse
non vérifiée plus loin (pas nécessaire, la mesure suffit à trancher) :
le noyau `nvfp4_gemm_grouped_mma`/`mma2` est construit pour le régime
MoE (beaucoup de petits groupes d'experts) ; en E=1 dense, il ne
retrouve pas son terrain — même limite structurelle que le chemin
`torch._scaled_mm`, sous un noyau différent.

Note en passant, PAS la mesure de qualité (volet B, à faire séparément,
sérieusement) : l'écart relatif moyen entre la sortie int8 réelle et la
sortie nvfp4-en-mémoire, sur une activation aléatoire non calibrée, est
étonnamment stable à ~13,9-14,0 % sur les 192 projections — un signal,
pas une mesure, à ne pas citer comme un chiffre de qualité.

## Verdict volet A

**Réfuté sans réserve.** Conforme au protocole écrit avant mesure : on
s'arrête là côté vitesse, pas de volet B1/B2 à lancer derrière —
« la qualité ne rachète pas une perte de vitesse ». Bead fermé côté
prefill W4A4 projections attention, sur ce noyau et cette approche
(E=1 dense sur un noyau conçu pour le groupé MoE).

# Repli W8A8 — diagnostic du chemin actuel + prédiction scellée (A')

Océane, 14/09/2026 soir, hors carte. Jérôme, après le calcul de coin de
table sur le volet A (48 couches × 2·2048²·(4096+512+512+4096) ≈ 3,7
TFLOP ; à ~150 TFLOPS bf16 réels ≈ 25 ms — nos 30,67 ms sont donc déjà
proches de la borne bf16 tensor cores, vLLM à 3,3 ms est à la borne
FP4) : diagnostiquer précisément le chemin actuel avant d'écrire quoi
que ce soit.

## Diagnostic — ce que fait le chemin int8 actuel en prefill

Lu directement dans le code (`acvram/kernels/__init__.py:625-663`,
`int8_matmul`) : bascule sur `n = x.shape[0]` (nombre de jetons) contre
`ACVRAM_INT8_GEMV_MAX` (défaut 80). À pp2048, `n=2048 ≫ 80` : chemin
`else` pris, ligne 662-663 —

```python
w = int8_dequant(t, x.dtype)                    # int8 -> bf16, dequant complete
return torch.nn.functional.linear(x, w.to(x.dtype))   # cuBLAS bf16
```

**Confirmé : déquantification int8 → bf16 complète, puis GEMM bf16 via
cuBLAS (`F.linear`).** Pas de tensor cores int8. Cohérent avec le calcul
de Jérôme : 30,67 ms ≈ la borne bf16, pas un défaut d'implémentation à
corriger — c'est la BONNE décision pour ce seuil (`ACVRAM_INT8_GEMV_MAX`
existe précisément parce que la déquantification l'emporte au-delà
d'~88 jetons, commentaire ligne 634-643) : au régime prefill, rien ne
manque, le chemin fait ce qu'il doit avec les octets qu'il a.

**Nuance à nommer avant d'écrire W8A8** : notre `INT8Tensor` est
AFFINE, pas symétrique — `qweight` en **uint8** (0-255) avec un
zero-point `zeros` par groupe (`formats.py:188-231`), pas un int8 signé
centré sur zéro. `torch._int_mm`/cuBLASLt calculent un produit
int8×int8→int32 SANS terme de zero-point : les utiliser directement sur
notre `qweight` tel quel donnerait un résultat faux (mauvaise
interprétation uint8 vs int8 signé, ET le décalage du zero-point non
corrigé). « Zéro requant » au sens strict n'est donc pas littéral : soit
(i) on replie le zero-point dans une correction post-GEMM (terme
`zero · Σactivations` par canal de sortie, standard pour un GEMM
affine — le poids qui alimente le produit matriciel reste NUMÉRIQUEMENT
le même octet, aucune perte), soit (ii) on centre le uint8 en int8
signé par un simple décalage `-128` (transformation reversible,
équivalente à re-baser le zero-point, PAS une perte de précision — un
bijection sur l'espace des 256 codes). Les deux gardent le poids
"tel quel" au sens où aucune information n'est perdue ; ça change
seulement l'arithmétique de reconstruction, pas le contenu quantifié.
Je pars sur (ii), plus simple à câbler et strictement équivalente à (i).

## Prédiction scellée A' (vitesse), Jérôme

- Départ : 30,67 ms (mesuré ce soir, volet A, chemin actuel confirmé
  inchangé).
- **Chemin testé** : poids int8 affine → décalage en int8 signé (zéro
  perte), activation int8 PAR JETON dynamique (échelle par ligne,
  calculée à la volée), produit via `torch._int_mm` (tensor cores int8,
  2× le débit bf16 en théorie).
- **Seuil de preuve : ≤ 16 ms** (borne haute de la fourchette 14-16 ms).
- **Seuil de réfutation : ≥ 24 ms.**
- Qualité (volet B', à faire séparément si A' tient) : PPL Coder-30B
  ≤ +0,3 % — activation int8 dynamique par jeton, perte attendue
  ≤ 0,1 % d'après Jérôme, à vérifier, pas supposée.

## Résultat A' (14/09 soir, carte, `outils/banc_1aj_prefill_w8a8_vitesse.py`)

192 projections (48 couches × q/k/v/o), pp2048, poids int8 requantifié
en int8 signé PAR CANAL DE SORTIE (amax par ligne, format W8A8 standard
SmoothQuant/TensorRT — confirmé par Jérôme avant la mesure : nos groupes
int8 courent le long de K, donc un seul `torch._int_mm` n'est correct
qu'avec une échelle par ligne, pas par groupe) :

```
SOMME actuel  (int8 -> bf16 -> cuBLAS, chemin reel)  : 30,85 ms
SOMME nouveau (int8 signe par ligne + torch._int_mm) : 42,28 ms
Seuil de preuve      : <= 16 ms
Seuil de refutation   : >= 24 ms
VERDICT : RÉFUTATION
```

**Réfuté, avec une marge confortable** (42,28 ms contre un seuil de
24 ms — 76 % au-delà). Motif visible dans le détail par couche : les
petites projections (k_proj/v_proj, 512 sorties) sont proportionnellement
les PLUS touchées (0,042 → 0,110 ms, ×2,6) — signature d'un coût FIXE
par appel (quantification d'activation + lancement `torch._int_mm`) qui
domine sur les petites matrices, plus visible ici que sur q_proj/o_proj.

**Réserve méthodologique à nommer, ne change pas le verdict** : q/k/v
partagent la MÊME activation d'entrée dans une vraie couche, mais ce
banc requantifie l'activation en int8 séparément pour chacun des 4
appels par couche (mesure isolée, pas une intégration réelle) — une
implémentation qui partagerait la quantification d'activation entre
q/k/v économiserait une partie du coût fixe observé sur k_proj/v_proj.
Non mesuré ici (pas nécessaire : 42,28 ms est à 76 % au-delà du seuil de
réfutation, une économie partagée plausible ne suffirait pas à
retraverser 24 ms). Si le chantier veut un chiffre plus juste pour cette
hypothèse précise, il resterait à mesurer, pas à supposer.

## Contrôle de Jérôme : décomposer int_mm seul / quantification seule

Objection reçue avant de fermer : le verdict A' porte sur « quantification
torch non fusée + int_mm », pas sur `int_mm` seul — la quantification
(amax/div/round/clamp/cast, plusieurs lancements séparés) pourrait
dominer artificiellement. Mesuré séparément (`outils/banc_1aj_w8a8_decompose.py`),
mêmes 192 formes, activations PRÉQUANTIFIÉES hors chronomètre pour (1) :

```
SOMME int_mm seul   (activations dejà int8) : 28,22 ms
SOMME quantification seule                   : 10,03 ms
Seuil de preuve (int_mm seul)      : <= 16 ms
Seuil de refutation (int_mm seul)   : >= 24 ms
VERDICT : FERMÉ POUR DE BON (28,22 ms >= 24 ms)
```

**`int_mm` seul dépasse déjà le seuil de réfutation.** La quantification
non fusée (10,03 ms) explique une partie de l'écart mais pas
l'essentiel : même en la retirant complètement du chronomètre,
`torch._int_mm`/cuBLASLt int8 ne descend qu'à 28,22 ms — proche de la
borne bf16 déjà en place (30,85 ms), pas le ×2 attendu des tensor cores
int8. L'estimation de coin de table de Jérôme (0,05-0,08 ms par appel,
≈10-15 ms au total) ne tient pas dans ce régime de formes (2048 jetons ×
512-4096) sur cette carte — cause non creusée plus loin (pas nécessaire :
la mesure suffit à trancher, et le protocole prévoyait explicitement
l'arrêt si `int_mm` seul ≥ 24 ms).

**Réserve (pas un chantier, juste nommer la cause)** : 28,2 ms pour
3,7 T-op ≈ 130 TOPS, moins de 10 % du pic int8 tensor cores de la 5090 —
contre le bf16 à 30,9 ms, ~60 % de son pic. `torch._int_mm` est mal
servi par cuBLASLt sur ces formes précises, ce n'est pas une limite
matérielle. Si ce levier revient un jour, ce sera par un GEMM CUTLASS
int8/FP4 dédié, jamais par `torch._int_mm`.

## Verdict repli W8A8

**Réfuté sans réserve utile.** Même symptôme que le W4A4 : un coût fixe
par appel qui ne s'amortit pas sur des projections de cette taille.
`torch._int_mm`/cuBLASLt int8 ne rattrape pas le cuBLAS bf16 déjà
utilisé aujourd'hui, dans ce régime. Bead 1aj fermé côté prefill
projections attention — les deux pistes de repli tentées (W4A4 noyau
groupé E=1, W8A8 `torch._int_mm`) perdent toutes les deux dans le même
régime, pour une raison structurelle proche (coût fixe par appel,
prefill à gros lot déjà proche de sa borne bf16).
