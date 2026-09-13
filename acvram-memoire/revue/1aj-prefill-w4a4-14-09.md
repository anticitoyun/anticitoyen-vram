# Bead 1aj, volet prefill (W4A4 projections) — prédiction scellée A (vitesse)

poste1, 14/09/2026 soir, AVANT mesure. Décision de chef après lecture de
deux précédents directement pertinents (message du soir, voir aussi
`acvram-memoire/revue/duel-mla-glm-14-09.md` pour le style du garde-fou
"regarde ce qui existe avant d'écrire") : découpler VITESSE et QUALITÉ
avant d'écrire quoi que ce soit dans le chemin chaud du modèle.

## Ce qu'on veut démontrer, au final (les deux volets réunis)

« W4A4 fusé sur les projections en prefill donne le temps de vLLM
(3,3 ms sur pp2048) à ≤ +0,5 % PPL. »

## Volet A — vitesse seule, qualité ignorée

**Prédiction scellée (chef) :** brancher `nvfp4_gemm_grouped_mma`/`mma2`
(noyau écrit main, PAS `torch._scaled_mm` — ce chemin a déjà perdu le
10/09, `acvram/kernels/__init__.py:719-755`, coût de quantification
d'activation proportionnel non amorti) avec **E=1** (un seul "expert" =
la projection dense entière) et quantification d'activation fusée
(`nvfp4_quant_act`, bloc 16) sur les projections q/k/v/o + `lm_head` en
prefill, Coder-30B, pp2048.

- Départ mesuré : 30 ms (poste4, poste4.md:4346 — `int8_gemv` dense,
  192 appels = 48 couches × 4 projections + lm_head compté à part).
- **Seuil de preuve : ≤ 8 ms.**
- **Seuil de réfutation : ≥ 20 ms.**
- Entre les deux : ni preuve ni réfutation, à documenter tel quel.

Si réfuté : on s'arrête là, note honnête, bead fermé côté vitesse — pas
de volet B à lancer (la qualité ne rachète pas une perte de vitesse).

## Volet B — qualité seule, vitesse ignorée (après A si A tient)

D'où viennent les +2,58 % PPL de poste2 (13/09, côté MoE, même geste
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
151 936 classes). Décision de chef après ce constat : mesurer quand
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

Le chemin actuel (30,67 ms) confirme le départ mesuré par poste4
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
