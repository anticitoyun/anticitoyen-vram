# Pièce 86 — b=2 sous b=1 en service (Océane, 23/09)

Constat (pièce 85, trois bras) : b=2 99,8-247,5 t/s sur lots de 1 024 jetons, contre ~560 sur les passes courtes du
même palier et 311 à b=1.

## Lecture à sec, avant la prise (relevés /metrics du bras D, `scratchpad/oceane-p85-23-09/metrics-D.jsonl`)

Au palier 2, ~53 pas/s (≈ 19 ms/pas contre ~3,2 en graphe) ; sur 9 s (horodatages 54766-54775) `graphes_replays`
figé à 568 pendant que `steps` passe de 665 à 885, `repli_eager` = 0, `gain_moyen` 3,3-3,5 : des pas spéculatifs
hors graphe, ~40 ms chacun, sans compteur.
Chaîne : `runner.py` `_speculative_decode` — en dense, chaque séquence garde SA proposition n-gramme (0 à k jetons) →
`_build_spec_batch` pose des `query_lens` différents → `graphs.py:573-574` rend False sans `_eager` (« longueurs
mixtes ») → `self.model(batch, logits_positions=…)` en eager. La garde (`speculative.py:85-92`) compte les jetons
émis par pas, pas le temps : 3,5 jetons/pas la garde active alors que le pas coûte ~12 × un pas de graphe.
À b=1, une seule longueur par lot : clé (1, ql, nblk) capturable, d'où b=1 sain.

## H86 et ce qui la réfuterait (écrit avant la prise)

H86 : à b=2, la vérification spéculative à longueurs mêlées tombe en eager muet et coûte plus qu'elle ne rapporte.
Deux bras, serveur neuf chacun, paliers 1 puis 2 (banc de la 85, fenêtre 10 s, 1 024 jetons) :
S0 défaut (lot_max 2) ; S1 `ACVRAM_SPECULATION_LOT_MAX=1` (pas de spéculation à b=2).
Prédit : S0 b=2 ≤ 300 t/s, Δreplays/Δsteps ≤ 0,7 au palier 2 ; S1 b=2 ≥ 500 t/s, Δreplays/Δsteps ≥ 0,95 ; b=1
identique entre bras à ± 5 %.
**H86 réfutée si** : (a) S1 b=2 < 400 t/s (la lenteur n'est pas la spéculation) ; ou (b) S0 Δreplays/Δsteps ≥ 0,9
(les pas spéculatifs passent en graphe) ; ou (c) b=1 diffère de > 5 % entre bras (autre chose a bougé).
Issues nommées : H86' le proposeur coûte en Python (le chronomètre hôte le dirait, pas les replays) ; H86'' capture
d'une clé par (ql, nblk) en cours de lot (graphes_captures monterait au palier 2 — lu, pas prédit).

## Prise 1 (3ade45e8, 11 h 17-11 h 20) : NE REPRODUIT PAS — non concluante pour H86

Paliers 1 puis 2 : S0 = S1 à 0,1 % (b=1 388,4 / 389,4 ; b=2 407,9 / 407,4), replays = steps, aucune vérification
spéculative à b=2 dans aucun bras. Condition manquée, lue après coup dans `speculative.py:77-81` : la garde ne se
réarme qu'à une DESCENTE d'un lot > lot_max vers ≤ lot_max. Dans la 85 la chauffe du banc est à B = max(slots) = 12,
puis palier 2 : réarmement. Ici B = 2, la garde désactivée à b=1 (gain 1,031 < 1,05) le reste. Rejeu : paliers 2
puis 12 (chauffe à 12 d'abord, comme la 85), mêmes bras, même prédiction et mêmes réfutations pour le palier 2.
Au passage : b=2 sans spéculation, en graphe = 408 t/s (4,9 ms/pas) ; b=1 à 388 porte la spéculation (gain 1,9-3,7).

## Question de Jerome : une capture échouée doit-elle couper les graphes à vie ? (analyse à sec, non jouée)

Aujourd'hui (`graphs.py:638-672`) : toute exception de capture → `enabled = False` (669), `graphs.clear()` (670) : on
perd aussi les graphes DÉJÀ capturés et sains, pour une faute qui ne concernait qu'une clé. C'est ce qui a changé
l'incident de la 85 (un synchronize étranger, transitoire) en panne permanente.

Mon avis : ni « à vie » ni « on réessaie tout » — trier par cause, et par clé.
1. **Transitoire** (`cudaErrorStreamCaptureInvalidated` / « previous error during capture » : un autre fil a touché
   le flux global) : garder les graphes existants, marquer la clé « à réessayer », reprise après N pas (ex. 64) avec
   au plus 3 essais par clé, chaque échec compté et nommé. Sûr à trois conditions : (a) `torch.cuda.synchronize()`
   rend sans erreur après l'échec (sinon l'erreur est collante, contexte perdu : l'arrêt nommé est le seul honnête) ;
   (b) nouveau pool pour les captures suivantes — le pool partagé (`self._pool`, 1011-1013) a vu une capture
   interrompue, je ne sais pas prouver à sec que PyTorch l'a laissé cohérent ; les graphes vivants gardent l'ancien ;
   (c) états récurrents restaurés (point 3).
2. **Déterministe** (OOM, `.item()`/synchronisation DANS la capture, opération non capturable) : refuser la SEULE
   clé, pour la vie du serveur, compté et nommé ; les autres formes restent en graphe. Réessayer reproduirait l'échec
   et paierait 40-130 ms (échauffement + capture) à chaque tentative.
3. **Défaut latent trouvé en lisant, indépendant de la reprise** : `_capture` photographie les états récurrents des
   hybrides (`instantane`, 992) puis joue deux pas d'échauffement (996-999) qui les font avancer ; la restauration
   (1017-1019) n'est que sur le chemin de SUCCÈS. Une capture qui échoue après l'échauffement laisse donc l'état GDN
   avancé de deux pas, et le pas eager de repli calcule sur un état faux : sortie fausse pour les séquences du lot,
   sur les modèles hybrides seulement (Qwen3-Next, Kimi-Linear…). Correctif simple : restaurer dans un `finally`.
   Test qui casserait : un faux `linear_attn` compteur + un `step` qui lève après l'échauffement → l'état doit être
   celui d'avant. Pièce à ouvrir (priorité haute : c'est une sortie fausse, pas une lenteur).
Le cas de la 85 (cause 2) est désormais évité à la source ; la reprise par clé est une robustesse, pas l'urgence.

## Rejeu (395954d8, 11 h 26-11 h 29) : H86 TENUE

Paliers 2 puis 12 (chauffe à 12). b=2 : S0 (défaut) 293,8 t/s contre S1 (`LOT_MAX=1`) 407,6 ; pendant les pas
spéculatifs de S0 (gain 1,5-3,5) replays figés à 1 058 et `repli_eager` = 0 ; Δreplays/Δsteps sur la fenêtre = 0,80.
Réfutations (a) (S1 < 400) et (b) (≥ 0,9) non déclenchées → H86 tenue. Deux chiffres prédits MANQUÉS, dits : S1
« ≥ 500 » (407,6 : b=2 sans spéculation vaut 4,9 ms/pas, je l'avais surestimé) et ratio « ≤ 0,7 » (0,80 : la garde
coupe la spéculation au milieu de la fenêtre, gain 1,047). b=12 après b=2 : 1 871,0 (S0) / 1 838,0 (S1).
Correctif (6926d423) : `runner.py` `_speculative_decode` — sous graphe, un lot dense à longueurs de proposition
mêlées décode sans spéculer, compté (`spec_longueurs_melees` dans /metrics), avant toute réservation de blocs ;
`graphs.py` nomme et compte le refus « longueurs mêlées ». Tests : `tests/test_piece86_spec_longueurs_melees.py` (3).
Reste nommé : la garde juge en jetons/pas, pas en temps — un pas spéculatif qui coûte 2 × un pas simple et rend
1,5 jeton la garderait active. Pièce à part.

## 86 bis — l'écart de la 87 (L 1 756,4 contre N 1 883,0, −6,72 %, repli_eager 0) : prédiction avant la prise

Indices déjà en main : horloge égale dans la 87 (2 671 / 2 669) ; ma séquence 2 → 12 donne 1 838-1 871, à −0,6/−2,4 %
de N ; les séquences à paliers 4 et 8 (85-D 1 742, 87-L 1 756) sont celles qui décrochent.
Bras, serveur neuf chacun, même séance, commit du correctif 86 : N1 (12) ; L (1,2,4,8,12) ; R (12,12,12 : même
forme, autant de lots et d'âge que L, sans histoire de formes) ; N2 (12, témoin de dérive).
- **Histoire des formes** (graphes 22 contre 12, ordre et pool des sorties, allocateur par forme) : prédit R ≥ N −2 %
  et L ≤ N −4 %. Réfutée si R ≤ N −4 %.
- **Usure / âge** (fragmentation des tables de blocs KV par le va-et-vient, état hôte qui croît, chaleur) : prédit
  R ≤ N −4 %. Réfutée si R ≥ N −2 % alors que L ≤ N −4 %.
- **Lien avec b=2** (vérifications eager au palier 2, allocations hors pool) : le correctif 86 les supprime ; si L ≥
  N −3 % sur ce commit (−6,72 % pour Manon sur 08ae6ac2), le lien est COMPATIBLE — non prouvé, deux séances.
- Séance invalide si |N1 − N2| > 3 %, ou toute cellule avec `repli_eager` > 0 ou graphes capturés pendant le palier
  12 mesuré (lu dans /metrics).
Je mets 55 % sur l'histoire des formes, 35 % sur l'usure, 10 % sur le lien b=2.

## Verdict 86 + 86 bis

instrument : `scratchpad/banc-llamacpp-16-09.py` (HTTP/SSE, serveur neuf par bras) + `/metrics` toutes les 2 s ; `scratchpad/oceane-p86-23-09/prise.sh`, `prise-bis.sh`
commit : 395954d8 (rejeu 86), f2488976 (86 bis, correctif 86 inclus), bd1fb4ee (preuve) ; alias Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c
régime : -lgc 2700 (horloge moyenne 2 642-2 679), plafond 400 W, max-batch 12, max-model-len 2 304, MAX_GRAPHS 64
scellé : H86 réfutée si S1 b=2 < 400, ou replays/pas ≥ 0,9 en S0 (3ade45e8) ; 86 bis : formes réfutées si R ≤ N −4 %, usure réfutée si R ≥ N −2 % avec L ≤ N −4 %, séance invalide si |N1 − N2| > 3 % (f2488976)
mesuré : b=2 — S0 avant correctif 293,8 t/s ; S1 sans spéculation 407,6 ; S0 après correctif 407,7 (replays = pas, spec_longueurs_melees 20, repli_eager 0). b=12 — N1 1 754,3 ; N2 1 733,1 ; L (1,2,4,8,12) 1 824,1 ; R (12 × 3) 1 726,1 / 1 888,6 / 1 806,1
verdict : 86 TENUE et corrigée (b=2 +38,8 %). 86 bis : l'écart de la 87 NE SE REPRODUIT PAS — L est 4,0 % AU-DESSUS de N (moyenne N1/N2 1 743,7) ; aucune des deux hypothèses ne tient, la séance est valide (|N1 − N2| 1,2 %)
durée : prévu ≤ 12 min + ≤ 12 min / tenu 143 s (prise 1 non concluante) + 174 s + 390 s + 79 s (journal `tenue=`)

Lecture de la 86 bis : l'instrument ne tranche pas 6,7 %. À b=12, une fenêtre de 10 s = UN lot de 12 × 1 024 jetons
(`lots` = 1). R, même serveur et même forme, varie de 1 726 à 1 889 d'un palier à l'autre (écart-type 4,5 %) ; la
différence de deux cellules isolées a donc un écart-type d'environ 6,4 %, et les −6,72 % de la 87 font environ
1 σ. Le scellé ± 3 % de la 87 était sous 2 × le bruit du témoin (REGLES § 3), faute de témoin mesuré avant.
Sur le lien avec b=2 : non établi. Le correctif 86 n'était pas exercé dans L (garde éteinte dès le palier 1,
spec_longueurs_melees 0), et L se tient pourtant au-dessus de N.
Pour Manon : une cellule qui compare deux serveurs à b=12 a besoin d'au moins 5 lots par bras (fenêtre ≥ 60 s), en
ABBA, avec l'écart-type intra-bras publié. Je ne rouvre pas le chiffre de la 87 : il est réfuté comme écart, pas comme
mesure.
