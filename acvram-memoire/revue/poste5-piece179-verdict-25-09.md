# Verdict — pièce 179 : B' int8 et admission en deux pas (poste5, 25/09)

* **instrument** : `scratchpad/poste5-p179-25-09/prise.sh` (tests + cassant, `diag172.py` mixte, banc ABC CBA ABC CBA ABC
  × 2 modèles, `iso179.py`, `eng179.py`) → `prise-1.txt`, `prise-banc-mixte.txt`, `prise-2.txt`, `prise-eng.txt`,
  `cellule.jsonl`, `*.json`
* **commits** : dda271d9 (tests, diag, banc mixte), b2b90a39 (iso, banc Qwen3.8), c84849f9 (eng)
* **régime** : défaut, -lgc 2700 ; banc : 30/30 passes `repli_eager=0`, 0 nulle
* **scellé** : `revue/poste5-piece179-scelle-25-09.md` (avant ; trois ajouts datés, chacun avant la mesure suivante)
* **faute** : j'ai modifié `prise.sh` pendant que la prise du banc Qwen3.8 le lisait. Bash relit un script au fil de
  l'exécution : la fin du script est tombée en erreur de syntaxe, APRÈS les 15 passes, qui sont complètes et
  valides. Aucun serveur n'est resté en vie.

## (1) B' int8
* Tests : 131 verts (dont `test_depaq_int8_179.py` : au bit, 1 + 7, témoin GEMM groupée ≠, octets retenus ≤ réserve de
  la 172) ; bras cassant (partage retiré du chemin int8) → ROUGE.
* Logits du mixte au bit (C1 et C2) ; bras cassant LOT=1 ≠.
* En processus, 8 × L : **L = 92 : 905 → 439 ms (−51,5 %)** ; L = 120 : 958 → 492 (−48,6 %) ; **L = 78 : 1 195 =
  1 199** (GEMV : 0 déquant à partager). Diag C1 (8 × 78) 0 %, C2 mêlées −21,5 %.
* **Banc mixte : B/A −0,8 % (prédit +5 à +8 : FAUX).** La cause est trouvée par `eng179` (le moteur sans HTTP, avec
  l'invite réelle) : **l'invite du banc fait 78 jetons**, sous le seuil GEMV de 80. L'int8 y prend le GEMV (3 456 appels
  par passage), et B' n'a rien à partager ; préfill 1,19 s par lot avec ou sans B'. **Mon « L = 92 » de la 177 était
  faux** : il moyennait les requêtes de chauffe, dont les invites diffèrent. La 177 a donc surestimé ce levier. Son
  « préfill = 79 % du hors-décodage » tient.
* Banc Qwen3.8 : B/A **+0,75 %** (prédit +0,5 à +2 : tenu ; B' nvfp4 déjà au défaut, A le coupe).

## (2) Admission en deux pas
* Cause lue dans le code (scellé) : ni budget de jetons ni blocs KV. Le fil moteur sonde la file toutes les 2 ms et fait
  un pas dès qu'une requête attend, et les 8 requêtes arrivent en quelques ms.
* **C (fenêtre de 5 ms, opt-in)** : mixte **C/B +2,7 %** (332,5 contre 323,7 t/s ; 5 pas de préfill par passe contre
  9-10) ; Qwen3.8 **C/B +2,5 %** (549,8 contre 536,4 ; 7 contre 12-13). Prédit +1 à +2 % et +0,5 à +2 % : au-dessus
  des fourchettes, en faveur. J/jeton −2,2 %.
* Sortie : l'arithmétique d'un lot est inchangée, mais la COMPOSITION des lots de préfill change. Elle dépend déjà
  aujourd'hui du moment d'arrivée des requêtes (témoin T1/T2 de la 165). Défaut : décision de chef ou de l'utilisateur.

## Le vrai levier du banc, trouvé en passant
À 78 jetons par séquence, le préfill du mixte passe par le GEMV int8 par tranches, et c'est PLUS LENT que la déquant
suivie d'un GEMM à 92 jetons (1 195 contre 905 ms pour 8 séquences, 439 avec B'). Le seuil `ACVRAM_INT8_GEMV_MAX` = 80
vient d'une mesure sur UNE séquence (docstring d'`int8_matmul` : croisement vers 88 jetons). Pour un préfill de
plusieurs séquences sous B', la déquant est payée une fois : le croisement descend nettement. Abaisser le seuil DANS la
portée de B' ramènerait le préfill du lot de ≈ 1,19 à ≈ 0,45 s, soit ≈ +13 % de débit servi à ce banc (estimation).
Mais GEMV et GEMM n'ont pas la même arithmétique : la sortie change, et il faudra une KL contre témoins.

## Verdict
* B' int8 : **au bit, tenu** ; gain nul à ce banc (invites ≤ 80), −49 à −52 % du préfill au-delà de 80 jetons.
* Fenêtre d'admission : +2,5 à +2,7 % de débit servi, opt-in ; défaut sur décision.
* Levier suivant proposé : seuil GEMV/GEMM int8 abaissé sous B' (KL requise).

### Suite complète (2a6ef690 contre la base, 05:0x-05:24:26)
HEAD 1 échec, 2 794 verts ; base 0 échec, 2 793 verts. L'échec propre à la branche était
`test_octets_retenus_dans_la_reserve` : vert seul, rouge dans la suite, parce que la différence « après − avant »
comptait des allocations persistantes faites pendant la portée (espace de travail cuBLAS au premier GEMM d'une forme ;
le compte exact tombait pile sur le terme de la réserve, 231,7 Mio). Corrigé : le retenu se mesure à la SORTIE de la
portée, ce qu'elle rend. Rejeu sous verrou (e79527fb) : 2/2 seul, 39/39 au milieu des tests GPU voisins. La suite
complète n'a pas été rejouée sur ce correctif, qui ne touche qu'un test.

### Relais de la 181 (poste1) : coût réel de l'admission en deux pas
La 181 mesure un préfill servi de 1,30 à 1,46 s par lot, contre « 0,907 s groupé ». Ce 0,907 vient de mon iso de la 177
à L = 92, qui est faux (L réel = 78). Au moteur, avec l'invite réelle, le lot groupé en UN pas coûte **1,19 s**
(`eng179`). L'admission en deux pas coûte donc **0,11 à 0,27 s par lot**, et non 0,4 à 0,5. C'est cohérent avec le
gain de la fenêtre de 5 ms : +2,7 % ≈ 0,17 s par lot.

## Addendum 179 b (25/09 06 h, AVANT la prise) — coût en SOLO de la fenêtre d'admission (demande de chef)
Changement de code, avant la mesure : la fenêtre ne s'ouvre qu'en RAFALE, c'est-à-dire avec ≥ 2 requêtes déjà en file
au réveil du fil (`server/app.py`, `_attendre_les_arrivees`). Dans la version mesurée au banc, une requête seule
attendait la fenêtre entière (≥ 5 ms). Au banc, le premier pas prenait 2 ou 4 requêtes (181) : la rafale y est vue.
Instrument : `scratchpad/poste5-p179b-25-09/prise.sh`, Qwen3.8 servi, A (0) / B (5 ms), A B B A A B B A A B ; TTFT
solo (`ttft-service-p145`, une requête à la fois, L = 78 et 512) et débit b=1 (banc chat b=1).
**Prédit** : TTFT solo B − A = 0 ± 1 ms aux deux L (la fenêtre ne s'ouvre pas) ; débit b=1 B/A = 1,000 ± 0,5 %.
**Critère de chef** : TTFT solo ≤ +5 ms et débit b=1 inchangé → défaut. **FAUX** si TTFT solo > +1 ms (la fenêtre
s'ouvrirait en solo). Le gain au banc b=8 de cette nouvelle version n'est PAS remesuré ici (limite).
**Résultat 179 b (solo, 05:3x-05:53:47, 10 passes, 0 nulle)** : TTFT solo L = 78 : A 93,24 / B 93,31 ms (+0,07) ;
L = 512 : 229,16 / 229,31 (+0,15) ; débit b=1 78,8 = 78,8 t/s. **Tenu** : critère de chef (≤ +5 ms, débit inchangé)
et prédiction (0 ± 1 ms).
**Avant la mesure suivante** : gain au banc b=8 de la version rafale (Qwen3.8, A 0 / B 5 ms, 10 passes,
`prise-b8.sh`). Prédit : B/A +1,5 à +3 % (version d'origine : +2,5 %) ; FAUX si < +1 % (la rafale serait manquée au
réveil).
**Résultat banc b=8, version rafale (293d2a9f, → 06:12:22, 10 passes, 0 nulle)** : Qwen3.8 A 538,3 t/s (σ 2,6) contre
B 549,2 (σ 1,2), **B/A +2,0 %**, J/jeton −2,0 %. Prédit +1,5 à +3 % : tenu. La version rafale garde ≈ 80 % du gain
de la version d'origine (+2,5 %), sans aucun coût en solo.
