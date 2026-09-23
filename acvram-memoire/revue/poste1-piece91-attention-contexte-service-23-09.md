# Pièce 91 — attention au contexte de service : acvram DERRIÈRE vLLM de +0,145 ms/pas sur le lot (issue A) ; la p76 comparait notre ctx 290 à leur moyenne de lot — 23/09 (poste1)

* **instrument** : `scratchpad/poste1-p91-23-09/par-contexte.py` (découpage de `outils/gpu/mesure/familles-comparees.py`, un pas = 48 marqueurs ; l'indice du pas dans son lot donne ctx = 256 + i ; frontière de lot = pas dont les NOYAUX dépassent 3 × le mur médian, c'est-à-dire un préfill ; les pas longs presque vides restent dans l'indice) et `murs-longs.py` ; à sec, sur deux traces déjà prises : acvram servi `scratchpad/poste1-p82-23-09/serve-w0_cuda_gpu_trace.csv` (serve, W13=0 = défaut, alias `…-nvfp4-qkvo-i8c` de la 89, b = 12, lots de 1 024 sur invite 256) ; vLLM `scratchpad/poste1-p76-23-09/vllm_cuda_gpu_trace.csv` (vLLM 0.29.0, même invite, même lot)
* **commit** : traces bb6ec0c3 (acvram) et 730b7075 (vLLM) ; `attn_paginee.py` inchangé depuis 2980bb3b ⊂ bb6ec0c3 ; le calcul de nblk (`graphs.py:642`) est le même à HEAD ; scellé 32427ea5
* **régime** : -lgc 2700 ; acvram 2 677 MHz (`/metrics` de la prise), vLLM 2 639 MHz ; pas de correction d'horloge (une correction élargirait l'écart d'environ 1 %) ; carte : 0 s
* **scellé** (32427ea5, avant lecture) : acvram en moyenne sur le lot 0,60-0,75 ms/pas ; A : Δ ≥ +0,10 → levier nommé ; B : |Δ| < 0,05 ; C : Δ < −0,05 ; alarme : un lot qui ne fait pas 1 024 ± 8 pas → découpage faux
* **mesuré** : lot acvram de **1 027 pas** (alarme non déclenchée ; lot vLLM de 804 pas visibles, la trace s'arrête vers ctx 1 060) ; attention en µs par couche, un lancement chez nous (`_partiel_reduit_kernel`), deux chez vLLM (`kernel_unified_attention` + `reduce_segments`) :

| ctx (tranche) | 256-383 | 384-511 | 512-639 | 640-767 | 768-895 | 896-1023 | 1024-1151 | 1152-1279 | moyenne du lot | ms/pas |
|---|---|---|---|---|---|---|---|---|---|---|
| acvram | 9,53 | 10,58 | **13,64** | 14,21 | 14,88 | 15,60 | **20,26** | 21,33 | 15,00 | **0,720** |
| vLLM | 8,46 | 9,83 | 10,33 | 11,49 | 12,66 | 13,79 | 14,33 | 15,0 ¹ | 11,99 | **0,575** |

  ¹ extrapolé (+0,6 µs par tranche) ; la trace vLLM s'arrête avant.
* **verdict** : **issue A, Δ = +0,145 ms/pas** ; ma prédiction pour acvram (0,60-0,75) est tenue avec 0,720. L'attention n'est pas un gain (−0,103 dans la p76) : c'est un **surcoût**. La p76 opposait notre trace Engine de 60 pas (ctx ≈ 290, 0,435 ms) à la moyenne de lot de vLLM, d'où ce biais de 0,25 ms/pas en notre faveur. Le bilan des familles devient MoE +0,15, projections +0,238, **attention +0,145**, tête −0,177, reste ≈ −0,1, soit ≈ +0,26 de noyaux : **les noyaux expliquent à eux seuls l'essentiel des 0,292 ms/pas de la 89**.
  Deux causes lues dans la table :
  1. **Marches aux ctx 512 et 1 024** (+3,1 et +4,7 µs/couche d'un coup ; presque rien à l'intérieur d'une tranche). `_tranches` (`attn_paginee.py:232-241`) fixe C et `chunk` depuis `N = nblk`, le godet de blocs arrondi à la puissance de deux (`graphs.py:642` → `kvcache.py:39` `bucket_blocks`), pas depuis la longueur réelle. À b = 12 et HKV = 4, C = 8 : à ctx 576 dans le godet 1 024, 5 programmes sur 8 travaillent et font chacun 2 tuiles au lieu d'une. Gain estimé en suivant l'enveloppe basse (ctx 448 et 960) : **≈ −0,06 ms/pas**.
  2. **Pente** : 0,0098 µs par jeton et par couche chez nous, 0,0073 chez vLLM, soit **≈ −0,08 ms/pas** de plus à gagner sur le rendement du noyau (tranches de 64 jetons à 8 warps).
  Justesse : changer le découpage change l'ordre de fusion des tranches du softmax. Ce n'est pas au bit du défaut actuel ; même critère que w13 (ulp contre le témoin `_partiel_kernel` + KL). Remarque : aujourd'hui, le résultat d'une séquence dépend du godet de SON LOT (le plus long de ses voisins fixe N) ; un `chunk` calculé depuis `slen` seul rendrait l'attention invariante au lot.
* **trouvaille annexe** : captures paresseuses (11 dans la prise, `graphes_captures` de `/metrics`). La première traversée de ctx 512 et 1 024 coûte 3 pas de 44/22/46 ms (trous de 16 à 41 ms, noyaux normaux). Cela fait ≈ 0,11 s par clé nblk neuve, soit ≈ 0,2 ms/pas sur le premier lot long d'un serveur, 0 ensuite. Une précapture au démarrage des godets servis les supprimerait.
* **durée** : prévue 1 h + 5 min de carte ; tenue ≈ 50 min, **0 min de carte** (traces existantes)

## Ce que cela ordonne (au chef)

* Levier attention, au rang du w13 : `chunk` calculé dans le noyau depuis `slen` et C porté par le godet. Prédit −0,06 à −0,14 ms/pas, 3-5 h, 10 min de carte ; critère ulp + KL.
* w13 au décodage seul (déjà ordonné) : −0,177. Les deux ensemble : −0,24 à −0,32 contre 0,292 d'écart.
