# Verdict — pièce 207 (1) C4 : l'aligneur dans route_fusee (Triton, dernier programme) est AU BIT mais PLUS LENT que les deux lancements — FAUX, rien n'est intégré (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p207-25-09/banc-c4.py` (carte : au bit contre `route_fusee` + `ext.moe_aligner_petit`
  sur T ∈ {1, 2, 5, 8, 12, 16}, k 8, E 128, égalités et fantômes ; µs par chaîne sous graphe, 300 rejeux) ; résultat
  `banc-c4-resultat.txt` ; à sec : référence Python de l'aligneur (4/4 au bit sous interpréteur).
* **commit** : 20bc7bb49 (poste6-207 : noyau `_route_fusee_alignee_kernel` + `route_fusee_alignee` dans `route_prep.py`,
  NON branché, à ne pas fusionner tel quel).
* **régime** : carte 0, horloge libre, prise poste6-p207-c4-banc (tenue 163 s après 6 min de file).
* **scellé** : `scelle-c4-banc.md` (20bc7bb49, avant) : fusionné 3,3-3,9 µs prédit, gain 1,4-2,0 µs/couche ; falsificateur
  « fusionné ≥ 4,7 µs → C4 mort ».
* **mesuré** : au bit **6/6** (topw, topi, eid, usage, sorted_ids, expert_ids, num_post) ; µs sous graphe : témoin 2 lancements
  6,19 (T=1 et T=8), route_fusee seul 4,11, **fusionné 8,21 (T=1) / 10,25 (T=8)** → gain **−2,0 / −4,1 µs**.
* **verdict** : **FAUX** — falsificateur déclenché ; l'alignement par UN programme Triton (matrices 128 × 64, 128 × 128, 64 × 64,
  cumsum, 8 écritures de sentinelle, barrières) coûte ≈ 6 µs là où l'aligneur CUDA à 256 fils en met 2,4 : le nœud de graphe
  économisé (0,19 µs) ne pèse rien devant. C4 seul est mort sous cette forme ; rien n'entre au défaut.
* **durée** : prévu ≤ 3 min ; tenu 163 s ; à sec 50 min (noyau + banc).

## Ce qui reste vrai, et la suite proposée (décision chef)
* L'exactitude entière est acquise : la convention de l'aligneur (rang stable, sentinelle G, −1 au-delà) est reproduite au bit.
* Le gain visé (aligneur 2,4 µs + nœud) n'est atteignable qu'en gardant 256 fils : c'est-à-dire en CUDA, dans le noyau de
  l'aligneur lui-même — qui pourrait alors absorber aussi la sélection top-k ET les logits (C3) : un seul noyau CUDA « logits +
  top-k + alignement » (256 fils : logits [8 × 128] depuis x [8 × 2048] et w [128 × 2048] en L2, un warp par ligne pour le top-k,
  bloc 0 = l'aligneur actuel) contre 8,6 µs + 3 nœuds aujourd'hui (cutlass 2,56 + splitK 0,93 + route_fusee 2,69 + aligneur 2,43) ;
  visé ≤ 4,5 µs → **≈ 0,2 ms/pas (4 %) à b=8**, **hors bit** (ordre des sommes fp32 des logits et de la sélection ≠ Triton) →
  c'est la pièce (2) C3, élargie ; elle exige le scellé KL à témoins d'échantillon égal avant, comme ordonné.
* Je ne code pas (2) sans feu ; le noyau Triton de (1) reste sur la branche comme témoin d'exactitude, hors main.
