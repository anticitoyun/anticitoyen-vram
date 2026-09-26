# Verdict — pièce 207 (2) C3 élargie : un noyau CUDA logits + top-k + alignement — exact, mais PLUS LENT que les quatre lancements (v1 14,35 µs, v2 12,31 contre 9-10) : FAUX, rien n'entre, famille close (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p207b-25-09/banc-c3.py` (carte : top-k exact sur logits entiers sans égalité T ∈ {1, 2, 8, 16} ;
  logits aléatoires (x ~ N(0,1) bf16, w ~ 0,02) : lignes de topi égales, |Δlogits|, |Δtopw|, alignement au bit ; µs sous graphe,
  300 rejeux, témoin = F.linear bf16 + `route_fusee` (warps=1) + `moe_aligner_petit`, 4 lancements) ; résultats
  `banc-c3-v1-resultat.txt`, `banc-c3-v2-resultat.txt` ; compilation de l'extension dans chaque prise (160 s).
* **commit** : 6f895c918 (v1), v2 = commit suivant (`moe_route_align_kernel<2>`, charges groupées) — poste6-207b, NON branché,
  à ne pas fusionner (comme le C4 Triton de la 207).
* **régime** : carte 0, horloge libre, prises poste6-p207b-c3-banc (tenue 8 min dont 2,7 de compilation) et -banc2 (162 s).
* **scellé** : `scelle-banc.md` (6f895c918) + addendum v2 avant la 2e prise : fusionné 4,0-5,5 µs prédit (v1), 5,0-7,0 (v2) ;
  seuil ≥ 2,1 µs/couche (0,1 ms/pas, chef) ; falsificateur v2 « ≥ 8,2 µs → C3 mort, sans v3 ».
* **mesuré** : exactitude TENUE (v1 = v2) : top-k exact 4/4 sans égalité ; aléatoire : topi égal 8/8 (T=8), 15/16 (T=16 : une
  égalité à un ulp bf16), |Δlogits| ≤ 2⁻⁶ (un ulp bf16), |Δtopw| ≤ 0,0029, alignement au bit à eid égal, usage égal ; µs sous
  graphe T=8 : témoin 10,28 / 8,98, **fusionné 14,35 (v1) / 12,31 (v2)** ; T=1 : témoin 8,23, fusionné 10,26 / 10,26.
* **verdict** : **FAUX** deux fois ; à T=1 le fusionné coûte déjà 10,3 µs pour 8 paires : le coût n'est pas la phase logits (v2 l'a
  réduite de 2 µs à T=8 sans bouger T=1) mais le dernier bloc (sélection à tableaux locaux indexés dynamiquement `pv[32]`/`sv[32]`
  → mémoire locale, plus l'alignement) : un seul bloc de 256 fils fait en ≈ 8 µs ce que Triton (8 programmes) + l'aligneur font en
  5. Le falsificateur v2 interdit une v3 : **la famille « fusionner routage et alignement » (C4 Triton, C3 CUDA) est close**.
* **durée** : prévu 2 × ≤ 8 min ; tenu 8 + 3 min ; à sec 1 h 40.

## Ce qui reste, et ce que ça vaut
* Les leviers par lancement sur cette chaîne rendent au mieux 1,5-2 µs/couche (0,07-0,10 ms/pas, ≤ 2 %) — sous le seuil de 0,1 :
  le routage + alignement du Coder à b=8 (8,6 µs/couche, 0,41 ms/pas) n'est pas un poste à pièce.
* Restent vrais dans la 206 : A (4 couches hors tensor, +0,13 ms/pas, pièce 157 bis) et B (frontière hôte 67-80 µs/pas).
* Deux noyaux exacts existent sur les branches (Triton `_route_fusee_alignee_kernel`, CUDA `moe_route_align`) : témoins
  d'exactitude, hors main.
