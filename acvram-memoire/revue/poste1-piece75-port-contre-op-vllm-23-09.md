# Pièce 75 — notre port Marlin MoE contre l'op `fused_marlin_moe` de vLLM, au même banc, sur les mêmes octets — 23/09 (poste1)

## Pourquoi

La 74 a fermé l'illusion d'octets : vLLM touche 32,80 experts distincts par couche, nous 32,46. Le même noyau
(la 73 : arguments identiques, notre port est leur source) tourne donc à 1,67 To/s chez eux et à 1,37 chez nous
en service. Deux familles de causes restent, que les compteurs ncu (48, sudo) ne sont pas seuls à séparer :

* **contexte du service** : L2 réchauffée par les noyaux voisins, disposition des poids en mémoire, forme en
  3 lancements chez nous contre 2 chez eux ;
* **binaire** : même source, compilée autrement (drapeaux nvcc, `sm_120` contre `sm_120a`, instanciations).

Les chiffres de banc existants ne tranchent pas : ils viennent de deux bancs différents (poste5 61/0 : op vLLM
57-67 µs/couche à L2 froide ; 62 A1 : notre port 49,8 à L2 tiède).

## Protocole

`outils/gpu/mesure/banc-marlin-moe-p75.py`, un seul fichier et trois sous-commandes :

1. `preparer` (venv vLLM) : une pile de 128 experts aux formes du Coder (K 2 048, N 768, gate‖up 1 536, down
   768 → 2 048), reconditionnée par **leur** `rand_marlin_weight_nvfp4_like` (groupe 16), écrite une fois sur le
   disque avec x, les 20 routages réels de la cellule b = 12 (`poste5-p62-23-09/routages-cellule-b12-20.pt`,
   27 → 41 experts distincts) et des poids de routage déterministes.
2. `vllm` (venv de la cellule, `/opt/ia/vLLM/.venv`) : `fused_marlin_moe` tel que vLLM l'appelle.
3. `acvram` (notre venv, PYTHONPATH sur ce worktree) : (A2) **notre port avec les arguments de vLLM**, 2 GEMM
   w13 puis down sur **les mêmes tenseurs** ; (A3) **notre forme servie**, `gemm_experts_tensor` + `moe_reduce`,
   gate et up découpées dans w13 (la disposition Marlin est locale par tuile de 64 colonnes, donc la découpe
   donne les mêmes octets, pas des octets voisins).

Harnais identique, octet pour octet, dans les deux processus :

* graphe CUDA de 8 appels, chacun sur une copie distincte de la pile (**L2 froide** : 8 × ≈ 78 Mo touchés,
  contre 96 Mo de L2) ; puis 8 appels sur la même copie (**L2 tiède**) ; 100 rejeux, médiane, ÷ 8 ;
* profileur torch : µs par lancement du noyau `Marlin`, séparé du reste de l'op ;
* sortie de chaque bras comparée à celle de vLLM (écart relatif max) : c'est la preuve que le même travail
  est fait.
* -lgc 2700, compute-apps au début et à la fin, ≤ 5 min.

## Prédiction et issues, écrites AVANT la prise

Grandeur jugée : **µs par lancement du noyau Marlin, à L2 froide, A2 contre V** (même source, mêmes arguments,
mêmes octets, même routage). Le total de l'op est publié à côté, mais ne juge pas : la glue diffère par
construction.

| issue | condition | ce qu'elle rend |
|---|---|---|
| **J1 — jeu égal** | \|A2 − V\| ≤ 5 % | le binaire n'y est pour rien : l'écart de la 73 est le **contexte du service** (L2, voisins, disposition, 3 lancements) ; la suite se cherche là, sans la 48 |
| **J2 — l'op vLLM plus rapide** | V < A2 de plus de 5 % | compilation ou instanciation : la 48 devient indispensable, et les drapeaux nvcc des deux .so passent avant |
| **J3 — notre port plus rapide** | A2 < V de plus de 5 % | le binaire est hors de cause, et l'écart de service est encore plus du contexte que J1 ne le dit |

* **prédiction nominale : J1**, avec A2/V entre 0,97 et 1,03.
* **ce qui me gênerait : J2.** J'ai écrit dans la 73 « même noyau » ; J2 dirait que la même source ne donne pas
  le même noyau, et que « mêmes arguments » ne suffisait pas.
* **alarme** : si la sortie de A2 s'écarte de celle de V de plus de 1·10⁻² en relatif, les deux bras ne font pas
  le même calcul (échelles, ordre gate/up) ; aucune conclusion, je le dis.
* **alarme 2** : si V à L2 froide s'écarte de plus de 15 % des 57-67 µs/couche de la 61/0, le harnais n'est pas
  le même régime que celui de poste5, et je le nomme avant de conclure.

## Verdict — 23/09 08 h 52 (poste1)

* **instrument** : `outils/gpu/mesure/banc-marlin-moe-p75.py`, chaîne `scratchpad/poste1-p75-23-09/chaine.sh` ; résultats `resultat-vllm.json`, `resultat-acvram.json`
* **commit** : 9a5dd937 (arbre mesuré) ; vLLM 0.29.0 / torch 2.13 (`/opt/ia/vLLM/.venv`) ; acvram torch 2.14, port `acvram_marlin.so`
* **régime** : -lgc 2700 (2 692 MHz relevés), b = 12, k = 8, E = 128, les 20 routages réels de la cellule (27 → 41 distincts), pile reconditionnée par `rand_marlin_weight_nvfp4_like`, relue du même fichier par les deux venvs ; graphe de 8 appels, 100 rejeux ; compute-apps début = fin = llama-server 4627
* **scellé** : J1 si \|A2 − V\| ≤ 5 % sur les µs du noyau Marlin à L2 froide ; J2 si V plus rapide de plus de 5 % ; J3 si A2 plus rapide de plus de 5 %
* **mesuré** (µs par couche, médianes sur les 20 routages) :

| bras | noyau Marlin, L2 froide | par lancement | hors Marlin | total, L2 froide | total, L2 tiède | écart de sortie contre V |
|---|---|---|---|---|---|---|
| **V** — `fused_marlin_moe` vLLM | **62,39** | w13 40,06 · down 22,40 | 7,76 | 69,45 | 36,80 | — |
| **A2** — notre port, arguments vLLM | **62,20 (−0,3 %)** | w13 39,98 · down 22,30 | 13,46 (glue torch du banc) | 75,57 | 43,55 | 1,18·10⁻² |
| **A3** — notre forme servie | 66,78 | gate 22,56 · up 22,27 · down 22,18 | 5,03 | 73,25 | 41,84 | 8,3·10⁻³ |

  Même instanciation des deux côtés (`marlin_moe_wna16::Marlin<1125899906909960l, …>`).
* **verdict : J1.** Sur les mêmes octets, avec le même routage et la même horloge, notre binaire fait exactement le travail du leur (−0,3 %). **La compilation et le noyau sont hors de cause** ; l'écart de la 73 n'est pas dans le binaire.
* **durée** : prévue ≤ 5 min ; tenue 3 min 01 (08:49:12-08:52:13)

### Ce qui reste de l'écart de la 73, et un confondant trouvé

* **Notre service colle au banc, celui de vLLM non.** Nous : 64,6 µs/couche en service (p73) contre 66,8 pour A3 au
  banc froid. vLLM : **53,1** en service (p59) contre **62,4** au banc froid.
* **Confondant : la trace vLLM de la p59 a tourné à horloge libre** (`vllm-b12.json.12.json` : horloge moyenne
  **2 905 MHz**, bridage puissance, pas de -lgc) ; la nôtre (p73) et ce banc tournaient à **2 692**. Le « 1,67 contre
  1,37 To/s » de la 73 compare donc deux régimes d'horloge (+7,9 % de fréquence SM chez eux). Ce décalage
  expliquerait au plus 7,9 % des 17 % d'écart, et seulement si le noyau suit la fréquence SM, ce qui n'est pas mesuré.
  Le reste, ~9 %, est du contexte de service, à chercher chez vLLM et pas dans notre noyau.
* **Notre forme en 3 lancements** coûte +4,6 µs/couche contre w13 fusionné (66,8 contre 62,2), soit **0,22 ms/pas**.
  C'est cohérent avec la 71 bis (−4,0) et sous le seuil de 0,3 ms.
* **La glue servie est plus légère que celle de vLLM** : 5,0 contre 7,8 µs/couche.

### Alarme 1 franchie de justesse par A2, et ce qu'elle vaut

A2 s'écarte de V de **1,18·10⁻²** (seuil écrit : 1·10⁻²) ; A3 tient (8,3·10⁻³). A2 et A3 lisent **les mêmes poids et
les mêmes échelles** par le même noyau ; ils ne diffèrent que par la glue du banc (silu·up en fp32 torch, somme en
bf16) — une erreur d'échelle ou d'ordre gate/up donnerait un écart d'ordre 1, pas 10⁻². Le seuil était trop serré
pour une sortie bf16 (1 ulp au maximum = 3,9-7,8·10⁻³, donc 1·10⁻² ≈ 1,3-2,6 ulp). Je le déclare franchi, je ne
le déplace pas. Le verdict tient par A3 (sous le seuil) et par l'identité des temps du noyau, qui ne dépend pas
des valeurs.

### Suite proposée (non engagée)

Retracer vLLM en service **à -lgc 2700** (même chaîne que la p59, nsys, ≤ 5 min) avant de toucher la 48 :
* si leur Marlin remonte vers 58-62 µs/couche, l'écart MoE de la 73 était surtout l'horloge. Les 9,1 % de la p64
  (même horloge) se cherchent alors dans les projections (+0,57 ms) et l'hôte ;
* s'il reste à ~53, le contexte de service de vLLM rend au MoE ~15 % que notre banc ne reproduit pas, et c'est
  leur service qu'il faut disséquer.
Dans les deux cas, la pièce 48 (ncu) ne nommerait qu'un noyau dont on sait maintenant qu'il est identique : elle
n'est plus indispensable pour cette question.
