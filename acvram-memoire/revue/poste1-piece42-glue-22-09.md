# Pièce 42 — attribution des 591 lancements de glue (trace du contrôle 2 de poste5), à sec — 22/09 (poste1)

## Prédictions et issues, écrites AVANT de lire la trace

* instrument : `scratchpad/poste5-p42-22-09/03b-graphe_cuda_gpu_trace.csv` (prise
  poste5, arbre `41a69f47`, alias B `…-nvfp4-qkv-alpha2-22-09`), relu avec le
  découpage EXACT de `outils/gpu/mesure/familles-noyaux.py` (marqueur
  `_route_fusee_kernel`, 48 couches, exclusion (2, 1)) et le motif `glue_torch`
  tel quel ; 0 min de carte, `nice 19`.
* **contrôle d'abord : ma population doit être la sienne.** Si je ne retrouve pas
  `glue_torch` à **0,964 ms/pas ± 3 %** ET **591 lancements/pas ± 2**, je ne
  décompose rien et je le dis : je n'aurais pas lu le même objet.
* excédent à attribuer : 0,964 − 0,044 = **0,920 ms/pas**, 591 − 17 = **574
  lancements/pas** (officiel : verdict `poste5-piece42-controle2-22-09.md`).
* méthode du site d'appel (déclarée, faute de NVTX dans cette prise) : appariement
  temporel — un noyau de glue est dit *solidaire des projections séparées* s'il
  s'exécute dans les 5 µs qui précèdent un `_dense_etroit_kernel`, et si son
  compte par pas est un multiple de 48 compatible avec 3 ou 4 GEMM par couche
  (144 ou 192). Tout autre site est nommé par son voisin.

| issue | condition chiffrée | ce qu'elle rend |
|---|---|---|
| **I1 solidaire** | ≥ 80 % de l'excédent (**≥ 0,736 ms/pas**) apparié aux `_dense_etroit_kernel` d'attention, et le reste ≤ 0,15 ms au-dessus des 0,044 officiels | (a) se scelle : la glue part avec les trois GEMM |
| **I2 défaut propre** | ≥ 20 % de l'excédent (**≥ 0,184 ms/pas**) hors de cet appariement, sur un site nommé fichier:ligne | (b) se joue sur un arbre corrigé ; la voie n'est pas close |
| **I3 les deux** | les deux seuils atteints | les deux parts chiffrées, verdict au poids dominant, aucune des deux tue |
| **I4 population fausse** | le contrôle ci-dessus échoue | aucun verdict ; je rends l'écart et je m'arrête |

Ce qui rendrait « solidaire » faux : un noyau de glue au-delà des 17 officiels
qui ne précède aucun `_dense_etroit_kernel`. Ce qui rendrait « défaut propre »
faux : la totalité des 574 lancements excédentaires appariée aux GEMM séparés.
