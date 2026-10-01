# ddw — décodage KDA b=1 : la tuile S[h] en mémoire partagée (poste5, 01/10) — borne et prédiction SCELLÉES avant toute mesure

Ordre de chef (bd ddw, P1). Base : 84123994d (fusion 61w, pas encore sur main 3f10000bd). Branche poste5-ddw.

## Borne (Kimi-Linear-35B, b=1)

* 20 couches KDA, 32 têtes, D = 128, S fp32 = 2 Mio par couche (poste1-p81-cake-kda-01-10.md:7).
* Octets incompressibles : lire puis réécrire S = 4 Mio = 4,19 Mo par couche, 83,9 Mo par pas.
* Plancher à 1,52 To/s (p81:22) : **2,76 µs par couche, 55 µs par pas**.
* Noyau actuel (61w, acvram_kernels.cu:6455-6546) : **≈ 11 µs par couche, ≈ 0,22 ms par pas** — INFÉRÉ par poste1
  (poste1-61w-verdict-01-10.md:16 : 0,487 → ≈ 0,22 ms si tout le gain de 61w vient du noyau), PAS mesuré.
  Soit ≈ 380 Go/s, ≈ 25 % du plancher.
* Ce que le code explique : `<<<32, 128>>>` fait 4 096 chaînes série sur j (l'ordre de j est imposé par le « au bit »).
  En vol : 32 SM × 4 warps × 16 lectures (unroll 16) × 128 o ≈ 256 Kio. À ~0,7 µs de latence, cela donne ≈ 370 Go/s,
  cohérent avec l'inférence. La deuxième passe relit S, en L2 vraisemblablement.

## Levier

Chaque bloc (une tête) commence par lancer en `cp.async` la copie de toute sa tuile S[h] (64 Kio) en mémoire partagée
dynamique. Les convolutions, normes et portes s'exécutent pendant que la copie est en vol. Les deux passes sur j lisent
ensuite la mémoire partagée, sans conflit de banc (le fil i lit le mot j·D + i). L'écriture reste coalescée en global.
Toute la couche (2 Mio) est en vol d'un coup. L'arithmétique est inchangée (mêmes intrinsèques, j = 0 de D = 128 gardé),
seule la provenance des opérandes change. b=12 n'est pas touché (chemin fla).

## Prédiction (scellée)

* nsys, `kda_decode_kernel` à b=1 : **4 à 7 µs par couche** (0,08 à 0,14 ms par pas). C'est 40 à 70 % du plancher,
  limité par la latence de démarrage et par les 32 SM seulement.
* Mur b=1 (témoin 3,381 ms, poste1-61w:13) : **−0,08 à −0,14 ms, soit −2,4 à −4,1 %**.
* b=12 : 0 ± bruit (≤ 0,5 %).
* **FAUX si** le noyau dépasse 8 µs par couche, ou si le gain au mur à b=1 est inférieur à 0,05 ms.
* Issue gênante, nommée : le noyau actuel serait déjà ≈ 5 µs (le gain de 61w venant d'ailleurs). La base serait alors
  fausse et la pièce ne vaudrait rien. La prise mesure donc AUSSI A au nsys, et la prédiction se relit sur le rapport B/A.
* Autre issue : 32 SM ne suffisent pas à tirer la bande, et le noyau plafonne à 7-8 µs. Levier suivant : 4 blocs par tête
  (grappes, DSMEM pour les sommes de bloc), ou lancement dépendant programmatique (PDL) pour précharger S pendant les
  projections.

## Au bit (règle 9)

* tests/test_kda_etat_kv.py:113-139, inchangé : D=64 sur 64 pas, D=128 sur 64 et sur 512 pas CHAÎNÉS, contre le témoin
  `kda_decode_vk` (état et sortie égaux au bit). Bras cassant :142.
* À sec : la suite des opérations flottantes du SASS (drapeaux de l'extension : -O3 --use_fast_math, sm_120f, nvcc 13.4)
  doit être identique entre l'ancien et le nouveau noyau, aux chargements près (leçon 61w).
