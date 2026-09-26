# Verdict — C9-M0 (bande PCIe hôte → RTX 5090 dans le moteur) : **22,6 Go/s épinglée H2D** (`bench_link_bandwidth` 512 Mo ; 22,3 à 21 Mo, la taille d'un expert 119B) — prédiction 18,7 **dépassée de 21 %**, très loin du seuil 40 : les coûts C9 restent ceux du x8 ; **52,8 est faux** (c'était un x16). Paginable 19,8 H2D / **13,9 D2H** ; lien `pcie gen 5 × 8` (max 16)

instrument : `scratchpad/c9-m0-19-09/m0.py` (`acvram.bench.bench_link_bandwidth` tel quel + copies plates 21 / 128 / 512 Mo épinglées et paginables, H2D et D2H, médiane de 20 événements CUDA, lien lu par `nvidia-smi`), prise `carte.sh` 00:27, horloge libre, llama-server pid 4284 5,6 Gio présent (ne copie rien)
scellé (poste7, poste7-c9-119b-cache-experts § 2 M0, avant) : 18,7 Go/s ; > 40 → tous les coûts ÷ 2,8 et les tiers réécrits (1 792 → 1 050)
mesuré : `bench_link_bandwidth` 512 Mo épinglée **h2d 22,6 · d2h 23,2** ; 21 Mo épinglée **22,3 / 24,7** (0,92 ms la copie) ; 128 Mo 21,6 / 21,6 ; 512 Mo 21,0 / 21,9 ; paginable 21 Mo 19,8 / 13,9 (1,04 ms), 128 Mo 19,1 / 13,2, 512 Mo 19,6 / 13,4 ; lien `5, 8, 5, 16`
verdict : **TENU (18,7 → 22,6, même classe x8)** — l'arithmétique de la fiche se recale à 22,6 : 1,9 Go/jeton sans cache = **84 ms → 11,9 j/s à b=1** (10 prédits) ; un expert de 21 Mo coûte **0,92 ms** épinglé ; la copie paginable H2D perd 12 %, la D2H paginable 40 % (à ne jamais employer pour un exil qui écrit) ; le tiers de 20 Go d'experts résidents reste la bonne unité
suite : poste7 : recaler les lignes M2/M3/C9-cellules sur 22,6 (pas 18,7, pas 52,8) ; poste1 : le chemin d'exil doit copier depuis de l'épinglé (13,9 en paginable D2H) ; ma file : cellule llama.cpp 119B b=1 experts en RAM (concurrent de référence), puis fin de nuit

## Rejouable
`ACVRAM_TYPE=mesure outils/carte.sh <python> scratchpad/c9-m0-19-09/m0.py` (30 s).
