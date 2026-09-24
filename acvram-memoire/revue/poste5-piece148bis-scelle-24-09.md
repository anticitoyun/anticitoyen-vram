# Pièce 148 bis — scellé (poste5, 24/09, écrit AVANT la prise) : le pas b=8 de l'alias mixte décomposé

Feu chef. Alias `Qwen3.8-27B-unsloth-mixte-i8c`, `ACVRAM_PROJ_MARLIN=1 ACVRAM_PROJ_MARLIN_DOUBLES=
ACVRAM_GEMV_MARLIN_V2=1 ACVRAM_GEMV_MARLIN_TPB=1 ACVRAM_GEMV_MARLIN_S=0`, b=8, -lgc 2700.

## Instrument (deux passages, une prise ≤ 10 min)

1. **Sans profileur** : `outils/gpu/mesure/frontiere-pas.py` (en processus, régime servi : graphes, pipeline, lot constant
   de 8, invite 256, ctx 2048, 300 pas). Par pas : `graphe` (rejeu, tête comprise), `echantillon`, `trou_gpu` (carte
   oisive entre deux pas = la part hôte). Leçon de la p69 : **aucune part hôte ne se lit sous nsys**.
2. **Sous nsys** (`-t cuda --cuda-graph-trace=node`, 60 pas) : noyaux des nœuds du graphe, classés par famille par un
   script à sec (`familles-148.py`) — GEMV nvfp4 (Marlin v2), GEMV int8, attention (16 couches pleines), GDN (48 couches
   linéaires : conv, récurrence, gates), normes / rope / glue, tête et échantillonneur. Somme des noyaux par pas contre
   `graphe` du passage 1 (écart > 10 % → instrument faux, je le dis).
3. **NInfer** (même grille, si ses noms de noyaux se classent) : trace nsys de `ninfer-serve` sous le banc chat b=8, 15 s,
   dans une seconde prise ≤ 5 min. Sinon, total seul (17,3 ms/pas, 139 c).

## Prédiction (ms/pas, b=8)

Le banc chat (139 bis) donne 27,0 ms/pas, HTTP et SSE compris. En processus, j'attends **graphe + échantillon + trou =
23-26 ms**. Découpe prédite des noyaux :

| famille | prédit | d'où |
|---|---|---|
| GEMV int8 (233 tenseurs, tête comprise) | 7,5-9,0 | banc 148 : 8,1 extrapolés |
| GEMV nvfp4 Marlin (MLP 0-55, 8,42 Go) | 5,0-7,0 | 1,2-1,6 To/s |
| GDN (48 couches) | 1,5-4,0 | conv + récurrence + gates à b=8 |
| normes / rope / glue élémentaire | 2,0-5,0 | ~6-10 noyaux par couche × 64, 5-10 µs chacun |
| attention (16 couches) | 0,3-0,8 | ctx ≤ 600 |
| tête hors GEMV + échantillonneur | 0,2-0,5 | |
| **trou hôte (frontière)** | **0,3-1,0** | graphes : ≤ 0,43 ms mesurés à b=12 sur Coder |

**Où sont les ~13 ms hors lecture de poids** (prédiction) : glue et normes 2-5, GDN 1,5-4, attention + tête 0,5-1,3,
frontière 0,3-1, et **2-4 ms de service** (HTTP/SSE, écart banc chat contre moteur en processus).

**Issues nommées.** (a) GDN > 5 ms → le levier est la couche linéaire, pas la glue. (b) Glue + normes > 6 ms → fusions
(le chantier C15 de Coder, ici sur un hybride). (c) Écart banc chat contre processus > 4 ms → le service lui-même.
(d) Somme des GEMV nettement au-dessus de 15 ms → ma lecture du banc 148 ne vaut pas sous graphe, et je le dis.
