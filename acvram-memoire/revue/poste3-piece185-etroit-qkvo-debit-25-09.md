# Verdict — pièce 185 : débit du GEMM int8 étroit, formes réelles q/k/v/o + GDN (poste3, 25/09)

* **instrument** : `torch.cuda.Event` sous graphe CUDA, 12 poids distincts en rotation (> L2,
  froid), M ∈ {8, 1}, `scratchpad/poste3-p185-25-09/banc.py` (réutilise `outils/gpu/mesure/
  banc-etroites-occupation.py::poids_int8`, plancher `occ.PLANCHER`=1,55 To/s).
* **commit** : c7bb05f8 (origin/main, worktree poste3-185 refait après perte du worktree /tmp).
* **régime** : horloge libre (pas de `-lgc` posé, comparaison en % du plancher, pas en énergie) ;
  carte tenue 14 s (`tenue=14s`, `/tmp/acvram-carte-0.lock.journal`).
* **scellé** : `scratchpad/poste3-p185-25-09/scelle.md` (avant la prise). Prédiction : idéal
  2,96 ms/pas pour q+k+v+o au plancher ; q/o 70-95 %, k/v 40-65 % ; falsificateurs k/v ≥ 85 %
  ou q/o ≤ 50 % ; total q+k+v+o hors bande 2,5-6,0 ms.
* **mesuré** (M=8, `resultat.json`) :

| forme | N | K | µs/appel froid | To/s | % plancher | BN32 à tranches BN64 | gain BN32 |
|---|---|---|---|---|---|---|---|
| q | 6 144 | 5 120 | 24,75 | 1,30 | 83,9 % | au bit | +0,5 % |
| k | 1 024 | 5 120 | 6,03 | 0,89 | 57,4 % | au bit | +2,9 % |
| v | 1 024 | 5 120 | 6,13 | 0,88 | 56,5 % | au bit | +1,9 % |
| o | 5 120 | 6 144 | 25,56 | 1,26 | 81,3 % | au bit | −1,3 % |
| gdn_qkv‖gate (176) | 16 384 | 5 120 | 65,07 | 1,32 | 85,1 % | au bit | **−13,1 %** |
| gdn_out | 5 120 | 6 144 | 25,23 | 1,28 | 82,3 % | au bit | −1,6 % |
| **ensemble/couche** | | | 152,77 | | 81,6 % | | |
| **× 64 couches** | | | **9,78 ms** | | (idéal 7,98 ms) | | |

  M=1 : même motif (k/v 69,2 %, gdn_qkv_gate 85,6 %, BN32 gdn_qkv_gate −10,1 %) — le débit
  n'est pas sensible à M (attendu : la grille ne dépend que de N/K, pas de M).

* **verdict** :
  1. **Falsificateurs non déclenchés** (k/v 56,5-69,2 % < 85 %, q/o 81,3-84,0 % > 50 %) :
     prédiction **TENUE** dans sa forme qualitative (q/o/GDN proches du plancher, k/v nettement
     en retrait), l'écart quantitatif de k/v est moins sévère que prédit (57-69 % contre 40-65 %
     annoncé — hypothèse latence-par-construction correcte en direction, pas en ampleur).
  2. **q+k+v+o seuls** : 62,47 µs/couche (M=8) → **4,00 ms/64 couches**, dans la bande prédite
     2,5-6,0 ms (idéal 2,96 ms, 74 % du plancher agrégé) — **30-41 % du total int8 étroit de 173**
     (8,30 ms/pas), confirme la fraction prédite.
  3. **GDN pèse autant que l'attention** (chef) : q/k/v/o = 62,47 µs, gdn_qkv‖gate + gdn_out =
     90,30 µs — GDN est le POSTE PLUS LOURD des deux, pas égal.
  4. **Réponse à poste1 (BN32 à tranches K de BN64)** : **BN32 NE RÉCUPÈRE PAS le débit de
     BN64**. Gain marginal (+1,9 à +2,9 %) UNIQUEMENT sur k/v à M=8, réversé à M=1 (−10,3 à
     −10,4 %) ; **gdn_qkv_gate, la forme la plus lourde, perd 10-13 %** en BN32 quel que soit M ;
     q/o/gdn_out quasi neutres (±1,6 %). **Le rééquilibrage par SM d'poste1 (≈ −0,4 ms/pas
     prédit si BN32 égalait BN64) n'est PAS rentable** : aucune forme ne regagne son coût par
     octet (+13 à +23 % mesuré par poste1) en changeant seulement le tuilage N à tranches K
     fixées — la largeur de tuile coûte net, l'occupation supplémentaire ne compense pas.
* **durée** : prévu ≤ 5 min (léger, sans balayage), tenu 14 s (`tenue=14s`).

## Suite

- Ensemble int8 étroit (193 appels, 173) − (q/k/v/o + GDN qkv‖gate/out, 6 formes, 152,77 µs ×
  quelque chose ≠ 64 forcément si tous les appels ne sont pas des couches identiques) reste à
  faire par qui prend le coût fixe (poste1) : q+k+v+o+GDN = 152,77 µs/couche × 64 = 9,78 ms,
  DÉJÀ AU-DESSUS des 8,30 ms/pas mesurés en service par 173 pour l'ensemble int8 étroit —
  écart à éclaircir (mon banc isolé n'est pas sous graphe de service réel, pas de recouvrement
  inter-couches ; à confronter au nsys de 173 avant toute conclusion d'agrégat, PAS scellé ici).
- k/v restent la piste la plus prometteuse (57-69 % du plancher) mais BN ne la résout pas :
  reste le coût fixe par appel (poste1, occupation/vagues) ou fusionner k+v en un seul appel
  (1 024+1 024=2 048, même K=5 120 — pas testé ici, hors périmètre débit pur).

## Fichiers

- `scratchpad/poste3-p185-25-09/{scelle.md,banc.py,resultat.json,journal.txt}` (éphémère, hors
  git, ne sera pas fusionné tel quel).
