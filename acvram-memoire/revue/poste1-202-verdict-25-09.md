# 202 — verdict (poste1, 25/09) : pas b=8 mixte 16,09 ms contre NInfer 15,98 (+0,7 %) ; banc chat 422,6 t/s ; 1er poste = le service

* instrument : `scratchpad/poste1-p202-25-09/` — `profil.py` + `familles.py` (copies 188 / 185 c) sous nsys `--cuda-graph-trace=node`,
  50 pas ; banc chat de la 102 (PYTHONPATH `outils/gpu/mesure`), 5 passes au défaut, serveur neuf par passe, fenêtre 20 s ; `prise.sh`
* commit : caba01b4 (poste1-202 = origin/main 9d646c19 + scellé), ATTENDU vérifié
* régime : Qwen3.8-27B-unsloth-mixte-i8c, b = 8, défaut de main SANS variable : `ab=auto(48) abflux` et `canal(table)` dans la ligne
  PAS, repli_eager=0 (5/5 passes) ; -lgc 2700 (horloge moyenne 2 521-2 557 au banc), ECO=off, cpu-safe 100 début/fin,
  compute-apps début = fin (4242 seul) ; bridage puissance dans les 5 passes du banc
* scellé : `scratchpad/poste1-p202-25-09/scelle.md` (caba01b4, avant la mesure)
* mesuré : pas hôte **16,090 ms** (497,2 t/s ; fenêtre nsys 16,082) ; banc chat **422,6 t/s** (421,1-425,2), J/jeton net 0,7587
* verdict : pas TENU (prédit 16,05, 15,75-16,40) ; étroit TENU (7,637 dans 7,45-7,75) ; banc TENU (424, 415-432)
* durée : prévu ≤ 10 min ; tenu 14:09:32-14:17:42 (8 min 10), après 8 s d'attente

## Pas contre NInfer (b=8, noyaux)
| | 185 c (17,365) | 202 | NInfer (148 bis) |
|---|---|---|---|
| pas / somme des noyaux | 17,365 | **16,09** (somme 17,76 : β‖α sur le second flux, 1,88 ms recouverts) | 15,98 |
| int8 étroit (+ tête) | 8,309 | **7,637** (canal 6,748 ; tête `_etroit_reduit` 0,848) | 6,80 + tête 0,79 |
| Marlin NVFP4 / W4A4 | 5,778 | 5,857 | 6,35 |
| GDN | 1,179 | 1,179 | 1,24 |
| β‖α | 0,688 (critique) | 1,884 hors chemin critique (39 µs/appel en concurrence) | 0 (fusionné) |
| attention | 0,435 | 0,464 | 0,26 + rope/kv 0,04 |
| normes + glue + copies | 0,669 | 0,742 | ≈ 0,30 |
Écart : **+0,11 ms/pas (+0,7 %)**. Banc servi : 422,6 t/s contre ≈ 463 chez NInfer (banc 17,27 ms/pas, 148 bis) : **−8,7 %**.

## Les 3 postes suivants (par gain max)
1. **Service (banc − pas) : 2,84 ms** (8 / 422,6 = 18,93 − 16,09) contre 1,29 chez NInfer → **gain max ≈ 1,55 ms/pas (+8 % au banc)**,
   au bit (aucune arithmétique). Non décomposé : admission des lots successifs, préfill par lot, HTTP/SSE, détokenisation. C'est
   désormais le plus gros écart à NInfer, plus que tous les noyaux réunis.
2. **int8 étroit : ≈ 0,8 ms/pas contre NInfer par forme** — pile GDN 59,42 contre 52,36 µs (0,34), o‖out 23,64 contre 20,79 (0,18),
   down 75,88 contre 55,99 (0,16), qkv attention 51,06 contre 45,95 (0,08), tête 0,85 contre 0,79 (0,06). **Hors bit** (toute autre
   géométrie du canal change l'ordre des sommes), protocole de la 195. Deux sous-postes à lire d'abord, sans changer la sortie :
   down servi à 75,9 µs contre 58,9 au banc de la 195 (+17 µs × 8 = 0,14) ; la pile GDN +3 µs sous la concurrence de β‖α (0,14).
3. **Marlin NVFP4 : ≤ 0,5 ms** (106-110 % du plancher, 180), hors bit ; à égalité, **normes + glue + copies : ≈ 0,44** contre
   NInfer, AU BIT par fusions (129 rmsnorm, 96 + 32 elementwise) ; attention ≈ 0,17 contre NInfer.
GDN : ≈ 0 (borné par la lecture de l'état).
