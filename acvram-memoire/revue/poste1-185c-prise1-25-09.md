# 185 c prise 1 — verdict (poste1, 25/09) : base b=8 17,37 ms tenue ; H indécise (R 0,799) mais contrôle pour H ; épilogue réfuté comme coût fixe → (b) clos

* instrument : `scratchpad/poste1-p185c-25-09/` — `banc-h-epilogue.py` (L2 froid, graphe, 60 rejeux, `banc-etroites-latence.chrono_froid`),
  `profil.py` + `familles.py` (copies de la 188) sous nsys ; `prise.sh` → `banc.jsonl`, `pas.txt`, `familles.txt`, `prise1.log`
* commit : 686312d8 (poste1-185c = origin/main 0c54811f + étape 1 de la 185 b + scellé), ATTENDU vérifié par `prise.sh`
* régime : Qwen3.8-27B-unsloth-mixte-i8c, b = 8, défaut de main SANS variable (preuve `ab=auto(48)` dans la ligne PAS), graphes,
  carte 0 seule, -lgc 2700, ECO=off, cpu-safe 100 début/fin, compute-apps début = fin (4242 seul)
* scellé : `scratchpad/poste1-p185c-25-09/scelle.md` (4e4bf01a, avant la mesure)
* mesuré : pas 17,365 ms (460,7 t/s ; fenêtre nsys 17,356) ; int8 étroit 8,309 ; R = 0,799 ; C = 0,946 ; réduction ≈ 0 µs
* verdict : base TENUE (dans 17,30-17,65 et 8,22-8,52 ; 187 ne touche pas le pas b=8, aucun `int8_gemv`) ; (a) H INDÉCISE ;
  (b) FAUX — issue gênante nommée au scellé (réduction < 0,3 µs) ; V3 déroulé au bit mais PLUS LENT
* durée : prévu ≤ 8 min ; tenu 10:53:31-10:56:49 (3 min 18), après 190 s d'attente (poste4-p153-ppl)

## Base (50 pas)
| famille | 188 (GDN_AB à la main) | 185 c (défaut) |
|---|---|---|
| pas | 17,476 | **17,365** |
| int8 étroit | 8,37 (`_reduit` 5,29, `_segments` 3,04) | **8,309** (5,249 ; 3,025) |
| Marlin NVFP4 | 5,80 | 5,778 |
| GDN | 1,18 | 1,179 |
| α/β bf16 | 0,70 | 0,688 |

## (a) H, K = 6144, `decouper_k` servi
| N | prog | gpt | max prog/SM | µs | µs/Mo |
|---|---|---|---|---|---|
| 2560 | 320 | 6 | 2 | 13,71 | 0,852 |
| 5120 (servi) | 400 | 10 | 3 | 25,91 | 0,805 |
| 5440 | 340 | 12 | 2 | 24,50 | **0,716** |
| 6400 | 400 | 12 | 3 | 30,66 | 0,762 |
R = t(5440)/t(6400) = 0,799 : entre 0,75 et 0,82, INDÉCISE (ma prédiction 0,84 : fausse du côté de H). Contrôle C = 0,946 :
5440 plus RAPIDE que le servi malgré +6 % d'octets, comme H le prédit (octets seuls : 1,06). Un tiers de SM à 3 programmes coûte
≈ 6-11 % par octet. Levier correspondant (tranches pour 2 prog/SM exacts) : change la partition K → hors bit, ± 1 ulp opt-in
seulement ; balayage des tranches fait par poste6 (193), pas doublé.

## (b) Épilogue, µs (V0 servi ; V1 sans réduction ; V2 store seul ; V3 déroulé ; V4 sans lecture des poids)
| forme | V0 | V1 | V2 | V3 | V4 | réduction | atomique |
|---|---|---|---|---|---|---|---|
| o/out 5120 × 6144 | 25,88 | 25,90 | 25,72 | 26,43 | **3,34** | −0,02 | 0,18 |
| down 5120 × 17408 | 66,31 | 67,14 | 65,85 | 67,64 | **3,98** | −0,83 | 1,29 |
La réduction du dernier arrivé ne coûte rien (masquée par les autres programmes) ; la dérouler coûte 0,5-1,3 µs. Mon hypothèse
« une latence L2 par itération » était fausse. Le plancher fixe d'un appel (V4, tout sauf la lecture des poids) vaut 3,3-4,0 µs :
≈ 0,65-0,75 ms/pas sur 193 appels, le reste est du débit. **(b) clos, aucun code.** Suite : (c) grille compacte des segments (prise 2).
