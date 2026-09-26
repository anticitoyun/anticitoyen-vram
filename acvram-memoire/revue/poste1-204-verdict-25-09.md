# 204 — service du banc chat b=8 mixte décomposé (poste1, 25/09)

* instrument : `scratchpad/poste1-p204-25-09/{serveur-trace,banc-trace,analyse}.py` (181 étendu), passes N T N T, 5 lots de mesure/passe
* commit : e22a55613 (poste1-204 = origin/main 1b56f68e9 + scellé/instrument)
* régime : Qwen3.8-27B-unsloth-mixte-i8c, ECO=off, -lgc 2700, cpu-safe 100 début/fin, abflux + canal(table) prouvés au journal serveur, repli_eager 0
* scellé : `scratchpad/poste1-p204-25-09/scelle.md` (573ae8a0 + addendum 7122241d1)
* mesuré : N 426,4 / 425,0 t/s ; T 419,7 / 419,5 ; lot 4,85-4,92 s ; décodage pur 15,49-15,54 ms/pas ; préfill 839-918 ms/lot
* verdict : service = PRÉFILL (87-95 %), mais préfill FAUX-haut (839-918 contre 450-650) : coût fixe ≈ 70-100 ms PAR SÉQUENCE
* durée : prévue ≤ 10 min / tenue 263 s (carte.sh ; deux prises avortées avant : SyntaxError 0 s, garde canal 247 s)

## Prédictions contre mesure (T2 / T4, médianes)
| poste | prédit | mesuré | issue |
|---|---|---|---|
| neutralité N/T | ± 1 % | T −1,4 % | FAUX (trace coûte ≈ 1,4 %) |
| décodage pur | 15,6-16,1 ms/pas | 15,49 / 15,54 | FAUX-bas de 0,1 (favorable) |
| préfill 8 × 78 | 450-650 ms/lot | 918 / 839 | **FAUX-haut** |
| préfill en 1 pas | ≥ 3 lots/4 | ≈ 1 lot/2 (T2 méd. 2 pas, T4 méd. 1) | FAUX |
| admission | < 2 ms/lot | 0,26 | tenu |
| captures en lots de mesure | 0 | 0 méd., 1 × 80 ms dans 1 lot/5 par passe | FAUX, nommé (≈ 80 ms, b=1) |
| pas mixtes | 0 | 1-2/lot, 3,5-4,0 ms/lot | FAUX, négligeable |
| surcoût hôte des pas | 0,05-0,2 ms/pas | 0,036 | tenu (sous la fourchette) |
| hors pas | 8-15 ms/lot | 9,2-9,4 | tenu |
| Σ postes = lot − 255 × décodage | ± 5 % | 97,4 % / 97,4 % | tenu |
| écart NInfer au préfill | ≈ 0,37 s/lot | service 0,88-0,97 s/lot contre 0,35 | tenu en direction, 1,6 × plus gros |

Alarme de l'addendum (201) : NON déclenchée comme cause — le lot de 202 (avant 201) valait déjà 4,846 s ; la 201 n'ajoute rien de visible.

## Le fait nouveau : coût fixe par séquence au préfill (pas `pas` > 100 ms, les deux passes T)
| préfill | ms |
|---|---|
| 1 × 78 | 179-266 |
| 7 × 78 = 546 | 743-760 |
| 8 × 78 = 624 | 813-899 |
| 1 × 2 046 | 848-851 |
| 1 × 4 094 | 1 730-1 741 |

Pente longue 0,435 ms/jeton (≈ 2 300 j/s) ; à cette pente, 624 jetons coûteraient ≈ 270 ms, on en paie 840 :
**≈ 70 ms par séquence** au-delà des jetons (7 → 8 séquences : +80 ms pour 78 jetons). La 1 × 78 isolée coûte 180-260 ms.
Second effet : la fenêtre d'admission de 5 ms coupe le lot en 1 + 7 dans ≈ la moitié des lots → deux pas de préfill
(190 + 750 = 940 ms contre 840 en un) : ≈ +100 ms sur ces lots.

## Suite proposée (à sec d'abord, décision chef)
1. Nommer le coût par séquence : trace nsys d'UN préfill 8 × 78 servi (≤ 5 min de carte), NVTX par couche ; hypothèses dans
   l'ordre : (a) préfill GDN/FLA par séquence (boucle hôte, lancements × 8), (b) attention préfill par séquence, (c) copie
   transitoire par appel (201, par séquence ?). Réfuté si le surcoût se répartit en proportion des jetons.
2. Fenêtre d'admission : ne pas couper un lot synchrone (≈ −50 ms/lot en moyenne, au bit).
