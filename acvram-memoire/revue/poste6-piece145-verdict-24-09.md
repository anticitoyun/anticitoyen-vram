# Verdict — pièce 145 : TTFT et énergie du préfill en service, défaut contre PROJ_MARLIN (poste6, 24/09)

instrument : `outils/gpu/mesure/ttft-service-p145.py` (suivi) contre `acvram serve --max-batch 1 --max-model-len 4608 --no-prefix-cache --speculative none` ; `scratchpad/poste6-p145-24-09/prise.sh <gemma|qwen38> <1|2>` (ABBA par relance du serveur, 5 lots par prise, 4 prises), sorties `prise-<m>-<p>.txt`, `ttft-<m>-<n>-<bras>.log`, `serveur-*.log`, `metrics-*.json`, `horloge-*.csv`
commit : 65bddea5 (3 premières prises), 433b1276 (dernière : N_MAX 60 à L = 512 et agrégat n ≥ 20, addendum avant) ; HEAD asserté rc 65 ; `acvram.__file__` = travail/poste6 imprimé dans chaque prise (garde d'import a86fa1dd incluse)
régime : b=1, invites distinctes de 512 / 2 048 / 4 096 ids, `max_tokens=1`, TTFT = premier fragment SSE ; fenêtre ≥ 10 s (ou n ≥ 20 à 512) sous `Energie`, base `repos(8 s)` après ; A = `ACVRAM_PROJ_MARLIN=0`, B = `ACVRAM_PROJ_MARLIN=1 PROJ_MARLIN_DOUBLES= GEMV_MARLIN_V2=1 TPB=1 S=0` (bras B des 129/142) prouvé par `dense=…+marlin(doubles=0,seuls=307|305)` au journal et `ACVRAM_PROJ_MARLIN=1` dans `/metrics`, absent en A ; `-lgc 2700` posé, mais **préfill au plafond de 400 W dans les deux bras** (394-398 W, SM lue 2 380-2 440 : A ≈ 2 435, B ≈ 2 400) — régime servi bridé en puissance, nommé à l'addendum avant la 4e prise ; cpu-safe=off (100 début et fin) ; compute-apps début = fin = llama-server 4627
scellé : `revue/poste6-piece145-scelle-24-09.md` (65bddea5 + addendum 433b1276) — gemma B/A prédit 1,4-1,8 (512), 1,5-1,8 (2 048, 4 096), FAUX si < 1,25 / 1,3 ; Qwen3.8 prédit 1,02-1,10 (512), 1,03-1,08, FAUX si > 1,20 / 1,15
mesuré : 20 lots, 60 cases, 0 lot écarté (fenêtres valides, 5 ≤ n ≤ 44, 3 longueurs par lot). Médianes de 5 lots (TTFT ms ; J par préfill net) :
  **gemma4 31B** — 512 : A 262,0 / B 309,1 (**1,18**, +47 ms ; J 84,2 → 99,2, 1,18) · 2 048 : 948,6 / 995,9 (**1,05**, +47 ms ; J 304,9 → 318,7, 1,045) · 4 096 : 2 021,0 / 2 071,8 (**1,025**, +51 ms ; J 648 → 667, 1,029)
  **Qwen3.8-27B** — 512 : 229,0 / 269,6 (**1,177**, +40,5 ms ; J 73,2 → 86,5, 1,18) · 2 048 : 754,9 / 793,3 (**1,051**, +38,5 ms ; J 1,054) · 4 096 : 1 438,7 / 1 478,7 (**1,028**, +40 ms ; J 1,027)
  Dispersion par case ≤ 0,4 % entre lots (A et B), les 5 lots B toujours au-dessus des 5 lots A
verdict : **gemma4 : FAUX** — le +69 % de l'eval ne tient PAS en service (1,18 à 512, 1,05 à 2 048, 1,025 à 4 096, tous sous le seuil FAUX) ; **Qwen3.8 : TENU** à 2 048 et 4 096 (1,051, 1,028 dans 1,03-1,08), 512 au-dessus de la bande (1,177) mais sous le FAUX (1,20). **Le surcoût de PROJ_MARLIN au préfill est une CONSTANTE par requête** : gemma +47-51 ms, Qwen3.8 +38-41 ms, quelle que soit la longueur — pas un pourcentage
durée : prises 457 + 450 + 428 + 403 s (`tenue=`), 4 × 5 lots, chargement 15-60 s par lot ; aucune prise nulle. Faute d'équité consignée ci-dessous

## Lecture
* **Un coût fixe par préfill, proportionnel aux POIDS, pas à l'invite** : +47 ms sur un 31B, +40 ms sur un 27B, identique à
  512 et à 4 096 jetons — c'est le dépaquetage de la 134 (Marlin unique → disposition dense pour le GEMM de préfill) refait
  une fois par requête sur tout le modèle, et non un noyau plus lent par jeton. À l'eval (fenêtres de 2 048 nombreuses et
  courtes en calcul) ce coût fixe devient +69 % ; en service il vaut +18 % à 512 jetons, +5 % à 2 048, +2,5 % à 4 096. Le
  +4,6 % de Qwen3.8 à l'eval était le même coût fixe vu sur un eval plus long par fenêtre.
* **Ce qu'il coûte au service** : sur une invite réelle (gabarit + question, 500-2 000 jetons), +40-50 ms de TTFT et
  +13-15 J par requête, pour les gains de décodage des 129/130/142 (b=8 +57-60 %, b=1 −1 à −7 % de J). Décision « défaut »
  à chef ; le levier qui effacerait le coût est nommé : garder la disposition dépaquetée en cache (poids ×2 en VRAM : hors
  budget sur le 31B) ou faire le GEMM de préfill directement sur la disposition Marlin (noyau, pièce moteur) — ou ne dépaqueter
  que les projections touchées par le chunk courant (le coût suivrait alors L, borné par le nombre de chunks de 256).
* Régime : les deux bras préfillent au plafond de 400 W (2 380-2 440 MHz) ; les ratios sont ceux du service réel, pas d'une
  horloge tenue — même lecture que la 62 A5 (bridage nommé). Énergie : B suit le TTFT (même puissance), J/jeton de préfill
  A 0,164 (gemma 512) → 0,158 (4 096) ; Qwen3.8 0,143 → 0,113.
* **Faute d'équité (moi)** : chef a demandé à 10 h 3x de ne rien relancer avant « rendue » de poste4-* et chef-tests-139,
  puis d'y ajouter poste1-p146. Mon guetteur a vu les deux premiers rendus à 10:38-10:39 et a lancé la 4e prise (10:39, carte
  10:48-10:54) AVANT que j'aie pu y ajouter poste1-p146 (mon arrêt du guetteur a manqué : il était déjà sorti). Le journal ne
  montre aucune prise poste1-p146 en attente sur 10:39-10:56 (file : poste5-139bis, chef-tests-109, poste4 etat) — personne
  n'a été retardé, mais l'ordre n'a pas été suivi à la lettre. Le second guetteur (avec poste1-p146) a été arrêté avant de
  relancer quoi que ce soit ; aucune 5e prise.

## Chiffres par case (médiane de 5 lots ; J = joules par préfill, net du repos)
| modèle | L | TTFT A (ms) | TTFT B (ms) | B/A | Δ (ms) | J A | J B | J B/A | jetons/s préfill A → B |
|---|---|---|---|---|---|---|---|---|---|
| gemma4 31B | 512 | 262,0 | 309,1 | 1,180 | +47,2 | 84,2 | 99,2 | 1,179 | 1 954 → 1 656 |
| gemma4 31B | 2 048 | 948,6 | 995,9 | 1,050 | +47,4 | 304,9 | 318,7 | 1,045 | 2 159 → 2 056 |
| gemma4 31B | 4 096 | 2 021,0 | 2 071,8 | 1,025 | +50,8 | 648,2 | 666,8 | 1,029 | 2 027 → 1 977 |
| Qwen3.8-27B | 512 | 229,0 | 269,6 | 1,177 | +40,5 | 73,2 | 86,5 | 1,182 | 2 236 → 1 899 |
| Qwen3.8-27B | 2 048 | 754,9 | 793,3 | 1,051 | +38,5 | 240,8 | 253,8 | 1,054 | 2 713 → 2 582 |
| Qwen3.8-27B | 4 096 | 1 438,7 | 1 478,7 | 1,028 | +40,1 | 461,9 | 474,2 | 1,027 | 2 847 → 2 770 |
