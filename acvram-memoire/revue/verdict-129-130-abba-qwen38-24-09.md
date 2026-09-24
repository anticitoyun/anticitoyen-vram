# Verdict — 129/130 (3) : ABBA Qwen3.8-27B-nvfp4, disposition Marlin + GEMV v2 contre défaut — 24/09 04 h 3x (poste1)

* **instrument** : `outils/gpu/mesure/certifie-b12.py` (fichier suivi), `CERT_PUR=1` (lot constant, ctx 2 560), un bras par processus, ordre A B B A A B B A A B ; `scratchpad/poste1-p129-24-09/prise-abba.sh`
* **commit** : b=8 9384fde3 (bras B = disposition UNIQUE + v2) ; b=1 2d5ebd1f (bras B = MIXTE + v2, `MIXTE=1`)
* **régime** : -lgc 2700 posé ; plafond 400 W ; cpu-safe=off (100/100) ; compute-apps début = fin (llama-server sur la 3080 Ti). Bras B prouvé par la ligne de régime (`dense=…+marlin(…)`), absent en A.
* **scellé** : `scratchpad/poste1-p129-24-09/scelle-abba.md` + addenda 04 h 0x et 04 h 1x (avant les lots) — b=8 pas B/A 0,63-0,72 ; b=1 0,99-1,01 ; FAUX si b=8 < +15 %, b=1 > +3 %, refus au chargement
* **mesuré** :
  * **b=8** (médianes de 5 lots par bras, tous NOMINAUX, 0/64 couche exilée) : A **25,30 ms/pas**, 316,2 t/s, 1,263 J/jeton, 400,0 W (bridé, SM ≈ 2 370 MHz) ; B **16,07 ms**, **497,7 t/s**, **0,738 J/jeton**, 367 W (SM 2 665). Écart entre lots ≤ 0,3 % par bras. Invalidations communes aux deux bras : « bridage pendant la fenêtre : puissance », « durées énergie/hôte divergent de 0,14-0,15 s » (≈ 0,7 % d'une fenêtre de 20 s).
  * **b=1** : A 13,19 ms, 75,8 t/s, 4,90 J/jeton ; **B 87,3 ms, 11,4 t/s** — régime **DÉGRADÉ** : `couches_exilées=12/64`, `graphes=off(repli eager : poids en flux depuis la RAM hôte (layers.52.mlp.gate_proj))`, 121 W.
* **verdict** :
  * **b=8 : TENU** — pas B/A **0,635** (prédit 0,63-0,72), débit **+57,4 %**, J/jeton **−41,6 %** (au-delà de la prédiction : A bute sur le plafond de 400 W et B non — gain réel DANS le régime servi, pas à horloge égale). Mesuré sur la disposition UNIQUE ; le MIXTE prend les mêmes noyaux à M = 8.
  * **b=1 : NON MESURÉ** (bras B invalide) — la réserve des 8,5 Gio doublés a fait exiler 12 couches, sans refus : **un trou de ma preuve mémoire**, qui ne regardait que le KV et la capacité. **Corrigé** : sous PROJ_MARLIN, un seul poids en flux = refus nommé au chargement (test + bras cassant ROUGE). Conséquence : **le MIXTE ne tient pas sur la 5090 pour Qwen3.8**, même à 1 × 2 560 ; il serait refusé désormais.
  * Faute de conduite : j'ai modifié `prise-abba.sh` pendant la prise b=8 ; bash lit au fil de l'eau, donc l'agrégat et le `-rgc` de fin ont sauté (horloges rendues ensuite sous verrou, lots intacts, agrégés depuis leurs JSON).
* **durée** : b=8 04:00:48 → 04:11:43 (tenue 655 s) ; b=1 → 04:23:03

## Au chef
Seule configuration qui tient : **UNIQUE + v2** (b=8 +57,4 %, capacité KV intacte ; KL 0,00545, NON qualifiée au seuil 0,00491).
Le MIXTE + v2, qualifié en KL, ne tient pas en mémoire. Ce qui manque : un ABBA b=1 de l'UNIQUE + v2 (prédit 0,99-1,01, banc 130 −0,58 %), et une réponse sur la KL de l'unique — soit un témoin plus large (plus d'invites), soit la source de l'écart par couche (préfill Marlin contre cuBLAS).

## Complément 24/09 04 h 4x — (a) ABBA b=1 de l'unique + v2 et (b) PPL, ordre du chef (scellé `scelle-ppl-b1.md`, commit 20cd76ab, avant la mesure)
* **(a) b=1** (prise 04:2x → 04:37:35, copie figée du script ; bras B NOMINAL, 0/64 couche exilée, `+marlin(doubles=0,seuls=305)`) :
  A **13,192 ms**, 75,80 t/s, 4,917 J/jeton, 372 W ; B **13,149 ms**, 76,05 t/s, **4,571 J/jeton**, 347 W (SM 2 677 contre 2 637).
  Pas B/A **0,9967** (prédit 0,99-1,01) → **TENU** ; débit +0,33 %, J/jeton **−7,0 %** (même vitesse, moins de puissance).
* **(b) PPL** (`acvram eval`, corpus `acvram/data/calibration-anglais.txt`, fenêtre 4 096, pas 2 048, min-context 0, 16 384
  jetons, 7 fenêtres — celles de la 102 ; `par_fenetre` ajouté à evaluate.py sans changer le calcul) : A **4,0938**
  (= la 102 au 10⁻⁴ : instrument inchangé), B **4,0950** → **+0,029 %** ≤ +0,5 % → **TENU**. Par fenêtre (jetons 4 095 puis
  6 × 2 048) : +0,014 / −0,022 / +0,148 / +0,122 / −0,025 / −0,225 / +0,220 %. L'écart de KL par pas de l'unique
  (0,00545) ne se traduit pas en PPL au-delà de ±0,23 % par fenêtre. Localisation par couche : non faite (prise de plus).
* Invalidations communes aux deux bras (b=1) : « bridage pendant la fenêtre : puissance », « durées énergie/hôte divergent » (≈ 0,15 s / 20 s).

## Bilan pour le chef
Disposition Marlin UNIQUE + GEMV v2 (opt-in `ACVRAM_PROJ_MARLIN=1 ACVRAM_PROJ_MARLIN_DOUBLES= ACVRAM_GEMV_MARLIN_V2=1
ACVRAM_GEMV_MARLIN_TPB=1`) sur Qwen3.8-27B-nvfp4 : **b=8 +57,4 % de débit, −41,6 % de J** ; **b=1 −0,3 % de pas, −7,0 % de
J** ; PPL **+0,029 %** (≤ +0,5 %) ; KL par pas **0,00545 > 0,00491 : NON qualifiée** au seuil du témoin (FAUX maintenu,
seuil non relevé) ; capacité KV intacte (aucun double). Le MIXTE, qualifié en KL, ne tient pas en mémoire (exil → refus).
