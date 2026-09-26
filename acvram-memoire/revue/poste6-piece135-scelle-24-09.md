# Scellé — pièce 135 : Qwen3.8-27B b=8, unique + v2 — borné par la mémoire ou par le calcul ? (poste6, 24/09, AVANT la mesure)

Ordre de chef. La 115 bis (poste2, Coder i8c) : débit ∝ horloge SM de 960 à 2 700 MHz (+150 % pour +181 % d'horloge, b=12),
minimum de J/jeton à 1 800 MHz (−11,1 %) — borné par le calcul. La 129/130 (poste1) : Qwen3.8-27B-nvfp4 à b=8, disposition
unique + GEMV Marlin v2 (`ACVRAM_PROJ_MARLIN=1`) : 16,07 ms/pas, 497,7 t/s, 0,738 J/jeton, 367 W, SM 2 665 (certifie, en
processus). À ~1,5 To/s lus, le pas peut être borné par la HBM (dont l'horloge ne suit pas `-lgc`).

## Instrument
`scratchpad/poste6-p135-24-09/bloc.sh` = `scratchpad/poste2-piece115bis-horloge-sm-23-09/bloc.sh` (instrument corrigé : `-i 0`,
`ACVRAM_ECO=off`, `set -euo pipefail`, contrôle horloge lue/demandée ± 50 MHz pendant la fenêtre, 5 répétitions du palier par
point) adapté : alias `Qwen3.8-27B-nvfp4`, `--max-batch 8`, env du bras B de la 129 (`ACVRAM_PROJ_MARLIN=1 ACVRAM_GEMV_MARLIN_V2=1
ACVRAM_GEMV_MARLIN_TPB=1 ACVRAM_GEMV_MARLIN_S=0 ACVRAM_PROJ_MARLIN_DOUBLES=`, lu dans `poste1-p129-24-09/banc-v2.json`), horloges
{1 800, 2 100, 2 400, 2 700}, `serve --regime` (la ligne de régime du serveur prouve `dense=…+marlin(…)`), banc HTTP
`banc-llamacpp-16-09.py decode` (BANC_SLOTS=8×5, BANC_JETONS=1024, BANC_FENETRE_S=10), énergie nvml nette de la fenêtre.
Sous `carte.sh` (mesure, ≤ 1 800 s), HEAD asserté ; max_perf_pct et compute-apps au début et à la fin ; cpu-safe=off.

## Mesures
Par point : t/s (`jetons_s`), J/jeton net, W, horloge lue. Ratios r_t(h) = t/s(h) / t/s(2 700) et r_J(h) = J(h) / J(2 700).

## Seuils, fixés ici
* Point NUL si horloge lue hors ± 50 MHz de la demandée, ou serveur DÉGRADÉ (ligne de régime), ou bridage puissance à 2 700
  (W ≥ 395 en moyenne : alors le point 2 700 n'est pas à son horloge et la pente est faussée — dit tel quel).
* **Borné par la mémoire** si r_t(1 800) ≥ 0,85 (l'horloge baisse de 33 %, le débit de ≤ 15 %).
* **Borné par le calcul** si r_t(1 800) ≤ 0,72 (proportionnel, comme la 115 bis : 0,667 attendu à horloge exacte).
* Entre 0,72 et 0,85 : **mixte**, dit tel quel avec le chiffre.
* Minimum de J/jeton : le point de J net le plus bas ; « gain » = 1 − r_J(min).

## Prédiction (avant)
* **Mixte tendant vers la mémoire** : r_t(1 800) = 0,80-0,88 ; r_t(2 100) = 0,88-0,94 ; r_t(2 400) = 0,95-0,99.
  Pourquoi pas franchement mémoire : les experts n'existent pas (dense), mais l'attention linéaire (GDN, fla) et la glue
  restent du calcul pur ; le GEMV Marlin v2 lit ~14 Go/pas à 16 ms = 0,9 To/s hors KV, 1,5 avec — sous le plancher 1,79.
* Débit à 2 700 en HTTP : 440-490 t/s (certifie 497,7 en processus ; le banc HTTP coûte 2-6 %).
* **Minimum de J/jeton à 1 800 MHz**, r_J(1 800) = 0,70-0,80 (la puissance suit V²·f, le débit moins que f) ;
  J(2 700) ≈ 0,72-0,80 J/jeton net.
* FAUX si r_t(1 800) ≤ 0,72 (toujours borné par le calcul : la thèse de chef tombe, la 115 bis se transpose) ;
  ou si le minimum de J est à 2 700 (aucun levier d'horloge).
* Ce qui me gênerait : r_t(1 800) ≥ 0,85 ET minimum de J à 1 800 avec r_J ≤ 0,70 — plus fort que prédit : alors l'éco 2 700
  servi par défaut (REGLES § 1) laisse 30 % de J sur la table pour ce modèle ; je le dirais et ce serait à chef.

## Durée
4 points × (chargement 27B ≈ 60-90 s + 5 × 10 s + fenêtres) ≈ 12-16 min ; prévu ≤ 20 min, une prise.

## Addendum 05 h 1x — deux prises nulles, main fusionné AVANT la prise (ordre chef), seuils et prédiction inchangés
04:51 : `setsid` depuis un chef de groupe forke et rend la main → `$!` n'était pas le serveur, ECHEC à tort après un chargement
réussi (régime NOMINAL, `dense=…+marlin(doubles=0,seuls=305)` : le bras v2 prend) ; corrigé 7bb7e39d (serveur par port + pgid).
05:08 : prise arrêtée par moi au chargement du point 2 700 (29 s de carte) parce que main avait avancé (134 : préfill exact dans
la disposition Marlin unique, 13474ff8) ; fusionné, la prise repart sur le commit du verdict. Aucun chiffre lu avant l'arrêt.
05:28 : prise nulle encore (304 s de carte, 0 chiffre) — `serve --regime` imprime la ligne de régime et QUITTE (cli.py:913, rc 0/1) :
c'était aussi la vraie cause du 04:51. Option retirée ; la preuve du bras v2 est la ligne `[acvram] régime … dense=…+marlin(…)`
du chargement, présente au niveau warning. Seuils et prédiction inchangés.
