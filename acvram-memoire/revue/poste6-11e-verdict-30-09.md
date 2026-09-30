# 11e (partie sans carte) — `--echelle` à quatre règles au CLI de conversion, WMSE depuis les stats de calibration, scellé PPL et script prêts (poste6, 30/09, branche poste6-11e, à sec)

instrument : pytest à sec (`CUDA_VISIBLE_DEVICES=""`), tiny checkpoint de la CI ; aucune carte
commit : ce commit, sur poste6-scalesweep 200463c3e (fusion locale de chef)
régime : à sec ; la prise sur carte attend l'ordre de chef (après edz poste1 et campagne poste2)
scellé : `revue/poste6-11e-scelle-ppl-30-09.md` — prédiction S1-S7 et issues (a)-(f) écrites AVANT
mesuré : rien sur carte ; 21 tests verts (cli, conversion tiny balayage/balayage-w, bootstrap)
verdict : LIVRÉ — `acvram convert --echelle {max6,4sur6,balayage,balayage-w}`, défaut max6 inchangé et testé au bit ; `balayage-w` pèse par (E|x|/s)² des ActStats, refus nommé sans calibration ; manifeste `part_balayes`/`sous_normales` ; script de prise et bootstrap apparié dans le dépôt
durée : 0 min de carte

## Fait
* `acvram/cli.py` : choices étendus, aide nommée. `acvram/quant/calibrate.py` (`quantize_with_calibration`) : sous `balayage-w`, importance =
  (E|x_j| / s_j)² transmise à `quantize_nvfp4` — E|x| vient des ActStats de la calibration, s de l'AWQ (x' = x/s est ce que le poids
  w·s voit) ; proxy nommé de E[x'²] ; sans stats → ValueError nommée (pas de secours silencieux). `convert.py` `_noter_echelle_nvfp4` :
  `part_balayes` et `sous_normales` par tenseur et au total, même contrat que `part_amax4`.
* `tests/test_echelle_cli_11e.py` (3) : parseur (4 règles, refus d'une 5e, défaut max6), conversion tiny `balayage` → manifeste,
  `balayage-w` refus sans stats / pondération effective avec ; `tests/test_ppl_appariee_11e.py` (2) : bootstrap apparié (identique → Δ 0, z 0 ;
  +3 % → conclusif ; fenêtres non appariées → ÉCHEC). Gardés : test_nvfp4_4sur6 (max6 au bit, 4sur6), test_nvfp4_balayage.
* `outils/gpu/mesure/ppl-balayage-11e.sh` : `convertir <échelle>` ×3, `evaluer`, `analyser` ; carte.sh, ACVRAM_POSTE=poste6, DUREE_MAX 1800,
  relevés nvidia-smi début/fin, HEAD et sha256 du corpus assertés, refus d'écraser un dossier. `ppl-appariee-bootstrap.py` : 138 bis dans le dépôt.

## Reste (sur carte, bd 11e, à l'ordre de chef)
Quatre prises du scellé, puis verdict PPL : prometteur / réfuté / non conclusif selon S3-S4. Aucun changement de défaut avant.
