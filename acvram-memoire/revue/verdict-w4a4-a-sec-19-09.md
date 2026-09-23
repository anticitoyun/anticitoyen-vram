# Verdict — perte W4A4 à sec (nuit sens 2, ligne Océane (2)) : **chantier FERMÉ pour la nuit** — la PPL fausse-quant est déjà mesurée sur la carte, 3 tranches, 17/09 : `both` **+0,0134** et `gateup` **+0,0100** au-dessus du défaut (échelle par ligne, le cas le plus favorable), contre un scellé ≤ 0,004 ; l'erreur relative à sec sur les activations réelles (§ ci-dessous) dit pourquoi

instrument : (a) `verdict-porte-a4-17-09` (Laure, 302025e) — `ACVRAM_PREFILL_A4=gateup|both` = `fausse_quant_nvfp4` de `model.py:2009` (échelle globale par LIGNE amax/(448×6), blocs de 16 en UE4M3, E2M1 au plus proche, après Hadamard, avant la GEMM groupée), `ppl-acvram-17-09.py`, 3 tranches privées Coder, sur carte ; (b) à sec aujourd'hui : `scratchpad/w4a4-a-sec-19-09/erreur-a4-activations.py` — la même `fausse_quant_nvfp4` enveloppée d'une sonde sous `PREFILL_A4=both`, prefill CPU de 512 jetons de la tranche 1 (sha256 `1316df41…`), couches {0, 23, 47}, entrée de gate/up et entrée de down, erreur relative ‖fq(x) − x‖/‖x‖ pour A4 (E2M1) et A8 (E4M3, `fake_quantize_e4m3_activation`, même bloc 16)
commit : `oceane-11` (ce commit) ; (a) au régime classé du 17/09 ; (b) `CUDA_VISIBLE_DEVICES=""`, `.venv` du worktree, 8 fils
régime : converti `Qwen3-Coder-30B-A3B-nvfp4` (manifeste : `awq: False`, `use_hadamard: auto`, `group_size 128`) — il n'existe pas de Coder NVFP4 « avec AWQ » : le bras « avec Hadamard/AWQ » demandé est celui-ci (Hadamard auto), le bras « sans Hadamard » n'a pas de converti et n'a pas été fabriqué (une conversion = 20 min de carte, hors ordre à sec)
scellé (Sage, `sage-nuit-sens2-19-09` § Ordre) : PPL fausse-quant − PPL défaut ≤ 0,004 → chantier W4A4 ouvert cette nuit ; > 0,004 → fermé pour la nuit, dit tel quel
mesuré : (a) Coder PPL/bf16 off **1,0148** (1,0102 / 1,0116 / 1,0229) · gateup **1,0248** (+0,0100) · both **1,0282** (+0,0134), 0 fenêtre explosée, bras qui doit différer tenu (gateup ≠ off sur 6/6 tranches) ; (b) à sec (598 relevés, un par expert appelé, 3 couches × 2 sites, prefill CPU 43 s) : erreur relative **A4 ≈ 9 %** (médianes gate/up 0,091 / 0,093 / 0,093 ; down 0,079 / 0,090 / 0,085 aux couches 0 / 23 / 47), **A8 (E4M3) ≈ 1,2-2,1 %** (gate/up 0,017 / 0,021 / 0,019 ; down 0,012 / 0,017 / 0,013) ; amax/rms médian 13-38, max 116 (couche 23, gate/up) — des sorties de norme à outliers, ce qu'une échelle par ligne ne rattrape pas en E2M1
verdict : **> 0,004 sur les deux bras mesurés (2,5× et 3,4× le seuil) : chantier W4A4 fermé pour la nuit**, tel quel ; ce n'est pas une nouvelle mesure, c'est la lecture d'un verdict existant que le scellé du jour ne renverse pas — refaire la PPL à sec sur 1 tranche (CPU, ~1 h) donnerait un chiffre moins bon (1 tranche contre 3) sans pouvoir passer sous 0,004 puisque la borne basse connue est +0,0100

## Erreur relative des activations réelles (à sec, complément)

| couche | site | n experts | A4 (E2M1 bloc 16) méd [min-max] | A8 (E4M3 bloc 16) méd [min-max] | amax/rms méd / max |
|---|---|---|---|---|---|
| 0 | gate/up | 122 | 0,0909 [0,081-0,096] | 0,0167 [0,010-0,020] | 20,1 / 24,7 |
| 0 | down | 122 | 0,0786 [0,035-0,091] | 0,0120 [0,004-0,017] | 37,9 / 108,6 |
| 23 | gate/up | 87 | 0,0934 [0,001-0,095] | 0,0209 [0,000-0,022] | 13,0 / 115,5 |
| 23 | down | 87 | 0,0896 [0,079-0,096] | 0,0169 [0,012-0,021] | 17,9 / 52,9 |
| 47 | gate/up | 90 | 0,0926 [0,090-0,097] | 0,0192 [0,017-0,022] | 20,2 / 25,7 |
| 47 | down | 90 | 0,0850 [0,045-0,094] | 0,0134 [0,004-0,018] | 21,4 / 79,7 |

Lecture : l'A4 coûte ~9 % d'erreur relative sur CHAQUE entrée de GEMM d'expert, uniformément (E2M1 n'a que 8 niveaux par signe : c'est le plancher du format, pas un outlier de couche — la valeur `fp4_gemm.py:247` « ~9,5 % » est confirmée sur les activations réelles) ; l'A8 en divise l'erreur par 5 à 7. Ce sont des erreurs par appel ; la PPL du 17/09 (+0,010 / +0,013) en est la conséquence intégrée sur 48 couches. Un schéma de migration d'échelles réduit l'amax/rms (ici 13-38), pas les 8 niveaux : il ne ramènera pas 9 % à ~1 %.

Instrument : le chemin CPU/référence n'appelle ni `fausse_quant_nvfp4` ni `_grouped` (0 appel aux deux premières passes — l'absence a été lue avant de conclure) ; sonde sur `MLP.forward` (experts un par un), MLP étiquetés par couche via `named_modules`, `fausse_quant_nvfp4` de `model.py` appliquée à x (entrée gate/up) et h = `_fusionner(gate, up)` (entrée down).

## Ce qui resterait admissible (pas cette nuit)

* `sage-w4a4-porte-fermee-17-09` § 3 : un seul autre schéma admissible — migration d'échelles (SmoothQuant : l'outlier d'activation transféré au poids avant la quantification), prédiction à écrire avant (`prediction-smoothquant-a4-sweep`), converti neuf, mesure sur carte ; ≥ 1 j.
* W4A8 (activations E4M3, `.kind::mxf8f6f4`) : erreur relative ≈ ×8 plus petite qu'A4 sur les mêmes activations (§ ci-dessus) ; c'est le repli qui garde les tensor cores 4 bits pour les poids et n'a jamais eu de porte PPL scellée.
