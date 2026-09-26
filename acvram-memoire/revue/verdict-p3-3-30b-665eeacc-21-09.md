# P3 (3) Qwen3-VL-30B nvfp4 vs AWQ déquantifié bf16 sur disque — 665eeacc — 21/09

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-3.sh` (référence = dossier déquantifié `Qwen3-VL-30B-A3B-awq-dequant-bf16` sur disque, défaut désormais)
* commit : 665eeacc (worktree poste2-w-21-09)
* régime : NOMINAL, graphes=on(hybrides≤4), kv=int8, deepstack=3, mrope=[24,20,20]
* scellé : haut_2se_pct ≤ 3 % (poste7 20/09)
* mesuré : `haut_2se_pct` = **21,823 %** (geo −11,561 %, sd 27,73) ; `greedy8_identiques` = 0/3 ; `premier_jeton_identique` = 3/3 mais `prefixe_commun_min` = 2 jetons (img02) ; détails par image : img00 diverge au jeton 4, img01 au jeton 7, img02 au jeton 2
* verdict : **RÉFUTÉ, établi pire que le seuil (× 7,3)** — divergence nette entre acvram nvfp4-vision et la référence bf16 déquantifiée, dès les tout premiers jetons de description pour 2 des 3 images. Référence construite proprement (`t_charge_s` 78,7 s, `modules` 53, sha256 publié).
* durée : 332 s (16:08:35–16:14:07), prévu ≤ 30 min

nvidia-smi propre après (seul PID 4286).

## Cohérence avec P3 (4)
Même arbre (665eeacc), même modèle Qwen3-VL-30B nvfp4-vision : P3 (4) donnait déjà TTFT ×45 et J ×4,4 le seuil, 17/20 top1 corrects (3 divergences). P3 (3) confirme et affine : la divergence commence très tôt (dès le 2e-4e jeton sur 2/3 images), cohérent avec un défaut dans le chemin vision/prefill de ce modèle sur ce moteur, pas un artefact de mesure isolé.
