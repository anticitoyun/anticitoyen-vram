# Pièce 139 — scellé de la carte (poste5, 24/09, écrit AVANT la conversion et toute prise)

Feu chef. Code : branche poste5 (dispatch par groupe 3065cd86, refus modelopt MIXED 2e commit après). Alias produit :
`models_acvram/Qwen3.8-27B-unsloth-mixte-i8c`, converti À SEC (`CUDA_VISIBLE_DEVICES=""`, `--profile
rig-14900k-5090-3080ti`, `--passage-direct --no-awq`, `ACVRAM_HFQUANT_PAR_GROUPE=1`) depuis
`/mnt/AI_GENERATOR/ninfer/sources/Qwen3.8-27B-NVFP4` (celle que NInfer convertit, scellé 102 bis).

Référence NInfer : `ninfer/models/qwen3_8_27b_nvfp4.ninfer` (recette officielle, **W4A4 MLP / W8A8 attention**, même
source unsloth) — l'artefact A16Only de la 102 bis ne démarre pas (`serveur-ninfer-102bis-b1.log` : « native input
requires one contiguous parent region »). acvram sert **W4A16 (nvfp4 direct, octets de la source) + W8A16 (int8 par
canal ré-encodé du fp8)** : mêmes poids nvfp4, fp8 ré-encodé (+1 % d'erreur en quadrature, dossier § 3), activations
NON quantifiées. Formats d'activation différents, nommés dans chaque ligne de verdict.

## (a) Conversion et chargement

* Manifeste : 168 tenseurs `passage_direct: true` nvfp4 ; 233 tenseurs `origine: fp8` en int8, `group_size` = largeur ;
  aucun `weight_scale`/`k_scale`/`v_scale` rendu comme tenseur. Sinon FAUX (défaut de code, rien à mesurer).
* Octets de poids (`weights_bytes` d'`acvram eval`) : prédit **21,1-22,1 Go** (nvfp4 8,42 + int8 10,63 + plongements
  bf16 2,54 + reste bf16 ~0,1) contre 24,35 Go pour `Qwen3.8-27B-nvfp4` (102). FAUX si > 23 Go (un groupe laissé en bf16).
* Chargement sous le régime servi, un jeton décodé au godet 1 (REGLES § 3) ; VRAM `nvidia-smi` relevée après
  chargement (serve, `--max-model-len 4096 --max-batch 8`). Prédit 24-27 Gio. Issue nommée : le chargeur ou un noyau
  refuse l'int8 par canal sur `linear_attn.in_proj_qkv/in_proj_z/out_proj` ou sur les piles gate/up des couches 56-63
  (jamais servis en int8 par canal) → repli nommé ou refus : je le dis, pas de mesure (b)/(c).

## (b) PPL — instrument de la 102, apparié à NInfer

`acvram.cli eval M --corpus acvram/data/calibration-anglais.txt --window 4096 --stride 2048 --min-context 0 --max-tokens
16384 --json` (`par_fenetre`), bootstrap apparié `scratchpad/poste4-piece138-sem-ppl-24-09/bootstrap_ppl.py`
(fonction de la 138 bis, 20 000 tirages) contre les fenêtres NInfer 4,2085.
* Prédit : **PPL 4,05-4,25** ; Δ apparié acvram−NInfer entre −3,5 % et +1 % (poids MLP identiques ; nous ajoutons
  1 % d'erreur de poids sur 40 % des paramètres, NInfer quantifie ses activations en 4 et 8 bits).
* Seuil : |z| > 2 conclut un écart ; sinon « non conclusif ». FAUX de code si PPL > 4,6 (repère : 4,0938 pour notre
  propre nvfp4, 3,6882 en bf16) — alors fp8 mal déquantifié ou ré-encodé, pas de (c).
* Issue qui me gênerait : acvram au-dessus de NInfer au-delà de 2 σ. Le ré-encodage int8 coûterait alors plus que
  leurs activations quantifiées → l'option C (fp8 servi nativement) se justifie par la mesure.

## (c) Débit et J/jeton, ABBA, face à NInfer

Instrument de la 102 : `scratchpad/poste2-piece102-etalon-hf-24-09/banc-chat-openai.py` (chat/completions,
`max_tokens` 256), ABBA ≥ 5 paires par b (≥ 5 lots par bras), `-lgc 2700`, serveur neuf par passe, cpu-safe relevé à
chaque passe, apps début/fin. b=1 et b=8, une prise ≤ 30 min par b. Repères de la 102 (même banc, même carte) :
NInfer 75,35 t/s / 4,365 J (b=1), 467,55 / 0,6933 (b=8) ; acvram `Qwen3.8-27B-nvfp4` 74,1 / 3,99 (b=1), 299,1 / 1,070 (b=8).
* b=1 : lecture 19,05 Go/pas au lieu de ~21,8 pour notre nvfp4 de la 102 ; mais int8 W8A16 moins efficace que le
  nvfp4 par octet (p57 : 1,11 To/s). Prédit **72-85 t/s**, J/jeton 3,6-4,1 ; FAUX si < 65 ou > 95.
* b=8 : NInfer garde ses tensor cores W4A4/W8A8. Prédit acvram **270-340 t/s**, J/jeton 0,95-1,20, NInfer devant
  de 35 à 75 %. FAUX si acvram < 240 (régression du chemin int8 par canal en lot).
* Horloge SM moyenne publiée par fenêtre ; écart > 30 MHz entre bras = cellule non comparable, je le dis.

## Ordre et durée

Conversion à sec (processeur, nice 19, 6 cœurs, hors carte) → (a) + (b) en UNE prise (chargement, jeton, VRAM, eval :
prévu ≤ 10 min) → (c) b=1 (≤ 15 min) → (c) b=8 (≤ 15 min). File carte : 109 poste3, 142 poste1, puis moi.

## Addendum 24/09 07 h 5x (après le diagnostic mémoire, AVANT la prise suivante ; ordre chef)

Diagnostic (`scratchpad/poste5-p139-24-09/diag-memoire.txt`, même prise, llama-server de l'utilisateur 5,5 Gio sur la
carte 0, jamais touché) : mixte **22,50 Go** de stockage unique, **23,9 Gio** alloués après chargement (102 : 17,28 Go /
19,18 Gio). (a) poids : 22,50 Go, AU-DESSUS de la fourchette prédite 21,1-22,1 (la tête MTP bf16, 0,53 Go, n'y était
pas), sous le FAUX de 23 Go. Il reste ~1,9 Gio libres : la chauffe 4096 × b=8 rend un refus NOMMÉ (« contexte non
tenu », `serve-a.log`) — pas un OOM muet ; le plan lit la VRAM totale (`detect.py:392`), le budget KV la VRAM libre
(`loader.py:1143`). Pas un défaut servi.

Changement, écrit avant : serve `--max-model-len 2048 --max-batch 8` pour (a) et pour les deux bras acvram de (c)
(le banc chat envoie < 1 024 jetons de contexte ; NInfer garde `--max-context 4096`, ce qui ne change que sa réserve
KV). (b) inchangé (`eval --window 4096`, une séquence) ; s'il ne tient pas, refus nommé au verdict, pas de fenêtre
plus courte improvisée. Comparaison des deux alias acvram à VRAM libre égale : même prise, même llama-server présent.

## Addendum 2, 24/09 08 h 0x (AVANT la prise suivante) — correction du premier addendum

* ERRATUM : le llama-server de l'utilisateur (pid 4627, 5,6 Gio) est sur la **3080 Ti (index 1)**, pas sur la 5090
  (`nvidia-smi --query-compute-apps=gpu_bus_id` : 02:00.0). La 5090 était ENTIÈREMENT à acvram ; le premier addendum
  et sa lecture « 1,9 Gio libres à cause du llama-server » sont faux.
* Cause des OOM (chauffe 2048 et `eval` seul, 29,16 Gio alloués par PyTorch contre 23,9 au chargement) : régime de
  préfill int8 par défaut `cublas` (`acvram/kernels/__init__.py:765`) — `_i8c_poids` (:770-785) construit et GARDE sur
  chaque poids int8 symétrique par canal une copie int8 signée de sa taille (conçu pour les q/k/v/o de Coder, 0,9 Gio) :
  ici 233 tenseurs, ~10,6 Go de plus → ne tient pas. Ce régime quantifie aussi les activations en A8 au préfill : la PPL
  mesurée sous le défaut serait W8A8 sur 40 % des paramètres, contraire à l'énoncé du scellé (« activations non
  quantifiées »).
* Changement, écrit avant : **`ACVRAM_PREFILL_INT8=bf16`** (régime existant : déquantification entière temporaire +
  GEMM bf16, W8A16, sans copie persistante) pour (a), (b) et les deux bras acvram de (c). `--max-model-len` revient à
  4096 (scellé d'origine). Prédictions inchangées. Défaut nommé, NON traité ici (poste moteur) : sous le défaut
  `cublas`, un alias à int8 par canal au-delà des q/k/v/o double sa mémoire int8 au premier préfill.

## Addendum 3, 24/09 08 h 1x (AVANT la prise suivante) — ordre chef : défaut réglé avant fusion

`ACVRAM_PREFILL_INT8=bf16` a échoué autrement (prise 07:53) : le repli bf16 refusait tout groupe ≠ 128
(`kernels/__init__.py:1284`), donc les poids par canal. Correctif 9082007a : le chargeur marque `prefill_bf16` les int8
« origine: fp8 » (`loader.py:_build_quant`) ; `_i8c_poids` ne leur fait AUCUNE copie ; ils prennent la déquant bf16
par tranches via `vue_g128` (au bit du par canal, test) ; ligne de régime `prefill_int8=bf16(origine fp8 ×233)`.
La prise revient au régime PAR DÉFAUT (aucune variable) : c'est ce qui sera servi. Format côté acvram sur ces 40 % :
**W8A16** (préfill et décodage) contre **W8A8** chez NInfer — à écrire tel quel au verdict.
