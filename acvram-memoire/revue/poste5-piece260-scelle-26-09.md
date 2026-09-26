# Pièce 260 — W8A8 int8 au préfill pour les 233 int8 d'origine fp8 du mixte : À SEC + SCELLÉ (poste5 26/09, avant mesure)

Ordre chef (0.7.1), suite des propositions (a) et (b) du verdict 255. Branche poste5-260 (depuis poste5-255 = fusion-070 + 255).

## 1. À sec — ce qui existe déjà
* **La quantification A8 fusionnée existe** : `kernels/gemm_w8a8.py:quantifier_a8` (un noyau Triton par jeton : amax, échelle,
  arrondi à demi éloigné de zéro, ± 127) et `epilogue_i8c` (f32(acc)·s_x·s_w → bf16 en un noyau), tenus au bit contre leur
  forme torch (`tests/test_prefill_a8_porte.py`). Le chemin servi `gemm_i8c_cublas` (kernels:~1063) les enchaîne autour de
  `torch._int_mm` : c'est le W8A8 int8 AU DÉFAUT des convertis -qkvo-i8c depuis le 19/09 (`poste7-p2-au-defaut-19-09`,
  PREFILL_INT8=cublas). Le bras I du banc 255 (quantification et épilogue en torch non fusionnés) le sous-estimait.
* **Seul un drapeau exclut nos 233 poids** : `loader.py:_build_quant` marque tout int8 « origine: fp8 » `prefill_bf16`
  (pièce 139) → `_i8c_eligible` rend False → déquant bf16. La raison de la 139 : la copie signée q − 128 était gardée À VIE
  (10,6 Go, OOM). **La 201 l'a rendue transitoire** (par appel, ou une fois par portée partagée) : la raison est tombée.
* **La copie coûte** : `(q.to(int16) − 128).to(int8)` = trois noyaux, ≈ 10 o de trafic par poids. **q ^ 0x80 relu en int8 vaut
  q − 128 sur les 256 valeurs** : un noyau, 2 o par poids, au bit.

Écrit dans cette pièce (avant mesure) :
* `ACVRAM_I8C_FP8_PREFILL` = **bf16** (défaut, 139) | **cublas** (opt-in, HORS BIT) : le chargeur ne pose plus la marque ;
  ligne de régime `prefill_int8=cublas(origine fp8 ×233)`.
* `ACVRAM_I8C_COPIE` = **xor** (défaut, AU BIT) | int16 (témoin) : `kernels.copie_signee`.
* `tests/test_i8c_copie_260.py` : xor = int16 sur les 256 valeurs et sur le GEMM cublas entier ; marque posée / non posée
  selon l'opt-in ; ligne de régime ; défauts.

## 2. Scellé — micro-banc (`scratchpad/poste5-p260-26-09/banc260.py`, harnais et poids de la 255)
Bras, tous par `kernels.int8_matmul` (chemin relevé par CHEMINS_INT8) : **T** servi (marqué, déquant bf16) ; **I** opt-in
(non marqué : cublas, copie xor) ; **J** idem copie int16 ; **I_nu** copie faite avant (borne : poids signés en mémoire).
`_seul` hors portée, `_part` dans `depaquetage_partage` (8 appels, coût / 8). n ∈ {17, 64, 78, 128, 624}.

### Prédictions (µs)
| forme | n=78 T_part (255) | I_nu | I_part | I_part/T_part | n=624 T_seul (255) | I_seul | I_seul/T_seul |
|---|---|---|---|---|---|---|---|
| qkv 10240×5120 | 174,2 | 50-60 | 58-70 | **0,33-0,40** | 1 030,8 | 260-290 | 0,25-0,28 |
| gate 6144×5120 | 102,4 | 45-55 | 51-61 | **0,50-0,60** | 599,0 | 175-200 | 0,29-0,33 |
| out 5120×6144 | 92,7 | 45-55 | 51-61 | **0,55-0,66** | 608,3 | 185-210 | 0,30-0,35 |
| down 5120×17408 | 281,0 | 90-100 | 105-117 | 0,37-0,42 | 1 756,4 | 500-560 | 0,28-0,32 |
* Copie : xor qkv 65-85 µs, int16 300-450 µs (J_part − I_part ≈ 30-45 µs sur qkv à n = 78).
* Justesse (x aléatoire) : I ≈ 1,29-1,45 % comme le bras I de la 255 (même arrondi à la quantification près), T 0,94-1,13 %.
* **I = J au bit** sur toutes les formes et tous les n (sinon la copie xor est fausse : arrêt).
* n = 17 hors portée : GEMV pour T comme pour I (seuil 80) → I_seul = T_seul à ± 3 % (contrôle de dispatch).

### Seuils (en RELATIF au témoin de la même prise)
* **S1 (celui de la 255, par forme)** : I_part ≤ 0,50 × T_part à n = 78 sur qkv ET out. **Prédit NON tenu sur out (0,55-0,66).**
* **S2 (pondéré, ce que paie le moteur)** : Σ I_part / Σ T_part sur les trois formes GDN à n = 78 (48 couches chacune)
  ≤ 0,50 — prédit 0,44-0,55 ; ET I_seul ≤ 0,50 × T_seul à n = 624 sur les quatre formes — prédit tenu.
* ~~Règle : S1 et S2 tenus → étape moteur. S1 faux et S2 tenu → arrêt, décision de chef. S2 faux → fermé.~~
* **DÉCISION DE chef (26/09, avant la mesure, remplace la règle ci-dessus)** : **le critère principal est S2** (pondéré
  GDN, parce que c'est le préfill moteur qui compte) ; S1 par forme est relevé à titre d'information seulement. **S2 tenu →
  j'enchaîne sans m'arrêter** : étape moteur (préfill par lot, § 3), puis PPL appariée et mini lm-eval. S2 faux → fermé.

## 3. Scellé — étape moteur (jouée seulement si la règle ci-dessus l'ouvre)
* **Préfill par lot** (eng243 : 8 × 78, invite réelle, ACVRAM_CHRONO_SYNC=1, bascule du drapeau à chaud A/B/A/B dans un
  processus) : témoin 243 0,410 s. Gain prédit : GDN Σ(T_part − I_part) ≈ 188 µs × 48 couches × 8 séquences ≈ 72 ms ;
  attention q/k/v/o (16 couches) et mlp 56-63 à n = 624 ≈ 52 ms → **0,410 → 0,27-0,33 s (−20 à −35 %)**. **Seuil −15 %.**
  Pic mémoire : + taille de la plus grande copie transitoire (qkv_gate 84 Mo, down 89 Mo) — ≤ + 200 Mio.
* **KL 2b** (protocole 243/195 : b=8, préfill 8 × 78, 32 pas forcés, logits fp32, un processus, chauffe) : A = servi, B = opt-in ;
  T_admis = bascule 243 à n = 96 (243 : max 0,00396). **Tenue si KL(A‖B) max ≤ 2 × KL T_admis max, argmax ≥ admis − 0,005,
  rejeux 0.** **Prédiction : NON tenue (≈ 70 %)** : l'activation quantifiée ajoute ≈ 0,8 % d'erreur par couche sur 233
  tenseurs, là où la bascule 243 ne changeait que l'ordre des sommes. Rapporté aussi : PPL des pas forcés A et B.
* **Décision de chef (même message)** : la KL est RELEVÉE mais ne juge pas ; **le critère d'acceptation est celui de la
  Q19 : récupération moyenne ≥ 99 % sur le mini panel lm-eval ET pire tâche publiée** (seuil de pire tâche à sceller avant
  le panel, dans l'addendum qui fixera tâches, n et réglages), PPL appariée en filtre.
* **Si la KL n'est pas tenue** (Q19, poste4 → duck.ai, 26/09) : l'opt-in reste opt-in ; la KL seule n'est pas le critère
  d'acceptation d'un changement de format dans la pratique (vLLM/llm-compressor) — le précédent de la maison est la P2 du
  19/09 (W8A8 int8 au préfill du Coder accepté sur PPL 3 tranches ≤ 1,020 et cinq autres lignes). Pièce suivante proposée :
  PPL appariée (filtre) + panel de tâches mini (MMLU, GSM8K) par lm-eval, récupération moyenne ≥ 99 % ET pire tâche publiée,
  réglages et BOS identiques entre bras.

### Issues nommées
(i) out reste lent (tuile de `_int_mm` à N = 5120, K = 6144, déjà vue en F_mm 255) → S1 faux ; (ii) le xor n'apporte rien
(copie bornée ailleurs) : I_part ≈ J_part ; (iii) la KL tient (activations réelles moins hostiles que prévu) ; (iv) un
préfill qui calcule les logits de toute l'invite (logprobs d'invite) ferait passer lm_head (1,27 Go) au cublas : copie
transitoire de 1,27 Go — à vérifier contre `ModelSpec.octets_transitoires_i8c_bytes` avant tout défaut ; (v) le gain moteur
est sous le seuil parce que la part des 233 poids dans le préfill est plus petite que mon estimation.

## 4. Durée
Micro-banc : une prise ≤ 10 min, tests 260 dans la même prise. Étape moteur : une prise ≤ 20 min.

## Addendum — instruments de l'étape moteur et du panel (avant ces mesures, après le banc)
* **Moteur** `eng260.py` (base eng243) et **KL** `kl-260.py` (base kl-243) : bascule À CHAUD des int8 d'origine fp8 dans UN
  processus (liste des tenseurs marqués au chargement, piles comprises ; bf16 = marque posée, cublas = marque retirée ; cache
  `_i8c` vidé) ; contrôle de prise : B doit compter `cublas` > A (assert). Témoin admis de la KL inchangé (243, n = 96).
* **PPL appariée (filtre)** : `acvram eval` wiki-gptq, fenêtre 2048/2048, 8 192 jetons, deux processus (A défaut, B
  `ACVRAM_I8C_FP8_PREFILL=cublas`). Référence : 7,2157 (153b). **Filtre : PPL_B ≤ 1,020 × PPL_A** (précédent P2 19/09) ;
  prédit PPL_B/PPL_A 1,000-1,010.
* **Mini panel lm-eval (critère d'acceptation, Q19 + décision chef)** : lm-eval 0.4.13 dans `~/.venvs/lm-eval` (hors venv de
  mesure), `local-completions` contre `acvram serve` neuf par bras (max-batch 8, 4 096, sans spéculation), tokenizer HF du
  converti (BOS absent des deux côtés, réglages identiques), glouton, seed 0 : **mmlu 0-shot `--limit 40` par matière**
  (57 matières, ≈ 2 280 questions, loglikelihood : tout au préfill, le chemin testé) et **gsm8k 5-shot `--limit 500`**
  (exact_match strict et flexible). Récupération = score_B / score_A par tâche.
  **Accepté si récupération moyenne (mmlu, gsm8k) ≥ 99 % ET pire tâche ≥ 97 %.** Publiés : scores, récupérations, pire tâche,
  discordances appariées (A juste/B faux, A faux/B juste) et 2 SE de la différence appariée ; **si un seuil tombe à moins de
  2 SE : « non résolu », pas « tenu »**. Prédit : mmlu 99,3-100,2 %, gsm8k 98-101 % (résolution gsm8k ≈ ± 1,8 % à n = 500).
* Chaîne : prise `poste5-p260-moteur` (eng, KL, PPL) puis prise `poste5-p260-lmeval` (A puis B), chacune ≤ 30 min.
