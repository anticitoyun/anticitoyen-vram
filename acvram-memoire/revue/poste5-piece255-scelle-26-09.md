# Pièce 255 — FP8 natif sur SM120 pour les 233 tenseurs d'origine fp8 du mixte : À SEC + SCELLÉ (poste5 26/09, avant mesure)

Ordre chef (0.7.1). Alias `Qwen3.8-27B-unsloth-mixte-i8c`, servi aujourd'hui en int8 symétrique par canal (W8A16).
Hérite de la 148 (`poste5-piece148-fp8-natif-24-09.md`, `poste5-piece148-banc-verdict-24-09.md`), de la 43
(`verdict-banc-etroites-noyaux-fp8-22-09.md`), de la 58 (`poste5-etroites-arithmetique-23-09.md`), de la 243 et de la Q17
(poste4 → duck.ai, `revue/poste4-duckai-26-09.md` § Q17).

## 1. À sec

### 1a. Le checkpoint d'origine a-t-il ses échelles ? OUI, intactes
Source (manifeste du converti, `source.chemin`) : `/mnt/AI_GENERATOR/ninfer/sources/Qwen3.8-27B-NVFP4`, un seul
`model.safetensors`, compressed-tensors `mixed-precision`. Groupe FP8 (`config.json` → `quantization_config.config_groups.group_0`) :
poids `float` 8 bits `strategy: channel`, symétriques, statiques ; activations `float` 8 bits `strategy: token`, **dynamiques** —
c'est un schéma **W8A8 FP8 par canal × par jeton**, celui que servent vLLM et NInfer. Recensement de l'en-tête :
**233 poids F8_E4M3** (lm_head 1 ; linear_attn in_proj_qkv/in_proj_z/out_proj 48 × 3 ; self_attn q/k/v/o 16 × 4 ;
mlp gate/up/down des couches 56-63 : 8 × 3), chacun avec `weight_scale` **BF16 [out, 1]**. = les 233 `origine: fp8` du
manifeste. (Les autres clés F8_E4M3 de l'en-tête sont les `weight_scale` e4m3 des mlp nvfp4 : 168 = 56 × 3.)
Notre conversion (`hfquant.py:_fp8_canal` → `convert.py:1777`) : w = e4m3 × s en fp32 EXACT, puis ré-encodage int8 par canal
(SNR 39,05 dB sur lm_head, ≈ 1,1 % d'erreur relative de poids). Un format fp8 neuf servirait les poids **au bit de la source**.

### 1b. Quels noyaux FP8 tournent réellement sur sm_120 dans notre pile
Pile : torch 2.14.0+cu130 (CUDA 13.0, arches compilées sm_75/80/86/90/100/**120**), triton 3.8.0 ; vLLM 0.29.0 dans
`/opt/ia/vLLM/.venv` (hors pile servie, banc seulement).

| chemin | échelles | sm_120 | preuve |
|---|---|---|---|
| `torch._scaled_mm` e4m3 **rowwise** (A [M,1] fp32, B [1,N] fp32) | par jeton × par canal = **notre checkpoint** | **oui, CUTLASS dédié** | symbole `f8f8bf16_rowwise_impl_sm100_sm120` dans `libtorch_cuda.so` (et `_sm89` à part) ; déjà EXÉCUTÉ sur la 5090 (p43 22/09, `banc-etroites-noyaux.py:117-135` ; `fp4_gemm.py:257-282`) |
| `torch._scaled_mm` e4m3 tensorwise | scalaire | oui (cuBLASLt 13.0) | message « FP8 tensorwise and rowwise… float32 » ; pas notre schéma |
| `torch._scaled_mm` blockwise 1×128 / 128×128 | par bloc | **non établi** sur sm_120 (chaînes présentes, dispatch non lu) | hors sujet : notre checkpoint est par canal, pas par bloc |
| `torch._scaled_mm` MXFP8 1×32 (e8m0) | micro-bloc | matériel oui (`mxf8f6f4`), torch non vérifié | hors sujet (format différent de la source) |
| vLLM `cutlass_scaled_mm_sm120_fp8` (dense, canal × jeton) et `cutlass_scaled_mm_blockwise_sm120_fp8` | canal / bloc | **oui** | symboles dans `vllm/_C_stable_libtorch.abi3.so` ; la Q17 (MoE groupé rejeté ≥ 110) ne touche PAS le dense |
| Marlin fp8 W8A16 (`apply_fp8_marlin_linear`, vLLM) | canal | oui | mesuré 148 : ± 1 % de Marlin u8, 1,36-1,58 To/s à M = 8 |
| Triton `tl.dot` e4m3 | libre | oui | mesuré 58 (`_fp8_kernel`) : qkv +4,9 %, o +16 % contre int8 aux formes étroites |
| acvram `.cu` | — | **aucune MMA fp8** | seule `kind::mxf4nvf4` (148 § 0) |

Piège Q17 (repli sm_89 pris pour du natif) : le banc relève le NOM du noyau lancé (torch.profiler) ; un nom `sm89` ou
cuBLASLt sur le chemin rowwise = bras INVALIDE.

### 1c. Ce qu'on sait déjà, et ce qui manque
* **Décodage n = 1-8** : fermé côté octets (1 o/poids dans tous les formats). W8A8 `_scaled_mm` : 0,30-0,61 To/s (p43, formes
  ÉTROITES du Coder) ; Marlin fp8 W8A16 = Marlin u8 (148), +3,1 % du pas b=8 au mieux. **Manque** : `_scaled_mm` rowwise sur les
  formes RÉELLES du 27B (leçon 148 : une efficacité se relève sur les formes du modèle visé).
* **Préfill n = 64-128** : **jamais mesuré** en FP8. Aujourd'hui (pièce 139, `loader.py:116`) les 233 poids passent en déquant
  bf16 + GEMM bf16 (W8A16) ; sous la portée partagée du préfill GDN (243, seuil 16) la déquant est payée une fois pour 8
  séquences. Banc 243 (qkv 10240 × 5120, n = 78) : déquant NON partagée + GEMM **733,7 µs**, partagée **171,7 µs**, GEMV 636,4.
  Le FP8 W8A8 supprime la déquant ET double le débit MMA (e4m3 contre bf16).

## 2. Scellé — micro-banc isolé (étape 3), avant toute mesure

Instrument : `scratchpad/poste5-p255-26-09/banc255.py`, harnais du banc 243 (poids RÉELS lus dans le converti ET dans la
source, L2 froid par rotation, médiane de 30 rejeux sous graphe CUDA), `prise.sh` sous `carte.sh`, -lgc 2700.
Formes : qkv 10240 × 5120, z 6144 × 5120, out 5120 × 6144, down 5120 × 17408 (couche 0 / 56 du mixte ; poids fp8 de la
source + leur `weight_scale`, poids int8 du converti).
Bras : **T** = servi (`kernels` tel que le moteur l'appelle : GEMV int8 à n ≤ seuil, sinon déquant + GEMM bf16) — **T_seul**
hors portée, **T_part** dans `depaquetage_partage` avec 8 appels (coût par appel) ; **F** = `torch._scaled_mm` rowwise
e4m3, quantification e4m3 PAR JETON de l'activation INCLUSE dans le temps ; **I** = `torch._int_mm` int8 W8A8 (activation
int8 par jeton, poids signés déjà en mémoire — copie non comptée, dite) : le concurrent int8 à activation quantifiée égale.
n ∈ {1, 8} (décodage) ; {64, 78, 128, 624} (préfill ; 624 = 8 × 78, attention et mlp 56-63).

### Prédictions (qkv sauf mention ; µs)
* **P1 décodage** : T n=1 33-37 (1,41-1,57 To/s, 148), n=8 36-40 ; **F n=1 et n=8 : 45-90 (0,6-1,2 To/s) → F/T 1,2-2,5,
  PLUS LENT**. Seuil : F ≤ 0,90 × T à n = 8 sur les QUATRE formes, sinon décodage fermé pour le W8A8 (prédit : fermé).
* **P2 préfill n = 78** : T_part 160-185 (243 : 171,7), T_seul 700-760 (243 : 733,7) ; **F 40-70** (poids 52 Mo au plancher
  ≈ 35 µs, 8,2 GFLOP à ≈ 400 TFLOPS e4m3 ≈ 20 µs, + quantification) → **F/T_part 0,22-0,45**. I : 40-80 (int8 MMA même débit
  que e4m3 sur la 5090) → I/F 0,8-1,3.
* **P3 préfill n = 64 / 128** : même ordre, F/T_part 0,2-0,5 ; T à n = 64 hors portée = GEMV 525,8 (243) → F/T 0,08-0,15.
* **P4 préfill n = 624** : T_seul ≈ 900-1 300 (déquant ≈ 650 + GEMM bf16 65 GFLOP), **F 170-300** → F/T_seul 0,15-0,3.
* **P5 justesse (contre x · W_fp8 exact en fp64, x bf16 réel)** : T (int8 W8A16) erreur relative RMS 0,8-1,5 % (ré-encodage
  des poids) ; F (W8A8) **3-4,5 %** (58 : 3,8 % ; activation e4m3 par jeton) ; I (int8 W8A8) 1,5-3 %. Contrôle qui peut
  rendre faux : si F < T en erreur, le ré-encodage int8 coûte plus que la quantification de l'activation.
* **P6 noyau réellement lancé** : F → noyau CUTLASS sm100/sm120 (nom relevé) ; s'il sort `sm89` ou cuBLASLt : bras invalide.

### Seuil de passage à l'étape moteur (préfill seul)
**F ≤ 0,50 × T_part à n = 78 sur qkv ET out, ET F ≤ 0,50 × T_seul à n = 624 sur les quatre formes.** Tous les seuils en
RELATIF au témoin de la même prise (leçon 243 : jamais un absolu tiré d'un témoin supposé).

### Issues nommées, y compris les gênantes
(i) F lent au préfill aussi (tuile CUTLASS mal choisie à M = 78, ou quantification de l'activation non fusionnée qui coûte
autant que le GEMM) → fermé, l'écart avec NInfer n'est pas là ; (ii) I ≈ F : alors l'int8 W8A8 (`_int_mm`, déjà dans la pile,
poids signés à stocker) atteint le même gain SANS format neuf — et la question devient « activation quantifiée ou non », pas
« fp8 ou int8 » ; (iii) le gain de temps passe mais la qualité non : W8A8 change la sortie au-delà de l'ulp et rend une part
des −5,67 % de PPL gagnés contre NInfer (139 b) — c'est l'issue la plus probable côté KL (voir 2b) ; (iv) VRAM : un préfill
FP8 et un décodage int8 exigent deux copies (+10,6 Go) — non servable ; seul un format fp8 UNIQUE (décodage W8A16 fp8 par
Marlin fp8 ou GEMV maison, préfill W8A8) tient en mémoire, soit un format neuf (convert, loader, dispatch, ≈ 1-2 jours) ;
(v) `_scaled_mm` exige K et N multiples de 16 et B colonne-majeure : formes du mixte conformes, lm_head (248 320) aussi.

### 2b. Scellé de l'étape moteur (n'est joué que si le seuil ci-dessus passe)
KL au protocole de la 243/195 (b=8, préfill 8 × 78, 32 pas forcés, logits fp32, UN processus, chauffe d'abord) :
A = servi, B = préfill FP8 W8A8 des 233 tenseurs (décodage inchangé, copie fp8 transitoire pour la mesure seulement) ;
témoin T_admis = bascule 243 (80/16) ; **tenu si KL(A‖B) max ≤ 2 × KL T_admis max et argmax ≥ admis − 0,005, rejeux 0**.
**Prédiction : NON tenue (≈ 65 %)** — l'erreur par couche de F (3-4,5 %) est ≫ celle de la bascule 243 (même poids, bf16).
Si non tenue : opt-in seulement, et la suite honnête est un format fp8 unique jugé à la PPL appariée (poids au bit de la source
: prédit PPL −0,1 à +1,5 % contre l'int8 servi), pas au bit.

## 3. Durée
À sec : 45 min. Micro-banc : une prise ≤ 10 min (chargement de 4 poids × 2 formats, pas du modèle).

## Addendum instrument (avant mesure, 26/09 05 h)
Harnais retenu = celui du banc 243 (témoin comparable) : événements CUDA, 8 appels par fenêtre, médiane de 5 fenêtres,
L2 vidé (192 Mo) avant chaque fenêtre, rotation sur 4 couches (0-3 GDN, 56-59 mlp) — PAS de graphe (le § 2 disait « 30
rejeux sous graphe » : écart d'écriture, corrigé avant mesure). Si `_scaled_mm` refuse un M brut, M est rembourré à 16 et
c'est relevé (`F_pad`). Poids int8 lus directement dans le converti (INT8Tensor marqué `prefill_bf16`, comme loader.py:116),
pas le modèle entier.
