# Pièce 243 (179 b) — seuil GEMV → GEMM int8 abaissé : SCELLÉ (poste5 26/09, écrit avant toute mesure)

Ordre chef : bascule du GEMV int8 vers le GEMM au-delà d'un seuil de n, HORS BIT (KL scellée). Point de départ :
verdict 221 (poste1) — à n = 78-80 le GEMV int8 est borné par le CALCUL (cœurs CUDA, W relu depuis le L2) ; 204 b :
préfill 8 × 78 du mixte 886 ms dont 578 d'`int8_gemv` ; eng179 (moteur, invite réelle 78 jetons) : 1,19 s par lot,
3 456 appels `int8_gemv`. Branche poste5-243 depuis origin/main 1d6755256.

## À sec — ce qui prend le GEMV aujourd'hui (fichier:ligne)
* `int8_matmul` (kernels/__init__.py:1402) : GEMV si n ≤ `ACVRAM_INT8_GEMV_MAX` = 80 (l. 828) ; au-delà, poids i8c
  éligibles → cuBLASLt W8A8 (`gemm_i8c_cublas`, A8 par jeton, l. 1485) ; sinon déquant bf16 + GEMM (l. 1500 ss), partagée
  entre séquences sous B' (`depaquetage_partage`, l. 588 ; pièce 179).
* Mixte : 281 tenseurs int8, tous par canal (groupe = K) ; 233 marqués « origine fp8 » (`prefill_bf16`, l. 857) → PAS de
  cuBLASLt, déquant bf16 au-delà du seuil ; les autres prennent cuBLASLt.
* Qui voit n = 78 : les couches GDN bouclent par séquence (couches.py, boucle de `forward` hybride) → chaque linéaire int8
  y est appelé 8 fois à n = 78 ; attention et MLP groupés (n = 624 > 80) prennent déjà le GEMM.
* Le seuil 80 vient d'une mesure à UNE séquence et à l'ancienne tranche 16 (docstring l. 1411 : croisement vers 88) ;
  depuis la 187 la tranche vaut 6 (W relu ~2,7 × plus souvent) et, sous B', la déquant se paie une fois pour 8.

## Prédictions (avant mesure)
P1 — banc isolé, poids réels du mixte (couche 0 : qkv 10240 × 5120, porte, sortie), n ∈ {17, 24, 32, 48, 64, 78} :
* GEMV qkv n = 78 : 655 µs (221, reproduit à ± 5 %) ;
* déquant + GEMM bf16, NON partagée : 120-220 µs à n = 78 ; croisement n* (non partagé) entre 16 et 40 ;
* déquant partagée sur 8 (coût déquant / 8 + GEMM) : 40-90 µs à n = 78 ; croisement ≤ 17 (plancher cuBLASLt M > 16) ;
* cuBLASLt W8A8 (tenseurs éligibles) : 20-60 µs à n = 78.
P2 — moteur (eng179, 8 × 78, invite réelle) avec seuil 16 SOUS B' (portée multi-séquences seulement) : préfill par lot
1,19 → 0,40-0,55 s (−54 à −66 %). **Seuil : ≤ 0,83 s (−30 %)**, sinon levier non tenu.
P3 — KL (décodage b=8 après préfill 8 × 78, 32 pas forcés, logits fp32, protocole de la 195) :
* témoin T_admis = la MÊME bascule déjà servie au défaut, à n = 96 (A : seuil 128 = GEMV, B : seuil 80 = GEMM) ;
* **tenu si KL(A‖B) max à 78 ≤ 2 × KL T_admis max à 96**, argmax AB ≥ argmax T_admis − 0,005, rejeux A et B = 0 ;
* rapporté aussi : T1 de la 195 (seq 0 seule b=1 contre sa ligne b=8), PPL des pas forcés.
P4 — service (banc chat du 179, mixte b=8, ABBA, -lgc 2700) : +6 à +14 % t/s. **Seuil +3 %** et gain > 2 × étendue de A.

## Issues nommées (y compris les gênantes)
(i) gain sous le seuil : la déquant bf16 est bornée par la mémoire plus que prévu, ou le GEMM bf16 à M = 78 tombe sur une
mauvaise tuile cuBLAS ; (ii) KL non tenue → opt-in seulement ; (iii) mémoire : sous B' la déquant partagée retient
davantage de poids 16 bits (pièces 172/201) → marge KV ou OOM, à relever ; (iv) hors B' (une séquence, invite courte à
b=1), un seuil abaissé régresserait si n* non partagé > seuil → le seuil global reste 80 sauf croisement mesuré plus bas ;
(v) Coder -qkvo-i8c (cuBLASLt éligible) : même bascule si la portée est globale, KL à refaire sur lui ; (vi) la RoPE du
1er lot (213) : KL mesurée dans UN processus, A puis B, après un préfill de chauffe → tables RoPE identiques des deux côtés.

Implémentation prévue (opt-in d'abord) : `ACVRAM_INT8_GEMV_MAX_PARTAGE` (défaut = `INT8_GEMV_MAX`, sortie inchangée),
seuil lu quand une portée `depaquetage_partage` est active. Q14 (seuils vLLM/SGLang/TRT-LLM, poste4 → duck.ai) :
intégrée si elle arrive avant la mesure.

## Addendum (avant mesure) — Q14 reçue (poste4 → duck.ai, poste4-224 d142d5f38, revue/poste4-duckai-26-09.md)
Aucun seuil universel ; convergence : M ≤ 16 GEMV, 16-32 transition, 32-64 GEMM int8 MMA en général meilleur, ≥ 64 GEMM
d'abord ; vLLM (SM90) ne choisit que la tuile CUTLASS par tranche de M, pas GEMV contre GEMM ; TRT-LLM : un croisement
mesuré ≈ 32 (H800, autre format). Cohérent avec P1 (croisement non partagé 16-40, partagé ≤ 17) ; rien ne change aux
prédictions. Point noté, hors portée : vLLM déconseille l'INT8 sur CC ≥ 10 (FP8 recommandé) — nos poids du mixte sont
d'origine fp8 et servis en int8 par canal ; à ouvrir en question séparée si chef le veut.
