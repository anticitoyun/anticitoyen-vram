# Sage — GEMM dense petit M : porte micro-banc FERMÉE telle qu'écrite ; palier Triton en situ AUTORISÉ comme étape scellée à part ; avant tout CUDA, un palier 2 Triton à sec (QKV empilée + split-K) que le banc lui-même suggère (17/09)

Entrée : Laure 21033b7 — M = 12, To/s par forme : q 0,72 · kv 0,47 · o 0,86 · gdn_qkv 1,01 · gdn_out 0,83 · gate_up 0,94 · down 1,09 ; min 0,47 < 0,9 ⇒ **porte fermée** (prédiction ≥ 1,3 réfutée sur q/k/v/o et gate_up, tenue sur down). Triton 3-6× la GEMV actuelle sur chaque forme ; bascule GEMV/GEMM à M ≥ 4 ; estimation en situ 20-25 ms/pas ≈ 350-400 t/s.

## 1. Porte : fermée, écrit tel quel

Le seuil disait « Triton atteint le mur de la bande » ; il ne l'atteint pas (0,47-1,09 selon N). On n'arrondit pas : la décision CUDA reste une décision séparée (§ 4). Ma prédiction est réfutée sur cinq formes sur sept — carnet.

## 2. Palier intermédiaire en situ : OUI, maintenant, comme étape scellée à part (Laure, 10 min)

Un gain de ×3 ne se laisse pas sur la table parce qu'une porte de ×7 est fermée — à condition de ne pas l'appeler « porte tenue ». Régime : `ACVRAM_DENSE_GEMM=triton|gemv` + seuil de bascule `M ≥ 4`, porté par `regime_ligne()` ; test d'équivalence logits vs boucle GEMV dans le même commit (bras cassant : échelle de bloc décalée → rouge).
Scellé, écrit avant : Qwen3.8 calibA b=12 **128 → ≥ 300 t/s** (estimation Laure 350-400 ; faux si < 250 : surcoût d'intégration — bascule, copies, forme non couverte — à lire au profil, pas à corriger à l'aveugle) ; `ppl-decode-kv` ± 0,0005 ; b=1 inchangé ± 3 % (M = 1 reste GEMV) ; J/jeton 3,12 → ≤ 1,3. Tenu ⇒ **défaut pour M ≥ 4**, cellule aux menus, source datée.

## 3. Avant CUDA : palier 2 Triton, à sec, que le banc lui-même désigne

Lire les sept formes : le taux monte avec N — kv (N petit) 0,47, q 0,72, o 0,86, gate_up 0,94, down 1,09, **gdn_qkv empilée 1,01**. Le noyau n'est pas mauvais, il est *sous-occupé* sur les petites N (170 SM à remplir avec peu de tuiles). Deux gestes à sec (Laurine, 2 h), puis le même banc :
1. **q/k/v des couches d'attention empilés en un seul poids** (comme `kda.py:102` et le `qkv` de `gdn.py:97` le font déjà) — une GEMM N = q + k + v par couche au lieu de trois ; prédiction : ≥ 0,95 To/s sur la forme empilée (le gdn_qkv est le témoin).
2. **Split-K sur les formes à N petit** (o, kv si encore séparée, gdn_out) : plus de CTA, réduction fp32 — prédiction : o 0,86 → ≥ 1,0.
Porte du palier 2 : **taux pondéré par les octets ≥ 1,0 To/s** (pas le min : c'est le pas qui compte, et gate_up/down sont 70 % des octets) ; faux si < 0,85. Tenu ⇒ en situ attendu 15-18 ms/pas ≈ 550-650 t/s — le ≥ 500 initial sans une ligne de CUDA.

## 4. CUDA : décision reportée après le palier 2, avec son prix écrit

Ce qu'il donnerait : de ~1,0 pondéré à 1,4-1,5 To/s (Marlin-like : dequant LOP3 en registres, pipeline cp.async/TMA), soit 15-18 → 12-13 ms/pas, **≈ 900 t/s, vLLM (621) dépassé** ; sur *tout* dense à b > 1. Ce qu'il coûte : Laurine 3-4 jours, un noyau de plus à maintenir hors Triton, et le précédent B1 (in-tuile réfuté deux fois) — mais ici la crête utile est la bande, pas les tensor cores, ce qui change la donne. Je tranche après le banc du palier 2 : si ≥ 1,0 pondéré, CUDA ne vaut que ×1,4 et passe **derrière** la GEMV experts ≥ 85 % (cellule phare classée) ; si < 0,85, CUDA est la seule voie et passe devant.

## 5. Ordre

Laure : § 2 en situ (10 min) → Laurine : § 3 à sec + banc → Laure : § 3 en situ (10 min) → décision § 4. Manon/Katy inchangés.
