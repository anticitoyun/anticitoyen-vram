# LeapQuant / STEPQuant × états GDN (Qwen3.8-27B, Qwen3.6-35B-A3B) : applicabilité à sec, octets, prédiction scellée (poste5, 30/09)
* instrument : `scratchpad/poste5-leap-30-09/octets-etat.py` (sortie `octets-etat.txt`), à sec ; articles arXiv 2609.38166 v1, 2609.38169 v1 ; dépôt github.com/Dreamer-Toby/STEPQuant
* commit : poste5-leap (voir git log)
* régime : à sec, CUDA_VISIBLE_DEVICES="", aucune prise, aucun pytest (campagne de poste2)
* scellé : § 3 ci-dessous, écrit AVANT toute mesure ; aucune mesure n'a eu lieu
* mesuré : rien (octets calculés ; temps ancrés sur les profils du 17/09 et du 24/09)
* verdict : APPLICABLE en opt-in seulement (la sortie change) ; gain prévu b=12 0,75-2,44 ms/pas (Qwen3.8), b=1 ≤ 0,2 ms (non résoluble)
* durée : 0 (à sec)
## 1. Notre chemin et ses octets (fichier:ligne)
État par séquence et par couche : `S [1, HV, 128, 128]` fp32 (gdn.py:12-14, créneaux `lot_etats`, gdn.py:478-484), décodé par
`fused_recurrent_gated_delta_rule_fwd_kernel` (fla 0.5.2) EN PLACE `h0 = ht = S` (F4, gdn.py:137-157) : une lecture et une écriture
par tuile → **8 o/élément/pas**. Qwen3.8-27B (48 GDN sur 64, HV 48) : 3,15 Mo/couche/séq, **6,29 Mo r+w/couche/pas, 302 Mo/séq/pas,
3,62 Go/pas à b=12** ; Qwen3.6-35B-A3B (30 GDN sur 40, HV 32) : 126 Mo/séq/pas, 1,51 Go à b=12. État de conv (≤ 0,25 Mo) hors champ.
Noyau mesuré (Qwen3.8) : b=8 22,6 µs/couche (poste5-piece156b-dossier-24-09.md:20), b=12 31,3 (1,50 ms/48, verdict-gemm-dense-fusion-
17-09.md:6) → 5,3 µs + 2,16 µs/séq. Débit apparent r+w b=12 2,42 To/s > 1,2-1,45 (REGLES § 9) : une part des écritures sort de la
fenêtre du noyau (réécriture différée) et se paie sur les noyaux suivants.
## 2. Ce que font les articles, rapporté à nous
| | LeapQuant (2609.38166) | STEPQuant (2609.38169) |
|---|---|---|
| principe | état INT8 figé sur une fenêtre p=16 ; jetons de la fenêtre tamponnés en fp16, sortie = état figé + tampon ; requantification au bord seulement ; 4 « Compensator Tokens » fp16 (rang 1, itération de puissance) + lissage par ligne de clé | quantification à CHAQUE pas ; bits par tête fixés hors ligne (durée de vie × erreur), têtes pivots fp16 (1,39 %) ; ajustement 2 axes + réécriture sur un flux séparé |
| o/élément | 1,19 au bord ; par pas (lecture + tampon moyen + bord amorti) **1,20 Mo/couche/séq** (ratio 0,191) | 6 bits nominaux, ×5,03 → **1,25 Mo** (ratio 0,199) |
| qualité publiée | INT8 = FP32 (AIME 87,9, Qwen3.5-9B) ; INT8 par pas 7,1 ; BF16 par pas 72,1 | @6 = FP32 sur **Qwen3.8-27B** (84,51 contre 84,61) ; @4 > INT8 uniforme |
| vitesse publiée (aucune à b ≤ 16) | noyau 4,25× sur RTX 5090 à **b=512** ; décodage ×1,22-1,57 à b ≥ 256 | +20,5 % à B=512 (A800 TP4, BF16) ; rien sous B=32 |
| code | aucun lien dans l'article ; TileLang dans vLLM | publié : SGLang 0.5.12, Triton 3.7.1, calibration hors ligne |
## 3. Prédiction et seuils (scellés AVANT mesure ; toutes les issues nommées)
Gain LeapQuant, Qwen3.8 : **b=12 0,75-1,01 ms/pas** si seul le temps du noyau compte (M1), **2,02-2,44** si le pas paie tous les
octets épargnés (2,93 Go/pas, M2), soit 2,7-9,0 % du pas de 27,2 ms (17/09) ; **b=1 0,06-0,20 ms** (0,4-1,4 % de 14,3 ms), sous 2 σ
d'une cellule. Qwen3.6 : b=12 0,31-1,02 ms, b=1 ≤ 0,09. STEPQuant : mêmes octets à 4 % près + 48 ajustements/pas sur un second flux,
coût non publié sous B=32. Mémoire : Qwen3.8 144 → 61 Mio/séq (LeapQuant) ou 29 (STEPQuant@6), −1,0 à −1,3 Gio de KV à b=12.
* A (banc noyau isolé, dimensions Qwen3.8, témoin fla fp32 en place dans la même prise, godets {1, 2, 8, 16} capturés, § 3) :
  fp32 7,5 ± 1 µs (b=1) et 31,3 ± 3 µs (b=12), sinon l'ancre est fausse et tout se recalcule ; INT8 b=12 ≤ 16 µs.
* B (pas servi, Qwen3.8 b=12, ABBA ≥ 2 paires, `certifie-b12`) : **GO si gain ≥ 0,75 ms/pas ET ≥ 2 σ du témoin**.
* Qualité (la sortie change : ni défaut ni « ± 1 ulp » → format d'état opt-in, décision de l'utilisatrice comme pour le KV) : KL
  contre fp32 ≤ 0,74 à b=12, court (2 048) **et long ≥ 16 384** (chef 30/09) ; ppl-decode-kv géo dans 2 SE ; FAUX si l'une échoue.
Issues : **I1** GO (B et qualité tenus) → format d'état INT8 opt-in, étiqueté. **I2** gain sous le bruit (< 2 σ) ou < 0,75 ms à b ≤ 12
→ abandonné (chef 30/09). **I3** gain ≤ 0 : l'ordonnée (5,3 µs/couche) et le bord (itération de puissance) mangent les octets.
**I4** qualité FAUX sur nos poids nvfp4 → abandon. **I5 (celle qui me gêne)** : un noyau fp32 en place mieux réglé, AU BIT, prend
l'essentiel sans risque — NInfer fait la même récurrence en 18,7 µs contre 22,6 à b=8 (−17 %) ; s'il rend ≥ 0,45 ms/pas à b=12
(60 % de M1), LeapQuant passe après. **Ordre proposé** : (1) ce noyau réglé (BV, num_warps, portes F1), test d'équivalence au bit
dans le même commit ; (2) banc A LeapQuant (portage Triton, code absent) si (1) laisse ≥ 0,5 ms ; (3) STEPQuant pour la mémoire.
