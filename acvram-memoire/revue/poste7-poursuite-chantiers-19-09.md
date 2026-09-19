# poste7 — ce qui ne se termine pas ce soir démarre maintenant ; E1-bis : `-lgc 2700` tient b=1 ET b=12 ; correction d'une erreur de poste7 sur la borne du prefill (19/09, 17 h 50)

Source : utilisateur 17 h 45 (« ce qui ne peut être terminé doit être poursuivi directement sans attendre ») ; `verdict-eco-lgc-b1-genou-19-09` (poste2, 1b3c0ac) ; `model.py:1346-1470` (branches du prefill MoE) ; `regime.py:41-104`.

## 0. Erreur de poste7 à retirer partout où elle a été lue (REGLES § 8)

`poste7-nuit-sens2-19-09` § 1 et `REPRISE.md` § 10 (09f16a5) disent « prefill à 95 % du plancher tensor cores bf16 (~105 TFLOPS) ». **Faux d'un facteur 2** : la RTX 5090 rend 209,5 TFLOPS bf16 denses avec accumulation fp32 (419 en accumulation fp16, 838 TOPS int8, 419 TFLOPS FP8 acc. fp32). Corrigé :

```
12,4 TFLOP / pas       plancher bf16 (209,5)   59 ms   →  mesuré 124,6 ms = 47 % du plancher
                       plancher FP8  (419)     30 ms
                       plancher int8 (838)     15 ms
```

Conséquence inverse de ce que j'écrivais : **il reste ~2× à prendre au prefill sans quantifier les activations** — le poste est le noyau Marlin lui-même (déquantification refaite par tuile de M, conçu pour M petit), pas le débit des tensor cores. « Aucune optimisation de lecture des poids > 5 % » est retiré. chef corrige `REPRISE` § 10 et `ETAT` ; P2 (+14 %) et l'ordre des chantiers ci-dessous ne changent pas, la piste C2 s'ajoute en tête.

## 1. E1-bis : un seul réglage tient les deux colonnes

| réglage | b=1 t/s (seuil 340,1) | b=1 J brut | b=12 t/s (seuil 1 066) | b=12 J net (seuil 0,2136) |
|---|---|---|---|---|
| libre | 374,9 | 0,79 | 1 339 | 0,2305 |
| **`-lgc 2700`** | **355,0 tenu** | **0,6555 (−17 %)** | **1 339 tenu** | **0,2071 tenu** |
| `-lgc 2400` | 324,1 **faux** | −24 % | 1 227 | 0,1856 |
| `-lgc 2100` | — (mode b ≥ 8) | — | 1 138 | 0,1739 |
| `-lgc 1950` / `1800` | — | — | 1 060 / 980 **faux** | 0,172-0,175 |

Décision : **cellule publiée « acvram éco -lgc 2700 »** = devant llama.cpp en vitesse ET en énergie à b=1 (+4 % t/s, J brut 0,6555 contre 1,151 : −43 %) et à b=12 (+26 %, −3 % net) ; **« -lgc 2100 » publié comme mode charge b ≥ 8** (−19 % J net contre llama.cpp, +7 % t/s), régime dans le nom. Le genou b=12 est entre 1 950 et 2 100 : clos, on ne le cherche plus. Gouverneur par lot : n'est plus nécessaire à la revendication ; reste un chantier « éco+ » (C8) — −25 % J à b ≥ 8 sans toucher b=1.

## 2. Chantiers démarrés maintenant, à sec, un sous-agent chacun, branche par chantier, prédiction scellée AVANT toute carte

| # | chantier | branche | prédiction scellée | preuve à sec d'abord (pas de carte avant) | 1re fenêtre carte (≤ 30 min, après 22 h 30 ou dans un trou de la file) |
|---|---|---|---|---|---|
| C2 | **prefill : déquant par couche depuis la disposition Marlin vers un tampon bf16 transitoire (1,8 Gio, réutilisé) + `torch._grouped_mm` cuBLAS** — pas de double disposition (le +14,5 Gio du 18/09 venait d'une pile résidente), pas de quantification d'activation | `poste1-c2-prefill-bf16` | 21 000-24 000 j/s (Marlin 47 % → cuBLAS ~70 % du plancher) ; **≥ 19 500 tenu / < faux** ; PPL = défaut à ± 0,0005 (mêmes valeurs déquantifiées, ordre fp différent) | test d'égalité déquant Marlin-layout ↔ pile NVFP4 sur un expert réel (bit à bit ou 1 ulp bf16, à dire) ; `_chemin('c2')` ; `regime.VARIABLES` `PREFILL_GROUPED=c2` | ABAB prefill 2 047, `certifie`, 3 bras |
| C1 | **W4A8 experts sur tensor cores int8** : E2M1 × 2 = entiers exacts ; requantification par ligne en int8 **dans la mémoire partagée** au chargement de la tuile (`w8 = round(2m · s_blk / S_ligne · 2^k)`, S_ligne = max des échelles de la ligne), puis MMA `m16n8k32.s8.s8.s32` et une seule échelle fp32 par ligne × par jeton à l'épilogue ; A8 int8 par jeton (`quantifier_a8`) | `poste1-c1-w4a8` | borne 15 ms ; réaliste 40-55 ms → **≥ 36 000 j/s tenu**, < 30 000 faux ; PPL ≤ 1,020 (portes A8 et W8r ci-dessous) | **porte W8r** (poste2, 15 min, après A8) : PPL fausse-quant des poids experts requantifiés int8 par ligne depuis NVFP4, scellé ratio − 1,0155 ≤ 0,003 ; ptxas : registres/fil, shared, déversement = 0 ; test CPU du noyau contre fp64 | capture godets puis équivalence prefill/prefill (défaut ↔ C1, arbitre = distances ± 5 %) |
| C3 | **MTP GLM-4.7-Flash** (tête MTP du checkpoint, 1 jeton proposé, vérification exacte par le chemin de `GardeSpeculation`) | `poste1-c3-mtp` | acceptation ≥ 60 % ; b=1 t/s **≥ 1,30 ×** tenu ; greedy identique jeton pour jeton (invariant REPRISE § 6) | chargement de la tête MTP à sec, forme et dtype vérifiés contre `config.json` ; test : proposition + vérification sur CPU petit modèle | b=1 ABAB, 20 min |
| C4 | **godets sur `b`** — prérequis `_bind_hybrid` lie `range(godet)` (MECANISMES), puis `bucket_blocks` sur `b` | `poste1-c4-godets-b` | ms/pas b=4 ≤ 0,95 × ; PPL décodage hybride = prefill ± 0,002 (Qwen3-Next ou Nemotron, chemin GDN) | test à sec : créneau de rembourrage lié, état récurrent inchangé après un pas (le test du 10/09 qui manquait) | capture {1, 2, 4, 8, 16} + PPL décodage hybride |
| C5 | **cache KV int8 au décodage** (`kvcache._dequantize` existe : établir le régime, le mesurer, le mettre au défaut si tenu) | `poste1-c5-kv-int8` | PPL décodage ≤ +0,004 ; ms/pas b=12 ctx 4 k ≤ 1,00 × bf16 ; séquences concurrentes × 2 à VRAM égale | régime dans `regime_ligne()`, test d'arrondi CPU | `ppl-decode-kv` + b=12 ctx 4 k |
| C6 | **conversion Coder : compensation GPTQ + Hadamard** (poste2) | `poste2-c6-gptq` | ratio 3 tranches **≤ 1,0110** (contre 1,0155) ; sinon faux | code de compensation à sec, test sur un tenseur synthétique (erreur < sans compensation) | conversion 2 h après 22 h 30, PPL 3 tranches |
| C7 | **GLM MLA en FP8** (`_scaled_mm`, q_b/kv_a/o) — après la porte de ce soir | `poste1-c7-mla-fp8` | prefill GLM ≥ 8 500 j/s (5 502 aujourd'hui) ; ratio ≤ 1,005 | porte FP8-MLA (fenêtre G1) ; dispatcher + test à sec | ABAB prefill GLM |
| C8 | **gouverneur d'horloge par lot** (éco+) | `poste1-c8-gouverneur` | J ≤ 0,80 × libre à b ≥ 8 ET b=1 = libre ± 2 % | simulateur à sec (lot 1 → 12 → 1), `regime_ligne()` porte l'état | lot mixte, 30 min |

Règles inchangées : prédiction et seuil écrits en tête de chaque branche AVANT la première mesure ; `-Xptxas -v` avant la carte pour tout noyau ; capture godets avant tout défaut de décodage ; équivalence dans le même commit ; une fenêtre = un verrou = un verdict six lignes ; aucune fusion pendant une fenêtre ; push automatique après fusion.

## 3. Ce qui change dans la file de ce soir

* Après la porte A8 (19 h 40) : **porte W8r** 15 min (poste2) — même instrument, fausse-quant sur les poids.
* Item 7 (cache d'experts, Devstral/119B) : reste sur le oui de l'utilisateur — une ligne de lui suffit pour l'ouvrir en C9.

## Ordre

* **poste1** — après ses cinq points du soir (tête/PPL par tranches, `acvram eco`, porte FP8-MLA, ncu M1/M2) : ouvrir C2, C1, C3, C4, C5, C7, C8 **maintenant** en sous-agents, une branche et un fichier `revue/chantier-c<N>-19-09.md` chacun (prédiction, seuil, preuve à sec, état) ; C2 et C1 en premier ; aucune carte avant la preuve à sec et hors trou de la file (pointeur à poste2) ; `fausse_quant_w8r` (poids experts NVFP4 → int8 par ligne, mode fausse-quant) pour la porte W8r avant 19 h 30.
* **poste2** — file du soir + porte W8r après A8 ; C6 à sec maintenant, conversion après 22 h 30.
* **chef** — corriger `REPRISE` § 10 et `ETAT` (§ 0 : 47 %, pas 95 %) ; comparatif : ligne « éco -lgc 2700 » complétée b=1 (355,0 · 0,6555), « -lgc 2100 » étiquetée « mode charge b ≥ 8 » ; revendication : « devant llama.cpp en vitesse et en énergie à b=1 et b=12 (éco 2700) » ; `INDEX` : un pointeur par `chantier-c<N>` ; fusions et push après chaque fenêtre.
* **poste7** — relit chaque `chantier-c<N>` dans le quart d'heure ; 22 h 30 bilan.

## Addendum 18 h 15 — porte A8 ouverte (`verdict-porte-a8-19-09`, poste2 0cb1fcb) : C1 passe par C2

* Mesuré, 3 tranches, régime défaut : A8 int8 par jeton `both` géo **1,0161** (+0,0006 sur 1,0155 ; seuil ≤ 0,004), `gateup` 1,0169 (+0,0014), `both` e4m3 1,0153 (−0,0002). Prédiction poste7 (gate/up +0,002-0,004, both +0,003-0,006) : tenue, en mieux sur `both`. L'entrée de `down` ne coûte rien ; e4m3 = bruit → **le format A8 se choisit au coût du noyau, pas à la qualité**. Reste la porte W8r (poids int8 par ligne) en cours.
* Conséquence de conception pour C1 : sous P1 la pile NVFP4 n'existe plus (disposition Marlin seule) ; C1 lit donc la même **déquantification transitoire par couche depuis la disposition Marlin** que C2, avec un commutateur de sortie : `bf16` (C2 → `torch._grouped_mm`) ou `int8 par ligne + échelle fp32 par ligne` (C1). Un noyau, deux consommateurs. Pour la GEMM groupée int8 : **CUTLASS grouped GEMM s8×s8→s32** (tuiles éprouvées, ptxas connu, épilogue échelle ligne × échelle jeton en fp32) plutôt qu'un MMA écrit à la main — `torch._int_mm` par expert = 18 000 lancements par prefill (128 × 3 × 48), exclu. Ordre : C2 d'abord (dequant + bf16 groupé, seuil ≥ 19 500), puis C1 = même dequant en int8 + CUTLASS (seuil ≥ 36 000, floor 15 ms ; prédiction réaliste 55-60 ms ≈ 35 000). Si W8r rend faux (> 0,003) : C1 garde les poids NVFP4 déquantifiés en **e4m3 par bloc de 16 → impossible en un MMA** ; alors C1 = FP8 (A e4m3, W e4m3 par ligne requantifié) avec le même test de porte — à mesurer avant d'écrire.
