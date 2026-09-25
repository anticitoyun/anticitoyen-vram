# Scellé — pièce 172 : B' au défaut, poids déquantifié partagé par la boucle par séquence (poste5, 25/09 01 h, AVANT la prise)

Ordre de chef. Code : `kernels.depaquetage_partage()` et `_w_partage` (quatre sites de déquantification NVFP4 au
préfill : naturel, échelle par ligne, disposition Marlin, vue Marlin), portée ouverte par la boucle par séquence de
`DecoderLayerGDN` (`couches.py`). `ACVRAM_DEPAQ_PARTAGE` vaut 1 par défaut, 0 = témoin. **Au bit par construction**,
donc sans KL.

**Tests** (`tests/test_depaquetage_partage_172.py`, formes réelles 6 144 × 5 120) : partage contre sans partage AU BIT
(naturel et Marlin, C1 et C2), compte des réutilisations (1 fabrication + 7 réutilisations) ; témoin : la GEMM groupée
DIFFÈRE, sinon le test casse ; au niveau d'une `DecoderLayerGDN` : au bit, et GDN_PREFILL_LOT=1 diffère. Bras
cassants : GEMM groupée réintroduite dans la couche → ROUGE ; partage perdu → ROUGE.

**Modèle** (`diag172.py`) : logits fp32 C1 et C2, A (partage coupé) contre B' AU BIT, Qwen3.8 et Qwen3.5-35B-A3B ;
bras cassant B' + LOT=1 ≠ A.

**Vitesse, prédiction** :
* forward de préfill en processus, A B' B' A : Qwen3.8 **−10 à −15 %** (169 : −13 % en C1, −10,5 % en C2) ;
  Qwen3.5-35B, où Marlin est refusé et où le chemin naturel déquantifie des GDN plus petits (hidden 2 048), **−3 à −8 %** ;
* TTFT servi sous 8 requêtes (A B B A A B B A A B par modèle) : Qwen3.8 −9 à −13 % (C1) et −7 à −11 % (C2) ;
  Qwen3.5-35B −2 à −6 %.
* **FAUX** si B' ≠ A au bit sur un modèle, ou si le forward de Qwen3.8 gagne moins de 5 %.
* Coût nommé d'avance : pic mémoire du préfill ≈ +230 Mo sur Qwen3.8, le temps de la boucle d'une couche (relevé).
