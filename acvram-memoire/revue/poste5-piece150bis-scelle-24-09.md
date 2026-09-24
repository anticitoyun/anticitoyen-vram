# Pièce 150 bis — scellé (poste5, 24/09, écrit AVANT toute prise) : préfill groupé des hybrides GDN

Ordre de reprise de chef (carnet, 24/09 12 h 5x, point (2)). Bead `anticitoyen-vram-zjw`.

## À sec : pourquoi le préfill des hybrides passe séquence par séquence

* `runner.py:1561` groupe déjà les admis d'un même pas en UN ForwardBatch (aucune frontière d'instantané à 78 jetons).
  L'attention pleine y fait ses projections sur le lot (seul le SDPA boucle, `attention.py:489`) ; le MLP aussi.
* **Mais `DecoderLayerGDN.forward` (`couches.py:125`) boucle `self.linear_attn(h[start:start + ql], etat)` par séquence**,
  et `GatedDeltaNet.forward` (`gdn.py`) fait les CINQ projections (qkv, gate, α, β, out) dans cette boucle : à b=8 × 78
  jetons, chaque projection tourne 8 fois à M = 78 au lieu d'une fois à M = 624. Sur l'alias mixte, un int8 par canal à
  M = 78 ≤ `ACVRAM_INT8_GEMV_MAX` = 80 prend le GEMV par tranches : c'est le 0,74 s/préfill de la trace 150.
* Changement (opt-in `ACVRAM_GDN_PREFILL_LOT=1`, `gdn.py:forward_lot`) : projections du lot en un appel, convolution et
  règle delta par séquence (état et causalité ne passent pas la frontière). Test d'équivalence et bras cassant (lot
  joué comme une séquence → rouge) : `tests/test_gdn_prefill_lot.py`.
* **Au bit ? Non attendu** : autre M, donc possiblement autre noyau (GEMV → cuBLAS au mixte ; nvfp4 : même famille de
  noyau probable, mais je ne l'affirme pas). S'il sort au bit sur le défaut, je le dirai et la KL devient sans objet.

## Instrument

1. **Qualité** (`scratchpad/poste5-p150bis-24-09/kl-lot.py`, un alias par processus) : même processus, drapeau basculé
   à chaud ; compositions C1 8 × 78 (banc), C2 8 longueurs mêlées 33-200, C3 [300, 17, 500], contenu README distinct par
   séquence. A (boucle) ×2, B (lot) ×2, A-seul (chaque séquence seule). Logits fp32 de toutes les positions.
2. **Vitesse** : banc de la 150 (`banc-service.py`, lots de 8 × 256, invite 78 jetons, 3 lots), serveur b=8, ordre
   A B B A par alias, `/metrics` : `prefill_seconds` par lot. -lgc 2700, cpu-safe. Pas de nsys.
   Alias : défaut `Qwen3.8-27B-nvfp4` ; mixte `Qwen3.8-27B-unsloth-mixte-i8c` sous PROJ_MARLIN (celui de la 150).

## Prédiction

* **Défaut** : préfill 0,41 s/lot (150) → **0,22-0,32 s/lot** (−0,09 à −0,19). Le poids GDN relu 8 fois ne coûte que
  ~15 ms (≈ 2,8 Go nvfp4 × 7 / 1,5 To/s) : le gain attendu vient des lancements (5 projections × 48 couches × 7 de
  moins par pas de préfill) et du rendement de GEMM à M = 624 ; les règles delta fla par séquence restent.
* **Mixte** : 1,37 s/lot → **0,45-0,8 s/lot** (les GEMV int8 par canal des couches GDN passent au GEMM cuBLAS).
* Débit du banc : défaut +1 à +3 % ; mixte +8 à +14 %.
* Qualité : voir « Critère KL » ci-dessous (remplace la formulation moyenne ± marge d'abord écrite ici).

## Critère KL (précision demandée par chef, 24/09 13 h, écrite AVANT la prise)

* **Le test d'équivalence n'est PAS au bit** : `tests/test_gdn_prefill_lot.py` compare à une tolérance relative 1e-4,
  sur processeur en fp32. Il garde la découpe (état, convolution et frontière entre séquences ; le bras cassant rend
  rouge), pas l'arithmétique. Sur la carte, la sortie est donc tenue pour CHANGÉE : autre M, autre noyau possible.
* **Témoins**, mesurés dans la même prise et le même processus, sur les mêmes compositions, KL max par position contre
  A-lot :
  T1 = chaque séquence seule, dans son propre passage (variation que le préfill groupé déjà servi accepte) ;
  T2 = le même lot en ordre inverse (autre ordre d'arrivée) ;
  T3 (mixte seulement) = tout l'int8 par déquant + GEMM (`_INT8_GEMV_MAX` = 0), soit l'arithmétique que B donne aux
  projections GDN.
* **Seuil par composition** : KL_max(A ‖ B) ≤ **2 × max(T1, T2, T3)** ; argmax A/B ≥ 99 % des positions ; rejeu A et B
  à 0 (écart absolu des logits). Si le max des témoins vaut 0, le seuil n'est pas utilisable et je le dirai. Aucune
  conclusion n'est tirée de la moyenne. ΔNLL(B−A) est rapporté sans servir de porte.
* **Si A = B au bit sur le défaut** : la KL est superflue pour cet alias, seul l'ABBA compte ; le mixte garde la KL.
* Prédit : T1 de l'ordre de 1e-3 à 1e-2 (le MLP et les projections d'attention y changent de M) ; KL_max(A ‖ B) du même
  ordre ; tenu sur les 3 compositions et les deux alias.

## Seuils et issues nommées

* **TENU** si : défaut −0,08 s/lot ou mieux ET mixte −0,4 s/lot ou mieux, critère KL tenu sur les 3 compositions, les deux
  alias. Alors proposition au chef : défaut à 1 après la suite CI et la capture des godets.
* (a) **Défaut < −0,05 s/lot** : le préfill du défaut n'est pas dans les projections GDN → le coût est dans les règles
  delta fla par séquence (8 × 48 appels) : levier suivant = fla en varlen (`cu_seqlens`), une règle pour le lot.
* (b) **Mixte < −0,3 s/lot** : les GEMV vus à n = 78 ne sont pas (tous) GDN → je me suis trompée de lieu ; relire la trace
  1213 par nom de couche.
* (c) **KL(A‖B) au-delà du témoin, ou rejeu ≠ 0** : faute de découpe (état, convolution, ordre des séquences) — le chemin
  reste éteint, quoi que dise la vitesse.
* (d) **Au bit sur le défaut** : la KL ne juge rien sur le défaut ; seul le mixte la porte.

## Durée prévue

Tests processeur ≤ 5 min ; KL 2 × ≤ 8 min ; banc 2 alias × 4 serveurs ≤ 25 min. Prises par `carte.sh`, file FIFO.
