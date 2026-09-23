# Sage — P3 (1) tenu, clause 4 tranchée : kv=int8 conservé sous réserve, chiffre à la fiche (20/09, 21 h 14)

Lu : `verdict-p3-1-rejeu-masque-20-09` (Manon 21 h 13, main 645f87df). P3 (1) **TENU** : clause 1 8,21 % ≤ 13,92 % (prédiction « le masque porte 70 % de l'excès » tenue : 22,2 → 8,21), clause 3 19/20 avec marges fp32 vraies, clause 5 au bit. Rien à reprendre.

## Clause 4 — ce que la mesure dit vraiment

kv=int8/kv=bf16 : géo +2,02 %, sd 6,87 %, n = 20 → SE ≈ 1,54 %, intervalle à 2 SE **[−1,1 ; +5,1] %** : il couvre les trois zones du scellé (≤ 1, entre, > 3). La mesure ne tranche pas ; à n = 20 elle ne pouvait pas (résolution 3 % pour un seuil à 1 %). Le sous-critère « greedy8 égal ≥ 18/20 » tombe : la paire de référence elle-même (acvram bf16 / HF fp32) donne 15/20, le témoin ne l'atteint pas — un contrôle que la référence ne passe pas ne sépare rien (REGLES § 5).

## Décision

* **kv=int8 conservé** pour Qwen3-VL-2B dans 0.6.34, ligne de fiche : « kv=int8 vs bf16 (vision, 20 images) : +2,02 % ± 1,5 % (SE), non résolu, sous réserve ». Coût de l'alternative `kv=bf16(vision)` : un régime de plus sur la ligne pour un gain non établi — refusé tant que non mesuré.
* **Réfutation, hors porte 0.6.34** (Manon, fin d'une prise, ≈ 1 min : 3 × (bf16 ×20 + int8 ×20), refs réutilisées) : n = 60, seuil écrit ici : géo − 2 SE > +1 % → `kv=bf16(vision)` pour l'alias en 0.6.35 ; géo + 2 SE ≤ +1 % → réserve levée ; sinon la réserve reste et se publie telle quelle.
* Instrument : `model.generate(output_scores=True)` + assert greedy = référence accepté ; `get_rope_index` 811≠803 = contournement, une ligne dans MECANISMES (forward manuel concaténé interdit sur Qwen3-VL).

## Ordre

1. Jérôme : ETAT — P3 (1) TENU, clause 4 « int8 sous réserve, fiche » ; fiche alias Qwen3-VL-2B : la ligne ci-dessus ; MECANISMES : get_rope_index.
2. Manon : S2 0 × 500 (en cours) → § 1b → feu vert 0.6.34 (rejeu GLM b=1 sur 645f87df) ; la mesure n = 60 après, jamais avant.
3. Femoceane : .deb 0.6.34 candidat déjà construit → installé seulement sur le feu vert de 2.
