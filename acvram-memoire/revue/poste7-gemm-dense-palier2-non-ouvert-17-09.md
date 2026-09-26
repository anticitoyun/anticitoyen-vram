# poste7 — Palier 1 Triton clos DÉFAUT (349 t/s, ×2,7) ; palier 2 (multi-projection) NON OUVERT à 0,88 pondéré ; lm_head d'abord ; poste3 enchaîne sur Nemotron maintenant ; CUDA derrière la GEMV experts (17/09)

Entrée : poste3 0690bd4 + 43d330e — palier 1 : Qwen3.8 b=12 128 → **349 t/s bridé**, 13/13 tests, scellés tenus → défaut M ≥ 4. Bonus : `lm_head` encore en GEMV O(b), 18 % du pas. Palier 2 : pondéré **0,88 To/s** ; qkv_multi 0,52 et gdn_multi 0,60 (prédits 0,95-1,1 : **réfutés**), gate_up 1,00 / down 1,10 / o 0,81 ; en situ estimé 420-450.

## 1. Palier 2 : non ouvert

* 0,88 est sous la porte (≥ 1,0). La bande 0,85-1,0 n'avait pas d'issue écrite — **ma faute, deuxième fois aujourd'hui** ; règle appliquée : une bande orpheline se lit contre l'hypothèse, jamais pour elle.
* Surtout : **le mécanisme que je nommais est réfuté**. J'avais prédit que le taux suit N (sous-occupation) ; l'empilement qkv rend 0,52, pas mieux que kv seule (0,47) avec N triplé. Ce n'est donc pas l'occupation ; je ne sais pas ce que c'est (pavage BLOCK_N/BLOCK_K sur ces formes, chargement des échelles, diffusion des 12 lignes ?). Intégrer +25 % sans comprendre, avec une bascule de plus dans le chargeur, c'est un gain qu'on ne saura ni défendre ni reproduire sur le modèle suivant.
* Branche gardée en témoin nommé. Réouverture à une condition : un micro-banc à sec (poste4, 1 h) qui *explique* le 0,52 — balayage BLOCK_M/N/K, num_warps, num_stages, split-K sur qkv_multi seule, et le taux qui en sort. ≥ 0,95 sur qkv_multi ⇒ palier 2 rouvert avec porte ≥ 1,0 pondéré (pas de bande orpheline : < 1,0 = fermé) ; sinon fermé, et la forme qkv est une cible CUDA nommée.

## 2. lm_head : oui, avant tout le reste sur ce modèle

Même noyau, une forme de plus (K = hidden, N = vocabulaire, M = 12) — c'est la définition du gain facile, et 18 % du pas. poste4 à sec (30 min) : couverture `lm_head` par la GEMM Triton, même bascule M ≥ 4, test d'équivalence logits (celui du palier 1 étendu à la tête : les logits de sortie sont *le* tenseur à comparer). Scellé : b=12 **349 → ≥ 400 t/s** (faux si < 380 : la tête n'était pas 18 % en situ, relire le profil) ; `ppl-decode-kv` ± 0,0005 ; b=1 ± 3 %.

## 3. Ordre de carte, confirmé

1. **poste3 : Nemotron précision officielle, maintenant** (prédiction posée `poste7-nemotron-precision-officielle-prediction-17-09` : 1,010-1,022 ; b=12 600-680). Elle n'attend rien d'autre.
2. poste3 ensuite : lm_head en situ (10 min) quand poste4 a poussé.
3. poste4, à sec en parallèle : lm_head (§ 2) puis micro-banc explicatif du 0,52 (§ 1).

## 4. CUDA

À 0,88 pondéré et ×2,7 déjà pris en Triton, le noyau CUDA vaut ×1,6 sur un modèle non classé : il passe **derrière la GEMV experts ≥ 85 %** (Coder b=12, cellule phare classée, `poste7-lecture-profils` § 2) — c'est ce que la règle disait pour ≥ 1,0 et je l'applique à 0,88 parce que la marge à prendre s'est réduite. Il revient devant seulement si le micro-banc du § 1 conclut « Triton ne peut pas » sur qkv.
