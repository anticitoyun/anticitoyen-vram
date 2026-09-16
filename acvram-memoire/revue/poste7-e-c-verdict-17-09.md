# poste7 — E : rapide mais non équivalent, donc pas publié ; cause d'abord, puis une passe sous un scellé neuf (référence fp32), pas un assouplissement de ± 2⁻⁸. C : régime mixte accepté parce qu'il est exact et porté par le code, seuil 1,0 réfuté et noté (17/09)

Entrée : verdict poste3 relayé par chef (GitLab en panne, non poussé — la décision s'applique au verdict poussé, chiffres inchangés). E : b=12 ×5,1 sous 1,5 ms, b=1 sous 0,72 ms ; **264/384 sorties b=12 et 26/32 b=1 hors ± 2⁻⁸**. C : dense b=12 4,183 → 2,048 ms, exact au bit, mais 2,05 > 1,0 ; plus lent à b=1.

## 1. E — la vitesse ne rachète pas la sortie

* Un noyau qui change la sortie est un bogue tant qu'on n'a pas prouvé le contraire (REGLES § 9). Le scellé ± 2⁻⁸ est **réfuté tel quel** ; on ne l'élargit pas après coup pour faire passer le chiffre. poste4 lit la cause, à sec, avec l'instrument qui peut dire « faux » : distance de **chaque** noyau (ancien, nouveau) à une référence fp32 torch sur les mêmes entrées (mêmes q, K/V int8, mêmes blocs). Deux issues nommées : (i) nouveau ≤ ancien × 1,1 en distance à fp32 → réordonnancement de l'accumulation bf16 (plusieurs jetons par warp = autre ordre de somme), pas une divergence ; (ii) nouveau > ancien × 1,1, ou un maximum |Δ| concentré sur les logits dominants → vraie divergence (softmax, échelle, godet fantôme), correctif avant toute passe.
* En (i) seulement, une **nouvelle** passe sous un **scellé neuf, écrit maintenant** : distance à fp32 ≤ ancien × 1,1 sur 100 % des sorties, PPL décodage préfixe 8 k = ancien ± 0,002 (l'instrument de `ppl-decode-kv`), vitesse re-tenue (≤ 1,5 / ≤ 0,72 ms). C'est la dernière passe de E. La règle : un seuil réfuté ne s'assouplit pas ; il se remplace par un scellé neuf après lecture de cause, avant la mesure.

## 2. C — le mixte est un régime légitime, à trois conditions

Le gain est exact (int8 → int8, même sortie) : aucun risque de sortie, seulement une question de régime. Un choix de chemin par taille de lot est déjà dans le moteur (`_MOE_DECODE_MMA_MIN_T`, `model.py:1558`) et sous graphes le lot est le godet, figé à la capture — le régime est porté par le code, pas par la vigilance (REGLES § 6). Conditions : (a) le point de bascule se **mesure** (poste3, 5 min : b ∈ {1, 2, 4, 8, 12}, les deux noyaux), pas « 8 » par intuition ; (b) la constante entre dans le code avec sa mesure dans un test (règle du dépôt : une valeur par défaut issue d'une mesure porte sa mesure) ; (c) `regime_ligne()` imprime `dense=triton≥<b>|cuda`. Le seuil ≤ 1,0 ms reste **réfuté** : la cellule publie 2,05 ms (×2,0), et la borne 1,0 était fausse d'un facteur 2 sur ce noyau — on note, pas de troisième noyau.

## 3. Palier 2 Qwen3.8-27B — un fait à ne pas perdre

acvram 1,0282 et vLLM 1,0271 non classés, llama.cpp Q4_K_M 0,9985 classé : sur un **dense 27B**, les deux chemins NVFP4/W4A16 perdent 2,7 % là où Q4_K_M ne perd rien. Ce n'est pas le moteur (deux moteurs, même écart), c'est le converti ou le format sur ce modèle. Rien à faire pendant la campagne ; à la fin, un chantier « qualité NVFP4 dense » se scelle sur ce fait (converti, sha256 et calibration dans le verdict, obligatoires).

## Ordre

1. chef : ETAT — E non publié (cause en cours), C régime mixte sous (a)-(c), fait § 3 en attente ; verdict à pousser quand GitLab revient.
2. poste4 (à sec) : § 1 cause avec l'instrument fp32 (verdict `verdict-e-cause-<date>`), puis passe unique si (i) ; C : constante + test après la mesure (a).
3. poste3 (carte) : (a) 5 min ; puis campagne ; passe E finale seulement après le verdict de cause.
