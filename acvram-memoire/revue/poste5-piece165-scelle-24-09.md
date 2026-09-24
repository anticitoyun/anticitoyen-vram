# Scellé — pièce 165 : requalifier GDN_PREFILL_LOT=1 contre témoins, et TTFT servi sous charge (poste5, 24/09 23 h 5x, AVANT la prise)

Ordre de chef. Source : étape 0 d'poste6 (164 : LOT 0 → 1 = −18 à −25 % du forward de préfill à 8 séquences, sous
Marlin comme au naturel). La 150 bis avait laissé LOT=1 en opt-in sur un seuil d'argmax (≥ 99 %) écrit sans témoin.

## Qualité — `scratchpad/poste5-p165-24-09/kl-lot165.py` (reprise de `kl-lot.py` de la 150 bis)
Même processus, A (LOT=0, défaut) et B (LOT=1) basculés à chaud (`couches._GDN_PREFILL_LOT`) ; logits fp32 de toutes les
positions du préfill ; compositions C1 8 × 78, C2 mêlées [40 … 200], C3 [300, 17, 500], textes NEUFS (décalage 4 000 au
lieu de 500). Témoins de la même prise : T1 = chaque séquence dans son propre passage, T2 = même lot en ordre inverse.
Modèles : `Qwen3.8-27B-nvfp4` (défaut du soir : F1-F6, Marlin) et `Qwen3.5-35B-A3B-srcQ4_K_M-nvfp4` (MoE hybride).
**Critère (chef), par composition** : rejeux A et B au bit ; KL_max(A‖B) ≤ 2 × max(KL_max T1, KL_max T2) ; accord
d'argmax A/B ≥ accord A/T1 − 0,5 pt ; |ΔPPL| de chaque séquence ≤ 2 × max des |ΔPPL| des témoins (séquence = fenêtre).
**Prédit** : tenu sur les 6 cellules ; Qwen3.8 KL_max(B) ≤ T1 (150 bis : 5 cas sur 6) ; accord A/T1 entre 98 et 99,5 %
(inconnu jusqu'ici) et A/B à ±0,5 pt de lui ; sur le MoE, T2 ≠ 0 (l'ordre change le MoE, addendum 2 de la 156 d).
Issue qui me gênerait : l'accord A/B de C1 sur Qwen3.8 (98,2 % en 150 bis) plus de 0,5 pt sous A/T1.

## TTFT servi sous charge — `ttft-charge.py`, `prise.sh ttft`
Qwen3.8 servi (`--max-batch 8 --no-prefix-cache`, -lgc 2700), 8 requêtes simultanées de 1 jeton, ids jamais revus,
C1 8 × 78 et C2 mêlées ; 2 tours de chauffe puis 12 tours ; A B B A A B B A A B, serveur neuf par passe ; passe NULLE
si repli eager, ou si B n'a pas `GDN_PREFILL_LOT=1` à sa ligne de régime.
**Prédit** : TTFT du tour (le plus lent des 8), médiane : A 380-480 ms en C1 ; B/A −15 à −25 % en C1, −12 à −22 % en C2.
**FAUX** si B/A > −8 % dans une composition. Issue nommée d'avance : les 8 requêtes admises sur deux pas de préfill
(la 150 mesurait 1,7 pas par lot). Le gain serait alors partagé, mesurable par le nombre de pas de préfill.

**Ajout 24/09 23 h 5x (après la KL, avant tout TTFT)** : 1re passe TTFT (23:45) tombée sur « aucun jeton reçu » —
`ttft-service-p145.ttft` exige un fragment de TEXTE non vide ; avec max_tokens=1 sur des ids aléatoires, le seul jeton
peut être vide. Prise arrêtée, aucun TTFT mesuré ; `ttft-charge.py` compte désormais le premier fragment portant
`choices`. Critère et prédiction du TTFT inchangés. (Les résultats KL de la même prise sont dans `prise-kl.txt`.)
