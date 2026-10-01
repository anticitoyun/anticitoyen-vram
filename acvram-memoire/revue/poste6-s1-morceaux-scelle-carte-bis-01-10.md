# kv31b levier 2, étape 1 — scellé bis de la preuve carte S1 : morceaux jugés par équivalence sous témoin reprise (REGLES § 4, chef 01/10), contexte court et long, bras INSTA_PAS (poste6, 01/10, écrit AVANT la prise)

Remplace le critère « au bit » du scellé du 30/09 (réfuté : `poste6-lic-relire-kv-01-10`). Scripts inchangés dans leur mécanique :
`prise-s1-morceaux-kv31b.sh` (`REQUETES=2` sur A1 → témoin reprise `completion-2.json` ; `INVITE_FICHIERS` ; `OPTIONS_SERVE`) et
`s1-morceaux-comparer.py` (P3 = Δ(B, A1) ≤ 2 × Δ(témoin reprise), lu à la position 0 si les ids divergent, sinon sur les 32 pas + top-10).

## Bras (un processus par bras, HEAD asserté, carte seule : 4436 sur la 3080 Ti)
| chaîne | modèle | invite | ctx | bras |
|---|---|---|---|---|
| court | gemma-4-31B 4sur6 (int8 KV) | `87d8bfe0a:README.md`, 7 953 jetons, sha 0640377149ab8d46… | 10 240 | A1 (REQUETES=2), A2, B |
| long | gemma-4-31B | README + REPRISE (`INVITE_FICHIERS="README.md REPRISE.md"`), 17 859 jetons, sha 0c7af2ff497a659a… | 20 480 | A1 (REQUETES=2), A2, B |
| insta | gemma-4-31B, court | idem court, `ACVRAM_INSTA_PAS=16384` (frontière au-delà de l'invite : aucune coupe) | 10 240 | A1 (REQUETES=2), B |
| dense | Devstral-24B (0 exilé) | court, 7 865 jetons | 10 240 | A1 (REQUETES=2), A2, B |
Lecture du témoin reprise : sur gemma le cache de préfixe ne sert rien (g9m) → `completion-2.json` = rejeu sans reprise (Δ 0 attendu) : le témoin
gemma est VIDE tant que g9m n'est pas corrigé ; le seuil se lit alors sur Devstral (témoin reprise réel, 0,004 le 01/10), dit dans le verdict.

## Prédictions et seuils
| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| Q1 | prise : `prefill_morceaux` B > 0 (chauffe comprise), A = 0, régime `(morceaux@4096)` | tenu sur les 4 chaînes | B = 0 |
| Q2 | témoin A1/A2 (même processus rejoué) | Δ = 0 | Δ ≠ 0 : carte non reproductible, dit |
| Q3 | témoin reprise Devstral | 0,002-0,01 (0,004 le 01/10) | > 0,02 : la reprise elle-même dérive |
| Q4 | Devstral B/A1 (0,041 le 01/10) | > 2 × témoin (0,008) → **FAUX attendu** : 10 × le témoin ; morceaux hors critère sur un dense | ≤ 0,008 : tenu (surprise, dite) |
| Q5 | gemma court sans coupe (`INSTA_PAS` 16 384) B/A1 | ≈ 0,088 (comme « sans cache » le 01/10) — la coupe à 7 936 explique le 0,60 | ≥ 0,3 : la frontière n'était pas la cause |
| Q6 | gemma long (17 859, 4 morceaux + reste) B/A1 | Δ pos 0 du même ordre que le court (0,05-0,3), ids divergents | Δ > 1 : dérive avec le nombre de morceaux |
| Q7 | exil MLP égal entre bras d'une chaîne ; durée ≤ 2 min par prise (long : ≤ 4) | tenu | sinon rejeu |
Verdict attendu, écrit avant : **le critère n'est PAS tenu** (Q4) — les morceaux relisent 4 096 K/V int8 pour ~3 800 lignes et composent l'erreur
sur 40-60 couches, là où la reprise ne relit que pour les lignes nouvelles ; l'option reste opt-in, et la pièce suivante est « morceaux sous KV
bf16 » (bead lic : budget bf16 à 325 blocs) ou l'acceptation d'un seuil absolu scellé par chef. Issues : (a) Q4 tenu → levier 2 ouvert ;
(b) Q5 faux → la frontière d'instantané n'explique pas 0,60 : chercher dans `_prefill_morceaux` × `coupe` (prefill_len/cached_len) ;
(c) témoin gemma vide (g9m) → dit, pas remplacé par un chiffre.
