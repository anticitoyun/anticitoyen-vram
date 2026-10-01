# Scellé : (2) gemma court/long sur poste6-g9m avec témoin reprise réel ; (1) Devstral KV bf16 sur main ≥ 37f955f84 (poste6, 01/10, écrit AVANT la prise, ordre chef)

Ordre de carte : poste2 (275) → poste5 (10 min) → moi. Scripts : `scratchpad/poste6-g9m-s1.sh` (worktree travail/poste6-g9m, HEAD aa3dec8f7 =
poste6-g9m + origin/main 37f955f84) et `scratchpad/poste6-bf16.sh` (worktree travail/poste6-menus, HEAD ef7d6ff4d = poste6-reserve-attention +
37f955f84). Même mécanique que S1 bis (REQUETES=2 sur A1, comparateur REGLES § 4).

## (2) gemma-4-31B sur poste6-g9m — chaînes court (7 953, ctx 10 240) et long (17 859, ctx 20 480), bras A1 (×2), A2, B
| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| G1 | `cached_prompt_tokens` de A1 après la requête rejouée | > 7 000 (court), > 17 000 (long) — g9m sert le cache | 0 : le correctif ne suffit pas (autre plafond) |
| G2 | régime : plus de `(coupé@256)` ; `est_hybride` faux | tenu | `(coupé@256)` présent |
| G3 | témoin reprise gemma (A1 rejoué, K/V int8 relus pour la dernière ligne) | 0,003-0,03 à la position 0 (Devstral 0,004 à 40 couches ; 60 ici) | 0 (cache non servi) ou > 0,1 |
| G4 | morceaux B/A1, court | ≈ 0,088 (S1 bis sans coupe) ; > 2 × G3 → **critère NON tenu** attendu | ≤ 2 × G3 : tenu (dit, surprise) |
| G5 | morceaux B/A1, long | ≈ 0,09, ids stables ≥ 20 pas | > 1 |
| G6 | ids du seul tenant A1 court = ceux de « insta » du S1 bis (d3449ae4…) — sans coupe, même chemin | tenu | diffèrent : g9m change autre chose que la coupe |
| G7 | durée ≤ 8 min (6 prises) | tenu | > 10 |
Issues : (a) G1 faux → g9m incomplet, fusion suspendue, à nommer ; (b) G4 tenu → l'étape 1 passe le critère sur gemma seulement — à
confronter à Devstral avant d'ouvrir l'anneau ; (c) G6 faux → bissection du chemin de g9m.

## (1) Devstral-24B KV bf16 (`ACVRAM_KV_FORMAT=bf16`), ctx 10 240, bras A1 (×2), A2, B
| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| D1 | blocs KV (journal) | 640 admis (poste1) ; invite 7 865 acceptée | refus « blocs KV » |
| D2 | témoin A1/A2 | au bit | Δ ≠ 0 |
| D3 | témoin reprise bf16 (A1 rejoué, cache servi) | ≤ 0,002 (plus de quantification ; reste la longueur des appels) | > 0,01 |
| D4 | morceaux B/A1 bf16 | **≤ 0,01** à la position 0 (l'int8 relu pesait l'essentiel des 0,041) | > 0,02 : le noyau GPU (flash/paginé, longueur-dépendant) domine, pas le format |
| D5 | critère REGLES § 4 : D4 ≤ 2 × D3 | non tenu si D3 ≈ 0 (2 × 0,002 = 0,004 < 0,01) — dit | — |
| D6 | durée ≤ 4 min (3 prises) | tenu | > 6 |
Issues : (a) D4 > 0,02 → la relecture n'explique rien, le chemin GPU des morceaux est à ouvrir (paged vs flash) ; (b) D1 faux → lic incomplet.
