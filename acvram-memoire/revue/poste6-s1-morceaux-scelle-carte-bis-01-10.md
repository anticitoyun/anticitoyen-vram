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

## Addendum du 02/10 15 h — rejeu de S1 bis sur fusion-2 (62396fcba), écrit AVANT la prise (ordre chef)

Ce qui a changé depuis le 01/10 et oblige à réécrire les bras et les prédictions (le reste du scellé tient) :
* **d19** : les morceaux ne relisent plus leurs K/V int8 du cache, ils gardent des K/V bf16 transitoires par couche
  (`attention._kv_transitoires`) — la cause du Q4 FAUX d'hier (0,041 sur Devstral, 0,436 sur gemma long) n'est plus là ;
* **g9m** : gemma sert son cache de préfixe → son témoin reprise n'est plus vide (0,144 court, 0,027 long le 01/10) ;
* **anneau** : à 20 480 gemma passe sous l'anneau en `auto`, où le cache de préfixe est COUPÉ — aucune reprise n'existe,
  donc aucun témoin : la chaîne longue se joue sous `ACVRAM_KV_ANNEAU=0` (KV plein, 16/60 MLP exilés à sec, DÉGRADÉ) ;
* **d19 encore** : un bras A laissé au défaut passerait lui aussi par morceaux au-delà du tenu → les bras A portent
  `ACVRAM_PREFILL_MORCEAU_AU_DELA=0` ; la chaîne « insta » (frontière d'instantané d'un faux hybride) n'a plus d'objet.

Script : `scratchpad/poste6-s1-bis/carte-s1-bis.sh [court|long|dense]` (relu : arbre propre exigé, HEAD asserté par la
prise, sha des invites vérifiés contre git, `set -euo pipefail`, comparateur par chaîne à sec).

| chaîne | modèle, régime prévu à sec | invite | ctx | bras |
|---|---|---|---|---|
| court | gemma-4-31B, KV plein, 0 exilé, NOMINAL | `87d8bfe0a:README.md`, 7 953 jetons | 8 192 | A1 (2 requêtes), A2, B |
| long | gemma-4-31B, `ACVRAM_KV_ANNEAU=0`, 16/60 exilés, table hôte | `a77970f45` README + REPRISE, 17 859 jetons | 20 480 | A1 (2 requêtes), A2, B |
| dense | Devstral-24B (19 Go, sur disque dur) | court, 7 865 jetons | 10 240 | A1 (2 requêtes), A2, B |

**Durée chiffrée** (d'après les prises d'aujourd'hui : 37-40 s par bras à 8 192 ; d'hier : 177 s le long A1, 28-60 s
Devstral) : court ≈ 2 min ; long ≈ 7 min 30 ; dense ≈ 3 min si le cache de pages est chaud, +4 min sinon (19 Go à
0,07-0,15 Go/s sur le disque dur : à préchauffer par une lecture hors carte et hors mesure). **Total ≈ 13 min, plafond 20.**
Segmentable : `court long` (≈ 9 min 30) puis `dense` (≈ 3 min).

| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| Q1' | B : `(morceaux@4096)`, `prefill_morceaux` > 0 ; A : 0 | tenu sur les 3 chaînes | A > 0 (le témoin découpe aussi) ou B = 0 |
| Q2' | A1 / A2 | Δ = 0, ids égaux | Δ ≠ 0 |
| Q3' | témoin reprise | gemma court 0,05-0,30 ; gemma long 0,01-0,10 ; Devstral 0,002-0,01 | nul (cache non servi) : témoin vide, dit |
| Q4' | B / A1 avec K/V transitoires | gemma court 0,02-0,15 → **tenu** (≤ 2 × témoin) ; gemma long 0,02-0,15 → incertain, plutôt FAUX au seuil (≈ 0,05) ; Devstral 0,003-0,02 → incertain, plutôt FAUX au seuil (≈ 0,008) | un écart ≥ celui d'hier (0,436 long, 0,041 Devstral) : les transitoires n'ont rien amélioré |
| Q5' | exil égal entre les bras d'une chaîne ; durée par bras ≤ 1 min (court, dense chaud), ≤ 3 min (long) | tenu | sinon dit, rejeu |

Issues nommées : (a) long A1 d'un seul tenant ne tient pas 17 859 jetons (réserve planifiée pour un plafond de 4 096,
≈ 5,7 Gio libres à sec) → 400, chaîne non jouable telle quelle, dite ; (b) Q4' tenu partout → les morceaux passent le
critère § 4 avec d19, à dire à chef pour la 0.7.18 ; (c) le premier bras Devstral lit 19 Go sur le disque dur avec la
prélecture B2 active — jamais mesuré sur disque dur (verdict B2) : son temps de chargement sera relevé, sans conclure.
