# ya1 — MoE 30B « Marlin refusé / 4 096 non tenu » : verdict (poste5, 29/09)

* instrument : `acvram serve` (chauffe du contexte), `scratchpad/poste5-ya1-29-09/prise.sh` ; à sec `recensement.py` (fonction du moteur)
* commit : poste5-ya1 6b391ca66 (B-bis) ; A/C paquet 0.7.13 ; modèle Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c, 4 096, ngram
* régime : 5090 seule, llama-server 5 606 Mio seul hors verrou, avant = après ; A et B sans graphes (comme edz rapide), C avec
* scellé : `scratchpad/poste5-ya1-29-09/scelle.md` (prise 1, puis prise 2 après B invalide, tous deux poussés avant)
* mesuré : A 3 072/4 096 (1 831 Mio libres) ; C 4 096/4 096 (0,7 s, 1 927 Mio) ; B (v1) INVALIDE, identique à A à l'octet ; **B-bis 4 096/4 096 (0,6 s, 1 927 Mio)**, preuve de pré-construction au journal (B.log:6) : « [acvram] piles d experts construites avant la chauffe : 48 couches », avant la ligne de chauffe (B.log:7)
* verdict : VRAI (B-bis tenu, A non tenu, même paquet de sources à la pré-construction près)
* durée : prévue 10 + 5 min / tenue 253 s (prise 1) + 16 s (B-bis) ; drapeau edz 09:38-09:45 puis 09:47-10:02 (sans carte : alias d'poste1 tenu 17 min) puis 10:03-10:08

## Réponses
1. **Arbre et paquet divergent-ils ?** Non. Le 7x8 portait sur un autre modèle (…-srcQ4_K_M, 32 768, avec graphes). Sur
   qkvo-i8c, l'échec vient du régime SANS graphes, en 0.7.12 (sans Marlin) comme en 0.7.13.
2. **Une couche refusée fait-elle tomber toutes les piles ?** Non. La ligne de régime est imprimée avant la construction
   paresseuse (runner.py:803-805) : `piles_ok=None`, `experts_layout=naturel` et « pas de piles Marlin (disposition
   naturelle) » (moe.py:430) sont des valeurs de départ. Recensement : 4/48 refus (L0, L1, L2, L4) ; B-bis : 48 couches
   décidées. Le message dit « même par ligne » alors que par_ligne vaut 0 par défaut (moe.py:494) : texte trompeur, laissé.
3. **Coût du repli ?** ≈ 0 en régime établi (pile naturelle et Marlin : 324 Mio/couche chacune). Le coût était
   TRANSITOIRE : sans graphes, pile + repack se construisaient dans le 1er forward (moe.py:1673-1674), soit la 1re passe
   de chauffe, par-dessus le préfill plein (7 échecs d'allocation de 96-192 Mio) → 3 072.

## Correctif
`ChauffeContexte._construire_piles_avant_chauffe` (contexte.py), en tête de `demarrer_service`, garde par appareil de
la COUCHE (comme graphs.py:474). Mêmes piles, même sortie : seul le moment change. Tests
`test_piles_avant_chauffe_ya1.py` (3, sur un bloc MoE SANS paramètre ; 2 rouges sur la v1 et sans l'appel).

## Leçon
v1 : la garde lisait le 1er paramètre du bloc MoE, mais les poids de QuantLinear sont des attributs ordinaires
(layers.py:438), donc aucun paramètre et une garde fausse partout. Le test passait parce que son bloc factice avait un
nn.Linear. Un bras identique au témoin à l'octet est un bras qui n'a rien changé : déclaré invalide et non lu, puis
une ligne de journal a été ajoutée pour prouver que le correctif tourne.
