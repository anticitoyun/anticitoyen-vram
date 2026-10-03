# evp — scellé vitesse : gpt-oss-20b, acvram contre llama.cpp, b=1 et b=12

poste1, 02/10, écrit avant toute mesure. La mesure est faite par poste2, pas par moi (REGLES § 3).
Branche `poste1-evp-parc`. Justesse tenue : `poste1-evp-justesse-02-10.md`.

## Question

Combien coûte aujourd'hui le chemin de gpt-oss dans acvram, comparé à llama.cpp servant les mêmes poids ? C'est
une ligne de base : elle chiffre ce que rapporterait un chemin MoE groupé qui connaisse les biais et la SwiGLU
bornée. Elle ne départage rien de publiable.

## Les deux bras

| | acvram | llama.cpp |
|---|---|---|
| poids | `gpt-oss-20b-srcmxfp4-nvfp4` : experts NVFP4 obtenus EXACTEMENT depuis le MXFP4 ; attention, routeur, biais et lm_head en bf16 | `ggml-org/gpt-oss-20b-GGUF`, `gpt-oss-20b-MXFP4.gguf` (12 109 566 624 o), dans `models_gguf/gpt-oss-20b-mxfp4/` |
| serveur | `python -m acvram serve <dossier> --port 8095 --max-model-len 4096 --max-batch 12 --speculative none` | `llama-server` officiel (`/mnt/AI_GENERATOR/llamacpp/officiel/bin`, 4c9233c) `-m … -ngl 99 --flash-attn on --jinja -np 12 -c 49152 --port 8096` (pour b=1 : `-np 1 -c 4096`), sans brouillon |
| carte | 0 seule (`CUDA_VISIBLE_DEVICES=0`) | 0 seule (`CUDA_VISIBLE_DEVICES=0`) |

Les deux serveurs sont lancés par `outils/gpu/mesure/serveur-bras.sh` (`bras_servir`, `bras_pret`, `bras_arreter`),
sous `carte.sh`, avec `ACVRAM_POSTE=poste2` et `ACVRAM_TYPE=mesure`.

Client : `outils/gpu/mesure/duel-moteurs.py URL CLE MODELE 9 CONC`, avec CONC=1 puis 12, température 0,
N=256 et des invites de texte réel (wiki). Le décodage vient de `usage.completion_tokens`, raisonnement compris des
deux côtés : il ne dépend pas de la façon dont chaque serveur découpe harmony. Le premier essai, froid, est écarté ;
on publie la médiane des 8 suivants. La puissance est lue par `nvidia-smi` sur la carte 0. Ordre : acvram b=1, b=12,
puis llama.cpp b=1, b=12, puis acvram b=1 rejoué (dérive thermique, < 3 % attendu).

## Ce que le chemin acvram fait aujourd'hui (à sec, lu dans le code)

* MoE : la pile est refusée nommément (`moe.py:202`, `_refus_pile`). Le moteur prend la boucle par expert :
  * b=1 : une synchronisation hôte par couche (`topi.tolist()`, `moe.py:1765`), puis 4 experts × 3 GEMV, avec les
    biais et la SwiGLU bornée en opérations séparées ;
  * b=12 : `unique().tolist()`, puis pour chaque expert présent un masque booléen, une indexation (une
    synchronisation de plus) et un `index_add_` (`moe.py:1799-1805`).
* Graphes CUDA : rien ne peut être capturé à travers un `.tolist()`, donc on attend un décodage sans graphe (eager).
* Plancher des octets à b=1 (identique aux deux bras) : experts actifs 4 × 24 × ≈ 14 Mo, plus attention bf16
  ≈ 1,27 Go, plus lm_head 1,16 Go, soit ≈ 3,8 Go par jeton. À 1,5 To/s, cela fait 2,5 ms, soit un plafond
  d'environ 400 t/s.

## Prédiction (avant la mesure)

| | acvram | llama.cpp | rapport A/L |
|---|---|---|---|
| b=1 décodage | 35-80 t/s (≈ 2 000 lancements par jeton, CPU d'abord) | 230-320 t/s | 0,12-0,30 |
| b=12 décodage agrégé | 80-200 t/s (≈ 25 experts distincts par couche, une synchronisation par expert) | 900-1 600 t/s | 0,06-0,20 |
| J/jeton | ≥ 3 × llama.cpp aux deux lots | — | — |

Le chiffre de llama.cpp vient de mémoire (fils publics sur la 5090), sans source vérifiée : c'est le moins sûr des deux.

## Seuils et décision

* Validité : à b=12, `servies_max` = 12 pour acvram (concurrence servie, pas seulement annoncée, lue sur
  `/metrics`) ; llama.cpp n'expose pas ce compteur, donc on s'appuie sur `-np 12` et sur un TTFT mural qui ne double pas ;
  dispersion du décodage ≤ 5 % pour chaque cellule ; acvram b=1 rejoué à moins de 3 % du premier ; aucune
  requête en erreur. Une condition manquée → la cellule est INDÉCIDABLE et n'est pas publiée.
* Décision : rapport b=1 < 0,5 → le chemin groupé avec biais (pile NVFP4 + biais + SwiGLU bornée dans le
  noyau, capturable) devient la pièce suivante de evp, avant le 120b. Rapport b=1 ≥ 0,8 → pas de pièce vitesse,
  on passe au 120b.

## Issues nommées

1. Prédiction tenue (b=1 dans 0,12-0,30) : le coût est le lancement et la synchronisation, pas les octets. Le
   chemin groupé est justifié.
2. **Celle qui me gênerait** : rapport b=1 ≥ 0,5. Mon modèle « 2 000 lancements » serait faux, peut-être parce que
   le moteur capture quand même un graphe ou fusionne la boucle. Le chemin groupé perdrait sa priorité, et il
   faudrait relire où passe le temps (nsys) avant toute pièce.
3. acvram ne sert pas : refus nommé à la capture ou au chargement, ou plantage sous 12 requêtes. Verdict ÉCHEC,
   avec la cause écrite ; pas de chiffre llama.cpp publié seul comme « comparatif ».
4. llama.cpp hors fourchette (< 200 ou > 400 t/s à b=1) : ma référence de mémoire était fausse. Le rapport reste
   valide ; on corrige la phrase.
5. Les longueurs de sortie diffèrent beaucoup (un bras s'arrête avant 256 jetons) : le débit reste comparable, car
   il est calculé sur `completion_tokens`, mais le nombre de jetons par requête est noté par bras.

## Coût

Environ 20 min de carte : deux chargements par bras et par lot, 9 essais par cellule, la cellule acvram b=12 étant
la plus longue (≈ 25 s par essai). Aucun pytest pendant la mesure.
