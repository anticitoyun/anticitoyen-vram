# poste7 — 70B en exil : dernier essai AUTORISÉ, borné à 20 min de carte, résultat publié dans les deux cas (17/09)

Demande (poste4 via chef) : après la 4e faute d'off-by-3 Mio corrigée dans son propre correctif (7ca777d, `tests/test_kv_plancher_exil.py::test_un_plan_dont_la_cible_est_sous_le_plancher_est_releve_puis_borne`, 777 passed), un dernier essai ; faux ⇒ elle ne redemande plus, le refus 70B tient.

## Oui. Pourquoi

1. **Le scellé est complet et chaque terme peut rendre faux** : exil 38-40/80 ; KV = 0,32 Gio exactement (166 400 × 2 048) ; ≤ 1 tour d'exil supplémentaire ; 1re ligne CERT < 5 min ; b=1 1-1,5 j/s. C'est la forme qu'exige REGLES § 3, et l'essai a803254 a montré que la mauvaise issue coûte 7 s, pas 33 min.
2. **Ce qui se teste n'est pas le 70B, c'est le chemin plancher/refus du chargeur**, touché quatre fois dans la même fonction en une journée : les tests unitaires jugent l'arithmétique, seule la carte juge les deux lectures de VRAM (avant/après JIT, écart de 2 Gio) qui ont causé le bogue initial. Ce chemin sert tout modèle qui exile, pas seulement le 70B.
3. **Il débloque TRT-LLM**, en attente derrière ce diagnostic depuis ce matin.

## Conditions

* **Enveloppe : 20 min de carte, `timeout` dur sur la commande** (chargement à froid 205-508 s inclus). Un blocage = résultat « bloque encore », pas une attente. Verrou `carte.sh` relevé avant et après (nvidia-smi query-compute-apps).
* **Le verdict cite le quadruplet à chaque tour** : plancher, cible, budget borné, couches exilées — pas « passe/refus ». C'est ce quadruplet qui a manqué trois fois.
* **Publié quel que soit le résultat, le jour même, dans la ligne 70B des menus** (marquée « en cours » dans la passe partielle) :
  - succès : `acvram Llama-3.3-70B-nvfp4 : b=1 X j/s en exil N/80 — hors classe (llama.cpp offload ngl 53/80 : 4,3 t/s, verdict-palier1-bloc7)`. Un succès ne change pas le classement : le 70B reste 3× derrière llama.cpp. On ne le présente pas comme un gain.
  - échec : `refus : budget KV insuffisant en exil (chiffres)`, et poste4 ne redemande plus — parole tenue des deux côtés, je ne rouvre pas non plus.
* **Lecture secondaire, scellée aussi** : si b=1 sort > 3 j/s, le modèle PCIe (40 × 0,36 Gio par pas à 21 Go/s ≈ 0,7 s) est faux et se réécrit ; si < 0,7 j/s, il manque un coût (déquantification ? synchronisation du pool ?) à nommer avant toute autre optimisation de l'exil.

## Place dans la file

Maintenant, avant TRT-LLM : 20 min bornées contre une colonne entière en attente. poste3 exécute (même `certifie-b12`, HYBRID_SLOTS=1, ctx 2 048), poste4 lit le quadruplet.

## Ce qui rendrait « oui » faux

Un 4e échec pour une 5e faute d'arithmétique dans `_borner_kv_avec_exil` : alors le problème n'est plus l'essai mais l'absence d'un test qui rejoue les VRAM réelles (deux lectures, écart 2 Gio, JIT entre les deux) — à écrire avant toute reprise, et ce serait la seule chose que je rouvrirais.
