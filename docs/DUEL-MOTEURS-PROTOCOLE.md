# Comparer acvram à un moteur concurrent

Protocole écrit après le duel du 9 septembre 2026 contre llama.cpp, et surtout
après la **première version de ce duel, qui était fausse en notre faveur de
38 %**. Chaque règle ci-dessous a coûté une mesure.

## Le résultat qui a servi d'étalonnage

    Qwen3-Coder-30B-A3B-Instruct, source Q4_K_M des deux cotes, ABBA

                    acvram    llama.cpp   rapport
    decodage        131,0      256,4       1,96x
    prefill        2363       5458         2,31x
    jetons / kJ      804       1174        1,46x
    puissance        163 W      218 W

Recoupement des deux manches acvram : **0,15 %**. Celles de llama.cpp : 10,25 %.

## Les six conditions, et ce que chacune a coûté d'avoir été oubliée

**1. Même source et même quantification d'origine.** `-Q4_K_M` contre
`-srcQ4_K_M-nvfp4` : on compare deux moteurs sur le même matériau, pas notre
conversion contre une autre source. 30 couples du parc le permettent.

**2. Les deux modèles doivent tenir en VRAM.** Le premier choix
(`Nemotron-Lightning-Q6_K`, 32 Go pour 30,9 libres) aurait opposé un moteur en
VRAM à un moteur qui déborde. Vérifier les tailles **avant** de valider le
modèle.

**3. Le bon binaire.** Les builds de `/mnt/AI_GENERATOR/llamacpp/` sont en
**Vulkan** — aucun `libggml-cuda.so`. Le CUDA est celui de Jan
(`~/.local/share/Jan/.../b9967/linux-cuda-13-.../llama-server.real`) et il
n'existe qu'à travers le `LD_LIBRARY_PATH` que pose `llamacpp-serveur`. Un
duel lancé sur le binaire évident opposerait notre CUDA à leur Vulkan.
**Nommer le binaire dans le résultat.**

**4. Même cache KV.** `CACHE_KV=q8_0` côté llama.cpp contre l'`int8` d'acvram.
Une manche a déjà comparé un cache `q4_0` à un cache 8 bits : 0,11 % d'écart à
1024 de contexte, **~12 % à 32k**. Écrire la valeur retenue.

**5. Même état de spéculation.** **acvram spécule par défaut**
(`--speculative ngram`), llama.cpp seulement sur `--draft`. Le duel opposait un
moteur spéculatif à un moteur qui ne l'est pas.

**6. Invites de texte réel, DIFFÉRENTES à chaque essai, des deux côtés.**
Deux raisons distinctes :

* `llamacpp-serveur` passe `--cache-reuse 256` et acvram a son cache de
  préfixe : sur des invites répétées **les deux moteurs mesureraient leur
  cache**. Un biais symétrique ne se voit pas — c'est le pire des cas ;
* la première version tirait 400 mots dans un vocabulaire de 24. Cette
  répétition **nourrit la spéculation ngram** et donnait acvram à 181 j/s au
  lieu de 131. Avec (5), les deux biais valaient **+38 %**, tous deux en notre
  faveur.

Le corpus wiki (sha vérifié) fournit ce texte : un extrait différent par essai.

## Ce qui rend le chiffre lisible

**ABBA, pas A-puis-B.** La dérive thermique vaut +8,78 W entre carte froide et
palier ; alterner une fois ne corrige rien.

**Attendre la libération RÉELLE entre deux manches.** Un `sleep` fixe ne
suffit pas : une manche a été refusée sur « carte non libre (1176 Mio) » parce
que le moteur précédent n'avait pas fini de rendre sa VRAM. Lire la VRAM en
boucle jusqu'au seuil.

**Écrire les deux tailles à côté du résultat.** acvram servait 16,5 Go contre
17,3 : **5 % d'octets en moins**, soit autant de débit gagné gratuitement sur
un décodage limité par la bande passante. Ce n'est pas disqualifiant — c'est un
avantage réel de notre format —, mais le lecteur doit pouvoir le soustraire.

**Séparer prefill et décodage sans instrumenter les moteurs** : deux requêtes
sur la même invite, à `max_tokens=1` puis `N`. Le débit de décodage vaut
`(N-1) / (tN - t1)`. Symétrique, donc équitable. Prendre **N ≥ 256** : à 128,
`tN - t1` est du même ordre que le bruit et la dispersion monte à 62 %.

**L'énergie fait partie du verdict.** La 5090 est bridée à 500 W. Un relevé
manquant n'est pas un relevé nul : rendre `None` et le dire.

## Le contrôle qu'aucun chiffre ne remplace

**Lire les deux textes avant toute mesure de débit.** Pas « le service
répond », pas « la sortie n'est pas vide » : les lire. Un moteur qui rend du
charabia plus vite n'a rien gagné. C'est aussi ce qui vérifie que le concurrent
sert réellement l'architecture — un modèle hybride peut être accepté puis
déroulé de travers.

## Deux questions, pas une

Ce protocole répond à **« moteur contre moteur, à configuration égale »**. Il
ne répond pas à **« ce que chaque outil donne par défaut »** — cadrage où la
spéculation ngram d'acvram, absente par défaut chez llama.cpp, vaut jusqu'à
+38 % sur du texte répétitif. Les deux sont légitimes ; dire laquelle on mesure.
