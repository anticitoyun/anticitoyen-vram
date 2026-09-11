# Objectif elargi : battre llama.cpp, TabbyAPI, vLLM, YALS, JAN

Pose par l'utilisateur le 9/09. Plus rapide ET plus econome que les cinq.

## Constat qui restructure l'objectif : TROIS des cinq sont le meme moteur

    llama.cpp   le moteur          -> ADVERSAIRE 1
    YALS        libllama.so + libggml-cuda.so sous enveloppe Deno
                (/mnt/AI_GENERATOR/YALS/lib/)
    JAN         backend llama.cpp b9967
                (~/.local/share/Jan/data/llamacpp/backends/b9967/)
    TabbyAPI    ExLlamaV3          -> ADVERSAIRE 2
    vLLM        moteur distinct    -> ADVERSAIRE 3

**YALS et JAN encapsulent llama.cpp.** Leur debit ne peut pas depasser celui du
moteur qu'ils embarquent : une enveloppe ajoute du cout (HTTP, serialisation,
ordonnancement), elle n'en retire pas. **Battre llama.cpp, c'est les battre par
construction.**

**Mais « par construction » n'est pas « mesure ».** Deux choses restent a
verifier, pas a supposer :

1. **Leur version de llama.cpp** peut differer de la notre (`e34f042`). Une
   enveloppe sur un llama.cpp plus recent peut battre notre reference sans que
   l'enveloppe y soit pour quelque chose. **Relever le commit embarque des
   deux.**
2. **Leurs defauts peuvent differer** — spéculation, `-fit`, `n_ctx`, type de
   cache KV. C'est le peche originel applique aux nouveaux venus : un
   concurrent qui specule par defaut contre nous qui ne speculons pas.

## Consequence sur le plan de travail

Trois moteurs a battre, pas cinq. Et deux verifications d'encapsulation, qui
coutent une lecture de source chacune au lieu d'une campagne.

    ADVERSAIRE 1  llama.cpp   -2,3 % debit / -8,5 % energie   MESURE le 9/09
    ADVERSAIRE 2  TabbyAPI    non mesure recemment            A FAIRE
    ADVERSAIRE 3  vLLM        non mesure recemment            A FAIRE
    YALS / JAN    encapsulent 1                               A VERIFIER, pas a mesurer

## Etat de l'outillage

Le banc couvre **4 moteurs** : `PORTS = {acvram: 8090, llamacpp: 8080,
vllm: 8000, tabby: 5000}`. Il manque YALS et JAN — mais s'ils encapsulent
llama.cpp, **les ajouter au banc est peut-etre du travail inutile** : une
mesure de leur surcout d'enveloppe suffirait, et elle ne demande pas de
comparaison de moteurs.

**A trancher avant d'ecrire les lanceurs.**

---

# ARBITRAGE d'poste1 (9/09) — « par construction » est FAUX

Mon raisonnement « une enveloppe ajoute du cout, donc battre llama.cpp les bat
tous les deux » est refuse, et pas pour les deux raisons que j'avais listees.

**Troisieme raison, qui tient a NOTRE protocole** : une enveloppe n'ajoute du
cout que si elle est un pur passe-plat. **Notre banc envoie cinq fois le meme
prompt et jette le premier passage.** Une enveloppe qui fait du **cache de
prefixe** rendrait les passages 2-5 quasi instantanes et paraitrait plus rapide
que le llama.cpp nu qu'elle encapsule — artefact de protocole, pas performance.
Ajouter la mise en lot : une enveloppe serveur peut regrouper des requetes
concurrentes et **augmenter** le debit utile. Invisible a lot 1, mais preuve
qu'une enveloppe n'est pas additive par nature.

## Ni « on saute », ni « deux campagnes » : un DEPISTAGE

Une execution de YALS et de JAN — meme modele, meme prompt, meme structure a
cinq passages — comparee a notre chiffre llama.cpp.

    plus lents   -> question close. « Nous battons llama.cpp, ils sont plus
                    lents que lui » suffit, sans campagne complete.
    plus rapides -> il se passe quelque chose (build recent, cache de prefixe,
                    mise en lot) et ca merite la campagne entiere.

**Le controle qui decide est gratuit** : comparer leur ecart passage 1 ->
passage 5 a celui de llama.cpp nu. **C'est la signature du cache de prefixe.**
Notre TTFT est plat a 75 ms des le second passage ; une enveloppe qui cache
s'effondrerait.

**Deux conditions sans lesquelles le depistage ne vaut rien** : le mode declare
des deux cotes (peche originel), et **le commit du llama.cpp embarque releve
sur le binaire**. JAN porte `b9967`, nous `e34f042`. Si JAN gagne, il faut
pouvoir dire si c'est l'enveloppe ou six mois de llama.cpp.

*Note du 9/09 : le binaire JAN ne demarre pas seul —
`libllama-server-impl.so: cannot open shared object file`. Son enveloppe le
lance autrement, ce qui complique le relevé de version par `--version`.*

# TabbyAPI et vLLM — deux asymetries a ecrire AVANT le protocole

## 1. La neutralite de format se brise, et pas symetriquement

    vLLM      lit les safetensors bf16  -> l'appariement nu de la manche 1
                                           est REPRODUCTIBLE, memes poids
    TabbyAPI  exige de l'exl3           -> AUCUN terrain neutre n'existe

Pour TabbyAPI, la seule comparaison honnete est **appariee en qualite** :
perplexite des deux cotes, **jamais un debit seul**. Sans cette ligne en tete,
on presenterait deux manches de nature differente sous le meme tableau.

## 2. vLLM a lot 1 est hors de son terrain — et c'est le MIROIR du bf16

vLLM est concu pour le debit **sous concurrence** (attention paginee, mise en
lot continue). Notre banc mesure a **lot 1**, ou il est notoirement quelconque.

**Gagner la serait exactement ce que nous avons refuse quand ca nous visait.**
Nous avons ecarte la lecture « acvram est a 2,3 % sur le format ou acvram n'a
aucun noyau » parce qu'elle nous arrangeait ; nous ne pouvons pas prendre la
symetrique contre vLLM.

**Decision : mesurer les DEUX regimes.** Lot 1 (usage d'un poste de travail) et
lot 16 (usage d'un service) — `max_batch_size=16` existe chez nous. L'objectif
de l'utilisateur ne dit pas lequel compte, donc les deux se publient.

---

# Defauts releves en preparant le depistage (chef, 9/09)

## YALS — `/mnt/AI_GENERATOR/YALS/config.yml`

    port            5010
    model_dir       /mnt/4TO_SATACMR_2022/Modeles/models_gguf
    max_seq_len     32768
    num_gpu_layers  999
    cache_mode_k    q8_0        <- CACHE KV QUANTIFIE
    cache_mode_v    q8_0        <- CACHE KV QUANTIFIE

**Le cache KV quantifie en q8_0 est un ecart d'appariement majeur.** Notre
cache est en f16 ; le leur occupe **moitie moins de memoire** et se lit deux
fois moins. Ce n'est ni la meme memoire ni le meme debit — et a contexte long
l'ecart grandit.

**Consequence pour le depistage** : un YALS plus rapide que llama.cpp nu
pourrait ne rien devoir a son enveloppe, seulement a son cache quantifie. **Le
depistage doit forcer le meme type de cache des deux cotes**, ou publier
l'ecart en toutes lettres.

C'est le troisieme mecanisme par lequel une enveloppe peut paraitre plus rapide
que le moteur qu'elle encapsule, apres le cache de prefixe et la mise en lot
(poste1) : **un reglage different, ni enveloppe ni moteur.**

## Note logistique

YALS lit ses GGUF dans `/mnt/4TO_SATACMR_2022/Modeles/models_gguf`, pas dans
`/mnt/2TO_SSD_2025_IA/gguf/` ou vit notre `Qwen2.5-Coder-14B-bf16-pur`. Un lien
suffira, mais **verifier que YALS suit les liens symboliques** avant de compter
dessus.

## Rappel : le meme controle vaut pour NOUS

`llamacpp-serveur` porte deja `--cache-type-k q4_0 --cache-type-v q4_0` sur le
service permanent 8081 (constate le 9/09 dans la ligne de commande du PID
5484). **Notre banc utilise-t-il ces options ?** Si oui, la manche 1 comparait
peut-etre deja deux caches differents — a verifier avant de publier quoi que ce
soit d'autre.
