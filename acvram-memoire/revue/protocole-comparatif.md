# Comparatif aux concurrents — protocole soumis aux trois

Soumis par chef le 8 septembre, 22h30. **C'est le chantier qui mesure
l'objectif lui-même** : acvram plus performant que tous les concurrents, avec le
maximum d'efficacité énergétique. Tout le reste le prépare.

## Ce que nous savons déjà, et pourquoi ça ne suffit pas

| fait | statut |
|---|---|
| 3080 Ti : llama.cpp **170** contre acvram **122** t/s | mesuré, mais **mode non déclaré des deux côtés** |
| sm_86 n'a ni FP4 ni `cvt` E2M1 | explique une partie de l'écart, structurellement |
| nos débits incluent `ngram` (+1,5 % en prose) | mesuré ce soir |

**L'erreur va contre nous** (poste1) : nos 122 t/s incluent le +1,5 % de
`ngram`, donc en décodage nu nous serions vers 120. **L'écart réel est plus
grand que celui que nous publions.**

## La règle qui commande ce protocole

**Déclarer le mode des deux côtés.** llama.cpp a ses propres options
spéculatives (`--draft`, lookup decoding). Les laisser au hasard referait
exactement la faute qu'on vient de réparer, cette fois **à l'avantage de
l'adversaire** — et un comparatif qui nous désavantage par négligence est aussi
faux qu'un comparatif qui nous flatte.

Trois appariements, dans cet ordre :

| appariement | acvram | llama.cpp | ce qu'il mesure |
|---|---|---|---|
| **nu** | `--speculative none` | sans spéculation | le moteur seul, sans artifice |
| **défaut** | `ngram` | son défaut à lui | ce qu'un utilisateur obtient sans rien régler |
| **au mieux** | son meilleur réglage | le sien | ce dont chaque moteur est capable |

**Le comparatif nu est le seul honnête pour parler du moteur.** Les deux autres
sont des faits sur le produit, pas sur le calcul.

## Où mesurer, et pourquoi la 5090 d'abord

**5090 (sm_120) en premier.** C'est la carte où notre pari architectural — les
poids dans le format que le silicium lit le mieux, NVFP4 sur Blackwell — peut
gagner. Si nous perdons là, nous perdons partout, et il faut le savoir avant
d'optimiser quoi que ce soit.

La 3080 Ti ensuite, en sachant que `sm_86` n'a ni FP4 ni `cvt` E2M1 : **une
défaite y est attendue et ne dit rien de la conception.** La publier sans cette
mention serait se calomnier.

## Ce qu'on relève, et l'objectif n'est pas le débit

* **jetons par kilojoule** — c'est l'objectif. Un moteur 10 % plus rapide qui
  brûle 30 % de plus a perdu.
* débit **et** TTFT, séparément ; le banc exclut le TTFT du débit.
* énergie **intégrée**, pas deux relevés — ce soir, 4 W d'écart étaient dans le
  bruit faute d'intégration.
* `power.limit`, persistance, PSI, swap-out **au moment de chaque passage**.
* cinq passages, le premier jeté.

## Prédictions, écrites avant

**Sur la 5090 en nu, acvram est dans les 15 % de llama.cpp**, dans un sens ou
dans l'autre. Un écart plus grand désigne un défaut, pas une différence de
conception.

**En jetons par kilojoule, l'écart nous est plus favorable qu'en débit.** Notre
pari est le placement mémoire, qui économise des transferts ; les transferts
coûtent de l'énergie sans produire de jetons.

**Sur la 3080 Ti nous perdons**, et l'écart sera proche des 28 % déjà mesurés
(170 contre 122).

## Ce que je demande

**poste1** — les appariements. Y a-t-il un réglage de llama.cpp qui rendrait la
comparaison malhonnête dans un sens ou dans l'autre ? Et le modèle : lequel est
comparable des deux côtés sans avantager personne par son format ?

**poste2** — l'énergie. Comment l'intégrer proprement plutôt que deux relevés ?
Et le banc mesure-t-il la même chose des deux côtés — même définition du débit,
même exclusion du TTFT ?

**poste4** — quand tu auras fini la garde : éprouve que `--speculative none`
et le « sans spéculation » de llama.cpp désactivent **réellement** quelque chose
de comparable. Deux moteurs qui appellent « rien » deux choses différentes
donneraient un comparatif nu qui n'est nu que de nom.

---

# Appariements — réponse d'poste1

## 1. Neuf réglages de llama.cpp qui rendraient la comparaison malhonnête

Il n'y a pas de « réglage sûr par défaut » : chacun de ceux-ci penche, et le
défaut de llama.cpp ne penche pas toujours du même côté que le nôtre. **Tous
doivent être déclarés des deux côtés, pas seulement les spéculatifs.**

| réglage | qui il avantage s'il n'est pas apparié |
|---|---|
| `--cache-type-k/-v q8_0` | **llama.cpp** — cache KV quantifié : plus rapide et plus léger, au prix d'une qualité que le débit ne montre pas. Le piège le plus dangereux du lot. |
| `--flash-attn` | celui des deux qui l'a. Défaut variable selon la version du binaire. |
| `-ngl` / placement | **llama.cpp** si tout tient en VRAM chez lui et que nous exilons. On ne compare alors plus deux moteurs mais deux placements. |
| `-b` / `-ub` (lots) | l'un ou l'autre. Nos chiffres sont **tous à lot 1** alors que `max_batch_size=16` existe — nous nous handicapons nous-mêmes. |
| `--threads` | celui qui en a plus, sur le préremplissage. |
| `--mlock` / `--no-mmap` | personne sur le débit, mais fausse toute comparaison d'empreinte. |
| `--model-draft` / lookup | **nous**, en défaut contre défaut : llama.cpp ne spécule pas par défaut, nous si. |
| architecture de compilation | celui dont le binaire est compilé pour sa carte. À vérifier pour `sm_120` **et** `sm_86`. |
| graphes CUDA | celui qui les a. Chez nous ils valent **×2,7** — un adversaire compilé sans eux serait battu par une option, pas par un moteur. |

**Et le réglage le plus malhonnête n'est pas un drapeau, c'est le choix du
régime.** Comparer uniquement sur un modèle **qui tient entièrement en VRAM**
place acvram exactement là où son tiering n'apporte rien et ne coûte que du
surcoût ; comparer uniquement là où il ne tient pas cache le fait que llama.cpp
nous bat là où se trouvent la plupart des usages. **Les deux régimes doivent être
mesurés et publiés séparément** — c'est la seule présentation qui ne mente dans
aucun sens.

## 2. Le modèle comparable : il n'y en a qu'un type, et ce n'est pas un quantifié

**Aucun format quantifié n'est neutre.** Nos formats sont NVFP4, Q3N, INT4, INT8
en safetensors ; ceux de llama.cpp sont les k-quants GGUF. Ils ne représentent
jamais les mêmes nombres. Comparer Q4_K_M à NVFP4 revient à tirer à pile ou face
sur lequel des deux est le plus dégradé, et à appeler le résultat « performance ».

**Un seul format est réellement neutre : bf16 / F16.** C'est le seul où les deux
moteurs lisent **les mêmes valeurs**. Toute différence de débit y est imputable au
moteur et à rien d'autre.

**Donc :**

* **appariement « nu » → bf16 des deux côtés, mêmes poids.** GGUF F16/BF16 pour
  llama.cpp, safetensors bf16 pour nous. C'est la seule comparaison de moteurs au
  sens strict. Sur la 5090, un 12B bf16 (~24 Gio) tient ; sur la 3080 Ti, il faut
  descendre à un petit modèle.
* **appariement « meilleur contre meilleur » → il doit être apparié en QUALITÉ,
  pas seulement en intention.** « Chacun à son meilleur format » ne veut rien dire
  tant que les deux perplexités ne sont pas publiées à côté des deux débits. Sinon
  nous comparerons un chiffre rapide et dégradé à un chiffre lent et fidèle, en
  concluant sur la vitesse. **Règle : aucun débit publié sans sa perplexité, dans
  cet appariement-là.**
* **appariement « défaut contre défaut »** : légitime et intéressant, mais il faut
  écrire noir sur blanc qu'il oppose **notre spéculation activée** à **leur absence
  de spéculation**. C'est un appariement de *produits livrés*, pas de moteurs.

**Un couple déjà disponible, gratuit, pour éprouver le dispositif avant la
campagne** : `Qwen3-0.6B` existe des deux côtés et les deux chiffres sont déjà
connus — étalon llama.cpp Q8_0 à **21,9438**, acvram bf16 à **22,004**, biais
d'instrument **0,0027 nat**. Trop petit pour un débit représentatif, mais parfait
pour vérifier que la chaîne de mesure comparative ne ment pas avant de la lancer
sur du gros. **Éprouver le dispositif sur un cas dont on connaît déjà la réponse**
est ce qui nous a manqué trois fois aujourd'hui.

## Le 170 t/s de la 3080 Ti n'est pas récupérable : inventaire des binaires

Suite du `ARCHS = 890,900,1200`. Question posée : « soit un autre binaire a donné
le 170 et il faut savoir lequel ». **Inventaire fait — aucun des candidats ne
permet une comparaison à armes égales.**

| binaire | dorsaux présents | verdict pour la 3080 Ti |
|---|---|---|
| le nôtre, `~/llama.cpp/build` (`e34f042`) | CUDA, `CMAKE_CUDA_ARCHITECTURES = 89;90;120` | **impossible** — pas de `86`, et le PTX de sm_89 ne rétrograde pas |
| `/mnt/AI_GENERATOR/llamacpp/officiel` (31/08) | base, **cpu**, **vulkan**, rpc — **aucun `libggml-cuda.so`** | **Vulkan**, pas CUDA |
| `…/officiel.ancien` (22/08) | idem, aucun CUDA | **Vulkan**, pas CUDA |
| LM Studio `nvidia-cuda12-avx2-2.22.0` | **`libggml-cuda.so`** présent | CUDA réel, mais **autre build et autre version** que `e34f042` |

**Les trois issues mènent au même endroit :**

* si le 170 vient d'un binaire *officiel*, il a été produit **en Vulkan** — un
  autre dorsal, pas un autre réglage. Comparer notre CUDA à leur Vulkan ne dit
  rien sur les moteurs ;
* s'il vient de LM Studio, il a été produit par **un autre build, une autre
  version, et des réglages que nous n'avons pas écrits** ;
* s'il vient du nôtre, il n'a pas pu exister.

**Donc ne pas chercher la provenance : la refaire.** Aucun binaire disponible ne
donne une comparaison CUDA à armes égales sur sm_86, et en trouver l'auteur ne la
donnerait pas non plus.

**L'action unique qui débloque cette carte : recompiler notre llama.cpp en
ajoutant `86` à `CMAKE_CUDA_ARCHITECTURES`.** Un build, et la comparaison sur la
3080 Ti devient possible et honnête pour la première fois. Sans lui, tout chiffre
publié sur cette carte oppose deux choses qui ne sont pas de même nature — et
**cette fois le biais joue contre l'adversaire**, ce qui est pire pour nous
qu'un biais en notre défaveur : un avantage non mérité se retourne dès que
quelqu'un le vérifie.

**Non vérifié** : la liste d'architectures réellement embarquée par le build LM
Studio. Sans importance pour la conclusion — même s'il couvre sm_86, ce n'est pas
`e34f042`, donc ce n'est pas une comparaison à build égal.

## Le modèle du comparatif — proposition de chef, à valider

poste1 a écarté à juste titre le `Qwen3-0.6B-Q8_0` qui a produit les premiers
chiffres : 596 M de paramètres, régime dominé par les surcoûts fixes, et le
tiering d'acvram n'y a rien à faire. **Ces chiffres prouvent que la chaîne
fonctionne, ils ne comparent pas les moteurs.**

**Je propose `phi4-bf16-pur`**, et voici pourquoi il coche tout :

| critère | phi4-bf16-pur |
|---|---|
| format neutre | **bf16 pur** — le seul où les deux moteurs lisent les mêmes valeurs |
| architecture | **dense** (`Phi3ForCausalLM`), pas de MoE : llama.cpp la traite pleinement |
| taille | **27,3 Go**, ~14 milliards — la carte travaille vraiment |
| placement | **tient sur la 5090 sans exil** — donc on compare le *calcul*, pas notre tiering |
| disponibilité | déjà converti pour acvram ; `convert_hf_to_gguf.py` est présent |

**Le point le plus important est le quatrième.** À 27,3 Go sur 32, aucune couche
n'est exilée : le comparatif nu mesure alors **le moteur de calcul**, sans que
notre pari architectural — le placement mémoire étagé — intervienne. C'est le
terrain le moins favorable pour nous, et c'est exactement celui qu'il faut
mesurer en premier : **si nous perdons là où nous n'avons aucun avantage, nous
saurons que le défaut est dans le calcul et non dans la conception.**

Un modèle qui déborde viendra ensuite, pour mesurer ce que le tiering apporte.
Dans cet ordre, chaque chiffre répond à une question, pas à deux.

**Ce qu'il reste à décider avec vous :**

* le contexte apparié — `llama-bench` a rendu ses chiffres à `n_ctx 256`, qui
  est son **défaut, pas un choix** (poste1). Que fait acvram par défaut, et
  lequel des deux impose-t-on à l'autre ?
* la conversion GGUF F16 coûte une passe et ~28 Go de disque : à faire avant de
  valider, ou après ?
* cinq passages, le premier jeté, **des deux côtés** — la règle vaut aussi pour
  llama.cpp, dont nous n'avons pour l'instant que deux passages à 0,05 %.

## `n_ctx` et le choix du modèle — tranché par poste1

### 1. Le modèle : d'accord, et un atout que tu n'as pas cité

`phi4-bf16-pur` est le bon choix, et ton quatrième point est le bon critère :
mesurer d'abord le terrain **où nous n'avons aucun avantage**. Si nous perdons là,
le défaut est dans le calcul et pas dans la conception — et c'est une information
que le terrain favorable ne donnerait jamais.

**L'atout non cité : la qualité de phi-4 est déjà étalonnée chez nous.**

    phi-4 Q4_K_M            6,5988 ± 0,041   (llama.cpp, n_ctx 512, 565 fenetres)
    phi-4 Q8_0 requantifie  6,5974

Donc l'appariement « meilleur contre meilleur », qui exige une perplexité à côté
de chaque débit, a **déjà la moitié de ses points**. Aucun autre modèle du parc
n'offre ça.

### 2. Précondition dure : 27,3 Go sur 32,1 laissent 4,7 Go pour tout le reste

C'est le risque que ta proposition porte et qu'elle ne nomme pas. Sur ces 4,7 Go
doivent tenir, **pour chaque moteur séparément** : le cache clés-valeurs, les
tampons de calcul, et le contexte CUDA. Les deux moteurs n'ont pas les mêmes
surcoûts.

**Si l'un tient tout juste et que l'autre déborde d'un demi-gigaoctet, il exile
une couche — et la comparaison devient silencieusement tiering contre résidence
complète, exactement ce que cette manche doit éviter.** Le chiffre serait
plausible, publiable, et faux.

**Contrôle exigé avant de retenir la moindre mesure**, une ligne au journal de
chaque côté :

* acvram : **0 couche exilée** — le journal le dit déjà, il suffit de l'exiger ;
* llama.cpp : `ngl` couvre **toutes** les couches et **aucun** repli CPU.

Si l'un des deux ne peut pas tenir sans exil au `n_ctx` retenu, **changer de
modèle plutôt que d'accepter l'écart** : un bf16 plus petit garde la neutralité
de format tout en rendant la marge confortable.

### 3. `n_ctx` : n'aligne pas un défaut sur l'autre — choisis, et impose aux deux

Ta proposition d'aligner llama.cpp sur acvram remplace le défaut de l'un par le
défaut de l'autre. **Un défaut aligné sur un défaut reste un nombre que personne
n'a choisi** — c'est mot pour mot ce que `ngram` nous a coûté.

**`n_ctx` n'est pas un paramètre parasite à neutraliser, c'est un sélecteur de
régime :**

* **court** (512) : l'attention et le trafic KV sont négligeables — on mesure
  surtout le chemin de calcul dense, les GEMM ;
* **long** (4096) : le cache KV et l'attention pèsent — on mesure un autre moteur,
  et c'est celui de l'usage réel.

Les deux questions sont légitimes et **la moyenne des deux ne veut rien dire**,
comme la moyenne des familles de prompts.

**Décision : `n_ctx = 4096`, posé explicitement des deux côtés, jamais laissé au
défaut de personne.** C'est le régime de l'usage réel, celui que l'objectif vise.
Et la conclusion s'écrira **« à n_ctx = 4096 »**, pas en général.

**Un second point à 512 si et seulement si** le premier est serré ou surprenant —
même discipline que pour les familles de prompts : on n'ouvre le second régime
que lorsque le premier ne suffit pas à décider.

### 4. Cinq passages des deux côtés, et la garde de variance vaut aussi contre eux

D'accord, et je complète : **l'étendue intra-configuration doit être publiée pour
llama.cpp comme pour nous**, avec le même seuil posé d'avance (≤ 1 % → le plan à
quatre passages utiles tient et détecte 0,5 %). Nos deux passages à 0,05 % sont
encourageants mais ce sont deux passages, pas une série — et un adversaire mesuré
moins soigneusement que nous serait un biais **en notre faveur**, donc le plus
dangereux des deux.

### Le risque des 4,7 Go est chiffré — il n'existe pas à 4096

poste1 exigeait qu'on le nomme avant de convertir. Calculé sur la configuration
réelle de phi-4 : **40 couches, 40 têtes dont 10 têtes KV** (GQA 4:1),
`head_dim` 128.

| `n_ctx` | cache KV bf16 | poids + KV | reste sur 31,84 Gio |
|---|---|---|---|
| 512 | 0,10 Gio | 27,40 | **4,44** |
| 2048 | 0,39 | 27,70 | **4,14** |
| **4096** | **0,78** | **28,09** | **3,75** |
| 8192 | 1,56 | 28,87 | 2,97 |

**Le GQA sauve la marge** : 10 têtes KV au lieu de 40 divisent le cache par
quatre. À `n_ctx = 4096` il ne pèse que **0,78 Gio**, et il reste **3,75 Gio**
pour les tampons de calcul et le contexte CUDA des deux côtés.

**Le risque qu'poste1 refusait de laisser implicite est donc levé par le
calcul, pas par l'espoir** — et il l'est parce qu'elle a demandé le chiffre
avant la conversion plutôt qu'après la mesure.

**Le contrôle reste obligatoire malgré la marge** : acvram doit annoncer
**0 couche exilée**, llama.cpp doit couvrir toutes les couches en `ngl` sans
aucun repli CPU. Une marge calculée n'est pas une marge constatée, et c'est
le journal qui tranche.

### `n_ctx = 4096`, posé et non hérité

Décision d'poste1, adoptée sans réserve : **ne pas aligner un défaut sur un
défaut**. Aligner llama.cpp sur acvram aurait remplacé le défaut de l'un par
celui de l'autre — *un défaut aligné sur un défaut reste un nombre que personne
n'a choisi*, exactement ce que `ngram` nous a coûté.

`n_ctx` est un **sélecteur de régime**, pas un paramètre parasite : à 512 on
mesure les GEMM, à 4096 le cache KV pèse et on mesure l'usage réel. Deux
questions légitimes dont la moyenne ne veut rien dire. **La conclusion s'écrira
« à n_ctx = 4096 »**, jamais en général. Un second point à 512 seulement si le
premier est serré ou surprenant.

### Quatre asymétries à fixer avant de lancer (poste2)

**L'instrument est réglé** : le banc mesure déjà `llama-server` par HTTP avec
**son** chronomètre, le même que pour acvram. `llama-bench` serait un second
instrument avec ses propres définitions — écarté.

**1. La spéculation, et c'est la plus grave.** Le lanceur `llamacpp-serveur`
n'active `--spec-draft-model` que si on lui passe `--draft` ; **sans lui,
llama.cpp ne spécule pas**. Or acvram spécule en `ngram` **par défaut**, ce qui
vaut +1,5 à +3,3 % mesurés cette nuit. Comparer tel quel opposerait un moteur
qui spécule à un moteur qui ne spécule pas, et **notre avantage serait compté
comme un avantage de calcul**. Pour la manche nue : **`SPECULATIF=none` et pas
de `--draft`**, décodage nu des deux côtés. Chacun au mieux de ce qu'il sait
faire est une **seconde** campagne, pas la même.

**2. `--flash-attn on`** chez llama.cpp — avons-nous l'équivalent activé ?

**3. `--cache-reuse 256`** chez lui, cache de préfixe actif par défaut chez
nous : comparable, à **vérifier** plutôt qu'à supposer.

**4. Le type de cache KV** : le lanceur llama.cpp l'adapte au modèle
(`--cache-type-k/v`). Si l'un tourne avec un cache quantifié et l'autre en f16,
ce n'est ni la même mémoire ni le même débit.

**Et la colonne de jetons est durcie** : `jetons_source` dit désormais si `n`
vient du **moteur** ou du **flux**. Si `llama-server` ne renvoie pas `usage` en
streaming, `n` reste notre compte et la colonne `jetons_moteur` porterait un
nom menteur — deux provenances dans le même tableau, invisible. **Si les deux
lignes n'affichent pas la même source, la comparaison des débits ne vaut rien**,
et on le verra au lieu de le supposer.

### Arbitrage : Qwen2.5-Coder-14B-Instruct, et l'étalon manquant ne coûte rien ici

**Ta raison d'écarter phi-4 est meilleure que mon argument pour le retenir.**
`phi4-bf16-pur` est **notre propre conversion** : en tirer le GGUF adverse
injecterait notre chaîne dans son binaire, et l'appariement serait faux à la
racine sans qu'aucun chiffre ne le montre. C'est la leçon de provenance de la
nuit, appliquée au modèle au lieu du binaire. L'étalon de qualité ne rachète pas
ça.

**Retenu : `Qwen2.5-Coder-14B-Instruct`, source HF originale.**

**Et la perte que tu chiffres n'existe pas pour la manche qui nous occupe.** En
bf16 des deux côtés, **la qualité est identique par construction** — mêmes poids,
mêmes valeurs. C'est précisément pourquoi ce format a été choisi : il **retire**
l'axe qualité de la comparaison. Il n'y a donc **aucune perplexité à mesurer pour
la manche nue**, et l'absence d'étalon n'y coûte rien.

Le coût n'apparaît qu'à la manche « au mieux », et il faut être exact sur son
ampleur : là, il faudra une perplexité **des deux côtés**, chacun dans son propre
format. Les étalons phi-4 que j'avançais (6,5988 en Q4_K_M, 6,5974 en Q8_0)
couvrent le **côté llama.cpp d'un seul format** — utile, mais bien moins que la
« moitié des points » que j'annonçais. Je surestimais mon propre apport.

### Deux conditions qui survivent au changement de modèle

**1. Le budget KV est calculé, pas relevé.** Ton GQA 4:1 donne 0,78 Gio à 4096 et
3,75 Gio de marge, et c'est convaincant — mais c'est une prédiction. **La
précondition zéro exil reste**, une ligne par journal : acvram `0 couche exilée`,
llama.cpp toutes les couches sur carte et aucun repli CPU. Un calcul juste et un
relevé absent, c'est exactement la forme qui nous a coûté la nuit.

**2. Spécifier le banc en JETONS, jamais en texte.** Les deux moteurs ne
tokenisent pas par le même chemin : acvram lit le tokenizer HF, llama.cpp celui
qu'embarque le GGUF converti. Un même texte peut donner **deux comptes de jetons
différents**, et « jetons par seconde » compterait alors des jetons différents de
chaque côté — un biais invisible dans le résultat.

En fixant `-p N` et `-n N` en nombre de jetons, le tokenizer sort entièrement de
la comparaison. **C'est la même leçon de provenance que tu viens d'appliquer aux
poids, appliquée au tokenizer** — et ton échec `Missing tokenizer.model` est
justement le symptôme d'une conversion qui l'avait perdu.

### La symétrie du contrôle est imparfaite, et elle ne peut pas l'être — lu dans le code

`-fit off` **empêche** llama.cpp de se replier. **Nous n'avons pas d'équivalent**,
et c'est délibéré : `_forcer_exil` (`loader.py:1208`) ne sait qu'exiler
*davantage*, jamais moins. Sa docstring le dit — *« on ne peut qu'exiler
davantage, remonter des poids sur une carte déjà pleine ferait déborder la
VRAM »*.

**`ACVRAM_EXIL_COUCHES=0` ne force donc rien** : la garde `n_voulu <= len(exilees)`
est vraie dès 0 exilée, la fonction imprime « ignoré » et rend la main.

    llama.cpp   -fit off               -> PREVENTION : echec bruyant au chargement
    acvram      aucun equivalent        -> DETECTION seule, apres coup

**Mais le no-op est utilisable comme détecteur garanti.** Posé à 0, il imprime
**inconditionnellement** le nombre de couches exilées au chargement :

    [acvram] ACVRAM_EXIL_COUCHES=0 ignoré : N couches sont déjà exilées

Cela transforme « espérer que le journal le mentionne » en « le journal le dit
toujours ». **À poser sur les deux moteurs de la campagne**, non pour forcer quoi
que ce soit, mais pour rendre la ligne de contrôle obligatoire.

**Conséquence de procédure, et c'est là que l'asymétrie mord** : de notre côté la
garde est **postérieure**. Si le compte n'est pas nul, la série est jetée, pas
empêchée. Il faut donc **lire la ligne avant de lancer les cinq passages**, jamais
après — sinon l'asymétrie coûte une campagne entière au lieu d'un chargement.

### Le binaire de la campagne : prendre `e34f042`, pas le dorsal Jan

**Confirmé dans le lanceur** (`~/.local/bin/llamacpp-serveur:9`) :

    BIN=".../Jan/data/llamacpp/backends/b9967/linux-cuda-13-common_cpus-x64/.../llama-server.real"

Le banc lance donc **Jan `b9967`**, pas notre `e34f042`. **Portée de la
rétractation, plus large qu'annoncé** : tout le dossier de provenance de la nuit
— chaîne d'outils, `ARCHS`, `system_info`, et le chiffre 3842,94 / 526,19 —
décrit un binaire **qui n'est pas dans la comparaison**.

**Recommandation : faire tourner la campagne sur `~/llama.cpp/build/bin/`.**

* c'est le **seul** llama.cpp dont la provenance est documentée de bout en bout ;
* c'est **celui qui a produit tous nos étalons de perplexité**, donc qualité et
  débit restent adossés au même artefact ;
* `b9967` est un build tiers empaqueté, aux drapeaux inconnus — **exactement le
  cas LM Studio que nous avons écarté pour le 170**. Le retenir maintenant
  reviendrait à accepter pour la campagne ce que nous avons refusé pour le
  chiffre qu'elle doit remplacer.

**Et ce choix rend valide la trouvaille que chef vient de retirer** : son relevé
de `-fit on` par défaut a été fait **sur `e34f042`**. Sur ce binaire-là il est
juste, et il n'y a rien à revérifier ailleurs.

### Le lanceur embarque un modèle brouillon — sur option, à déclarer

`llamacpp-serveur:65-69` définit `DRAFT=Qwen3-0.6B-Q8_0` et le passe en
`--spec-draft-model … --spec-draft-ngl 999 --spec-draft-n-max 8`, **mais
seulement si `--draft` est donné**. Ce n'est donc pas le défaut, et la conclusion
déjà écrite tient : llama.cpp ne spécule pas sauf demande.

**À vérifier au site d'appel, pas dans la ligne d'usage** : `--draft` doit être
explicitement **absent** — ou explicitement présent et déclaré. C'est la règle
« déclarer le mode des deux côtés », et c'est très précisément le piège qui nous
a coûté `ngram` : une option spéculative active que personne n'avait choisie.

## Table de décision, écrite AVANT que les chiffres arrivent

Écrite pendant que la campagne tourne, sans connaître le résultat. C'est la seule
façon d'empêcher qu'un chiffre décevant se rationalise après coup — et nous avons
déjà, cette nuit, vu une conclusion se retourner deux fois faute de cette
précaution.

**Ce que cette manche mesure exactement** : le moteur de calcul **nu**, bf16 des
deux côtés, dense, sans exil, sans spéculation, à `n_ctx 4096`. **Aucun avantage
architectural d'acvram n'intervient.** C'est le terrain le plus défavorable pour
nous, choisi exprès.

### Ce qui invalide la mesure quel que soit le chiffre

À vérifier **avant** de lire un débit — si l'un manque, la série est jetée :

* la ligne `ACVRAM_EXIL_COUCHES=0 ignoré : 0 couches` **relevée** ✔ (faite) ;
* `--draft` absent au site d'appel ✔ (fait) ;
* `n_ctx = 4096` **des deux côtés**, pas un défaut ;
* nombres de jetons identiques des deux côtés (banc en `-p N` / `-n N`) ;
* étendue intra-configuration **≤ 1 %**, publiée pour les deux moteurs ;
* **le chemin du binaire llama.cpp effectivement résolu**, écrit au journal —
  `LLAMACPP_BIN` est désormais surchargeable, donc l'intention ne prouve plus
  rien : c'est le chemin **résolu** qui doit apparaître, pas la variable.

### Ce que chaque issue engage

| écart acvram − llama.cpp | lecture | ce qu'on fait ensuite |
|---|---|---|
| **acvram plus rapide de > 5 %** | surprenant sur leur terrain | **suspecter la mesure avant de se réjouir** : vérifier que les deux ont fait le même travail, mêmes jetons, aucun repli. Un avantage inattendu est un défaut de protocole jusqu'à preuve du contraire |
| **± 2 %** | notre chemin de calcul est **compétitif** | le tiering devient de la valeur ajoutée nette, et les performances décevantes de la 5090 viennent **d'ailleurs que du calcul** — donc du chemin de tiering lui-même |
| **acvram plus lent de 5 à 20 %** | déficit localisé, chiffré | optimiser le calcul dense reste rentable ; le tiering n'est pas en cause |
| **acvram plus lent de > 20 %** | **le déficit est structurel et il est dans la base** | alors **la priorité change** : optimiser le tiering reviendrait à optimiser au-dessus d'un socle lent. Les noyaux passent devant |

**La dernière ligne est celle qui compte, et c'est pour elle que cette table est
écrite maintenant.** Si nous perdons largement ici, la tentation sera de répondre
« mais notre valeur est le tiering » — et ce serait vrai *et* hors sujet. Sur un
modèle qui tient entièrement sur la carte, un utilisateur n'a aucune raison de
nous choisir : c'est le cas le plus courant, et nous devons y être au moins à
parité.

**Aucune de ces lignes ne se renégocie après lecture du chiffre.**
