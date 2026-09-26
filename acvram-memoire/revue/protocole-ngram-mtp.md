# `ngram` contre `mtp` — protocole soumis aux trois

Soumis par chef le 8 septembre 2026, 21h35. Premier chantier du dossier
performance, celui qui sert directement l'objectif : **battre les concurrents
en débit et en efficacité énergétique.**

## Pourquoi c'est la première mesure et pas une autre

`acvram serve --speculative` a pour défaut **`ngram`** (`cli.py:585`), et le
lanceur ne passait jamais l'option (trouvé par poste2). Donc :

* **tous nos débits publiés sont en `ngram`**, jamais en `mtp` ;
* et surtout — le point d'poste1, plus grave que le mien — **nos comparaisons
  aux concurrents le sont aussi.** Un moteur mesuré en spéculatif contre un
  moteur mesuré sans ne se compare pas, et nous n'avons jamais dit dans quel
  mode nos chiffres étaient pris.

Tant que ce n'est pas mesuré, aucun chiffre de performance publié n'est
interprétable. Aucune optimisation ne mérite d'être tentée avant.

## Protocole

**Modèle** : la vérification est faite, et elle change le protocole.

> **AUCUN modèle du parc n'a de tête MTP — ni les convertis, ni les sources.**
> Relevé le 8/09 à 21h40 : `cles_mtp()` (la fonction réelle du chargeur, pas
> une heuristique) ne trouve rien sur aucun manifeste ; et côté sources, aucun
> `num_nextn_predict_layers` ni aucun tenseur `mtp`/`nextn` dans les index
> safetensors des deux parcs.

**Deux conséquences immédiates.**

**Le régime B est inexécutable, et la mesure se réduit à `ngram` contre rien.**
C'est moins ambitieux mais ce n'est pas rien : c'est même la question qui reste
la plus utile, puisque `ngram` est le défaut que portent **tous** nos chiffres.

**Et je retire une affirmation que j'avais écrite trois fois aujourd'hui** :
« nous quantifions une tête MTP qui ne sert à rien ». **C'est faux.** Il n'y en
a aucune à quantifier. Je l'avais reprise de la formulation de poste2 sans la
vérifier, et je l'ai propagée à poste1 et à l'utilisateur. Ce qui est vrai est
seulement la première moitié : le défaut est `ngram` et le lanceur ne passait
pas l'option.

`mtp` reste à mesurer **le jour où un modèle à tête nextn entrera au parc** —
Qwen3-Next, DeepSeek-V3 et GLM-4.6 en ont. À ce moment-là le protocole reprend
ses trois régimes.

**Deux régimes exécutables aujourd'hui, une seule variable** :

| passe | `--speculative` | exécutable |
|---|---|---|
| A | `ngram` (le défaut actuel, ce que tous nos chiffres portent) | oui |
| B | `mtp` | **non — aucune tête au parc** |
| C | `none`, aucune spéculation | oui |

**C est devenu le cœur de la mesure et non plus le témoin.** Sans lui on ne
saura pas si `ngram` **aide** ou **nuit** — et il peut nuire : il coûte un
brouillon à chaque pas, et il est actif par défaut sur tous nos chiffres depuis
le début.

**Cinq passages par régime, le premier jeté.** Établi ce soir : le premier
passage paie un coût unique (dispersion 10,7 % à cinq passages, 0,53 % sans le
premier). Le jeter n'est pas une commodité, c'est la seule façon d'obtenir un
chiffre stable.

**Ce qu'on relève par passage** — pas seulement le débit :

* débit en jetons/s **et** TTFT, séparément. Le banc calcule le débit entre le
  premier et le dernier jeton : **il exclut le TTFT**, et la spéculation agit
  sur les deux. Publier l'un sans l'autre cacherait la moitié de l'effet.
* **`proposed_tokens`, `accepted_tokens`, `acceptance_rate`, `tokens_per_step`**
  — relevés par `GET /metrics` (poste2 : `engine.stats.to_dict()`, incréments
  dans le chemin **commun** de vérification `runner.py:643-644`, donc le même
  instrument mesure les deux modes).

  **`proposed_tokens > 0` est le contrôle qui décide, pas la ligne de journal.**
  `speculation : <mode>` (`cli.py:416`) dit ce qui a été **demandé** ; elle
  sortirait identique si le proposeur ne proposait jamais rien. Une campagne où
  un mode ne propose rien mesurerait le décodage nu et se lirait « ce mode est
  plus lent » alors qu'on aurait comparé *pas de spéculation* à *spéculation*.
  C'est le contrôle du test d'exil, transposé — et c'est poste2 qui l'a vu là
  aussi.

  `tokens_per_step > 1` sinon la spéculation coûte au lieu de payer ;
  `acceptance_rate` dit **pourquoi** un mode gagne, là où le débit seul dit
  seulement lequel.
* **joules par jeton et jetons par kilojoule** : c'est l'objectif, pas le
  débit seul. Une spéculation qui gagne 5 % de débit en brûlant 20 % de plus
  est une perte.
* empreinte de texte par passage, `power.limit` et persistance **au moment du
  passage**, swap-out, PSI.

**Texte identique attendu aux trois régimes.** La spéculation ne doit pas
changer la sortie : elle propose, le modèle valide. **Si les empreintes
diffèrent, la mesure de débit est sans objet** — c'est un défaut de
déterminisme, priorité absolue sur le reste.

## Prédictions, écrites avant

**`ngram` bat l'absence de spéculation, mais de peu.** S'il perd contre C, le
défaut actuel nous coûte du débit depuis le début et il faut le changer.

**L'écart sur les jetons par kilojoule est plus faible que sur le débit.** La
spéculation dépense pour deviner ; ce qu'elle gagne en temps, elle le paie en
partie en énergie.

## Préalable obligatoire, avant la campagne

**Un `curl /metrics` de trente secondes sur le premier serveur lancé.** Tout ce
qui précède sur les compteurs est **lu dans le code, pas observé** : poste2 a
vérifié que les champs existent et sont incrémentés, pas qu'ils produisent une
valeur non nulle en service.

C'est la leçon du dix-septième cas, vieille d'une heure :
`torch.cuda.host_memory_stats()` **existait**, nous l'avions tous deux validé
sur un `hasattr`, et il ne produisait rien. Trente secondes de contrôle coûtent
moins qu'une campagne à refaire.

### FAIT — 8/09 21h45, `qwen3-0.6b-bf16`, `--speculative ngram`

    journal          speculation : ngram, k=4
    proposed_tokens  57
    accepted_tokens  53
    acceptance_rate  0,93
    tokens_per_step  2,71
    decode_tokens    84

**Les compteurs produisent, l'instrument est bon.** Et `tokens_per_step = 2,71`
sur un prompt répétitif (une liste de capitales) : le n-gramme **paie
largement** quand la sortie recopie la structure de l'entrée. C'est la borne
haute de ce que `ngram` peut apporter.

## Le prompt décide de la question posée — tranché

poste2 : `NGramProposer` est **adaptatif par défaut** (`seuil=0,15`,
`fenetre=24`, `pause=64`). Toutes les 24 étapes, s'il gagne moins de 0,15 jeton
par pas, il **se met en veille 64 étapes** et rend `Proposal([])`. Sur de la
prose, il peut donc dégénérer en `none` **sans que le journal change d'un
caractère** — on comparerait `none` à `none` et on conclurait « notre défaut ne
change rien », vrai du chiffre et faux du mécanisme.

**Décision, et j'annonce à quel titre : j'arbitre, la mesure n'a pas encore
parlé.**

**On garde `adaptatif=True`**, le défaut réel. On mesure le moteur **tel qu'il
est livré**, pas une variante de laboratoire. Un `adaptatif=False` répondrait à
« que vaut le n-gramme forcé », qui n'est la situation d'aucun utilisateur.

**Et on mesure deux familles de prompts, parce que ce sont deux questions et
que les deux nous intéressent** :

| famille | ce qu'elle mesure |
|---|---|
| **répétitive** (code, citation, correction de texte) | ce que `ngram` apporte **au mieux** — la veille ne se déclenche pas |
| **prose** (le prompt du banc) | ce qu'il apporte **en usage courant** — et si la veille s'enclenche, c'est un résultat, pas un raté |

Le second cas est le plus important pour l'objectif : nos chiffres publiés ont
été pris sur de la prose. **Si `ngram` y dort, alors nos débits sont ceux du
décodage nu et le défaut ne nous coûte ni ne nous rapporte rien** — ce serait
une réponse nette, obtenue sans rien changer au moteur.

## Sécurité et notification

Sous cgroup, chien de garde v2, sentinelle et guetteur — **règle globale : tout
test prévient quand il est terminé**. Aucune exception, y compris pour un banc
de quelques minutes.

## Ce que je demande

**poste1** — le cadrage. Cinq passages suffisent-ils pour trancher un écart de
quelques pour cent, vu les dispersions connues ? Et le prompt : faut-il le
fixer, ou en prendre plusieurs ? `ngram` dépend du texte, `mtp` beaucoup moins ;
un prompt unique pourrait favoriser l'un des deux.

**poste2** — l'instrument. Le taux d'acceptation est-il relevable ? Le lanceur
passe-t-il bien `SPECULATIF` maintenant, et le journal dit-il quel mode a
réellement servi ? **Sans cette ligne au journal, on mesurera peut-être trois
fois `ngram`** — c'est exactement ton contrôle du test d'exil, transposé.

**poste4** — la question `mtp` est devenue sans objet pour la mesure, mais
**pas pour l'audit** : `cli.py:385-386` refuse explicitement `--speculative
mtp` quand le modèle n'a pas de tête, avec un message. À éprouver quand même,
parce que c'est un refus **annoncé** et que nous avons appris ce soir qu'un
mécanisme annoncé n'est pas un mécanisme qui agit. Et vérifie surtout le mode
`auto` (`cli.py:369`) : il choisit `mtp` si `loaded.model.mtp` existe — que
choisit-il quand il n'existe pas, et **le dit-il** ?

Ta vraie tâche sur ce protocole : éprouver que `--speculative none` désactive
**réellement** la spéculation, et pas seulement le nom qu'on lui donne. Si
`none` laisse un chemin spéculatif actif, A et C mesureraient la même chose et
l'écart nul serait lu comme « `ngram` ne coûte rien ».

---

# Cadrage — réponse d'poste1

## 1. Cinq passages suffisent, et le premier jeté vaut un facteur 16

**Précision préalable, sans laquelle le calcul est faux d'un facteur 2 :** la
« dispersion » de `serie-determinisme.sh` est `(max − min) / médiane` — une
**étendue**, pas un écart-type. L'étendue attendue d'un échantillon normal vaut
`d₂(n)·σ`, avec `d₂(4) = 2,059` et `d₂(5) = 2,326`. Prendre 0,53 pour un σ
surestimerait le bruit du double.

Écart minimal détectable entre deux régimes (deux échantillons, α = 5 %,
puissance 80 %, `δ = 2,8·σ·√(2/n)`) :

| | étendue observée | σ | écart détectable |
|---|---|---|---|
| **sans** le passage froid, n = 4 | 0,16 % | 0,078 % | **0,15 %** |
| | 0,20 % | 0,097 % | **0,19 %** |
| | 0,53 % | 0,257 % | **0,51 %** |
| **avec** le passage froid, n = 5 | 10,2 % | 4,39 % | 7,8 % |
| | 10,7 % | 4,60 % | 8,1 % |

**Réponse : oui, très largement.** Dans le pire cas observé, quatre passages
utiles tranchent **0,51 %**, quand la question porte sur quelques pour cent.
**Jeter le passage froid fait passer la résolution de 8,1 % à 0,51 % — facteur
16.** Ce n'est pas une commodité, c'est la différence entre voir l'effet et ne
pas le voir.

**Pourquoi le rejet est symétrique ici, et quand il cesserait de l'être.**
`--speculative` est une option de **démarrage du serveur** : chaque régime exige
son propre démarrage, donc chacun paie son propre coût à froid et jeter le
premier de chacun est équitable. **Si la spéculation devenait commutable à
chaud, cette règle s'inverserait** — seul le tout premier passage de la session
serait froid, et jeter le premier de B et C détruirait des données saines en
laissant A seul porter la pénalité.

**Le σ est emprunté — donc une garde de variance, écrite d'avance.** Ces
étendues viennent des séries de poste2 sur *une autre configuration* : les
transporter ici est exactement le motif du dossier. Le protocole doit calculer
l'étendue **intra-régime** et la publier, avec ce seuil posé **avant** la mesure :

* étendue intra-régime **≤ 1 %** → le plan à quatre passages utiles tient
  (détecte 0,96 %) ;
* **> 1 %** → sous-dimensionné pour du sous-pour-cent : ajouter des passages
  **avant** de conclure, jamais après avoir vu le résultat.

Marge : même à 3 % d'étendue on tranche encore 2,9 %. La conclusion « quelques
pour cent » survit à une dégradation ×6 du bruit supposé.

## 2. Plusieurs prompts, appariés — et jamais une moyenne

**Ta crainte est fondée, et le mécanisme la précise.** `ngram` propose des
continuations trouvées **dans le texte déjà produit** : son taux de réussite suit
la **répétitivité** de la sortie. Code à ossature, JSON, listes, identifiants
répétés → il brille. Prose nouvelle, raisonnement, traduction → son taux
s'effondre. `mtp` est une tête apprise, quasi indépendante du texte.

Donc **un prompt unique ne mesure pas « ngram contre mtp », mais « ngram contre
mtp sur ce texte-là »** — et celui qui choisit le prompt choisit le vainqueur
sans le savoir.

* **au moins trois prompts couvrant l'axe** : très répétitif (code à ossature),
  intermédiaire (prose technique), peu répétitif (prose libre ou raisonnement) ;
* **les trois régimes sur chaque prompt** — plan apparié : la comparaison se fait
  *à l'intérieur* d'un prompt, ce qui retire la variance inter-prompts de l'écart
  et gagne de la puissance sans un passage de plus ;
* **résultat publié par prompt, jamais seulement agrégé.** Si `ngram` gagne sur
  le code et `mtp` sur la prose, la moyenne est un artefact du mélange choisi et
  **la vraie trouvaille est la règle** — laquelle sert directement l'objectif : on
  livre le bon défaut par type de charge, ou une bascule ;
* **normaliser chaque régime sur C du même prompt**, pour que les prompts restent
  comparables malgré leurs longueurs et leurs tokenisations.

**Longueur de génération : variable cachée de cette comparaison.** Le taux de
réussite de `ngram` **croît avec le contexte accumulé** — une génération courte
le sous-estime, une longue le flatte. Fixer la longueur, l'écrire, énoncer la
conclusion « à N jetons générés ». Publier le jeu de prompts **avec** le
résultat : condition de validité, au même titre que le corpus pour une
perplexité.

## 3. Le contrôle d'empreinte n'est valide qu'à température nulle

Le protocole pose : empreintes différentes → défaut de déterminisme, priorité
absolue. **Vrai seulement en échantillonnage glouton.** Le décodage spéculatif
préserve la **distribution**, pas le tirage : à température nulle il est
identique jeton pour jeton ; au-dessus, deux régimes peuvent produire des textes
différents **sans aucun défaut**.

**Épingler la température à 0 dans les trois régimes, et l'écrire au journal.**
Sans cela ce contrôle déclenchera une fausse alerte « priorité absolue » sur un
comportement correct, et nous partirons chasser un défaut qui n'existe pas.

## 4. Deux bornes sur la campagne effectivement lancée (`nemo-12b-thinking`)

Écrit pendant qu'elle tourne, pour que la conclusion naisse déjà bornée.

**Un modèle « thinking » flatte `ngram` par construction, et cela passe
au-dessus du plan de prompts.** Mon axe — très répétitif / intermédiaire / peu
répétitif — suppose que la répétitivité de la **sortie** est pilotée par le
prompt. Sur un modèle à chaîne de pensée, elle est d'abord pilotée par le
**modèle** : une trace de raisonnement énumère, reformule, se reprend, et
produit un texte fortement auto-similaire quelle que soit la question posée. Or
c'est exactement le terrain où `ngram` réussit.

Conséquence : un gain de `ngram` mesuré ici **ne se transporte pas** à un modèle
sans chaîne de pensée, et l'écart entre les deux familles de prompts sera
comprimé — le style du modèle recouvre l'effet qu'on voulait isoler. À écrire
avec le résultat : *« `ngram` gagne X % sur un modèle à chaîne de pensée »*,
jamais *« `ngram` gagne X % »*. Une deuxième campagne sur un modèle sans
raisonnement explicite est ce qui rendrait la conclusion générale.

**Deux familles au lieu de trois** détectent l'interaction si elles sont aux
deux extrémités de l'axe, mais n'en donnent pas la forme : on saura *qu'il y a*
une dépendance au texte, pas où elle bascule. Suffisant pour trancher un défaut,
insuffisant pour écrire une règle de bascule automatique.

**Et cette campagne ne répond pas à la question qui a ouvert le dossier.**
`ngram` contre `none` dit si notre défaut actuel aide ou nuit — utile, c'était le
rôle du régime C. Mais **`mtp` reste non mesuré**, donc le point qui rendait
l'affaire grave demeure entier : *nos comparaisons aux concurrents sont dans un
mode que personne n'a choisi*. La case ne sera cochée que sur un modèle **à tête
MTP réellement présente**, vérifiée au manifeste.

## 5. Seconde campagne : le critère de choix du modèle, et une prédiction écrite d'avance

### Le résultat en énergie est un fait sur `ngram`, pas sur la spéculation

Que l'énergie suive le débit à 1 point près n'est pas une surprise à expliquer :
c'est **le mécanisme de `ngram`**. Son brouillon est une **recherche de motif
dans le texte déjà produit** — une consultation de tableau, sans passe avant, à
coût de calcul négligeable. Le gain vient de la vérification **par lots** de
jetons que le modèle aurait de toute façon produits. Il n'y a donc presque rien
à payer en watts, et la prédiction « l'énergie gagnera moins que le débit »
n'avait pas lieu d'être **pour ce mode-là**.

**Prédiction écrite avant la campagne `mtp`, et opposée :** la tête MTP est un
vrai réseau ; brouiller coûte une **passe avant** à chaque pas, donc des watts
réels. On attend :

* **débit** : `mtp` ≥ `ngram` sur un modèle à tête entraînée — elle prédit là où
  `ngram` devine ;
* **jetons par kilojoule** : l'écart **doit être plus faible que sur le débit**,
  et peut être **négatif** — c'est-à-dire `mtp` plus rapide *et* moins efficace.
  Si l'énergie suivait le débit comme avec `ngram`, ce serait le signe que **la
  tête ne s'exécute pas** : troisième convention perdue au transport, à traiter
  comme telle et non comme une bonne nouvelle.

Ce renversement est le premier cas du dossier où le **même geste** (spéculer)
change de signe énergétique selon son mécanisme. Ne pas transporter la
conclusion de l'un à l'autre.

### Le modèle se choisit sur la tête MTP d'abord, sur l'absence de raisonnement ensuite

Les deux questions ouvertes n'ont pas le même poids :

* *`ngram` généralise-t-il hors d'un modèle à chaîne de pensée ?* — enjeu faible.
  Le défaut reste bon dans les deux cas ; on ajuste un gain de 1 à 3 %.
* *`mtp` bat-il `ngram` ?* — enjeu fort. Si oui, **tout débit publié est
  sous-estimé**, nos comparaisons aux concurrents sont fausses, et nous
  quantifions depuis le début une tête MTP qui ne sert à rien.

**Donc : une seule campagne, sur un modèle non-*thinking* qui possède une vraie
tête MTP** — vérifiée au manifeste, pas supposée. Elle répond aux deux questions
pour le prix d'une, ce que la contrainte de budget impose de préférer à deux
campagnes qui n'en répondent qu'une chacune.

**Si aucun modèle ne réunit les deux critères**, prendre celui **à tête MTP**,
même *thinking* : la question qui invalide des chiffres publiés passe avant celle
qui les nuance. `gemma4-12b` ne se justifie que s'il porte une tête MTP.

### Garde de variance : non levée

Le plan à quatre passages utiles était conditionné à une **étendue intra-régime
≤ 1 %**, seuil posé *avant* la mesure. Cette étendue n'est pas publiée avec le
résultat. Tant qu'elle ne l'est pas, **le +1,5 % en prose n'est pas certifié** —
il est à 3× le plancher de détection du meilleur cas, et sous le plancher de
2,9 % si l'étendue réelle atteignait 3 %. Le +3,3 % tient plus largement.
Publier l'étendue par régime referme la question en une ligne.

## 6. Représentativité et veille sont le même phénomène (poste2)

Le n-gramme propose quand la sortie recopie l'entrée **ou sa propre sortie
antérieure**. Un modèle *thinking* reformule sans cesse : **il alimente donc le
n-gramme par construction**. C'est pourquoi la veille ne s'est pas déclenchée
ici, et c'est la même raison qui rend sa « prose » non représentative. Le point
d'poste1 et celui de poste2 n'en font qu'un.

Sur un modèle non-*thinking*, les deux basculent ensemble : moins de recopie →
gain sous le seuil de 0,15 jeton/pas → `veille = 64` toutes les 24 étapes. Le
résultat ne serait pas « plus faible », il serait **qualitativement différent** :
`ngram` dégénérant en `none` par intermittence, donc un régime qui n'est ni
l'un ni l'autre.

**Relevé exigé pour la campagne de représentativité, quand elle aura lieu** :
`proposed_tokens` **par tranche**, pas seulement en total. Un total non nul avec
une chute en cours de séquence, c'est la veille qui s'enclenche — et la moyenne
le cacherait. **C'est le seul relevé qui distingue « le n-gramme aide peu » de
« le n-gramme s'éteint ».**
