# Série déterminisme — d'où vient une dispersion de 21 et 26 pour cent

Écrite le 8 septembre 2026, à exécuter telle quelle après la conversion. Elle
répond à une question précise et se refuse à en trancher d'autres.

## D'où elle vient

Le premier comparatif énergétique a rendu, sur les deux modèles Huihui servis
par acvram, des dispersions de **21,3 %** et **26,0 %** entre passages, contre
5,2 et 5,5 pour llama.cpp sur les mêmes modèles. À ce niveau, l'écart annoncé
sur Huihui-Opus (+41 %) ne vaut que 1,9 fois la dispersion : il ne conclut
rien.

L'explication avancée — « la spéculation par n-grammes rend le débit dépendant
du texte » — **est incompatible avec les conditions de mesure**. Le prompt est
fixe, la température est nulle, et le texte sort identique mot pour mot entre
deux exécutions dans les vérifications faites la veille. Même texte, mêmes
n-grammes, mêmes acceptations : la spéculation ne peut pas faire varier le
débit d'un cinquième.

Donc l'une de ces deux choses est vraie, et aucune n'est encore mesurée :

1. **les textes ne sont PAS identiques d'un passage à l'autre** sur ces
   modèles — le chemin n'est alors pas déterministe à température nulle, ce
   qui est un défaut plus grave que le débit qu'on venait mesurer ;
2. **les textes sont identiques** et la dispersion vient d'ailleurs : premier
   passage froid, état du cache, ordonnancement, ou une variabilité de timing
   propre à la spéculation, indépendante du texte.

Le banc écrit désormais une empreinte du texte par passage
(`empreintes`, `textes_identiques`) : la question est instrumentée, il reste à
l'exécuter.

## Ce qu'il faut relever à chaque passage

Par passage, et non par cas : empreinte du texte, nombre de jetons, débit,
temps au premier jeton, joules, watts de repos, horloges minimale et maximale,
température maximale, motifs de bridage, mémoire disponible et cache de la
machine, et l'empreinte du code chargé par le serveur.

## Les trois cas, dans cet ordre

**Cas A — spéculation allumée, cinq passages, sur les deux Huihui.**
Cinq et non trois : avec trois, un premier passage froid et deux passages
proches donnent la même image qu'une bimodalité, et on ne les distingue pas.

**Cas B — spéculation éteinte, cinq passages, mêmes modèles, même session,
rien d'autre changé.** À n'exécuter que si le cas A montre des textes
identiques ; si les textes diffèrent, le déterminisme se corrige d'abord, le
reste ne veut rien dire.

**Cas C — témoin, cinq passages sur un modèle qui disperse peu.**
Coder-Next dispersait à 1,3 % dans le même comparatif. Il tourne dans la même
session, entre A et B. **S'il disperse cette fois, la machine n'est pas calme
et toute la série est nulle** — c'est le contrôle qui empêche d'attribuer au
moteur ce qui appartient à l'environnement.

## Table de décision, écrite d'avance

| Ce qu'on observe | Ce qu'on en conclut |
|---|---|
| Cas C disperse > 5 % | Série nulle. La machine bouge. Rien d'autre ne se lit. |
| Cas A : empreintes différentes | **Défaut de déterminisme du moteur à température nulle.** Le débit de ces modèles n'est pas mesurable tant qu'il n'est pas corrigé. Priorité au défaut, pas au comparatif. |
| Cas A : empreintes identiques, dispersion tombe sous 5 % en écartant le premier passage | Le coût est celui du **démarrage à froid**. Le protocole ajoute un passage de chauffe jeté ; rien à corriger dans le moteur. |
| Cas A : empreintes identiques, dispersion reste haute sans le premier passage, et cas B tombe sous 5 % | **La spéculation introduit une variabilité de timing indépendante du texte.** À documenter comme telle ; le comparatif publie alors les deux régimes, avec et sans. |
| Cas A : empreintes identiques, dispersion haute, cas B haute aussi | Ce n'est ni le texte ni la spéculation : chercher dans l'ordonnancement, l'allocateur ou la mémoire hôte. La série a alors éliminé deux causes, ce qui est son travail. |

## Ce que la série ne dira pas

Elle ne dit rien du débit d'acvram contre llama.cpp : elle mesure la stabilité
d'un moteur, pas sa vitesse. Aucun chiffre de comparaison ne doit en sortir.
Et elle ne vaut que pour les deux modèles Huihui : une dispersion mesurée sur
un modèle ne se transporte pas à un autre — c'est exactement l'erreur qui a
fait transporter une conclusion d'une fenêtre de 120 secondes à une fenêtre de
25 le même jour.

## Résultat, 8 septembre 2026 — cas A

Exécutée sur les deux Huihui 35B-A3B servis par acvram, cinq passages,
spéculation `ngram`. Le filtre a gardé trois entrées de parc : deux d'entre
elles pointent vers le **même dossier** sous le même alias, ce qui donne un
contrôle involontaire de reproductibilité entre deux chargements.

Débits dans l'ordre d'exécution, en jetons par seconde :

| modèle | 1er | 2e | 3e | 4e | 5e |
|---|---|---|---|---|---|
| Opus-abl (1re entrée) | **112,9** | 150,5 | 150,2 | 151,2 | 152,3 |
| Opus-abl (2e entrée) | **114,5** | 151,0 | 150,9 | 151,0 | 151,5 |
| abliterated | **141,9** | 159,8 | 160,6 | 160,1 | 159,8 |

| modèle | dispersion | sans le 1er passage | coût du démarrage à froid |
|---|---|---|---|
| Opus-abl (1re entrée) | 10,7 % | **0,53 %** | 25,3 % |
| Opus-abl (2e entrée) | 10,2 % | **0,16 %** | 24,2 % |
| abliterated | 4,7 % | **0,20 %** | 11,4 % |

**Empreintes de texte identiques sur les cinq passages, pour les trois
modèles.** Le passage lent est le premier, toujours.

**Conclusion, par la table de décision écrite d'avance** : troisième ligne —
*empreintes identiques, dispersion sous 5 % en écartant le premier passage* →
le coût est celui du **démarrage à froid**, le protocole ajoute un passage de
chauffe jeté, **rien à corriger dans le moteur**.

**Le cas B devient sans objet.** Il ne se lisait que si la dispersion restait
haute après retrait du premier passage. L'explication avancée en août — « la
spéculation par n-grammes rend le débit dépendant du texte » — est réfutée
deux fois : le texte est identique bit pour bit entre passages, et la
dispersion disparaît en écartant un seul passage.

**Le cas C n'a pas été exécuté.** Lancé à 18h26, il a rencontré 1951 Mio de
VRAM libre — une autre session occupait 29 Go — et a été interrompu. Il reste
à faire sur machine calme ; il ne remet pas en cause le cas A, dont les trois
lignes ont été mesurées quand les cartes étaient libres, mais il n'apporte pas
encore sa validation d'environnement.

## Deux défauts d'instrument trouvés en route

**Le placement était tiré au sort à chaque chargement.** Les deux entrées vers
le même dossier ont d'abord rendu **15,8 puis 152,2 t/s**, facteur 9,6, avec
des textes différents. Le journal du serveur donne la cause : au premier
chargement le plan exilait 5 puis 6 MLP de plus en RAM hôte (« poids réels
8.7 Gio pour 11.1 Gio »), au second aucun. `arreter()` n'attend que la
fermeture du port plus trois secondes ; le serveur précédent avait rendu son
port sans avoir rendu sa mémoire, et le planificateur du suivant a calculé sur
les restes. **Tout chiffre de débit pris juste après un changement de modèle a
pu être mesuré sur un plan dégradé, sans que rien ne le signale.**

**Le régime de puissance a changé pendant la série.** La colonne `plafond_W`
(somme des deux cartes) porte **775 W** sur la première ligne et **875 W** sur
les deux suivantes : la limite de la 3080 Ti est passée de 275 à 375 W entre
18h23:57 et 18h24:45. Le comparatif du 3 septembre porte 675 W. Trois régimes,
trois séries qui ne se comparent pas en énergie. Sans cette colonne, l'écart
aurait été attribué aux modèles.

## Ce que la série a coûté, et la garde qui en est sortie

Le cas C a été lancé alors qu'il restait 1951 Mio de VRAM libre. Un modèle de
44 Gio dont la VRAM ne peut prendre que 8,7 exile le reste en RAM hôte : la
RAM a saturé, la machine a redémarré, et le worktree comme les fichiers de
mesure ont été perdus — ils vivaient dans `/tmp`. Les chiffres ci-dessus ont
survécu parce qu'ils étaient sous les yeux au moment du redémarrage.

La garde qui existait, `attendre_memoire()`, attend que la mémoire **cesse de
bouger**, pas qu'elle soit **disponible** : elle a rendu la main sur 1951 Mio
parce que 1951 Mio était stable. `verifier_place()` compare désormais le poids
du modèle à la VRAM libre et à la RAM hôte disponible, et refuse plutôt que de
charger sur les restes ; `--forcer-exil` lève le refus.

**Ce que cette garde ne prouve pas.** Ses deux constantes — 8 Gio de réserve
système, 20 % du poids pour les tampons — ne sont pas mesurées. Et la RAM
disponible à 18h26 n'a pas été relevée : je ne peux donc pas affirmer que la
garde aurait refusé ce chargement-là. Elle refuse les mêmes conditions dès que
la RAM libre descend sous ~52 Gio, ce qui était vraisemblablement le cas, sans
que ce soit établi. La mesure qui trancherait : relever le MemAvailable
**minimum** pendant un chargement dont l'exil est connu, sur trois tailles de
modèle.

## Limite de l'avant/après du mode persistant, trouvée après coup

L'avant/après annoncé — dispersion 10,7 → 8,7 % et 10,2 → 8,3 %, coût du froid
de ~25 % à ~20 % — **ne compare pas un seul facteur.** La machine a redémarré
à 18h36:01, entre la série d'avant (18h23-18h25) et celle d'après
(18h54-18h57). Au moins deux autres choses ont bougé dans l'intervalle : le
cache de pages du noyau, vide au premier lancement d'après reprise, et l'état
des cartes. Et pour le premier modèle seulement, le plafond de puissance est
passé de 775 à 875 W ; les deux autres étaient déjà à 875 des deux côtés.

Le verdict — la persistance retire une part du froid sans l'expliquer — n'est
pas renversé : un gain partiel reste partiel quel que soit le reste. Mais la
propriété qui faisait la force de la mesure, « un seul facteur change », est
fausse, et le chiffre d'un cinquième ne doit pas être cité comme s'il était
isolé. À refaire dans une seule session, persistance activée puis désactivée,
si le chiffre doit servir à décider.

## Le froid n'est ni de la compilation ni du chargement

Deux explications ont été avancées puis écartées par les faits.

**Ce n'est pas de la compilation** : `~/.nv/ComputeCache` n'a pas été écrit
depuis le 24 août et `~/.triton/cache` pas depuis 13h07, alors que les séries
courent de 18h25 à 19h. Rien n'a été compilé pendant les mesures.

**Ce n'est pas du chargement non plus** — lecture des caches, initialisation
du contexte, allocation de l'arène. Les cinq passages sont cinq requêtes au
**même processus déjà démarré** : quand le premier passage commence, tout cela
est fait depuis longtemps.

**Le candidat qui reste est la capture des graphes CUDA**, paresseuse : elle a
lieu à la première requête qui rencontre chaque forme, coûte une exécution de
traçage, n'écrit rien sur le disque — ce qui explique les dates figées des
caches — et ne se paie qu'une fois, d'où le plateau plat dès le deuxième
passage. La mesure des segments extensibles l'appuie déjà sans avoir été faite
pour ça : quand la capture échoue, le premier passage cesse d'être aberrant
(45,6 · 48,1 · 47,9 · 48,1 · 47,7) et la dispersion tombe à 0,4 %. Voir
`docs/SEGMENTS-EXTENSIBLES.md`.

Si cela se confirme avec `--no-cuda-graphs`, le froid ne se supprime pas mais
se **déplace** : capturer au démarrage du serveur plutôt qu'à la première
requête le retire du chemin de service, ce qui est exactement ce qu'on veut
d'un serveur.
