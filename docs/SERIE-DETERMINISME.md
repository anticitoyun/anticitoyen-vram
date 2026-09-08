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
