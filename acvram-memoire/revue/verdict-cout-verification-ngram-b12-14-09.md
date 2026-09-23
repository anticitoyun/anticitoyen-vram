# Verdict — le n-gram COÛTE la moitié du débit à b=12 (item 2, dernière inconnue A4)

Laure, 14/09/2026. Suite à
[`protocole-cout-verification-ngram-b12-14-09.md`](protocole-cout-verification-ngram-b12-14-09.md),
prédiction écrite avant mesure : **confirmée, et largement au-delà du seuil
prévu.**

## Montage exécuté

Qwen3-Coder-30B-A3B-nvfp4, 12 séquences réellement concurrentes
(`add_request`+`step()`, pas 12 `generate()` séquentiels), 12 des 20 invites
de code réelles, 128 jetons/séquence, carte exclusive
(`ACVRAM_TYPE=mesure`, verrou tenu du chargement à la fin), repos 30 s avant
chaque bras.

## Chiffres

    bras                    tok/s     J/jeton   pas   mur
    A — ngram inactif       481,37    0,6896    127   3,006 s
    B — ngram actif, k=4    241,35    0,9946    126   6,008 s

    ratio tok/s  B/A = 0,5014   (-49,9 % de débit)
    ratio J/jeton B/A = 1,4423  (+44,2 % d'énergie par jeton)

Les deux bras ont tourné sous le même bridage (`puissance`, cohérent entre A
et B — pas une contamination différentielle). Un avertissement OOM
transitoire (allocation de 96 Mio refusée un instant, `free: 41943040`
octets ≈ 40 Mio) est apparu pendant le bras A, sans faire échouer la
mesure (l'allocateur a récupéré) — signalé, pas caché ; la marge VRAM sur ce
modèle à ce lot est visiblement fine.

## Le verdict

**ratio = 0,50, très en dessous du seuil de 0,90** fixé dans le protocole
comme limite du « coût mineur ». Ce n'est pas un coût mineur : **activer
`ngram` à b=12 divise le débit par deux et augmente l'énergie par jeton de
44 %.** Le chiffre de Jérôme (« peut coûter 31 % ») était une borne basse —
la mesure réelle est pire.

**L'hypothèse du protocole est confirmée** : à b=12 le lot occupe déjà la
carte à largeur 1 par séquence ; la vérification n-gram élève la largeur
effective jusqu'à `spec_k+1=5` par séquence sans que la carte ait de marge
pour l'absorber — contrairement à b=1 où le même mécanisme remplit un noyau
sous-occupé. Ceci referme définitivement le point laissé ouvert par
[`chantier-speculation.md`](chantier-speculation.md) §2 (« LE LOT FAIT DEJA
MIEUX — à BQ=12 la carte est remplie sans spéculer », jamais mesuré avant
aujourd'hui) : ce n'est pas seulement que spéculer n'aide pas à b=12, **ça
coûte cher.**

## Défaut du serveur — la question posée par Jérôme

`acvram/cli.py:521-524` : `--speculative auto` (le défaut de `acvram serve`)
choisit `ngram` dès que le modèle n'a pas de tête MTP — **sans condition sur
la taille du lot**. Ce modèle (Qwen3-Coder-30B-A3B-nvfp4) n'a pas de tête
MTP → `ngram` s'active par défaut, y compris pour un service qui reçoit
constamment b≥12 concurrentes. **C'est bien un défaut du serveur** au sens
de Jérôme : le réglage par défaut est correct au décodage à faible
occupation et actively nuisible à occupation pleine, et rien ne bascule
entre les deux régimes.

## Ce que je ne tranche pas ici

Le seuil de lot à partir duquel `ngram` bascule de gain à coût (b=1 gagne
selon `verdict-taux-ngram-code-13-09.md`, taux 1,6137 — b=12 coûte -50 % ici)
n'est pas mesuré : cette campagne n'a que deux points (b=1 implicite via le
taux d'acceptation, b=12 explicite). Un balayage b∈{2,4,6,8,12} serait
nécessaire pour situer le seuil de bascule si un correctif conditionnel
(`ngram` actif seulement sous un lot réel < N) est envisagé.

## bd

Item (2) de la dernière inconnue A4 : **fermé.** Recommandation : ne pas
laisser `--speculative auto` activer `ngram` sans condition de lot pour un
service — sujet à trancher par Jérôme (créer le bead de correctif si retenu ;
je ne le fais pas ici, hors du périmètre de la mesure demandée).
