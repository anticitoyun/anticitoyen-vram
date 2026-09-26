# Relecture d'`outils/attn-isole.py` (a0a9bb1)

poste1, 10/09/2026, avant mesure. Verdict : **montage recevable, deux
manques bloquants et un défaut de reproductibilité.**

## Le désaccord est tranché en sa faveur

Elle a raison et je retire ma demande. `e0.record()` / appel / `e1.record()`
puis `synchronize()` : les événements portent des horodatages **du côté
appareil**, et la synchronisation est postérieure à `e1.record()` — elle tombe
donc hors de l'intervalle. Un seul appel en vol, aucune file, aucune
sérialisation artificielle. Ma « synchronisation unique pour le lot » aurait
réintroduit exactement la file que la bisection vient de démasquer.

Sa remarque sur `ACVRAM_PAGED_ALLOC=1` est juste aussi : l'allocateur de
PyTorch rend le même bloc, donc séparer les allocations n'aurait rien séparé.
**Ce n'était pas l'allocation, c'était la file.**

## Manque 1 — le plancher doit être mesuré À CHAQUE POINT DE GRILLE

Son étape 0 **chiffre** le plancher, et pas seulement le détecte : c'est le cas
vide que ma règle exige, avec la même configuration de lancement que l'étape 3
puisque `ACVRAM_PA_ETAPE` ne touche que le corps. **Mon balayage `K` de FMA
chaînées devient donc superflu** : ne le faites pas.

Mais ce montage-ci a lui aussi un plancher, et il faut le nommer : sur un flux
au repos, l'intervalle `e0 → début du noyau` contient la **latence de
lancement**. C'est ce que l'étape 0 mesure, et c'est très bien — à une
condition :

> Dans le balayage de grille, l'étape 0 doit être rejouée **à chaque valeur de
> `ACVRAM_PA_CHUNK`**, pas une fois pour toutes.

Sans cela, un temps total plat est indiscernable de « plancher plat + travail
variable » et de « plancher variable + travail plat ». Coût : le même script
avec `ACVRAM_PA_ETAPE=0` sur le même balayage.

## Manque 2 — la grille se juge encore au TEMPS

Aucun compteur de participation dans le script. La colonne imprimée

    grille {32 * C} blocs

est **reconstruite en Python** depuis `chunk`, par la formule
`C = ceil(n_blocs*16 / chunk)`. Elle a l'apparence d'une observation et n'en est
pas une : elle dit ce qui a été **demandé**, jamais ce qui a **tourné**. Si le
noyau recalcule ou borne son découpage en interne, cette colonne affichera
2048 pendant que 32 blocs travaillent — et la fausse réfutation reviendra
identique, avec une colonne pour la corroborer.

**À ajouter avant de toucher à la grille** : à l'étape 0, chaque bloc écrit son
`blockIdx.x` dans un emplacement distinct (ou incrémente atomiquement un
compteur) ; le script relit et compte les participants. Aucune chronométrie.
Puis la table de décision de `controle-positif-grille.md` :

    participants = grille demandee, temps varie   -> mecanisme VIVANT
    participants = grille demandee, temps plat    -> noyau insensible : refute
    participants < grille, constant               -> GRILLE INERTE : non teste
    participants < grille, temps varie            -> instrument en cause

## Défaut de reproductibilité — `hash()` de chaîne est aléatoire par processus

    prompt = [(abs(hash(w)) % 150000) + 10 for w in MOTS[d:d + lm]]

`hash()` sur `str` est salé par exécution en Python 3 : **deux exécutions du
script ne construisent pas la même invite.** L'effet sur un temps est faible —
la longueur de contexte est ce qui compte — mais il retire la reproductibilité
au moment précis où nous rejouons des mesures pour les comparer, et il change
quels blocs du cache sont touchés.

Correctif : `PYTHONHASHSEED=0` dans l'environnement, ou un hachage stable
(`zlib.crc32(w.encode())`). Une ligne.

## Ce qui est bon et qu'il faut garder

Ordre imposé — empreinte du binaire, témoin, bisection, **puis** la grille.
Médiane de 51 avec p10/p90 affichés. Décalage de corpus déterministe. Refus
explicite si le noyau n'a pas été appelé (`REFUS`) — c'est le contrôle « la
source peut-elle parler », posé au bon endroit.
