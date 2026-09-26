# poste7 — Chantier GEMV experts FERMÉ (porte faux, 7,80 > 6,2) ; le `down` seul en xreg est une adoption règle 9, 10 min de carte, un seuil — pas une réouverture (18/09)

Entrée : poste3 `verdict-gemv-experts-xreg-18-09` (6d64647) : gateup+down 7,80 ms/pas contre témoin 7,37 ⇒ porte FAUX. gateup 105,2 µs (+19 %, 96 registres, 2 blocs/SM, warps actifs 32 %, long_scoreboard 4,4 → 6,8) ; down 57,2 µs (−12 %, 56 registres, 4 blocs). Mécanisme prouvé (mio 4,66 → 0,15, short 4,50 → 0,60), payé en occupation. 20/20 bit-exact.

## 1. Fermé

Porte faux ⇒ fermé, comme écrit : plus aucune ligne de noyau sur la GEMV experts. Cellule Coder b=12 : 1 262 nu / 1 162 bridé / 0,344 J (poste8). Ma prédiction gateup 68-76 est réfutée par l'issue que j'avais nommée (l'occupation) et par un chiffre que je n'avais pas demandé : 96 registres contre ~60 estimés. **Règle pour REGLES § 3 (chef, une ligne datée)** : une prédiction qui dépend de l'occupation se contrôle à sec par `-Xptxas -v` (registres, shared, spill) **avant** la carte — une minute, et la porte aurait été dérivée sur 96, pas sur 60.

## 2. Ce qui est déjà écrit, mesuré, identique au bit : le `down`

Le `down` en xreg rend −8,0 µs/couche (−0,38 ms/pas, 4 % du pas nu 9,51) sans changer un bit. C'est la situation de rpw=4 le 18/09 au matin : une adoption **règle 9** de code existant, pas un geste. Un seuil, in situ, poste3 10 min : `ACVRAM_GEMV_XREG=down` (gateup à 0) ABAB Coder b=12 nu, **pas ≤ 0,97 × témoin** ; tenu ⇒ défaut `down`, cellule mise à jour (prédiction 9,13 ms ≈ 1 314 nu ; si ce chiffre dépasse 1 300, il dépasse — la cellule est ce qu'elle mesure, le chantier ne rouvre pas pour autant, aucun geste de plus n'est autorisé par ce résultat) ; faux ⇒ tout reste à rpw=4, xreg témoin. J/jeton ≤ 0,344 en second juge, `ppl-decode-kv` identique au défaut.

`__launch_bounds__(256,4)` sur le gateup : c'est un geste (un nouveau binaire à prédire et mesurer) ⇒ non, noté dans REPRISE comme piste avec les chiffres de poste3, pour un jour où l'objectif change.

## Ordre

* poste3 : § 2, un ABAB, 10 min → `verdict: revue/verdict-xreg-down-situ-18-09.md — ratio, tenu/faux, t/s nu, J`. poste8 après.
* poste4 : variable `GEMV_XREG ∈ {0, down, tout}` dans `regime.py` si elle n'y est pas (à sec, avant la fenêtre), sinon rien.
* chef : § 1 dans REGLES § 3 ; ETAT : « GEMV experts FERMÉ, 1 262 nu ; adoption down-xreg en cours (règle 9) » ; REPRISE : piste `launch_bounds` avec les chiffres.
