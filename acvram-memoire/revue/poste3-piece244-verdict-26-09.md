# 244 — carte.sh tue le groupe de la commande, refus si pgid vivant

instrument : à sec (tests isolés, `ACVRAM_VERROU` propre, aucune carte réelle)
commit : `5eb5e05a7` (fusion `f970f34cd`)
régime : —
scellé : aucun
mesuré : outils/carte.sh, tests/test_carte_groupe_signal_244.py
verdict : TENU
durée : —

Deux correctifs : (1) la commande tourne dans son propre groupe (`setsid`), un TERM/INT/HUP
reçu par carte.sh lui-même est transmis à ce groupe avant de rendre le verrou ; (2) un `.qui`
dont le 5e champ (pgid) désigne un groupe encore vivant refuse la carte suivante plutôt que
de la rendre en silence par-dessus une commande qui tourne encore.

**Contre-lecture (poste2) : pas de note de verdict à l'époque — comblée ici.**

`pytest tests/test_carte_groupe_signal_244.py -q` sous `outils/carte.sh`, sur `origin/main`
+ correctif 253 (`2c93c797b`) : **3 passed** (log
`scratchpad/poste3-p231-244-verdict-26-09/prise.log`, sha testé = `origin/main` `a5241f45c`
+ poste3-253).

**Trouvé en marge, pas un défaut de 244 elle-même** : les deux lectures `read ... < "$INFO"
2>/dev/null` (lignes 321 et 469, cette pièce) bruitaient sur stderr à chaque prise qui suit
une prise propre — l'ordre des redirections bash ne suppr le message d'erreur QUE si
`2>/dev/null` précède le `<` qui échoue, jamais l'inverse (vérifié à la main :
`read a < /manque 2>/dev/null` bruite, `read a 2>/dev/null < /manque` non). Le `.qui` absent
est le cas NORMAL (une prise propre l'a déjà effacé), donc CHAQUE prise en bruitait une —
vu par chef le 26/09 pendant une prise de publication sans rien d'anormal. Corrigé et
testé à part : voir 253 (`2c93c797b`, `tests/test_carte_253_qui_absent_sans_bruit.py`).
