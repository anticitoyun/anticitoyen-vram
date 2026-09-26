# 256b — `_INVITE_TEXTE_REELLE` allongée à 277 jetons (banc-llamacpp-16-09.py)

instrument : à sec (tokenizer Qwen3-Coder-30B-A3B-nvfp4, aucune carte réelle)
commit : (voir `git log origin/poste3-256 -1`)
régime : —
scellé : aucun
mesuré : `_INVITE_TEXTE_REELLE`, `invite_reelle()`, `tests/test_banc_repetition_256.py`
verdict : TENU
durée : —

La 256 avait rendu `invite_reelle(gguf, 256)` refusée par construction : la phrase réelle
d'origine (28 jetons) tuilée à 256 dépasse toujours le seuil (0,89 de trigrammes répétés).
Corrigé en allongeant `_INVITE_TEXTE_REELLE` à sept questions techniques réelles et
DISTINCTES (architecture machine, systèmes — même domaine que la phrase d'origine), 277
jetons mesurés au tokenizer réel : ratio de trigrammes répétés **0,0218** (tournures
communes entre questions, ex. « et », « de la » — pas une répétition structurelle), bien
sous le seuil 0,5.

`pytest tests/test_banc_repetition_256.py -q` sous `outils/carte.sh` : **6 passed** (log
`scratchpad/poste3-p256-26-09/prise-c.log`). `invite_reelle(gguf, 256)` est maintenant
ACCEPTÉE (256 ≤ 277, aucun tuilage) ; le témoin (`invite_reelle(gguf, 2000)`, au-delà de la
longueur naturelle) tuile encore et reste REFUSÉ — le mécanisme de détection n'a pas changé,
seule l'invite par défaut du banc est devenue assez longue pour ne plus le déclencher.

**La 229 et la 237 (invite tuilée à 256 jetons de la phrase courte de 28 jetons) ne sont donc
plus reproductibles À L'IDENTIQUE avec ce banc — voulu (chef, 26/09) : reproduire une
invite répétée n'a plus de sens une fois la règle qui l'interdit écrite (REGLES § 4, 256).**
Si un rejeu exact de la 229/237 est un jour nécessaire (contre-preuve, archéologie d'un
défaut), il faudra reconstruire l'ancienne phrase courte et la tuiler à la main hors de
`invite_reelle` — pas en assouplissant cette garde.
