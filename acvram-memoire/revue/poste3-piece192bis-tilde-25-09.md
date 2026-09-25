# 192 bis — chaînes `~/…` non développées après la purge

instrument : correction mécanique (regex + AST), `py_compile`, `bash -n`,
exécution directe de `banc-chat-openai.py --help`
commit : sur `poste3-192b` depuis `origin/main` (2ee125e6a)
régime : à sec, sans carte (CUDA_VISIBLE_DEVICES="")
scellé : aucun (correctif mécanique, pas une mesure)
mesuré : 45 fichiers `.py` + 66 fichiers `.sh` de `scratchpad/` portant une
chaîne `~/…` en dur ; `acvram/`, `outils/`, `tests/` déjà propres (confirmé)
verdict : TENU — 45/45 `.py` enveloppés dans `os.path.expanduser(...)`
(1 cas de concaténation implicite de chaînes sur deux lignes corrigé à la
main, `campagne-20s-llamacpp-14-09.py`) ; 66/66 `.sh` : `"~/` → `"$HOME/`,
1 cas en quotes simples (`kl2.sh`) corrigé en `"$HOME/...`. `py_compile`
0 échec sur les 45 ; `bash -n` 0 échec sur les 66. `banc-chat-openai.py
--help` : passe l'import (`energie` trouvé), échoue plus loin sur un
`--help` non géré par le script (hors périmètre, pas le bug signalé).
Second ordre (tee||echo) : 10 fichiers `scratchpad/*/prise*.sh` remplaçaient
un banc en échec par une ligne texte et continuaient (code 0, jsonl vide) —
`|| echo "BANC ÉCHOUÉ..."` → `|| { echo "BANC ÉCHOUÉ..." >&2; exit 1; }`,
et les 5 qui n'avaient que `set -uo pipefail` durcis en `set -euo pipefail`.
Garde ajoutée : `tests/test_tilde_sans_expanduser.py` (AST sur tout le
dépôt suivi, 1018 fichiers `.py`, 1 fichier au brouillon syntaxiquement
invalide ignoré hors périmètre) — 0 chaîne `~/…` nue restante dans
`sys.path.insert`/`open`. Non rejouée sous pytest (carte tenue en continu
par le circuit) ; validée par le même script en dehors de pytest, résultat
identique (liste vide).
durée : ~45 min (prévu : pas borné, correctif mécanique multi-fichiers)

## Fichiers touchés (second ordre, tee||echo)
scratchpad/poste5-p139-24-09/prise-cbis.sh
scratchpad/poste5-p177-25-09/prise.sh
scratchpad/poste5-p179-25-09/prise.sh
scratchpad/poste5-p179b-25-09/prise-b8.sh
scratchpad/poste5-p179b-25-09/prise.sh
scratchpad/poste5-p183-25-09/prise.sh
scratchpad/poste5-pninfer-24-09/prise.sh
scratchpad/poste1-p173-25-09/prise-abba.sh
scratchpad/poste1-p182-25-09/prise-abba.sh
scratchpad/poste1-p188-25-09/prise-abba.sh
