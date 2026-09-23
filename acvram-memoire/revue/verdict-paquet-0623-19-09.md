# Verdict — `test_paquet_charge_utile` sur `acvram_0.6.23_amd64.deb` (main 28441096, blob edee0fa9, sha256 afb06ed88760ba0e…, 799 Ko) : **4 passed en 0,34 s** — la charge utile refuse toujours ce qui n'est pas pour qui installe (journaux, outils hors `carte.sh`, mémoire) ; l'ETAT peut lever « sans verrou mesure : à faire »

instrument : `tests/test_paquet_charge_utile.py` tel quel, `DEB` pointé sur `acvram_0.6.23_amd64.deb` (le test vise `acvram_0.6.0_amd64.deb` en dur et construit le paquet s'il manque : copie temporaire du test avec le nom 0.6.23, supprimée après), `dpkg-deb -c` / `-f`, hors carte (`CUDA_VISIBLE_DEVICES=`), 01:44
scellé (ETAT du 19/09) : le test doit passer sur le paquet publié, sans verrou mesure
mesuré : `4 passed in 0.34s` ; `dpkg-deb -f Version` = 0.6.23 ; le blob de l'arbre manon = celui d'`origin/main` (edee0fa9)
verdict : **TENU** — rien à corriger ; remarque : le test lit un nom de paquet figé (`0.6.0`) : à chaque version il faut le copier à la main, ce qui est pourquoi il était « à faire » depuis le 19/09 — le paramétrer (`ACVRAM_DEB=` ou le plus récent `acvram_*_amd64.deb` de la racine) coûte trois lignes
suite : Jérôme : ETAT ; Océane ou moi à la reprise : `DEB` paramétrable ; ma file : capture du parc (en cours) → poste E

## Rejouable
`sed 's|acvram_0.6.0_amd64.deb|acvram_0.6.23_amd64.deb|' tests/test_paquet_charge_utile.py > tests/tmp_0623.py && CUDA_VISIBLE_DEVICES= <python> -m pytest tests/tmp_0623.py -q ; rm tests/tmp_0623.py` (1 s).
