# 7gb + ked — carte.sh récolte et nomme tout descendant qui survit à la prise ; les tests carte récoltent les leurs : verdict (poste5, 30/09)

* instrument : `tests/test_carte_descendants_7gb.py` (4) + fixture `recolte_carte` (tests/conftest.py) posée sur les 14 fichiers qui lancent de vrais carte.sh ; verrous isolés (ACVRAM_VERROU sous tmp_path), processus `sleep`, aucune carte ; détecteur de restes : tout processus vivant dont ACVRAM_VERROU est sous /tmp/pytest-of-*
* commit : poste5-7gb (voir git log)
* régime : à sec, aucune prise réelle
* scellé : (a) sans `_reaper_descendants`, les 2 tests de descendant rougissent (vérifié ×2) ; (b) la faute ked réintroduite (`popen.terminate()` au lieu du groupe) fait rougir PAR LA RÉCOLTE exactement les deux tests nommés dans ked (test_deux_mesures_ne_coexistent_pas, test_partage_refuse_pendant_mesure) ; (c) un processus étranger non marqué et un service détaché restent intacts
* mesuré : famille carte + sortir_version_281 + kv_puits : 61 verts, 8 sautés, 0 reste après la suite ; coût d'une prise mesure à vide 0,32 → 0,78 s (un seul grep sur les environ ; la version un-grep-par-processus coûtait 2,3 s)
* verdict : VRAI
* durée : 0 (à sec)

## 7gb
Le reaper c7w (`_reaper_setsid_orphelins`) ne visait que les descendants marqués DÉJÀ présents dans compute-apps à
la sortie : un descendant pas encore sur la carte (contexte CUDA ouvert plus tard — le PID 248433 du 27/09 est né
juste après la prise) lui échappait. `_reaper_descendants` (outils/carte.sh), appelé par le trap EXIT des classes
mesure/état et partage après c7w : tout processus vivant marqué ACVRAM_CARTE_TENUE=<la prise>, né après elle (champ
22 de /proc/<pid>/stat) et qui n'est plus sous elle (les aides du trap le sont encore), est NOMMÉ — journal
« ORPHELIN … descendant pid P (cmd) TERM (7gb) », stderr — puis TERM, KILL après 5 s. Service non visé (il rend la
main avant ce trap, par contrat). Au passage : le releveur de charge laissait son `sleep 1` orphelin (tué sans ses
enfants) → `sleep & wait` + trap TERM ; sinon chaque prise nommait ce faux orphelin.
Note : la note d'poste1 du 27/09 (« repli eager reproduit sans PID hors verrou ») reste vraie — ce correctif ferme la
fuite de processus, pas nécessairement la cause du repli eager de la 284c.

## ked
État au 30/09 : la 291 (28/09) avait déjà remplacé `.terminate()` par la mise à mort du groupe ; 0 reste mesuré sur
les 65 tests carte avant cette pièce. Restaient : `pkill -f "sleep 10"` (tuait tout « sleep 10 » du POSTE, d'une autre
session comprise) et un service tué par son seul PID → `_tuer_service` (groupe setsid). Et surtout aucun garde-fou :
`recolte_carte` fait désormais ÉCHOUER tout test carte qui laisse un processus vivant (délai de grâce 3 s, guetteur de
service récolté sans échec : il sort seul dans les 5 s et ne tient rien), et le tue.
