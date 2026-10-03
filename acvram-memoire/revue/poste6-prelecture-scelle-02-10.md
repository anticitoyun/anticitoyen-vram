# B2 — prélecture tamponnée des fragments avant le chargement : scellé AVANT mesure et code (poste6, 02/10)

Ordre : chef 02/10, après `poste6-colibri-flux-experts-verdict-02-10.md` (B2 tranché ; conditions : réglage déclaré au
régime, défaut décidé sur la mesure ; test d'identité à l'octet avec et sans ; chargement à froid mesuré sans sudo —
`posix_fadvise DONTNEED` ; prédiction avant ; sans carte d'abord).

## Ce qui est connu (verdict 076-3, FN960, modèle témoin Coder-30B nvfp4, 16,8 Go en 5 fragments)

Notre chemin à froid 0,62 Go/s (`_ShardReader.get` → `safe_open().get_tensor`, vue du mmap, lue par défauts de page, un
fil, `read_ahead_kb` 128) ; à chaud 13,2 Go/s. Lectures tamponnées parallèles, UNE passe chacune : 1,46 (2 fils) / 1,20 (4)
/ 1,80 (8) Go/s — bruit non caractérisé. `fadvise WILLNEED` sur 4 Go : 0 % en cache — lu depuis : le noyau borne chaque
appel à une E/S (`max_sectors_kb` 256 Kio), il faut donc l'appeler par pas.

## Conception

Au début de `load_model`, des fils lisent les fragments du manifeste par régions disjointes (`os.pread`, tampon jeté) : le
cache de pages se remplit pendant que le chargeur avance. Aucun octet du chemin de chargement ne change (les tenseurs
viennent toujours de `safe_open`) : l'identité est par construction, et prouvée par test. Sauté si le fragment est déjà en
cache (`mincore`) ou si la RAM disponible ne peut pas le tenir. `ACVRAM_PRELECTURE` = nombre de fils (0 = jamais).
Variante à mesurer avant de choisir : `readahead(2)` par pas de 256 Kio (remplit le cache sans copie vers l'espace
utilisateur).

## Prédictions et seuils

| | prédit | faux si / seuil |
|---|---|---|
| P1 `pread` tamponné à froid, 8 Gio, médiane de 3 : 2 / 4 / 8 / 16 fils | 1,2-1,6 / 1,3-1,9 / 1,5-2,1 / ≤ 8 fils + 15 % | 8 fils < 1,3 Go/s |
| P2 `readahead(2)` par pas de 256 Kio, 1 et 4 fils, jusqu'à 99,9 % en cache | 2,5-3,2 Go/s (le plafond O_DIRECT) | < 1,8 : sans intérêt, `pread` reste |
| P3 lecteur d'acvram à froid, modèle entier (16,8 Go, chaque tenseur lu puis jeté) | sans : 25-29 s ; avec : **≤ 12 s** | défaut ACTIF si gain ≥ 1,5 × ET ≥ 5 s ; sinon défaut 0 |
| P4 à chaud (tout en cache), avec prélecture | surcoût ≤ 0,3 s (contrôle `mincore`, aucune lecture) | > 1 s |
| P5 identité | tous les tenseurs égaux à l'octet avec et sans ; sha256 des fragments inchangé | une différence |

Issues qui me gêneraient : P2 ne remplit pas le cache plus vite (reste `pread`) ; les 8 fils de prélecture ralentissent le
chargeur lui-même sur 8 cœurs logiques (le gain P3 tombe sous 1,5 ×) ; le contrôle `mincore` coûte plus qu'il ne rapporte.
Mesure sur carte (temps de chargement réel, copie vers la carte comprise) : à ma fenêtre, pas ici.
