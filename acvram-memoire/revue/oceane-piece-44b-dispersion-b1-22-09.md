# Pièce 44 (b) — dispersion de 35 % entre sondes b=1 de 128 jetons : cinq hypothèses côté moteur, leur test départageant, et ce que la FORME de la série tranche (22/09, Océane, à sec)

Rien ici n est mesuré : ce sont des mécanismes lus dans le code, chacun avec le
chiffre qu il implique et le contrôle qui le rend faux. Les compteurs dont ils
ont tous besoin existent déjà : `GraphRunner.captures` et `.replays`
(`engine/graphs.py:292-293`), plus `len(runner.graphs)` (clés vivantes).
**Premier geste, commun aux cinq** : relever ces trois nombres AVANT et APRÈS
chaque sonde, à côté du temps. Une sonde qui ne capture rien et n en garde pas
moins 35 % d écart élimine H1, H2 et H5 d un coup.

## H1 — franchissement d un godet de blocs PENDANT la sonde (le plus probable)
Mécanisme : la clé de graphe est `(b, ql, nblk, lb)` (`graphs.py:602`) où
`nblk = bucket_blocks(blocs)` — puissances de deux **à partir de 8**
(`memory/kvcache.py:39-49`, `BLOCK_SIZE = 16`). Une sonde de 128 jetons de
sortie sur une invite de P jetons parcourt `⌈(P + t)/16⌉` blocs : elle franchit
un godet quand P + t passe 128, 256, 512… À chaque franchissement, clé neuve →
**capture** (`graphs.py:619-626`), dont le coût est hors du pas mesuré mais
tombe DEDANS.
Chiffré : une capture de décodage coûte ~10² ms ; répartie sur 128 jetons à
~1,3 ms/jeton (b=1), une seule capture au milieu d une sonde ajoute 30 à 60 %
au temps total de cette sonde — l ordre de grandeur observé.
Test : compter les captures par sonde (ci-dessus) ; et rejouer les mêmes sondes
avec des invites **calées** pour que P + 128 reste dans un seul godet (par
exemple P tel que P + 128 ≤ 128, ou ≥ 256 et P + 128 < 512). Réfuté si la
dispersion survit à godet constant et captures = 0.

## H2 — plafond `MAX_GRAPHS = 16` atteint, repli eager DURABLE
Mécanisme : `graphs.py:123` et `:613-617` — au-delà de 16 clés vivantes, toute
clé neuve **ne capture plus** et le pas part en eager, sans que rien ne le
rende au rejeu suivant. Une série de sondes qui varient invite et longueur
fabrique des clés à la chaîne.
Chiffré : eager contre graphe à b=1, c est le trou hôte par pas (levier 1 :
165 µs avant, 24,5 après) plus la préparation — de l ordre de +15 à +40 %,
**permanent** à partir de la sonde qui franchit le plafond.
Test : `len(runner.graphs)` par sonde, et la ligne `[graphe] limite 16
atteinte` qui est déjà imprimée sous trace. Réfuté si le plafond n est jamais
atteint (≤ 16 clés sur toute la série).

## H3 — rapatriement épinglé (levier 2)
Mécanisme : `engine/pipeline.py:74-96` — un tampon par (parité, n), réalloué
seulement si `n` change. **À b=1, n vaut 1 pour toute la série** : deux
allocations épinglées au tout premier pas, plus jamais ensuite.
Chiffré : ≤ 2 `pin_memory()` sur la série entière, soit un surcoût sur la
PREMIÈRE sonde seulement, invisible ailleurs.
Test : la dispersion persiste-t-elle sous `ACVRAM_RAPATRIEMENT=flux` (témoin) ?
Si oui, H3 est hors de cause. C est l hypothèse la moins coûteuse à écarter, et
celle que je tiens pour la moins probable.

## H4 — travail hôte concurrent (le verrou n est pas une garantie d exclusivité)
Mécanisme : le verrou `outils/carte.sh` protège la CARTE, pas les cœurs. À b=1
le pas est dominé par l hôte (lancement, tokeniseur, asyncio : ~0,6 ms/pas
mesuré le 22/09, `verdict-m2`), donc une conversion AWQ ou un pytest qui tourne
à côté déplace le pas sans toucher au GPU.
Chiffré : une charge hôte à 100 % d un cœur suffit à ajouter 20-40 % au temps
d un pas dominé par l hôte, et elle varie d une sonde à l autre.
Test : relever `os.getloadavg()` et l occupation GPU (`nvidia-smi
--query-compute-apps`) au début et à la fin de CHAQUE sonde, et publier la
série à côté des temps. Réfuté si la charge est plate pendant que les temps
dispersent. (Je suis moi-même une source de ce bruit aujourd hui : ma suite
complète de 5 min a tourné pendant la prise — raison de plus pour que la sonde
le mesure au lieu de le supposer.)

## H5 — allocateur de blocs KV et fragmentation
Mécanisme : `BlockAllocator.allocate/free` (`memory/kvcache.py:105-140`) sert
des listes de blocs ; une sonde qui libère puis reprend ses blocs obtient des
indices différents, donc des `block_tables` différentes — jamais une forme
différente, mais une localité différente à la lecture du cache KV.
Chiffré : effet de cache, quelques pour cent au plus. Il ne produit PAS 35 % à
lui seul ; il peut en revanche moduler H1 (un cache plus dispersé rend chaque
pas un peu plus lent).
Test : refaire la série sans libérer entre sondes (une seule séquence longue
découpée), et comparer la dispersion. Réfuté si elle est inchangée.

## Ce que la forme de la série de Manon implique
| forme | lecture | hypothèses compatibles | hypothèses éliminées |
|---|---|---|---|
| **décroissante puis plateau** (les premières sondes lentes) | les captures se paient une fois, les graphes servent ensuite | H1 | H2 (qui dégrade au contraire), H4 (sans raison de décroître) |
| **croissante, sans retour** | quelque chose s accumule : clés de graphe jusqu au plafond, fragmentation | **H2**, H5 | H1 (une capture ne revient pas), H3 |
| **sans tendance, dispersion large** | bruit extérieur au moteur | **H4** | H1, H2 (toutes deux ont une direction) |
| **périodique / bimodale** (une sonde sur k lente) | un franchissement de godet revient à période fixe — c est la signature exacte de H1 avec des invites de longueurs régulières | **H1** | H4 (qui n a pas de période) |
| dispersion qui **disparaît** sous `ACVRAM_DISABLE_CUDA_GRAPHS=1` | le coût est dans la capture ou le rejeu | H1, H2 | H3, H4, H5 |

## Ordre proposé (coût croissant, chacun peut clore)
1. compteurs `captures` / `replays` / `len(graphs)` par sonde — gratuit, tranche H1 et H2 ;
2. charge hôte relevée par sonde — gratuit, tranche H4 ;
3. invites calées à godet constant — une série de plus, tranche H1 ;
4. `ACVRAM_RAPATRIEMENT=flux` et série sans libération — deux séries, closent H3 et H5.
Si les quatre passent sans expliquer 35 %, l hypothèse suivante n est plus dans
le moteur mais dans la mesure elle-même (horloge, chauffe, fréquence GPU) — et
c est alors `nvidia-smi --query-gpu=clocks.sm,temperature.gpu` qu il faut
relever par sonde, ce qu aucune des cinq ne couvre.
