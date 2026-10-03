# g6r stabilité, carte — gemma-4-31B à 65 536 : deux chargements consécutifs NOMINAL, 0/60 exilé, mêmes ids, pic sous la réserve — l'alternance a disparu

instrument : `scratchpad/poste6-g6r/carte-g6r-3.sh` → `outils/gpu/mesure/prise-s1-morceaux-kv31b.sh A1` (une prise par chargement, sous `carte.sh`), lecture à sec des lignes `[acvram]`, de `metrics.json` et des empreintes d'ids (texte généré jamais affiché)
commit : poste6-gemma-anneau d50d56912 (HEAD asserté par la prise, provenance `…/poste6-gemma-anneau/acvram/__init__.py`, arbre propre sous `acvram/` et `outils/`) ; code du correctif : 1568623dd ; origin/main fusionné (ee01f0272)
régime : RTX 5090 seule (15 Mio occupés au départ et à la fin, verrou vide avant), `acvram-gemma-4-31b-it-nvfp4-4sur6-vision-nvfp4`, kv int8, b=1, plafond 400 W ; carte 1 : llama-server permanent seul ; charge 1,5-1,8 ; fichier de chauffe sans excès avant le premier chargement
scellé : `poste6-g6r-stabilite-scelle-02-10.md` (seuil : même régime, mêmes ids, pic ≤ réservé, aux DEUX chargements ; poussé à 13 h 1x, avant le code)
mesuré : 2 chargements consécutifs à 65 536, 2 requêtes chacun (invite de 61 942 jetons), 12 min 12 de carte
verdict : **TENU, les trois conditions aux deux chargements** — NOMINAL 0/60 et graphes actifs aux deux ; sha des 32 ids 5aabb6d5… aux deux (et aux deux requêtes de chacun, comme les prises NOMINAL de midi) ; pic 4,09 Gio ≤ réserve 4,69 puis 4,31 Gio.
durée : 13:40:27 → 13:52:39, 12 min 12 de carte sur 12 accordées

## Mesuré

| chargement | réserve de chauffe appliquée (ligne de régime) | plan | MLP exilés | régime | chauffe | pic / formule | réserve de préfill du plan | décodage | 32 ids (sha) |
|---|---|---|---|---|---|---|---|---|---|
| 3a (13:40) | `formule` | anneau R=67, table hôte, réserve complète | **0/60** | **NOMINAL**, 6 captures, 76 rejeux | 65 536 tenus, 197,0 s | 4,09 / 3,35 Gio → +11 Kio/jeton | 3,35 + 0,22 + 1,12 = **4,69 Gio** | 30,91 j/s | 5aabb6d5… (requêtes 1 et 2) |
| 3b (13:46) | `+11Kio/jeton(2026-10-02T13:44:01)` | anneau R=67, table hôte, **tampons denses non comptés** | **0/60** | **NOMINAL**, 6 captures, 76 rejeux | 65 536 tenus, 197,8 s | 4,09 / 3,35 Gio → +11 Kio/jeton | 3,35 + 0,74 + 0,22 = **4,31 Gio** | 30,97 j/s | 5aabb6d5… (requêtes 1 et 2) |

À midi (e3c5d15b7, sans le correctif) : 2a NOMINAL, 2b **2/60 exilés, DÉGRADÉ**, ids 47fea92e….
Chaque requête : 47 s (préfill de 61 942 jetons + 32 jetons), aux deux chargements ; 4 263 Mio libres après la chauffe aux deux.

## Lecture

* Le second chargement applique bien l'excès (ligne de régime), et le plan le loge sans exil parce que la réserve des
  tampons denses (1,12 Gio, « le pool n'existera pas ») n'est plus comptée quand aucun poids dense n'est exilé.
* Le troisième chargement est déjà décrit par le second : 3b enregistre le même excès que 3a (11 Kio/jeton), donc le même
  plan — point fixe, plus d'alternance.
* **« pic ≤ réservé »** : la réserve du plan n'est pas imprimée telle quelle ; je l'ai recomposée des lignes du journal
  (activations 3,35 Gio au plafond 5 120, excès 0,74) et du rejeu à sec du matin (termes Marlin 0,22, tampons 1,12). Au
  second chargement elle vaut le pic à la division entière près, plus les termes Marlin : 0,22 Gio de marge seulement.
  L'issue que je redoutais au scellé (réserve 3,81 < pic) ne s'est pas produite : le plan a gardé le plafond 5 120.
* **Écart au rejeu à sec, dit** : il prédisait un plafond de 6 144 au second chargement ; la carte a gardé 5 120 (la VRAM
  libre réelle n'est pas celle de ma simulation). Le régime, lui, est celui prédit.
* **Défaut de journal trouvé, non corrigé ici** : la ligne « réserve de préfill calée sur la chauffe » n'apparaît plus dans
  le journal du service — elle est imprimée une fois, dans la première chaîne de plan, dont la sortie est jetée quand un
  autre plan est retenu. L'information reste sur la ligne de régime (`reserve_chauffe=`).

## Reste

* Fusion par chef : l'arbre mesuré est d50d56912 (+ ce verdict). Suite sans carte complète non jouée depuis le correctif
  (lot ciblé : 49 verts à 13:18 ; lot élargi 821 verts à 13:05 avant le correctif, 2 échecs `test_menus` dus au disque).
* La formule de réserve sous-estime toujours le chemin Marlin résident de 0,74 Gio à 65 536 : c'est la chauffe qui la
  rattrape au second chargement. À corriger à la source (pièce à part) plutôt que de s'en remettre à l'excès mesuré.
* Non expliqué depuis midi : le DÉGRADÉ à 2 MLP exilés décodait à 35 j/s (62 jetons), plus vite que le NOMINAL (31).
* G6' (débit à ± 1 % table hôte / carte, b=1 et b=12) : instrument de cellule, poste2.
