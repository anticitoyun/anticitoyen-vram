# (1) Devstral-24B, KV bf16 : les morceaux tiennent 2 × le témoin reprise, mais la relecture int8 n'explique que la moitié de l'écart — le noyau GPU (longueur d'appel) porte l'autre (poste6, 01/10 08:00-08:03, main 1ab2fffa0)

instrument : `scratchpad/poste6-bf16.sh` → `prise-s1-morceaux-kv31b.sh` (`ACVRAM_KV_FORMAT=bf16`, REQUETES=2 sur A1, HEAD asserté, relevés) + `s1-morceaux-comparer.py`
commit : 1ab2fffa0 (origin/main : lic d'poste1 fusionné, KV bf16 budgété à 16 bits) — arbre figé (modifications 11e mises de côté par `git stash` le temps de la prise)
régime : 5090 seule (4436 et leann-core 386005 sur la 3080 Ti) ; Devstral-24B srcawq-nvfp4, ctx 10 240, kv=bf16, **640 blocs KV** (10 240 jetons par couche), 0 exilé, graphes on, invite de 7 865 jetons admise
scellé : `poste6-s1-g9m-bf16-scelle-01-10.md` § (1), D1-D6
mesuré : 3 prises (A1 avec requête rejouée, A2, B), 29-124 s chacune, 08:00:18-08:03:20 ; carte rendue 08:03:20
verdict : **D1 TENU** (640 blocs, comme poste1 l'a prédit : lic corrigé) ; **D2 TENU** (A1 = A2 au bit, 352 valeurs) ; **D3 FAUX** : le témoin reprise en bf16 vaut **0,022** à la position 0 et change l'argmax (quasi-égalité −3,19/−3,15), pas ≤ 0,002 — relire des K/V bf16 EXACTS pour une seule ligne déplace déjà les logits de 0,02 : c'est le noyau (paginé pour la reprise, flash pour le seul tenant), pas le format ; **D4 FAUX à la lettre** : B/A1 bf16 = **0,0215** à la position 0 (seuil 0,01 ; alarme > 0,02 → « le noyau GPU domine »), ids 32/32 identiques, Δmax 0,042 sur 32 pas, top-10 10/10 ; **D5** : 0,0215 ≤ 2 × 0,022 = 0,045 → critère REGLES § 4 **TENU** ; **D6 TENU** (3 min 2 s)
durée : 3 min 2 s de carte
provenance du code (bd jdp, chef 01/10) : `PYTHONPATH=$DEPOT` = travail/poste6-menus (1ab2fffa0) exporté par le script, non retiré par carte.sh (import vérifié depuis /tmp) ; pas de ligne de chemin dans le journal Devstral, mais `blocs KV : 640` et `kv=bf16` exigent le lic d'poste1 (37f955f84), absent de l'arbre principal 3f10000bd (325 blocs sinon) — code du worktree prouvé

## Chiffres (Devstral, même invite, int8 du S1 bis contre bf16 d'aujourd'hui)
| cache KV | témoin reprise (Δ pos 0 ; ids) | B/A1 (Δ pos 0 ; ids ; Δmax préfixe) | critère 2 × | cache servi |
|---|---|---|---|---|
| int8 (S1 bis, 07:05) | 0,004 ; div. 0 | 0,041 ; div. 0 | FAUX (0,041 > 0,008) | 9 712 |
| bf16 (08:03) | 0,022 ; div. 0 | 0,0215 ; 32/32 ; 0,042 | TENU (0,0215 ≤ 0,045) | 9 712 |

## Lecture
* Passer le cache en bf16 divise l'écart des morceaux par deux (0,041 → 0,0215) et rend les 32 ids : **la relecture quantifiée pèse la moitié**,
  l'autre moitié est le chemin — attention par morceaux (`cache.gather` + `attention(q_offset)`) contre un seul tenant flash, à longueurs
  d'appel différentes : exactement ce que la bissection lic a montré sur processeur (1 ulp par ligne, longueur-dépendant), ici sur 40 couches GPU.
* Le témoin reprise en bf16 (0,022) est 5 × celui en int8 (0,004) : surprenant, dit. Hypothèse la plus probable : en int8 la reprise lit les
  K/V par le noyau paginé int8 (`rope_kv` fusé + `paged_attention`, attention.py:402-413), en bf16 par un autre chemin (gather + attention
  à masque) — deux noyaux, deux arrondis ; la quasi-égalité −3,19/−3,15 de la position 0 amplifie tout en basculement d'argmax. À séparer
  par un bras « reprise int8 contre reprise bf16 sur une position sans quasi-égalité » si le chef veut ce chiffre.
* Le critère REGLES § 4 se tient en bf16 parce que le témoin y est grand, pas parce que les morceaux y sont petits : à prendre comme ce que
  la règle dit (équivalence AU témoin), pas comme « morceaux inoffensifs ».

## Reste
* `ACVRAM_PREFILL_MORCEAU` : opt-in ; critère tenu sur Devstral bf16 et gemma court int8 (g9m), non tenu sur Devstral int8 et gemma long —
  pas d'anneau tant que le chef n'a pas tranché entre « tenu dans 2 bras sur 4 » et un seuil absolu.
* Si on veut descendre les morceaux sous le témoin partout : réduire la différence de chemin (même noyau d'attention pour le seul tenant et
  le morceau, c'est-à-dire forcer le seul tenant en appels de longueur fixe — coûte en débit) ou accepter la règle telle quelle.
* Sorties : `scratchpad/bf16-devstral/`.
