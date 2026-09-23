# colibrì — lecture de code : taux de succès, cache apprenant, double banque (Océane, 13/09/2026)

Lu (`externes/colibri/c/`, fd93c41) : `expert_store.h` (103 l.), `route_trace.h`
(427 l.), `deepseek_v4_bank_pair.h` (52 l.), et les points d'appel dans
`colibri.c`/`deepseek_v4.c`/`tier.h` que `revue/colibri-hierarchie-experts-disque.md`
désignait sans les citer.

## Comment ils comptent le taux de succès d'experts

Trois compteurs cumulatifs globaux sur `Model` : `hits`, `miss`, et leur
ventilation par tier (`hit_vk`, `hit_pin`, `hit_ecache`). Ils s'incrémentent
**au point de lookup**, dans la boucle qui résout chaque expert unique demandé
par le batch courant (`colibri.c:5569-5586`) :

```c
if (registre VRAM le sert)      { m->hits++; m->hit_vk++;    continue; }
use[j] = pin_indexed(...);       if (use[j]) { m->hits++; m->hit_pin++; }
if (!use[j]) use[j] = ecache_indexed(...); if (use[j]) { m->hits++; m->hit_ecache++; }
if (!use[j]) { /* miss : va au disque */ m->miss++; }
```

Trois tiers testés dans l'ordre (registre VRAM → pin RAM → cache LRU RAM),
premier qui répond gagne — c'est un cache à niveaux, pas un booléen.

**Le taux par requête** n'est PAS un compteur séparé : c'est un **instantané
avant/après** des compteurs globaux. À l'entrée d'une requête (`colibri.c:8880`) :
`r->hits0 = m->hits; r->miss0 = m->miss;` — deux lectures d'entiers. À la sortie,
la différence `m->hits - r->hits0` sur `m->hits - r->hits0 + m->miss - r->miss0`
donne le taux de la requête (`colibri.c:8667`, motif répété à 9132-9223). Coût :
zéro allocation, deux soustractions ; le compteur cumulatif, lui, coûte un
incrément entier **par expert unique du batch**, déjà sur le chemin qui teste
sa présence — pas un coût ajouté, un comptage de ce qui se fait déjà.

`ColiExpertStoreStats` (`expert_store.h:29-38`) généralise ce schéma pour le
backend V4 découplé : `requests/hits/misses/prefetched/prefetch_hits/
bytes_read/resident_bytes/capacity_bytes` — un `stats()` de la vtable le
renvoie à la demande, jamais poussé.

## Comment le cache apprenant décide

Deux mécanismes séparés, pas un seul :

1. **`.coli_usage`** (`route_trace.h`) : un histogramme **persistant, jamais
   remis à zéro entre lancements**, `uint32_t rt_c[couche][expert]`, incrémenté
   par `rt_count()` à chaque sélection de routeur — un compteur pur, distinct de
   la trace textuelle optionnelle (`rt_trace()`, sous `ROUTE_TRACE=`). Sauvegardé
   en texte creux (`"<couche> <expert> <compte>"`, lignes non nulles seulement),
   avec deux enregistrements d'en-tête (dimensions, identité du moteur qui a
   écrit) compatibles avec les lecteurs plus anciens qui les ignorent (`l>=0`
   filtre). `COLI_USAGE_DECAY` (défaut 1,0 = pas de décroissance) permet un
   demi-vie en tours pour que le classement suive un changement de charge —
   mesuré : 18,2 M sélections accumulées, un tour n'en bouge le classement que
   de 0,2 %, donc **sans décroissance l'historique se fige**.
2. **Re-pin à chaud** (`repin_pick`, `colibri.c:8271-8304`, opt-in `REPIN=n`) :
   entre deux tours, au point sûr (aucun MoE en vol), échange les pins les
   plus froids contre les non-pinnés les plus chauds d'une **carte de chaleur
   séparée** (`m->eheat`, décroissante à chaque passe) qui laisse `.coli_usage`
   intact. La décision élémentaire est `tier_pick_lfru` (`tier.h:40-64`) :

   ```c
   score(e) = (heat[e] << 8) | min(255, 255 - (horloge - dernier_acces[e]))
   ```

   fréquence prime (poids 256), la récence ne fait que départager (poids ≤ 255)
   — *« a merely recent expert cannot displace a genuinely hotter one »*.
   Hystérésis **25 % + 4** avant tout échange (`hs <= cs + cs/4 + 4·256`) pour
   éviter le ping-pong ; max 4 échanges par passe (~20 Mio de disque chacun).
   Le compteur `.coli_usage` sert à l'AUTOPIN de démarrage (avant toute mesure
   en ligne) ; `m->eheat`/`repin_pick` prend le relais pendant l'exécution.

## Comment la double banque recouvre le transfert

`deepseek_v4_bank_pair.h` — deux fonctions pures, testables sans GPU (c'est le
point du fichier : « the sequencing — where double-buffer bugs actually live —
is unit-tested on machines with no GPU »). Pendant que le GPU calcule la couche
`L` depuis la banque active, un thread transfère la **couche `L+1` ENTIÈRE**
(tous ses experts, pas seulement ceux que le routeur choisira — le routage de
`L+1` n'est pas encore connu) dans l'autre banque, sur le flux auxiliaire du
device. Au changement de couche :

```c
coli_v4_bank_pair_decide(double_on, other_layer, incoming_layer)
  -> V4_BANK_SWAP   si l'autre banque a EXACTEMENT la couche qui arrive
  -> V4_BANK_LEGACY sinon (retour au remplissage à la demande, route-aware)
```

Un préchargement partiel aide quand même : l'échange porte la carte de
validité par expert, et le remplissage à la demande ne comble que les trous.
`coli_v4_bank_pair_prefetch_target` ne précharge jamais que `L+1` (jamais
plus loin) — un redémarrage de segment revient à la couche 0, absorbé par le
test de correspondance ci-dessus, pas par une logique séparée.

## Le premier compteur à poser chez nous

**Ce qui existe déjà chez nous et joue le rôle de `route_trace.h` : rien de
comparable pour le COMPTE cumulatif, mais `acvram/memory/trace_routage.py`
existe pour la TRACE textuelle** — une ligne par (jeton, couche, experts
choisis), désactivée par défaut (`ACVRAM_TRACE_ROUTAGE=`), un test de booléen
quand éteinte, jamais de synchronisation implicite en service (le module le
dit explicitement : « Elle ne doit jamais provoquer de synchronisation
implicite »). C'est le bon voisinage, mais ce n'est PAS un compteur cumulatif :
colibrì sépare exprès `rt_count()` (comptage, toujours actif, pas cher) de
`rt_trace()` (trace textuelle, opt-in, plus chère) — chez nous les deux sont
confondus dans un seul mécanisme opt-in.

**Où** : `MoEBlock._route()`, `acvram/engine/model.py:827-858`, juste après
`topi` (indices des `top_k` experts par jeton, ligne 836/844/848 selon le
chemin). C'est l'unique point qui voit tous les chemins (noyau fusionné,
`torch.topk` sigmoid, `torch.topk` softmax) avant qu'ils divergent en aval.

**Comment, sans reproduire le coût de la trace** : `topi` est un tenseur CUDA
(`[t, top_k]`). Un `.tolist()` synchronise — c'est justement ce que
`trace_routage.py` documente comme le coût à éviter. La bonne forme : un
histogramme **sur device**, `torch.bincount(topi.flatten(), minlength=n_experts)`
accumulé dans un tenseur persistant par couche (`self._usage: Tensor` sur
`x.device`, alloué au premier appel) — zéro `.item()`/`.cpu()` avant la fin de
la requête. Coût : un scan de `t · top_k` éléments, déjà lus par le noyau de
routage qui vient de les produire (`ext.moe_route` ou `torch.topk`) — négligeable
devant le GEMM du MLP qui suit dans la même couche, du même ordre que l'analyse
faite pour `_trace_routage.noter()` (« un test de booléen par appel » devient
« un bincount par appel », toujours sans synchronisation).

**Taux de succès par requête** : comme colibrì, un instantané avant/après —
mais chez nous il n'y a pas encore de tier de résidence PAR EXPERT (nos
`LayerPlacement.mlp_storage` est par COUCHE entière, pas par expert : geste 1
de jt5 raisonne au même grain). Le compteur `_usage` par (couche, expert) est
donc la première brique nécessaire à toute décision par-expert future — cache
apprenant, double banque colibrì-style — mais il ne calcule PAS encore de
« hit » tant qu'aucun ensemble résident par expert n'existe côté acvram pour
s'y comparer. **Ce que ce geste permet dès maintenant, sans cache par expert** :
mesurer, sur Qwen3-Coder-30B ou tout MoE chargé, la **concentration réelle** du
routage — combien d'experts distincts couvrent 80 % des sélections par couche —
le chiffre que la revue *colibri-hierarchie-experts-disque.md* réclamait et que
nous n'avons pas. C'est la mesure qui dirait si un cache par expert vaut le
détour chez nous avant d'écrire un seul octet de mécanisme.

## Ce qui ne se transpose pas tel quel

Le format texte de `.coli_usage` (lecture C, trois formats historiques
compatibles) n'a pas d'équivalent à créer : nous avons déjà un format à nous
(`trace_routage.py`) et pas de contrainte de compatibilité inter-version C.
`tier_pick_lfru` est directement portable en Python le jour où un cache par
expert existe — c'est une fonction pure sur trois entiers par candidat, aucune
dépendance à l'ABI colibrì.
