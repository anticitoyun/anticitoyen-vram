# Chiffrage — VRAM de `MAX_GRAPHS` 16→64 (item A9, audit poste7)

poste3, 13/09/2026, à sec (en attente de carte, occupée par poste4). Suite à
`poste7-audit-connaissances-14-09.md` item A9 : `MAX_GRAPHS` 16→64 réduit la
dispersion (65,11 % → 20,37 %, `poste4.md:4040-4082`) mais son coût VRAM
n'a jamais été chiffré. Critère de poste7 : `< 25 % de dispersion à ≤ +1 Gio
→ adopter 64`.

## 1. Réserve à poser avant de calculer quoi que ce soit

`REGLES.md:123` : « la dispersion elle-même disperse : 31,74 puis 65,11 %
sur **le même bras** ». Le chiffre `65,11 %` qui motive tout l'item A9
**n'est lui-même pas reproductible d'un tour à l'autre** — une seconde
mesure du même réglage (`MAX_GRAPHS=16`) a donné 31,74 %, pas 65,11 %.
**Le gain de dispersion à démontrer est donc lui-même incertain d'un
facteur ~2 avant même de toucher à `MAX_GRAPHS`.** Ce chiffrage porte sur
le COÛT (VRAM), qui n'a pas ce problème ; le BÉNÉFICE (dispersion) devrait
être confirmé par la manche de 1 h / 10 modèles déjà prévue par poste7, pas
supposé acquis sur la seule mesure du 10/09.

## 2. Ce que `graphs.py` alloue réellement par graphe — lu, pas supposé

`graphs.py:493-582` (`_capture`) : **deux catégories de mémoire bien
différentes**, à ne pas confondre.

**(a) Les tampons d'entrée (`entry`), un jeu PAR CLÉ CAPTURÉE** —
`graphs.py:498-506` :

    x          [b*ql, hidden]     dtype modèle (bf16 = 2 octets)
    positions  [b*ql]             int64 (8 octets)
    slots      [b*ql]             int64
    tables     [b, nblk]          int64
    seq_lens   [b]                int64

Ces tenseurs sont alloués **avant** `torch.cuda.graph(...)` (lignes 501-506,
574-579) : ils vivent **hors du pool partagé de graphe**, un jeu séparé par
clé, jamais libéré tant que la clé reste captée.

**(b) Les activations internes du graphe capturé** — allouées **dans**
`torch.cuda.graph(graph, pool=self._pool)` (`graphs.py:571-579).
`self._pool` est **UN SEUL POOL PARTAGÉ PAR TOUS LES GRAPHES** (assigné une
fois à la première capture, ligne 576, réutilisé pour toutes les suivantes,
ligne 578). **PyTorch ne multiplie pas ce pool par le nombre de graphes** :
les allocations internes de deux graphes qui ne s'exécutent jamais en même
temps (jamais deux `replay()` concurrents) partagent le même espace — le
pool grandit jusqu'à la plus grande empreinte d'activation rencontrée parmi
TOUTES les formes capturées, pas la somme des 64.

**Conséquence directe : le coût de 48 clés supplémentaires (16→64) est
dominé par (a), pas par (b).** (b) ne grandit que si une forme plus grande
qu'aucune vue jusque-là est capturée — ce qui peut arriver aussi bien à la
17e clé qu'à la 64e, indépendamment du plafond.

## 3. Calcul de (a) — le coût qui scale avec le nombre de clés

Modèle de référence : GLM-4.7-Grande-Heretic-42B (`hidden_size=2048`,
lu dans `config.json`), `BLOCK_SIZE=16` (`kvcache.py:33`), `b≤12`
(`ACVRAM_HYBRID_SLOTS`), `nblk≤128` (ctx 2048 / 16).

    x          12 x 1 x 2048 x 2 octets      =  49 152 octets  (48 KiB)
    positions  12 x 8                        =      96 octets
    slots      12 x 8                        =      96 octets
    tables     12 x 128 x 8                  =  12 288 octets  (12 KiB)
    seq_lens   12 x 8                        =      96 octets
    -----------------------------------------------------------
    total par clé (hors "out")               ≈  61 728 octets  (60,3 KiB)

`entry["out"]` (sortie de `decode_fixed`, capturée ligne 579) : si ce sont
des logits `[b*ql, vocab_size]` plutôt qu'un état caché, le poste dominant
possible — à vérifier sur le modèle réel (`vocab_size` GLM non relu ici,
mais même à 150 000 et fp32 : `12 x 150000 x 4 = 7,2 Mio` par clé, pire cas
plausible).

**Pour 48 clés supplémentaires (16→64)** :

    hypothèse basse (out ≈ hidden state, pas logits) :  48 x 60,3 KiB  ≈ 2,8 Mio
    hypothèse haute (out = logits complets, pire cas) : 48 x 7,2 Mio   ≈ 346 Mio

**Dans les deux cas, très largement sous le seuil de 1 Gio de poste7** — y
compris dans l'hypothèse la plus pessimiste envisagée (par un facteur ≥ 3).

## 4. Ce que ce calcul NE couvre pas — et pourquoi ça ne change pas la conclusion

- **Le tampon MTP** (`m.reserver_hidden`, `graphs.py:569-570`), s'il existe,
  est dimensionné à `b*ql` **une fois**, pas par clé — ne scale pas avec
  `MAX_GRAPHS`.
- **Les caches RoPE** (`mod.reserver(...)`, `graphs.py:516-518`) sont
  dimensionnés à `max_model_len`, **partagés entre toutes les couches et
  tous les graphes** — ne scale pas non plus.
- **Le pool partagé (b) lui-même** pourrait grandir si une des 48 clés
  supplémentaires capture une forme (b, ql, nblk) plus grande qu'aucune vue
  à 16 — possible, mais borné par le même espace de formes qu'à 16
  (`bucket_batch`/`bucket_blocks` produisent le même ensemble de valeurs
  discrètes quel que soit le plafond ; ce n'est pas le plafond qui invente
  de nouvelles tailles, c'est le TRAFIC RÉEL qui décide quelles clés
  apparaissent). **Le risque existe mais n'est pas structurel à
  `MAX_GRAPHS`** — il existerait pareillement à 16 si le trafic avait
  d'abord présenté les grandes formes.

## 5. Proposition

**Chiffrage arithmétique (§3) : le coût VRAM de `MAX_GRAPHS=64` est de
l'ordre de quelques mégaoctets à quelques centaines de mégaoctets, pas de
gigaoctets** — largement sous le seuil de 1 Gio fixé par poste7, dans
l'hypothèse basse comme dans l'hypothèse haute. **Ce chiffrage n'est PAS
une mesure** (`torch.cuda.memory_reserved()` avant/après capture de 64
graphes vs 16, sur carte, reste à faire — 10 min, pas l'heure complète déjà
prévue puisque seul le VOLET VRAM est visé ici).

**Ce qui reste à faire avant d'adopter 64 par défaut** :
1. Confirmer §3 par une mesure directe (`memory_reserved`) — rapide, ne
   nécessite pas les 10 modèles.
2. **Refaire la mesure de dispersion** (pas seulement la relire) — la
   réserve du §1 dit qu'un seul tour à 65,11 % ne suffit pas à établir le
   bénéfice, quel que soit le coût.

**Ma recommandation, sous réserve du point 2** : le coût VRAM ne s'oppose
pas à `MAX_GRAPHS=64` par défaut — si la dispersion se confirme sous 25 %
sur plusieurs tours (pas un seul), rien dans ce chiffrage ne justifie de
garder 16.
