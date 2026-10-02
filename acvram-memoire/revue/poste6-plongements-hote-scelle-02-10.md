# Levier plongements : la table bf16 quitte la carte avant tout MLP — scellé à sec AVANT le code (poste6, 02/10, poste6-gemma-anneau 5f1bec7ae + carnet)

Ordre : chef 02/10 (« plongements en RAM hôte : oui, scellé avant le code » ; deux points à vérifier d'abord). Suite de g6r
(`poste6-g6r-scelle-02-10.md`) : à 65 536 sur gemma-4-31B il manque 1,95 Gio après que le KV a cédé jusqu'à son plancher.

## Les deux points de chef, vérifiés à sec

1. **La tête liée relit-elle la table bf16 ?** Non. `loader.py:1029` : `lm_head = QuantLinear(_tete_liee(embed.to(head_dev)))` ;
   `_tete_liee` (`loader.py:1161`, défaut `ACVRAM_TETE_LIEE=int8`) quantifie par tranches et rend des tenseurs NEUFS — la tête a sa
   copie int8 (1,34 Gio), la table ne sert qu'au gather d'entrée. Table en RAM hôte : `embed.to(head_dev)` fait une copie
   transitoire sur la carte, rendue après la quantification. **Repli nommé** : si `libre < 2 × besoin` (2,8 Go) à cet instant,
   `_tete_liee` rend la copie bf16 elle-même (2,62 Gio gardés sur la carte, `loader.py:1175-1179`, ligne « tête liée laissée en
   bf16 ») — le levier ne gagnerait alors rien ; à cet instant les caches KV ne sont pas encore alloués (≈ 14 Gio libres).
2. **Coût du gather hôte par pas** (mesuré à sec, 2 cœurs, nice 19, table 262 144 × 5 376 bf16) : **3,1 µs à b=1, 10,6 µs à
   b=12** ; à transférer 10,7 Kio / 126 Kio (0,5 / 6 µs à 21 Go/s). Le nombre de transferts par pas ne change pas : aujourd'hui
   les indices montent sur la carte, demain ce sont les lignes. Morceau de préfill de 4 096 jetons : 5,3 ms de gather, 42 Mio.
   Prédit : **≤ 20 µs par pas, soit ≤ 0,15 % d'un pas de 15,7 ms** ; le coût fixe du transfert n'est pas mesurable à sec.

## Ce que la lecture a trouvé d'autre (fichier:ligne)

* `_reajuster_plan` (`loader.py:1681`) n'exile que des MLP ; la table (2,62 Gio) est placée avant eux (`tiering.py:590`).
* Quatre endroits prennent `embed_tokens.device` pour « l'appareil du modèle » : tour de vision (`runner.py:762` — table en RAM
  hôte = tour sur le processeur), chemin épinglé (`runner.py:2095`), **chauffe** (`contexte.py:103` et `:163` : hors carte,
  `_libre_apres_chauffe` rend (2^40, 2^40) — la chauffe ne juge PLUS la réserve et déclare tout tenu). Défaut latent pour tout
  modèle déjà servi avec `embed_device: cpu`.
* Ligne de régime : NOMINAL exige `len(cartes) ≤ 1` et `cartes` contient `plan.embed_device` (`runner.py:934`, `:1112`) : une
  table en RAM hôte ferait DÉGRADÉ à elle seule.
* `speculative.py:561` et `:588` : `e.to(hs.dtype)` ne change pas d'appareil (tête de brouillon contre table hôte).

## Correctif

* `_reajuster_plan(embed_exilable=…)` : à la planification seulement, après que le KV a cédé et AVANT le premier MLP, la table
  passe en RAM hôte si le plan déborde encore. `ACVRAM_EMBED` = auto (défaut) | hote (toujours, bras de mesure) | carte (jamais).
* `_plafonner_mlp_prefill` compte la table exilée comme un exil : le plafond de préfill reste préféré (plans qui tenaient :
  identiques).
* `ACVRamModel.appareil` (celui de la première couche) remplace `embed_tokens.device` aux quatre endroits ; `cartes` ne compte
  que des cartes ; la ligne de régime dit `plongements=hôte`.

## Prédictions

| | prédit | faux si |
|---|---|---|
| E1 à sec, 65 536 | table en RAM hôte, **0 MLP exilé**, anneau R=67, KV 5,45 Gio, ≥ 4 096 blocs, pas de refus | ≥ 1 MLP exilé, refus |
| E2 à sec, témoins 4 096 / 8 192 / 16 384 / 27 648 / 41 984 | table sur la carte, résumés IDENTIQUES à g6r (5f1bec7ae) | un champ diffère |
| E3 à sec, 46 080 (2 exilés sous g6r) | table en RAM hôte, 0 exilé | ≥ 1 exilé |
| E4 à sec, plus grande fenêtre sans MLP exilé | entre 67 584 et 75 776 | hors intervalle |
| E5 jouet (CI) | table hôte / table carte : mêmes ids et logits ; un plan qui exilait 1 MLP n'exile plus que la table | divergence |
| G1' carte, 65 536 | 65 536 tenus, 0 exilé, graphes actifs, **NOMINAL** | DÉGRADÉ, clamp |
| **G5' carte, invite de ~62 k jetons** | **≥ 20 j/s** (attendu 30-55 : 10 couches pleines lisent 62 k clés) | < 20 j/s |
| G6' carte, A/B `ACVRAM_EMBED=hote` contre `carte`, 4 096, b=1 et b=12 | débit à ± 1 % (2 σ du témoin si plus large), ids identiques (sha) | > 1 % plus lent, ids différents |

Issues qui me gêneraient, nommées : (i) la capture des graphes à 65 536 manque de mémoire (marge 1,59 Gio) → clamp ou graphes
off, G1' faux ; (ii) le repli bf16 de la tête liée se déclenche → +1,28 Gio, exil ; (iii) le coût fixe du transfert des lignes
dépasse 1 % à b=1 → G6' faux, le levier ne serait plus gratuit et resterait réservé au cas « sinon exil ».

## Résultats à sec (02/10, APRÈS le scellé ; rien ci-dessus n'a été retouché)

| | prédit | à sec (`scratchpad/poste6-g6r/sim_g6r.py`, vrai manifeste, VRAM de G1) | |
|---|---|---|---|
| E1 65 536 | table hôte, 0 MLP exilé, KV 5,45, ≥ 4 096 blocs | table hôte, **0 exilé**, 5,45 Gio, 4 096 blocs, 1 créneau, plafond 5 120, pas de refus | tenu |
| E2 témoins | identiques à g6r | 4 096 / 8 192 / 16 384 / 27 648 / 41 984 : budget, blocs, plafond, anneau identiques, table sur la carte | tenu |
| E3 46 080 | table hôte, 0 exilé | table hôte, 0 exilé (2 sous g6r) | tenu |
| E4 fenêtre sans MLP exilé | 67 584-75 776 | **70 656** | tenu |
| E5 | jouet : ids/logits ; exil 1 MLP → table seule | réplique : 7 MLP exilés (`ACVRAM_EMBED=carte`) → 0 ; l'égalité des ids n'a de sens que sur carte (G6') — NON jouée à sec, dit | à moitié |

Ajouts au correctif scellé, dits : la sortie d'une chaîne de plan n'est imprimée que si son plan est retenu (le plan plein
écarté annonçait des exils qui n'ont pas lieu) ; le bras forcé `ACVRAM_EMBED=hote` ne compte pas comme un exil.
Tests : 5 de plus dans `tests/test_cible_kv_anneau_g6r.py` (12 en tout) ; cassure vérifiée sur copie (table jamais exilée : 2
rouges ; plafond qui ignore la table : 1 ; chauffe sur l'appareil de la table : 1). Suite ciblée 81 fichiers, 2 cœurs, nice 19,
46 s, carte tenue par poste2 en `service` : 561 passés, 0 échec.

Reste carte : G1' / G5' (`scratchpad/poste6-g6r/carte-g6r.sh`) ; G6' à ± 1 % ne se résout pas avec l'instrument de prise
(32 jetons décodés) — à mesurer par l'instrument de cellule b=1 / b=12 (poste2), `ACVRAM_EMBED=hote` contre `carte` ; ma prise
ne donne que l'égalité des ids (sha) entre les deux bras à 4 096.
