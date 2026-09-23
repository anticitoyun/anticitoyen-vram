# Correctif `godet_hybride` — bead anticitoyen-vram-x0s clos, 13/09/2026

Laure. Suite à `part-reelle-mla-sous-graphes-13-09.md` (découverte du bug) et
à la demande de Jérôme de corriger tout de suite, puisque c'est moi qui l'ai
en main.

## Le correctif

`acvram/engine/graphs.py` :

1. **Fonction pure `godet_hybride(b_reel, max_slots)`** (remplace le calcul
   inline `b > self.max_slots` sur le godet bucketé) : `None` si
   `b_reel > max_slots` (vrai refus, hors des tampons), sinon
   `min(bucket_batch(b_reel), max_slots)` — `max_slots` devient lui-même un
   godet valide au-delà de la puissance de deux qui le précède. Un lot qui
   tient dans les tampons (`self.statics`) n'est plus JAMAIS refusé.
2. **`GraphRunner._eager(raison)`** : tout repli eager depuis la branche
   hybride s'annonce désormais, une fois par raison distincte (même patron
   que `_eligible()` pour la désactivation globale, jamais étendu à ce point
   d'entrée jusqu'ici).
3. **Bug connexe trouvé en écrivant l'épreuve du point 4** :
   `_bind_hybrid` donnait le **même** sid de rembourrage (`-1`) à tous les
   créneaux vides d'un pas. `static_bind` (model.py) suppose qu'un sid ne vit
   que dans un seul créneau : le second rembourrage avec `sid=-1` croyait que
   « -1 vit déjà ailleurs », exportait le premier créneau
   (`static_owners → None`) et pouvait lui faire hériter un état non-neuf.
   Sans conséquence tant qu'un seul créneau de rembourrage existait par pas
   (`bucket_batch` ne dépassait jamais `b_reel` de plus d'une unité près de
   la puissance de deux) — devenu systématique dès que `godet_hybride`
   autorise plusieurs rembourrages simultanés (`b_reel=9`, `max_slots=12` :
   trois). **Corrigé** : un sid distinct par créneau de rembourrage
   (`-1, -2, -3, ...`).
4. **Tests** (`tests/test_godet_hybride.py`, CPU, sans carte) :
   - `godet_hybride` sur `b_reel=9..15`, `max_slots=12` : plus aucun refus
     pour `b_reel ≤ 12`, refus franc au-delà ;
   - non-régression : comportement inchangé quand `max_slots` est déjà une
     puissance de deux, et sous la puissance de deux qui précède `max_slots` ;
   - méta-test : preuve que l'ancien calcul aurait refusé ces mêmes lots ;
   - sentinelle distincte : `_bind_hybrid` réel (pas le raccourci du test
     existant) sur `b_reel=10`, `max_slots=12` — trois créneaux réels
     conservent leur sid, deux créneaux de rembourrage reçoivent `-1` et `-2`
     sans collision.
   16 passed, 1 skipped (GPU) dans `test_godet_hybride.py` +
   `test_graphes_raison.py`. Suite complète : **538 passed, 1 failed
   (`test_le_gemv_nvfp4_ne_part_pas_en_emulation`, 99 Go/s — débit d'un
   micro-banc CUDA, sans rapport avec ce correctif, contention probable avec
   une autre session sur la carte), 5 skipped**.
5. **Documentation** : commentaire ajouté sur `ACVRAM_HYBRID_SLOTS` à sa
   définition (graphs.py) — n'importe quelle valeur entière ≥ 1 est valable
   depuis ce correctif, plus besoin d'une puissance de deux.

## Piège trouvé en mesurant : deux copies d'`acvram` dans l'environnement

Les scripts sous `scratchpad/` ont `scratchpad/` comme `sys.path[0]`, pas le
dépôt : sans `sys.path.insert(0, ".../travail/laure")` en tête, `import
acvram` retombe sur le venv partagé
(`anticitoyen-vram/.venv` → `anticitoyen-vram/acvram`, la copie **hors
worktree**), pas sur `travail/laure/acvram`. Une première tentative de mesure
du correctif a silencieusement mesuré l'ANCIEN code (aucune erreur, juste un
`ImportError` différé qui a fini par se révéler sur `godet_hybride`
manquant). Corrigé dans les scripts de mesure ; **à vérifier par quiconque
réutilise un script sous `scratchpad/` avec ce venv**.

## La mesure qui compte (Jérôme) : `ACVRAM_CHRONO_SYNC=1`, vrai `b_reel=12`

GLM-4.7-Grande-Heretic-42B, `ctx=2048`, `ACVRAM_HYBRID_SLOTS=12` (valeur
nominale, sans contournement — le correctif suffit désormais) :

    Régime A — 12 générations SÉQUENTIELLES, b_reel=1 à chaque fois
      (reproduit exactement le régime des campagnes "slots=12" d'avant le 13/09)
      1800 jetons en 20,217 s = 89,03 tok/s agrégé

    Régime B — lot CONCURRENT réel, b_reel=12, corrigé
      1792 jetons décodés en 15,156 s = 118,24 tok/s agrégé
      149 pas, voie graphe = 149/149 (0 pas en eager)
      replay_median (ACVRAM_CHRONO_SYNC=1) : 91,86 ms
      pas_total_median                    : 92,80 ms  (replay = 99,0 % du pas)

    GAIN DU BATCHING CONCURRENT : ×1,328

**C'est le premier chiffre de débit concurrent réel jamais mesuré sur ce
modèle hybride** — aucune campagne antérieure n'avait exercé le chemin
graphe à plus de 8 séquences simultanées. Le rapport replay/pas (~99 %) se
maintient depuis le régime à une séquence (98 %, campagne du 11/09) : ce
n'était donc pas faux en soi, seulement mal étiqueté quant au `b_reel` réel.

Script : `scratchpad/mesure-debit-concurrent-13-09.py`.

## Notes corrigées

`campagne-chrono-sync-11-09.md`, `duck-laure-12-09.md`,
`occupation-mla-decode-batching-slots.md` portent désormais chacune une
section « CORRECTION — 13/09/2026 » pointant ici.
`ou-nous-sommes-10-09.md` (Qwen3-Coder-30B-A3B, MoE non hybride) porte une
note confirmant qu'il n'est **pas** concerné par ce bug — ses chiffres
`b=12` sont d'authentiques mesures concurrentes, obtenues par
`outils/duel-moteurs.sh`, pas par `engine.generate()`.

## bd

`anticitoyen-vram-x0s` : fermé, correctif + tests + mesure dans ce document.
`anticitoyen-vram-6wa` : reste ouvert (implémentation du noyau batché,
Laurine) ; la prédiction de gain du 13/09 matin (-5/-18 %) n'a pas encore été
recalée sur la mesure ×1,328 — à faire quand Laurine reprend le bead.
