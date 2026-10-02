# evp — verdict vitesse : gpt-oss-20b, acvram contre llama.cpp (poste2, 02/10)

- **instrument** : `outils/gpu/mesure/serveur-bras.sh` + `outils/gpu/mesure/duel-moteurs.py`
  (scellé `revue/poste1-evp-vitesse-scelle-02-10.md`), script
  `scratchpad/poste2-evp-vitesse-02-10/mesure.sh`.
- **commit** : scellé lu sur `origin/poste1-evp-parc` (`84f27ef0f`), rien à fusionner côté moi.
- **régime** : ordre chef (file poste5 → poste1 120b → moi → poste6 → mon rejeu e50.2).
- **scellé** : `revue/poste1-evp-vitesse-scelle-02-10.md`, issue 3.
- **mesuré** : 3 tentatives acvram (b=1, b=12, b=1 rejoué) toutes mortes avant d'être prêtes ;
  llama.cpp b=1 et b=12 tenus.
- **verdict (CORRIGÉ 02/10, cause réelle trouvée par poste1)** : **pas un échec moteur — mesure
  INVALIDE, arbre servi sans les pièces evp.** `mesure.sh:14` posait
  `BRAS_CWD="$RACINE"` où `$RACINE` est ce worktree (`poste2-banc`, branche `poste2-banc`) : ni
  558999068 (gabarit gpt-oss) ni les correctifs `marlin_port`/`moe.py` spécifiques à evp n'y
  sont fusionnés. `outils/carte.sh` résout `ACVRAM_ARBRE` depuis le cwd de l'appelant
  (`outils/carte.sh:33-34`), donc le serveur acvram a chargé gpt-oss-20b comme un MoE
  ORDINAIRE, sans le chemin evp — le noyau Marlin qu'il a appelé n'est pas celui que le scellé
  mesure. La trace brute reste exacte et reproductible 3/3
  (`scratchpad/poste2-evp-vitesse-02-10/acvram-{b1,b12,b1-rejoue}.log:8`) :
  ```
  RuntimeError: marlin_mm, .../marlin_moe_wna16/ops.cu:495, Invalid thread config:
  thread_m_blocks = 4, thread_k = -1, thread_n = -1, num_threads = -1
  for MKN = [16376, 2880, 2880] and num_bits = 4, group_size = 16, has_act_order = 0,
  is_k_full = 1, has_zp = 0, is_zp_float = 0, max_shared_mem = 101376
  ```
  mais elle décrit le GEMM groupé de `poste2-banc` (`moe.py:1031` → `marlin_port/__init__.py:767`
  → `ops.cu:495`), pas celui d'evp — **ne pas l'attribuer à poste1 ni à `marlin_port` evp**, ce
  point de la note initiale était faux. Correctif posé : `mesure.sh` refuse désormais de
  mesurer si l'arbre servi n'est pas EXACTEMENT `ATTENDU_SHA` (sha de main donné par chef
  après sa suite). Rejeu à refaire depuis ce sha, après la preuve de service d'poste1.
  Conformément à l'issue 3 du scellé (mesure invalide = rien de publiable) : aucun rapport A/L,
  le chiffre llama.cpp seul n'est **pas** un comparatif, donné ici pour mémoire seulement :
  - b=1 : décodage 389,22 tok/s (dispersion 2,72 %), préfill 9950 tok/s, 955 jetons/kJ.
  - b=12 : décodage agrégé 828,61 tok/s (dispersion 17,16 %), TTFT mural 0,495 s,
    2403 jetons/kJ.
- **durée** : prévue ≈ 20 min (scellé) ; tenue 334 s (journal carte, `tenue=334s`), plus courte
  que prévu car les 3 manches acvram échouent au démarrage (pas de décodage à attendre).

## Issues nommées (scellé)

Issue 3 retenue : « acvram ne sert pas : refus nommé à la capture ou au chargement... Verdict
ÉCHEC, avec la cause écrite ; pas de chiffre llama.cpp publié seul comme « comparatif ». »
C'est exactement cette branche — la cause est écrite ci-dessus, fichier et ligne.

## Suite

Mesure à refaire une fois le sha de main (après poste6 → poste5 → chef → preuve de service
d'poste1) donné ; `ATTENDU_SHA` obligatoire dans `mesure.sh`, refus sinon. Sans rapport avec ma
pièce e50.2 ni le correctif ChatML (`poste2-e50.2-verdict-relance-02-10.md`, `fcf81b22f`), qui
restent valides. Mon rejeu e50.2 suit, après poste6 et poste5.
