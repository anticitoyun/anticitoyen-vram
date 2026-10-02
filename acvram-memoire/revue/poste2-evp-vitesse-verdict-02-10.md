# evp — verdict vitesse : gpt-oss-20b, acvram contre llama.cpp (poste2, 02/10)

- **instrument** : `outils/gpu/mesure/serveur-bras.sh` + `outils/gpu/mesure/duel-moteurs.py`
  (scellé `revue/poste1-evp-vitesse-scelle-02-10.md`), script
  `scratchpad/poste2-evp-vitesse-02-10/mesure.sh`.
- **commit** : scellé lu sur `origin/poste1-evp-parc` (`84f27ef0f`), rien à fusionner côté moi.
- **régime** : ordre chef (file poste5 → poste1 120b → moi → poste6 → mon rejeu e50.2).
- **scellé** : `revue/poste1-evp-vitesse-scelle-02-10.md`, issue 3.
- **mesuré** : 3 tentatives acvram (b=1, b=12, b=1 rejoué) toutes mortes avant d'être prêtes ;
  llama.cpp b=1 et b=12 tenus.
- **verdict** : **ÉCHEC côté acvram — issue 3 du scellé.** Cause identique et reproductible aux
  3 tentatives (`scratchpad/poste2-evp-vitesse-02-10/acvram-{b1,b12,b1-rejoue}.log:8`) :
  ```
  RuntimeError: marlin_mm, .../marlin_moe_wna16/ops.cu:495, Invalid thread config:
  thread_m_blocks = 4, thread_k = -1, thread_n = -1, num_threads = -1
  for MKN = [16376, 2880, 2880] and num_bits = 4, group_size = 16, has_act_order = 0,
  is_k_full = 1, has_zp = 0, is_zp_float = 0, max_shared_mem = 101376
  ```
  Pile d'appel : `acvram/engine/moe.py:1031` (`_forward_prefill_grouped`) →
  `acvram/kernels/marlin_port/__init__.py:767` (`gemm_moe`) → noyau Marlin MoE (`ops.cu:495`).
  Le serveur meurt AVANT d'ouvrir le port (jamais atteint `/v1/models`) : la cause est dans la
  CHAUFFE interne au démarrage (prefill groupé synthétique, M=16 376 constant aux 3 essais,
  indépendant du prompt réel), pas dans une requête du client. `num_bits=4, group_size=16`
  pointe la quantification MXFP4→NVFP4 des experts de gpt-oss-20b spécifiquement (modèle
  jamais encore chargé par ce chemin avant aujourd'hui — bd evp). **Pas une pièce vitesse à
  moi** : hors mandat de cette mesure, à remonter à qui tient `marlin_port`/`moe.py`
  (poste1 probablement, conception initiale du GEMM groupé).
  Conformément à l'issue 3 : aucun rapport A/L publié, le chiffre llama.cpp seul n'est **pas**
  un comparatif, donné ici pour mémoire seulement :
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

Bug moteur distinct de ma pièce e50.2 et du correctif ChatML (`poste2-e50.2-verdict-relance-
02-10.md`, `fcf81b22f`) — à ouvrir en bead séparé si chef confirme, propriétaire probable
poste1 (`marlin_port`). Mon rejeu e50.2 suit, après poste6.
