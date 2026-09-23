# Protocole (en attente) — contrôle sur carte du scaling yarn/llama4 (Laurine 496b148, validé à sec par Sage) : Devstral, quand Manon aura converti (téléchargement source : décision utilisateur en attente)

instrument : `ppl-decode-kv-17-09.py` sur une séquence de **8 192 + 2 048** jetons (teacher forcing sous graphes, decode_fixed) contre le préfill du même texte ; ratio PPL décodage / préfill ; régime classé, graphes on ; ligne [régime] doit passer de « llama4_scaling=validé à sec » à la valeur mesurée.
scellé (Sage) : ratio dans ± 0,002 tenu ; bras cassant `ACVRAM_LLAMA4_BETA=0` → ratio hors ± 0,002 au-delà de 8 192 jetons (sinon la garde ne voit rien, contrôle non prouvé) ; corpus : privé (tranche longue à assembler ≥ 10 240 jetons, sha256 dans le verdict, jamais affiché).
prérequis : converti Devstral (Manon) ; `max_model_len ≥ 10 240` au chargement ; capture godets si un noyau de décodage neuf.
