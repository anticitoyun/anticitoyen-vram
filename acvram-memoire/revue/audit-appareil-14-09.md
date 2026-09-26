# Audit appareil — 14 septembre 2026

Bancs et scripts appelant Energie() ou puissance_nvml sans CUDA_VISIBLE_DEVICES.

## Résultat

**Fraction :** 6/36 appels (16.7 %)

**Fichiers non conformes :**

1. outils/banc_prefill_vllm.py:19
2. outils/campagne-a6-snrfloor0.py:38
3. outils/banc_llamacpp.py:39
4. outils/banc_tabbyapi.py:26
5. outils/banc_decode_vllm.py:15
6. outils/banc-4moteurs.py:44

## Conséquence

Tous les bancs listés mesurent l'énergie brute de toutes les cartes (hôte GPU complète) au lieu de la 5090 seule (CUDA_VISIBLE_DEVICES=0 manquant). Mesures pollués par la 3080 Ti au repos (23 W) ou charge résiduelle.

Priorité : ajouter CUDA_VISIBLE_DEVICES=0 ou passer à Energie(devices=[0]) dès que possible.

## Audit scope

- Répertoires : outils/, outils/gpu/mesure/, scratchpad/
- Cherche : Energie(), puissance_nvml
- Total trouvé : 36 appels, 6 non conformes
