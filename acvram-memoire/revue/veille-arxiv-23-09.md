# Veille arXiv — cinq leviers ouverts d'acvram (23/09, chef)

Demandé par l'utilisateur le 23/09 : « fais des recherches en profondeur sur arxiv.org afin d'améliorer le
projet ». Cinq recherches parallèles, chaque référence ouverte sur arxiv.org (abs/ ou HTML) et chaque chiffre
relevé sur la page ; ce qui n'a pas pu être vérifié n'est pas cité. Aucun de ces chiffres n'a été mesuré chez
nous : ce sont des PRÉDICTIONS à sceller avant nos propres prises (REGLES § 4), jamais des résultats.

## 1. Énergie par jeton — le levier le plus fort pour notre objectif

* 2501.08219 (RTX PRO 6000 Blackwell = puce GB202 de la 5090, lots 1/4/8) : horloge SM 2 842 → basse = −42 %
  d'énergie en moyenne pour +1 à 6 % de latence ; gain plafonné sous ≈ 1 000 MHz, optimum ≈ 960 MHz.
* 2605.11999 (H200, décodage) : le plafond de puissance ne se déclenche jamais au décodage (137-300 W sur 700) ;
  -lgc 780 MHz = −24 à −32 % de puissance pour < 1 % de débit ; au-dessus de 1 590 MHz, débit plat et +7 à 13 %
  de puissance. Pièges : -lgc peut être ramené en silence (1 980 → 1 830), l'horloge mémoire demandée ignorée.
* 2601.22076 : FP8 contre BF16 coûte +30 % d'énergie à lot 8-16 (0/7 cas gagnants) par la (dé)quantification
  non fusionnée, et gagne −11 % à lot 65-256 — notre régime b=12 est dans la zone perdante.
* 2312.02741 : nvidia-smi n'échantillonne la puissance que 25 % du temps (A100/H100) → fenêtres longues.
* 2608.28044 : le J/jeton dépend de la longueur de sortie (7,46 → 0,72 J de 10 à 512 jetons) → publier aussi J/requête.

**Pour acvram** : nous mesurons à -lgc 2700, presque au maximum. Balayage d'horloge au décodage = pièce 115.

## 2. Projections d'attention en FP4 — notre KL 0,511 est anormal

* 2603.08747 (NVFP4/MXFP4 sur Qwen2.5) : l'attention est BIEN MOINS sensible que le MLP ; q la moins sensible,
  puis k ; v/o plus fragiles (cohérent avec notre S3 0,234), mais pas au point de 0,511 contre 0,045.
* 2509.23202 (MR-GPTQ) : les blocs de 16 de NVFP4 neutralisent les techniques anti-aberrantes ; ajouter Hadamard
  à NVFP4 en arrondi au plus proche DÉGRADE ; l'arrondi simple NVFP4 est bon.
* 2512.02010 (Four Over Six) : choisir par bloc l'échelle max→6 ou max→4 au plus petit MSE rapproche AWQ du BF16
  de 19,9 % (mais GPTQ s'en éloigne) ; l'erreur des échelles E4M3 compte peu ; lm_head gardé haut par usage.
* 2608.28113 (H-Scale, Qwen3-30B-A3B) : échelles affinées par la diagonale de la hessienne, NVFP4 au niveau BF16.
* 2407.03211, 2311.09755 : une calibration sans la langue visée dégrade fortement cette langue.
* 2506.09501 : la sortie gloutonne varie avec la taille de lot et le découpage → KL à lot et split-K fixés.

**Pour acvram** : soupçon sur AWQ (ou sa calibration, ou le report d'échelles v→o en GQA), pas sur le format.
Diagnostic à 0 min de carte : projections NVFP4 en arrondi simple sans AWQ, assemblées (pièce 114).

## 3. Noyaux MoE au décodage

* 2609.04244 (MonoMoE, Qwen3.5-35B-A3B, H200, B=1) : les deux GEMM fused_moe (23 µs) n'atteignent que 22 % de la
  bande passante ; les noyaux auxiliaires (routage, top-k, alignement, quantification) ≈ 28 µs, plus que les GEMM ;
  noyau persistant fusionné 1,17-1,54× à B=1, 1,02-1,17× à B=8.
* 2511.02237 (OEA, Qwen3-30B, lot 16) : routage conscient du lot, −39 % de latence de couche MoE sans perte
  significative — mais la sortie change (REGLES règle 9 : mode distinct, jamais le défaut).
* 2512.22219 (MPK) : les graphes CUDA retirent le coût hôte, pas les bulles entre noyaux.

**Pour acvram** : le retard b=1 sur llama.cpp (5,8 %) vient peut-être des noyaux auxiliaires autour du Marlin.
Profil d'une couche avant tout code (pièce 116).

## 4. Spéculation MTP en lot sur MoE

* 2505.19645 (MoESD), 2506.20675 (Cascade) : sur MoE à petit lot, les jetons de brouillon activent plus
  d'experts ; vérification 2-3× plus lente, jusqu'à 1,5× de ralentissement ; Cascade coupe quand l'utilité < 1.
* 2412.19437 (DeepSeek-V3) : MTP profondeur 1, 85-90 % d'acceptation, 1,8×.
* 2609.23900 (GDN Tree-Scan, Qwen3.6-27B, b=1) : arbre +27 % sur MTP en chaîne au décodage, +4 % bout à bout.
* Règle chiffrée : gain ≈ τ / R, R = temps(passe à b·(γ+1) jetons) / temps(passe à b jetons) sur NOTRE MoE ;
  gain nul prédit si R ≥ τ. Avec 128 experts top-8 (notre modèle), le nombre d'experts distincts chargés croît
  vite avec γ.

**Pour acvram** : mesurer R (b = 8, 10, 12 ; γ = 1-3) AVANT tout code du proposeur par lot (112) ; seuil
R < 0,8 τ ; γ = 1 dès b ≥ 8, γ ≤ 3 seulement à b = 1-2.

## 5. Cache KV 4 bits (pièce 104)

* 2404.00456 (QuaRot, K/V seuls) : K4V4 5,51 · K4V3 5,54 · K3V4 5,65 · K2V4 8,06 contre 5,47 — K bien plus
  sensible que V ; Hadamard par tête sur V fusionnable dans W_v/W_o sans coût à l'exécution.
* 2402.02750 (KIVI) : 4 bits par jeton en groupe de 32 quasi sans perte (Llama-2-13B) — mais asymétrique.
* 2403.01241 (IntactKV), 2605.12464 (ScaleSearch) : garder les premiers jetons (puits d'attention) en pleine
  précision améliore nettement le KV quantifié.
* 2502.04420 (KVTuner) : Qwen2.5 plus fragile (4,0 bits requis contre 3,25 pour Llama-3.1).
* 2504.19874 (TurboQuant) : rotation + quantifieur scalaire optimal, sans perte dès 3,5 bits sur LongBench.
* 2510.25602 : avec Hadamard, INT4 ≥ NVFP4 ; sans rotation, NVFP4/MXFP4 souvent meilleurs qu'INT4.

**Pour acvram** : K8V4 groupe 32 confirmé ; seuil PPL +0,30 % serré pour Qwen ; premier repli = puits d'attention
(premiers jetons) en int8, puis V avec point zéro, puis Hadamard fusionné + INT4, puis E2M1.

## Pièces ouvertes par cette veille

| Pièce | Poste | Objet | Carte |
|---|---|---|---|
| 114 | poste6 | projections NVFP4 en arrondi simple SANS AWQ (alias assemblé), dans le filtre 2 | minutes |
| 115 | poste2 | balayage -lgc 960-2700 au décodage b=1 et b=12, J/jeton et débit, horloge appliquée relue | ≈ 30 min |
| 116 | poste3 | profil nsys d'une couche MoE à b=1 : part des noyaux auxiliaires contre les GEMM | minutes |
| 104 | poste1 | § 5 : ordre des replis révisé (puits int8 d'abord) | — |
| 112 | poste1 | mesure de R avant tout code du proposeur par lot | minutes |
| — | reste | routage conscient du lot (OEA) en mode optionnel, après 116 | — |
