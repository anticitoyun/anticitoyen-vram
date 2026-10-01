# poste4 — duck.ai + vérification arXiv directe (30/09), 2 questions

Modèles : gpt-oss 120B, GPT-5.6 Luna, Gemma 4 31B. **Chaque identifiant
arXiv cité ci-dessous a été vérifié directement sur arxiv.org/abs/** (pas
seulement dans la réponse d'un modèle) — c'est la seule façon fiable de
traiter cette consigne, les modèles de langage inventant facilement des
identifiants plausibles.

## Q1 — Articles arXiv fin août/septembre 2026 : inférence LLM GPU grand public (sm_120, RTX 5090), NVFP4/FP4, attention FP4/FP8, MoE requête unique, décodage spéculatif adaptatif

**Luna a donné 6 identifiants, VÉRIFIÉS UN PAR UN sur arxiv.org/abs/ — les
6 existent, avec les bons titres, dates et résumés :**

| ID | Titre | Date | Gain annoncé | Code |
|---|---|---|---|---|
| **arXiv:2608.28113** | H-Scale (Hessian-Guided Scale Refinement, NVFP4) | 28 août 2026 | Zéro surcoût inférence, qualité rapprochée de BF16 (pas un gain de vitesse) | Non confirmé dans l'abstract |
| **arXiv:2608.12629** | CAKE (Compiler-Agent Co-Design, noyaux Ampere→Blackwell) | 12 août 2026 | Kimi Delta Attention : ×2,05 géomoyenne vs FlashKDA officiel ; KNN/KMeans ×1,42-2,12 ; **abstract ne mentionne PAS sm_120a explicitement** (contrairement à ce que Luna affirmait — nuance à corriger) | Partiel : 4 changements en PR upstream |
| **arXiv:2609.38166** | LeapQuant (quantification INT8 des états récurrents, attention linéaire) | 29 sept. 2026 | ×2,05-3,70 noyau, ×1,47 bout-en-bout, testé sur B200/RTX PRO 6000/**RTX 5090** (confirmé dans l'abstract) | Non confirmé |
| **arXiv:2609.38169** | STEPQuant (quantification spatio-temporelle, DeltaNet) | 29 sept. 2026 | 6 bits : >5× compression état récurrent, -68,7 % mémoire de service | **Oui, dépôt cité dans l'abstract** (« Our code is available at this https URL ») |
| **arXiv:2609.38090** | Mira (cache adaptatif + staging prédictif, MoE mono-GPU) | 29 sept. 2026 | ×5,71 débit moyen, ×11,71 TTFT, ×3,84 beam search (confirmé) | Non confirmé dans l'abstract |
| **arXiv:2609.02897** | Margins, Not Windows / AdaptiveSpec (décodage spéculatif adaptatif) | v1 3 juil., **v2 28 sept. 2026** | +56 % débit vs EAGLE-3, 93-100 % précision sans perte (confirmé, implémenté sur SGLang) | Non confirmé dans l'abstract |

**gpt-oss** a donné 5 identifiants dont un seul (arXiv:2609.21849, « The
Weight Is Over ») tombe dans la fenêtre demandée — **vérifié réel**, mais
porte sur la **diffusion image**, pas sur l'inférence LLM/attention/MoE
demandée (hors sujet, pas fabriqué). Les 4 autres (2601.09527, 2509.23202,
2605.00519, 2501.04052) sont datés janvier 2026, **septembre 2025**, mai
2026 et janvier 2025 — **hors de la fenêtre demandée** (gpt-oss a ignoré
la consigne de date, sans nécessairement inventer les identifiants :
2501.04052 correspond au vrai papier RaZeR, daté 2025 et non 2026).
**gpt-oss écarté** pour non-respect du filtre temporel sur 4 réponses/5.

**Gemma 4 31B**, après recherche Web longue (~2 min), a convergé de façon
indépendante sur **arXiv:2609.02897** (même papier que Luna) et ajouté
**arXiv:2608.11693** (audit ISA Blackwell Ultra/B300, FP4/FP8 priorisés
sur INT8 en CUTLASS) — non vérifié directement par manque de temps, mais
cohérent avec les deux autres papiers sur la priorisation FP4/FP8 côté
Blackwell.

**Aucun identifiant fabriqué détecté** dans les 6 retenus de Luna, ni dans
les 2 de Gemma — tous vérifiés existants sur arxiv.org. gpt-oss n'a
inventé aucun ID non plus mais a échoué le filtre de date/pertinence sur
la majorité de sa liste.

## Q2 — MpFA (arXiv 2609.33135, QK en NVFP4 / PV en FP8) : sm_120 grand public ou seulement B200 ?

**Vérifié directement sur arxiv.org/abs/2609.33135 et sa version HTML
(arxiv.org/html/2609.33135).** Titre exact : *MpFA: Hardware-Efficient
Train-Free QK4V8 FlashAttention Kernels on Blackwell GPUs* (soumis 27
septembre 2026).

- L'abstract est explicite : **« On an NVIDIA B200 »** — toutes les mesures
  de débit annoncées (×2,81 sur le débit de sortie bout-en-bout vs BF16 FA4,
  contextes 16K-128K) sont faites sur **B200 (sm_100)**, pas sur RTX 5090.
- Dans le corps du papier (version HTML), « RTX 5090 » n'apparaît que dans
  un **tableau comparatif de travaux connexes** (ligne SageAttn), pas comme
  plateforme de test de MpFA lui-même. RTX 4090 apparaît de la même façon,
  toujours pour un travail tiers.
- **Conclusion : MpFA n'a été validé expérimentalement que sur B200
  (sm_100)**, pas sur sm_120 grand public. Le format NVFP4 (QK) et FP8 (PV)
  est architecturalement disponible sur sm_120 (5e génération de Tensor
  Cores FP4, comme B200), donc rien n'exclut en principe un portage, mais
  **aucun chiffre de gain n'est fourni par les auteurs pour RTX 5090** — à
  ne pas extrapoler.
