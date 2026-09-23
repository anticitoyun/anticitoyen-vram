# Bande passante mesurée RTX 5090 — recherche 23/09/2026

Laurine · recherche internet · duck.ai + web

---

## 1. Spec officielle (pic théorique)

**1 792 GB/s** (= 1,792 TB/s)  
Bus : 512 bits · GDDR7 28 Gbps · 32 Go

Sources indépendantes :
- Page produit NVIDIA — https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- codesota.com, fiche avril 2026 — https://www.codesota.com/hardware/rtx-5090  
  Note explicite : *"Bandwidth is peak; sustained will be lower."*

---

## 2. Mesures applicatives (décodage LLM, sources internes acvram)

Ces chiffres viennent de nos propres bancs sur la carte du groupe ; ils ne constituent pas une source externe indépendante, mais restent les seules mesures de bande passante **effective** disponibles à ce jour pour cette carte.

| Régime | Bande passante effective | Efficacité |
|--------|--------------------------|------------|
| Plancher décodage acvram (poids lus) | 1,52–1,55 TB/s | 85–86 % |
| Décodage vLLM b=12 (backend auto) | ~1,67 TB/s | 93 % |
| GEMM FP8 W8A8 M=12 (TRT-LLM, depuis L2) | 1,81 TB/s | > 100 % du DRAM → L2 hit |

Le chiffre 1,81 TB/s de TRT-LLM dépasse le débit DRAM théorique. **Hypothèse** : la lecture provient en partie du cache L2 (96 Mo activés sur la RTX 5090 ; la puce GB202 complète en a 128 Mo — source : TechPowerUp, die-shot GB202, 27 jan. 2025), pas directement de la GDDR7. Cette hypothèse n'est pas vérifiée par une mesure de hit L2 sur cette carte.

---

## 3. Mesures indépendantes publiées — nvbandwidth / STREAM triad

**Aucune trouvée.**

Recherches effectuées :
- nvbandwidth NVIDIA/nvbandwidth (dépôt GitHub discussions : HTTP 404)
- STREAM triad / BabelStream RTX 5090 — résultats non indexés publiquement
- gpu-benches — aucun résultat pour sm_120
- Tom's Hardware, GamersNexus, AnandTech (accès HTTP 403 ou résultats spec uniquement)
- Arxiv 2605.00519 « Silicon Showdown » (mai 2026) — référence NVFP4 débit 151 t/s vs 92 t/s BF16, pas de mesure de bande passante brute

---

## 4. Conclusion

La bande passante pic (1,792 TB/s) est confirmée par au moins deux sources indépendantes (NVIDIA officiel + codesota.com).

Les seules mesures de bande passante **soutenue** disponibles sont internes au groupe (1,52–1,55 TB/s plancher, 1,67 TB/s vLLM). Aucune mesure externe de type nvbandwidth/STREAM n'est publiée pour cette carte à ce jour.

**Source externe indépendante : 2 (spec théorique confirmée par 2 sources).**  
**Mesure soutenue externe : aucune — source unique interne.**
