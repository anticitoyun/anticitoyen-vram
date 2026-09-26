# Questions NVIDIA sm_120 (RTX 5090 Blackwell) — Formats & Instructions

## Question 1 : wgmma / TMA en sm_120

**Statut** : ✓ VÉRIFIÉ  
**Priorité** : Haute

### Ce qu'on cherche
- **wgmma** (Warp Group Matrix Multiply-Accumulate) : disponible en sm_120 ?
- **TMA** (Tensor Memory Accelerator) : disponible en sm_120 ?

### Sources officielles NVIDIA (consultées)
1. **PTX ISA 9.4** : `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html`
   - Section **9.7.18 TensorCore 5th Generation Family Instructions**
   - Instruction : `tcgen05.mma.*` (TensorCore generation 5)

### Affirmations vérifiées

**Résultat 1 : wgmma n'existe PAS en sm_120**
- `wgmma` est une instruction **Hopper** (sm_90 uniquement)
- Blackwell sm_120 utilise **`tcgen05`** (TensorCore 5th Generation) à la place
- **Source** : PTX ISA section 9.7.18 — seules les instructions `tcgen05.mma.*` sont listées pour Blackwell

**Résultat 2 : TMA (Tensor Memory Accelerator) existe EN sm_120**
- TMA est utilisé avec les instructions `tcgen05.mma` pour accéder à la **Tensor Memory** en Blackwell
- Instruction : `tcgen05.mma.ws`, `tcgen05.mma.sp`, etc.
- Qualifieur : `.cta_group::1` ou `.cta_group::2` pour spécifier l'accès à Tensor Memory
- **Source** : PTX ISA 9.7.18.10 — « TensorCore 5th Generation Matrix Multiply and accumulate Operations »

### Conclusion
✓ **wgmma n'est pas sur sm_120 (c'est Hopper)**  
✓ **TMA est sur sm_120 (avec tcgen05)**  
✓ **Architecture : tcgen05 + TMA pour Blackwell grand public**

---

## Question 2 : Formes mma pour FP8 (e4m3) et FP4 en sm_120

**Statut** : ✓ PARTIELLEMENT VÉRIFIÉ  
**Priorité** : Haute — détermine les chemins matériels pour NVFP4 et INT8 dans le cache KV

### Ce qu'on cherche
- Quelles formes `tcgen05.mma` existent en sm_120 pour FP8/FP4 ?
- Formes natives pour **FP8 (e4m3)** et **FP4 (NVFP4)** ?
- Contraintes de forme : alignement mémoire, tailles M/N/K, empaquetage ?

### Sources officielles NVIDIA (consultées)
1. **PTX ISA 9.4** : section 9.7.18 — TensorCore 5th Generation
   - Table 66 : « Valid Combinations of Container size and floating point types »

### Affirmations vérifiées

**Résultat 1 : FP8 (e4m3) a un chemin NATIF en sm_120**
- Forme supportée : **`.kind::mxf8f6f4`** — groupe d'instructions pour FP8, FP6, FP4
- Instruction : `tcgen05.mma.cta_group.kind::mxf8f6f4 [d-tmem], a-desc, b-desc, idesc, [scales...], enable-input-d`
- Utilise **Tensor Memory** pour A, B, D et échelles (scales)
- **Source** : PTX ISA 9.7.18.10.4 — « Packing format for matrix A and B »

**Résultat 2 : FP4 (NVFP4) a un chemin NATIF en sm_120**
- Forme spécifique : **`.kind::mxf4nvf4`** — FP4 avec format NVIDIA FP4
- Également supportée : **`.kind::mxf4`** — FP4 générique
- Instruction : `tcgen05.mma.cta_group.kind::mxf4nvf4 [d-tmem], a-desc, b-desc, idesc, [scales...], enable-input-d`
- Empaquetage en Tensor Memory : **8-bits container, 32 éléments K-dim** (Table 66)
- **Source** : PTX ISA 9.7.18.10.4.3 — « Packing format used for matrix A by `.kind::mxf8f6f4` in Tensor Memory »

**Résultat 3 : Statut matériel de FP8 vs INT8**
- **FP8** : chemin matériel direct via `tcgen05.mma.kind::mxf8f6f4`
- **INT8** : également soutenu (voir `.kind::i8` dans les instructions)
- **Résumé** : FP8 et INT8 sont au même niveau — tous deux ont des chemins matériels directs

**Résultat 4 : Support vérifié par vLLM**
- Paper arXiv 2601.09527 (« Private LLM Inference on Consumer Blackwell »)
- Confirme : **NVFP4 has native hardware acceleration on Blackwell Tensor Cores**
- Formats : NVFP4 (poids + activations) vs W4A16 (poids seul)
- **Source** : Section 3.3 « Quantization Schemes »

### Conclusion
✓ **FP8 (e4m3) : chemin natif via `.kind::mxf8f6f4`**  
✓ **FP4/NVFP4 : chemin natif via `.kind::mxf4nvf4` (8-bit container, 32-elem K-dim)**  
✓ **INT8 et FP8 au même niveau hiérarchique en matériel**  
✓ **Formes m/n/k variables selon le descripteur d'instruction (idesc)**

---

## Notes du circuit

- **Règle IA** : deux sources divergentes = apprentissage ; concordance = pas fiable
- **Rapport** : distinguer affirmation / vérification / à vérifier
- **Prévenir** : seulement si contradiction avec nos croyances actuelles

## Mémoire du projet

- sm_120f obligatoire pour FP4 conversion E2M1 (sinon émulée) — voir [[sm120f-conversion-fp4]]
- 3080 Ti (sm_86) n'a ni FP4 ni cvt E2M1 — voir [[3080ti-llamacpp-mieux]]
