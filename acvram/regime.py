"""Régime des noyaux : quelles variables ``ACVRAM_*`` choisissent un chemin de
calcul, ce qu'elles valent RÉELLEMENT, et comment envoyer chaque chemin vers
son jumeau torch (bissection).

Né de poste7-prefill-a8-verdict-17-09 § 3 : le 1 % perdu contre Marlin sur GLM
était ``ACVRAM_PREFILL=a8`` par défaut — un régime que personne n'avait posé,
qu'aucun en-tête de mesure n'imprimait, et que les masques par symbole
d'extension ne pouvaient pas voir (chemin torch derrière une porte
``get_extension() is not None``). Trois réponses, dans ce module :

* ``regime_noyaux()`` : chaque variable de chemin avec sa valeur effective —
  celle que le module a lue à l'import quand il la lit là, pas celle de
  l'environnement au moment de l'appel ; ``regime_ligne()`` pour l'en-tête
  d'une mesure. Un verdict sans cette ligne n'entre plus dans INDEX.
* ``masquer(noms)`` : envoie vers torch les chemins nommés — par variable
  (``PREFILL``, ``MOE_MMA``…), par backend (``cuda-fusionne``) ou par noyau
  de l'extension (``rmsnorm``). Se fait AVANT le chargement du modèle ; pour
  une variable déjà lue à l'import, l'attribut du module est réécrit aussi.
* ``VARIABLES`` : la table, tenue à jour par un test qui casse quand une
  variable ``ACVRAM_*`` sélectionnant un chemin n'y est pas.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Variable:
    nom: str                      # sans le préfixe ACVRAM_
    defaut: str
    lu_a: Optional[tuple[str, str]] = None   # (module, attribut) si lue à l'import
    torch: Optional[str] = None   # valeur qui envoie ce chemin vers torch ; None : seuil
    note: str = ""

    @property
    def env(self) -> str:
        return "ACVRAM_" + self.nom


VARIABLES: tuple[Variable, ...] = (
    # --- tout ou rien ---------------------------------------------------
    Variable("DISABLE_KERNELS", "", None, "1", "aucune extension : backend reference partout"),
    Variable("DISABLE_CPU_KERNELS", "", None, "1"),
    Variable("DISABLE_CUDA_GRAPHS", "", None, "1"),
    Variable("DISABLE_FP4_GEMM", "", None, "1"),
    Variable("DISABLE_PAGED_ATTN", "", None, "1"),
    Variable("PAGED_ATTN", "triton", ("acvram.kernels", "_PAGED_ATTN"), "cuda",
             "attention paginée du décodage : triton (poste E, K/V lus une fois par groupe GQA, défaut depuis poste7-e-c-verdict-17-09) | cuda (ancien défaut, témoin)"),
    # --- projections NVFP4 non groupées --------------------------------
    Variable("PREFILL", "bf16", None, "bf16", "bf16 | w4a16 (B1 Triton, NVFP4 dans la tuile) | w8a8 | w4a4 au-delà de NVFP4_GEMV_MAX lignes"),
    Variable("PREFILL_INT8", "cublas", ("acvram.kernels", "_PREFILL_INT8"), "bf16",
             "linéaires INT8 au préfill : cublas (DÉFAUT depuis poste7-p2-au-defaut-19-09 : poids symétriques par canal des convertis -qkvo-i8c, A8 par jeton puis torch._int_mm cuBLASLt, M > 16 ; un poids affine par groupes — les classés — garde la déquant bf16, sortie inchangée) | bf16 (témoin : déquant entière + cutlass partout) | a8 (P0 : activation int8 par jeton, W8A8 Triton sur tout poids int8 ; poste7-profil-verdict-18-09)"),
    Variable("COLLE_MOE", "torch", ("acvram.engine.model", "_COLLE_MOE"), "torch",
             "colle du préfill MoE : torch (argsort + bincount + _tuiles) | triton (P0 : tri + histogramme et grille en deux lancements, mêmes tenseurs)"),
    Variable("NVFP4_GEMV_MAX", "32", ("acvram.kernels", "_NVFP4_GEMV_MAX")),
    Variable("INT8_GEMV_MAX", "80", ("acvram.kernels", "_INT8_GEMV_MAX")),
    Variable("NARROW_GEMM", "0", ("acvram.kernels", "_NARROW_GEMM"), "0"),
    Variable("NARROW_KERNEL", "mixte", ("acvram.kernels", "_NARROW_KERNEL"), None,
             "linéaires INT8 à b ≤ 16 : mixte (défaut, Triton dès b≥NARROW_TRITON_MIN_B, verdict-coder-c-mixte-17-09) | cuda | triton | tete"),
    Variable("NARROW_TRITON_MIN_B", "2", ("acvram.kernels", "_NARROW_TRITON_MIN_B")),
    Variable("NARROW_NVFP4", "0", ("acvram.kernels", "_NARROW_NVFP4"), "0"),
    Variable("DENSE_NVFP4", "triton", ("acvram.kernels", "_DENSE_NVFP4"), "gemv",
             "linéaires NVFP4 denses (et tête) à DENSE_NVFP4_MIN_M ≤ b ≤ 32 : triton (gemm_dense_etroit, poids lus une fois par pas, défaut depuis verdict-gemm-dense-palier1-situ-17-09) | gemv (témoin, poids relus par séquence)"),
    Variable("DENSE_NVFP4_MIN_M", "4", ("acvram.kernels", "_DENSE_NVFP4_MIN_M")),
    # lues dans acvram_kernels.cu (getenv, figées au premier lancement : un
    # PROCESSUS par valeur — poste7-gemv-experts-rpw-18-09)
    Variable("GROUPED_RPW", "4", None, None,
             "GEMV groupée des experts : lignes par warp (4 défaut depuis poste7-rpw-defaut-18-09, Coder b=12 1 262 t/s nu ; 1 | 2 témoins) ; l'activation est étagée une fois par bloc de GW_WARPS×RPW lignes"),
    Variable("GROUPED_OLD", "", None, "1", "témoin : l'ancien noyau groupé à 4 lignes par bloc"),
    Variable("GROUPED_XREG", "down", None, "0",
             "GEMV groupée des experts, K ≤ 2048 : down (défaut depuis verdict-gemv-experts-xreg-down-18-09 : ABAB × 0,944 nu, Coder b=12 1 307 t/s) = x en registres par tranche sur la projection down seule | 1 = gate/up aussi (témoin réfuté : 96 registres, 2 blocs/SM, +19 %) | 0 = x relu en shared (témoin) ; sortie identique au bit dans tous les cas"),
    Variable("GEMV_SPLITK", "0", None, "0",
             "GEMV Marlin (b) : split-K par lot aux petits lots (gate/up b=1 : 96 → 384 blocs, réduction du dernier bloc, déterministe) ; 0 défaut = noyau d'avant | 1 = opt-in : +3,1 % b=1 (385,6 contre 374,1 t/s ABAB), PPL décodage +0,0042 vs témoin graphes/eager 0,0026, non tranché 19/09 (verdict-splitk-b1-19-09) | n ≥ 2 = S forcé (diagnostic) ; lue une fois par processus"),
    Variable("GEMV_LAYOUT", "marlin", ("acvram.engine.model", "_GEMV_LAYOUT"), "marlin",
             "P1 disposition UNIQUE (forme (b)), DÉFAUT depuis l'adoption du 18/09 : marlin = pile Marlin seule (préfill GEMM classe Marlin ET GEMV du décodage relisant les tuiles 16 k × 64 n ; la pile NVFP4 est rendue après le repack, experts_layout=marlin ; va avec PREFILL_GROUPED=marlin, sinon refus à l import ; scellé ≤ 0,97 × GEMV à b=1 et b=12, fp32 par ligne) | naturel = pile NVFP4 seule (témoin, avec PREFILL_GROUPED=groupe ; b=1 366 t/s contre 351 en marlin)"),
    Variable("MARLIN_DISTINCT", "0", ("acvram.engine.model", "_MARLIN_DISTINCT"), "0",
             "C10 : 1 = la disposition unique Marlin sert aussi les MoE à gate/up distincts (tables AWQ séparées : GLM k48-calibA) — décodage par le GEMV Marlin à une projection, gate puis up ; 0 défaut = refus nommé, pile naturelle gardée, chemin d'avant (verdict-glm-b12-19-09) ; scellé GLM b=12 ≥ chemin d'avant × 1,05"),
    # --- capacités, plafonds, modes du moteur (19/09 : sortis de HORS_REGIME, poste7) ---
    Variable("HYBRID_SLOTS", "", None, None,
             "plafond des créneaux hybrides (GDN/KDA/Mamba2/MLA) par graphe ; vide = le --max-batch du moteur (≥ 4, graphs.py plafond_hybride) ; posé = valeur fixe (l'ancien défaut 4 mettait GLM en eager dès b=5)"),
    Variable("DENSE_SLOTS", "4", None, None, "créneaux denses par graphe"),
    Variable("WARM_SPEC", "1", None, None, "chauffe des graphes spéculatifs"),
    Variable("PLAN_FIGE", "", None, None, "1 = le plan de placement ne se recalcule pas"),
    Variable("SANS_REPLAN", "", None, None, "1 = pas de replanification au chargement"),
    Variable("SANS_PRECHARGE", "", None, None, "1 = pas de préchargement des couches exilées"),
    Variable("POOL_SYNC", "", None, None, "1 = pool d'experts synchrone"),
    Variable("PIPELINE", "", None, None, "1 = lot préparé pendant le rejeu (runner)"),
    Variable("PREFILL_BATCH", "", None, None, "prefills groupés"),
    Variable("SPECULATION_LOT_MAX", "2", None, None, "lot maximal sous spéculation"),
    Variable("MTP", "", None, None, "tête MTP (auto | none | mtp)"),
    Variable("DEQUANT_TRANCHE_MAX", "268435456", None, None, "pic (octets) d'une tranche de déquantification"),
    Variable("DENSE_ETROIT_BN", "64", None, None, "tuile N du GEMM dense étroit Triton"),
    Variable("DENSE_ETROIT_BK", "128", None, None, "tuile K du GEMM dense étroit Triton"),
    Variable("DENSE_ETROIT_WARPS", "4", None, None, "warps du GEMM dense étroit Triton"),
    Variable("DENSE_ETROIT_STAGES", "3", None, None, "étages du GEMM dense étroit Triton"),
    Variable("INSTA_PAS", "256", None, None, "pas entre deux relevés instantanés"),
    Variable("GRAPHES_TABLE", "", None, None, "graphes sur les piles à table (placement par expert)"),
    Variable("MLP_HOTE_CPU", "", None, None, "1 = MLP exilé calculé sur le processeur"),
    Variable("KDA_CHUNK", "1", None, None, "taille de bloc KDA"),
    Variable("MAMBA_CHUNK", "1", None, None, "taille de bloc Mamba2"),
    Variable("ALLOC_EXTENSIBLE", "", None, None, "1 = allocateur CUDA à segments extensibles"),
    Variable("BUDGET_JETONS", "0", None, None, "budget de jetons du prefill (0 = illimité)"),
    Variable("EXIL_COUCHES", "", None, None, "couches exilées forcées"),
    Variable("EXIL_EXPERTS_FRACTION", "", None, None, "fraction d'experts exilés forcée"),
    Variable("SEUIL_EXIL", "0.20", None, None, "seuil d'exil du planificateur"),
    Variable("REPIN", "64", None, None, "période (pas) du ré-épinglage des experts"),
    Variable("MAX_GRAPHS", "16", None, None, "graphes CUDA gardés (éviction au-delà)"),
    Variable("INSTA_MAX", "3", None, None, "relevés instantanés gardés"),
    Variable("KV_PLAN_OVERRIDE", "", None, None, "budget KV forcé (plan)"),
    Variable("LISTE_CLE", "", None, None, "clé de la liste de promotion"),
    Variable("LISTE_PROMUS", "", None, None, "tenseurs promus en int8 forcés"),
    Variable("MAX_PROMUS", "0", None, None, "plafond de tenseurs promus (0 = illimité)"),
    Variable("ORDRE_SAC", "snr", None, None, "ordre du sac à dos de placement (snr | …)"),
    Variable("ORDRE_SAC_INVERSE", "", None, None, "1 = sac à dos inversé"),
    Variable("PAGED_ALLOC", "", None, None, "allocateur paginé"),
    Variable("PA_CHUNK", "", None, None, "bloc de l'attention paginée"),
    Variable("PA_ETAPE", "", None, None, "étape de l'attention paginée"),
    Variable("PA_SANS_COMPTEUR", "", None, None, "attention paginée sans compteur"),
    Variable("INT8_GEMV_WARP", "", None, None, "warps du GEMV int8 (lu dans le .cu)"),
    Variable("INT8_TRANCHE", "", None, None, "découpage du GEMV int8 (12 = ancien, témoin ; lu dans le .cu)"),
    Variable("ECO", "", None, None,
             "poste7-eco-2700-defaut-19-09 § 1 : mode éco d'horloge DEMANDÉ par un instrument pour ses bras A/B (2700 | 2100 | off), jamais pour le service — le service lit config.json (\"eco\": \"2700\" par défaut) ; l'effectif est sur la ligne de régime (eco=<demandé>(<effectif>)) et un instrument ne publie pas si demandé ≠ effectif"),
    Variable("DOUBLE_DISPOSITION_DIAG", "0", None, "0",
             "diagnostic seulement (bissection du biais GEMV (b), poste7-p1-situ-verdict-18-09) : 1 = les deux dispositions gardées, préfill {groupe|marlin} × décodage {naturel|marlin} sur les mêmes piles ; jamais un régime servi"),
    Variable("MOE_GEMV", "v1", ("acvram.engine.model", "_MOE_GEMV"), "v1",
             "GEMV groupée du décodage MoE : v1 (une passe de poids par paire expert-jeton) | v2 (paires triées par expert, poids lus une fois pour ≤ 4 jetons, sortie identique au bit)"),
    Variable("MULTI_PROJ", "0", ("acvram.engine.model", "_MULTI_PROJ"), "0",
             "témoin (palier 2 non ouvert, 0,88 To/s) : q/k/v et qkv/gate/α/β du GDN en un lancement, chacune avec son scaler"),
    Variable("NARROW_MIN_M", "2", ("acvram.kernels", "_NARROW_MIN")),
    Variable("NARROW_ROWS", "32", ("acvram.kernels", "_NARROW_ROWS")),
    Variable("NARROW_MLA", "1", None, "0"),
    Variable("SEUIL_FUSION", "256", ("acvram.engine.model", "SEUIL_FUSION")),
    Variable("PREFILL_GROUPED", "marlin", ("acvram.engine.model", "_PREFILL_GROUPED"), "marlin",
             "GEMM groupée du prefill MoE : marlin (P1, classe Marlin sur la disposition unique, défaut depuis l'adoption du 18/09 : Coder 15 987 j/s, GLM 5 502) | groupe (B0 Triton bf16 persistant, défaut du 17/09 au 18/09, témoin) | w4a16 (B1, NVFP4 lu dans la tuile, opt-in jusqu'au scellé) | grouped_mm (torch, ancien défaut, témoin) | bmm par seaux (réfuté 0edc3b9)"),
    Variable("PREFILL_A4", "off", ("acvram.engine.model", "_PREFILL_A4"), None,
             "porte qualité W4A4 du prefill MoE : fausse quantification NVFP4 des activations en torch — off | gateup (entrée de gate/up) | both (+ entrée de down) ; poste7-lecture-profils-coder-17-09"),
    Variable("PREFILL_A8", "off", ("acvram.engine.model", "_PREFILL_A8"), None,
             "porte qualité W4A8 du prefill MoE (poste7-w4a4-clos-w4a8-porte-19-09) : fausse quantification des activations en torch — off | gateup | both ; format par PREFILL_A8_FMT ; exclusive de PREFILL_A4 ; scellé ratio − 1,0155 ≤ 0,004"),
    Variable("PREFILL_W8R", "0", ("acvram.engine.model", "_PREFILL_W8R"), "0",
             "porte qualité W8r (poste7-poursuite-chantiers-19-09) : experts déquantifiés par _pile_bf16 re-arrondis en int8 symétrique par ligne — 1 = porte ; ne s'applique qu'aux chemins PREFILL_GROUPED=grouped_mm|bmm (pile naturelle, GEMV_LAYOUT=naturel) ; aucun noyau"),
    Variable("PREFILL_A8_FMT", "int8", ("acvram.engine.model", "_PREFILL_A8_FMT"), None,
             "format de la porte A8 : int8 (par jeton, amax/127, l'arrondi de quantifier_a8 au bit) | e4m3 (E4M3 bloc 16, témoin mxf8f6f4)"),
    Variable("SANS_FUSION", "", None, "1"),
    Variable("SANS_FUSION_BF16", "", None, "1"),
    Variable("TETE_LIEE", "int8", ("acvram.engine.loader", "_TETE_LIEE")),
    Variable("TETE_FP32_ENTREE", "", None, "1"),
    # --- MoE ------------------------------------------------------------
    Variable("MOE_MMA", "1", ("acvram.engine.model", "_MOE_MMA"), "0", "prefill W4A4 sur MMA FP4"),
    Variable("MOE_DECODE_MMA_MARLIN", "0", ("acvram.engine.model", "_MOE_DECODE_MMA_MARLIN"), "0",
             "C17 (chantier-c17-mma2-lit-marlin-19-09) : sous la disposition unique Marlin, le décodage MoE par la MMA groupée (MOE_DECODE_MMA, t ≥ MIN_T) lit les TUILES MARLIN au lieu de laisser la GEMV Marlin ; 0 = jamais (défaut jusqu'au scellé) ; Mesure 1-ter sous 2 700 : ×0,89 à 45 distincts, ×1,23 à 8"),
    Variable("MOE_MMA_BT", "64", ("acvram.engine.model", "_MOE_MMA_BT")),
    Variable("MOE_MMA_ETAGES", "4", ("acvram.engine.model", "_MOE_MMA_ETAGES")),
    Variable("MOE_MMA_KS", "128", ("acvram.engine.model", "_MOE_MMA_KS")),
    Variable("MOE_DECODE_MMA", "1", ("acvram.engine.model", "_MOE_DECODE_MMA"), "0"),
    Variable("MOE_DECODE_MMA_BT", "16", ("acvram.engine.model", "_MOE_DECODE_MMA_BT")),
    Variable("MOE_DECODE_MMA_MIN_T", "5", ("acvram.engine.model", "_MOE_DECODE_MMA_MIN_T")),
    Variable("NORME_FUSEE", "0", ("acvram.engine.model", "_NORME_FUSEE"), "0",
             "poste F (3b) : norme d'entrée dans le GEMV int8 q/k/v — RÉFUTÉ 6dbb1bb (+0,31 ms/pas, norme recalculée par bloc), témoin"),
    Variable("ROPE_KV", "0", ("acvram.engine.model", "_ROPE_KV"), "0",
             "poste F (3a) : normes + RoPE + kv_write int8 en un noyau Triton — RÉFUTÉ a3f1b7e (corrompt sous graphe / codes ≠ kv_write_int8), témoin"),
    Variable("ROUTE_PREP", "2", ("acvram.engine.model", "_ROUTE_PREP"), "0",
             "poste F : 2 = moe_route + route_prep fusionnés (F2, défaut, verdict-f2-topk-17-09) | 1 = route_prep seul (F1) | 0 = torch"),
    Variable("MOE_DECODE_FUSED", "0", ("acvram.engine.model", "_MOE_DECODE_FUSED"), "0"),
    Variable("MOE_FUSED_TN", "64", ("acvram.engine.model", "_MOE_FUSED_TN")),
    Variable("MOE_FUSED_ATOMIQUE", "0", ("acvram.engine.model", "_MOE_FUSED_ATOMIQUE"), "0"),
    Variable("MOE_FUSED_ETAGES", "3", ("acvram.engine.model", "_MOE_FUSED_ETAGES")),
    Variable("MOE_ROUTE_PACK", "1", ("acvram.engine.model", "_MOE_ROUTE_PACK"), "0"),
    Variable("MOE_GEMM_MAX", "48", ("acvram.engine.model", "_MOE_GEMM_MAX")),
    Variable("MOE_GROUPED_MAX", "32", ("acvram.engine.model", "_MOE_GROUPED_MAX")),
    Variable("MOE_GLUE_TORCH", "", None, "1", "moe_act + moe_reduce_trie en torch"),
    Variable("PREFILL_DEQUANT", "", None, "1", "experts routés : déquant + _grouped_mm au lieu des GEMM 4 bits"),
    Variable("MOE_DECODE_MASQUES", "", None, "1"),
    Variable("MOE_AWQ_TEMOIN", "0", ("acvram.engine.model", "_MOE_AWQ_TEMOIN")),
    Variable("QA_COMPTE", "0", ("acvram.engine.model", "_QA_COMPTE")),
    # --- MLA ------------------------------------------------------------
    Variable("MLA_BATCH", "2", ("acvram.engine.model", "_MLA_BATCH"), "0"),
    Variable("MLA_BUCKET", "128", ("acvram.engine.mla", "MLA_BUCKET")),
    Variable("MLA_UNE_PASSE", "1", ("acvram.engine.mla", "_MLA_UNE_PASSE"), "0"),
    Variable("MLA_PREP_NOYAU", "1", ("acvram.engine.mla", "_MLA_PREP_NOYAU"), "0"),
    Variable("MLA_GLUE", "0", ("acvram.engine.mla", "_MLA_GLUE"), "0",
             "C15 (chantier-c15-19-09) : 1 = glue torch du décodage MLA b=1 retirée au bit (v_b fp32 une fois, cat kvp, stack RoPE, demi-tables cos/sin, résidu différé add_norm des couches MLA : −6 lancements/couche ; MoE : tok int64 servi, eid converti une fois, x[tok] une fois, tok_g = seq : −3 Marlin / −6 distincte) | 2 = en plus b=1 par decode_static_batch_complet (mla_prep_batch, numérique du lot, ≤ 1 ulp) | 0 = témoin"),
    Variable("MLA_CORE", "tf32", ("acvram.engine.mla", "_MLA_CORE"), "tf32",
             "C13 (poste7-c13a-defaut-19-09 § 1 + addendum) : précision des deux premiers produits du cœur d'attention MLA au PRÉFILL (scores, o_lat), softmax et noyau mla_1p en fp32 inchangés — tf32 DÉFAUT (C13-a tenu : 7 191 j/s, ΔPPL géo +0,00066) | fp32 (référence de qualité) | bf16 (C13-b : entrées bf16, acc fp32 ; scellé contre fp32 : prefill GLM ≥ 9 000 j/s ET ΔPPL géo ≤ +0,002 → défaut)"),
    Variable("MLA_CORE_VB", "0", ("acvram.engine.mla", "_MLA_CORE_VB"), "0",
             "C13 (poste7-c13a-defaut § 2) : 1 = le 3e produit du préfill y = v_b·o_lat (8ae21997) suit MLA_CORE ; 0 = fp32 ; scellé prefill ≥ 7 450 j/s ET ΔPPL géo ≤ +0,001 contre 2 produits → défaut 1"),
    Variable("MLA_CORE_DECODE", "fp32", ("acvram.engine.mla", "_MLA_CORE_DECODE"), "fp32",
             "C13 niveau 2 (poste7-c13a-defaut § 2) : régime du cœur au DÉCODAGE (y = v_b·o_lat, sgemm fp32 1,5 ms/pas à b=12), indépendant de MLA_CORE — fp32 défaut | tf32 | bf16 ; scellé sgemm ≤ 0,6 ms ET ppl-decode-kv 3 tranches ± 0,001 ET capture {1,2,8,12,16} 5/5 → défaut"),
    Variable("MLA_A8", "off", ("acvram.engine.mla", "_MLA_A8"), None,
             "porte qualité FP8-MLA (poste7-cloture-23h59-19-09) : fausse quantification torch de l'ENTRÉE de q_b, kv_a et o — off | e4m3 (E4M3 bloc 16, format de la MMA mxf8f6f4) | int8 (par jeton, témoin) ; aucun noyau"),
    Variable("MLA_LATENT_FP8", "0", ("acvram.engine.mla", "_MLA_LATENT_FP8"), "0"),
    Variable("KV_LM4_SEUL", "", None, None, "diagnostic lm4 (kvcache.write) : lm4 sur k ou v seulement, int8 ailleurs"),
    Variable("KV_LM4_PUITS", "", None, None, "diagnostic lm4 : positions < N gardées int8 ; 0 = contrôle (lm4 partout par le diagnostic)"),
    Variable("KV_FORMAT", "", ("acvram.memory.tiering", "_KV_FORMAT"), None,
             "cache KV des paliers carte : vide = capacités (int8) | lm4 4 bits par rotation | lm3, lm2 témoins"),
    Variable("MLA_NORME_NOYAU", "1", None, "0"),
    Variable("MLA_EAGER_TORCH", "", None, "1"),
    Variable("MLA_DEBUG_ECART", "", None),
    Variable("HYBRID_KERNELS", "1", None, "0", "KDA / GDN : noyaux hybrides ou torch"),
    Variable("GDN", "fla", ("acvram.engine.gdn", "_GDN_VOIE"), "torch",
             "Gated DeltaNet : fla (noyaux Triton de flash-linear-attention, prefill par blocs et décodage du lot en un lancement) | torch (référence transformers, séquence par séquence) ; 0 = refus des hybrides"),
    # --- autres chemins de calcul ----------------------------------------
    Variable("FUSION_NVFP4", "1", None, "0", "témoin de mesure : fusion gate/up NVFP4"),
    Variable("FUSION_PARTIELLE", "0", None, "0"),
    Variable("LOGITS_BF16", "", None, "", "1 : tête en bf16 (précision-de-sortie-invisible-à-la-PPL)"),
    Variable("GRAPHS_EAGER", "", None, "1"),
    Variable("GODETS_B", "1", ("acvram.engine.graphs", "_GODETS_B"), "0",
             "clé de graphe CUDA, dimension b : 1 = lot arrondi au godet (puissances de deux, plafond HYBRID_SLOTS ; en place depuis le 11/09) | 0 = lot exact, témoin de mesure du chantier C4 (revue/chantier-c4-19-09) ; même sortie dans les deux cas"),
    Variable("PA_ARM", "A", None, "A"),
    Variable("SCALER_SANS_CACHE", "", None, "1"),
)

# Variables ACVRAM_* qui ne choisissent PAS un chemin de calcul : le test
# `test_regime_noyaux.py` exige que toute autre variable lue soit dans VARIABLES.
HORS_REGIME = frozenset({
    # (19/09, poste7) : ici SEULEMENT ce qui observe, compile ou nomme un chemin de fichier —
    # jamais une capacité, un plafond, un mode : ceux-là vont dans VARIABLES (HYBRID_SLOTS y
    # manquait : le régime servi de GLM — eager dès b=5 — était invisible dans regime_ligne).
    # tests/test_regime_noyaux.py::test_hors_regime_ne_cache_aucun_regime le garde.
    "ACVRAM_MODELS_DIR", "ACVRAM_TRACEBACK", "ACVRAM_VERBOSE_BUILD", "ACVRAM_WARM_GRAPHS",
    "ACVRAM_GRAPHES_MUETS", "ACVRAM_REGIME_MUET", "ACVRAM_MARLIN_CACHE",          # journaux et cache : observation
    "ACVRAM_TRACE_CRENEAUX", "ACVRAM_TRACE_ENTREES", "ACVRAM_TRACE_PTRS",
    "ACVRAM_TRACE_ROUTAGE", "ACVRAM_TRACE_ROUTAGE_PT", "ACVRAM_TRACE_STEPS", "ACVRAM_TRACE_COUCHES", "ACVRAM_CHRONO_SYNC", "ACVRAM_SYNC_COUCHES",
    
    "ACVRAM_PPL_TRANCHE",
    
    
    
    # compilation, placement, parc, mémoire : pas des chemins de calcul
    "ACVRAM_ARCH_FAMILY", "ACVRAM_CUDA_HOME", "ACVRAM_GW_WARPS",
    "ACVRAM_KERNEL_CACHE", "ACVRAM_BANC_ACCEPTE_REPLAN", 
    
    "ACVRAM_FOND_COOL", "ACVRAM_FOND_ZEN", "ACVRAM_GALERIE_DIR", "ACVRAM_MODELES", "ACVRAM_PARC",
    "ACVRAM_VERROU_GLOB",   # test seul (ajout GUI n°1) : chemin du glob, pas un chemin de calcul
    
    
    
    "ACVRAM_DUMP_MOE",   # dossier de recopie des tampons MoE (diagnostic a2711bc) : n'aiguille aucun calcul
    # garde-fou d'admission, pas un chemin de calcul (poste7-reprise-ordre-18-09
    # §Suite) : Engine.__init__ refuse max_batch_size > plan.kv_planned_seqs,
    # ce flag force le lancement en connaissance de cause. Visible dans
    # regime_ligne() par kv_plan_override=1 quand posé, pas ici.
    
    # lues dans acvram_kernels.cu (getenv) : témoins A/B et réglages d'instrument,
    # jamais mesurés comme défaut — à monter dans VARIABLES le jour où l'un l'est
    
    
          # dossier de compilation du port Marlin (P1), pas un chemin de calcul
    # exportée par outils/carte.sh à ce qu'il lance (son PID) : eco.py s'en sert
    # pour ne pas refuser sa propre prise de la carte ; n'aiguille aucun calcul
    "ACVRAM_CARTE_TENUE",
})


def _format(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _effective(var: Variable) -> str:
    if var.lu_a is not None:
        mod, attr = var.lu_a
        m = importlib.import_module(mod)
        return _format(getattr(m, attr))
    return os.environ.get(var.env, var.defaut)


def regime_noyaux() -> dict:
    """Valeur effective de chaque variable de chemin, plus l'état des
    backends et des masques. Les entrées hors défaut sont ce qu'un lecteur
    doit voir en premier : `regime_ligne` les met en tête."""
    from . import kernels
    from .kernels import backends
    vals = {v.env: _effective(v) for v in VARIABLES}
    ext = kernels.get_extension()
    return {
        "variables": vals,
        "hors_defaut": {v.env: vals[v.env] for v in VARIABLES if vals[v.env] != v.defaut},
        "extension": ext is not None,
        "extension_raison": "" if ext is not None else kernels._ERROR,
        "backends_masques": sorted(backends._MASQUES),
        "noyaux_masques": kernels.noyaux_masques(),
    }


_LIRE_HORLOGE = None   # tests : remplace acvram.eco.lire_horloge quand non None


def _horloge() -> Optional[str]:
    """Étiquette de l'horloge SM (acvram.eco) : lgc<MHz> | libre | ?, ou None
    sans carte (CUDA_VISIBLE_DEVICES vide, ou torch sans CUDA) — à sec la
    ligne de régime ne change pas."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:                                     # noqa: BLE001
        return None
    try:
        from . import eco
        if _LIRE_HORLOGE is not None:
            return eco.etiquette_horloge(_LIRE_HORLOGE(eco.index_carte()))
        return eco.etiquette_horloge(eco.lire_sous_charge(eco.index_carte()))   # jamais au repos
    except Exception:                                     # noqa: BLE001
        return "?"


def regime_ligne() -> str:
    """Une ligne pour l'en-tête d'une mesure : ce qui diffère du défaut,
    puis extension et masques. « défaut » seul veut dire : tout au défaut."""
    r = regime_noyaux()
    parts = [f"{k}={v if v else repr('')}" for k, v in r["hors_defaut"].items()
             if k != "ACVRAM_GEMV_LAYOUT"] or ["défaut"]
    # la disposition lue par le GEMV des experts est toujours nommée (P1
    # disposition unique, poste7-p1-disposition-unique-18-09) : marlin | naturel
    parts.append("ACVRAM_GEMV_LAYOUT=" + str(r["variables"].get("ACVRAM_GEMV_LAYOUT", "?")))
    # la voie GDN est toujours nommée, défaut compris : c'est elle qui sépare
    # 97 de 621 j/s sur Qwen3.8 (poste7, 17/09), et « fla » demandé ne vaut
    # rien si fla est absent ou la carte aussi — la voie EFFECTIVE est écrite
    try:
        from .engine.gdn import gdn_regime
        parts.append("ACVRAM_GDN=" + gdn_regime())
    except Exception as exc:                                 # noqa: BLE001
        parts.append(f"ACVRAM_GDN=?({type(exc).__name__})")
    parts.append("extension=" + ("oui" if r["extension"] else f"non({r['extension_raison']})"))
    if r["backends_masques"]:
        parts.append("backends_masques=" + ",".join(r["backends_masques"]))
    if r["noyaux_masques"]:
        parts.append("noyaux_masques=" + ",".join(r["noyaux_masques"]))
    # Un pip install dans le venv de mesure change l'arithmétique sans qu'aucun
    # défaut ACVRAM_* ne bouge (poste7-glm-etendue-canal-saillant-18-09 § 5) : un
    # JSON sans ces versions ne distingue pas un noyau Triton d'une autre
    # version. torch toujours présent ; triton et fla optionnels.
    try:
        import torch
        parts.append("torch=" + torch.__version__)
    except Exception as exc:                                  # noqa: BLE001
        parts.append(f"torch=?({type(exc).__name__})")
    try:
        import triton
        parts.append("triton=" + triton.__version__)
    except Exception:
        parts.append("triton=absent")
    try:
        import fla
        parts.append("fla=" + getattr(fla, "__version__", "?"))
    except Exception:
        parts.append("fla=absent")
    # L'horloge SM verrouillée (`nvidia-smi -lgc`, mode éco, poste7-e1-eco-tenu-
    # 19-09 § 2) est root et hors processus : aucun défaut ACVRAM_* ne bouge,
    # et un chiffre éco passerait pour un chiffre défaut. Nommée seulement
    # quand une carte est visible : à sec la ligne ne change pas.
    try:
        from .engine import mla as _mla
        if _mla._MLA_CORE != "fp32":
            parts.append(f"mla_core={_mla._MLA_CORE}(≤{_mla._MLA_CORE_MAX_CLES} clés)")
    except Exception:                                     # noqa: BLE001
        pass
    try:
        from . import eco as _eco
        h = _eco.horloge_du_processus()
    except Exception:                                     # noqa: BLE001
        h = None
    if h is not None and h.etat != "sans carte":
        parts.append(h.etiquette())                       # eco=<demandé>(<effectif>[: état])
    else:
        horloge = _horloge()
        if horloge is not None:
            parts.append("horloge=" + horloge)
    return "[régime] " + " ".join(parts)


def masquer(noms) -> dict[str, str]:
    """Envoie chaque chemin nommé vers torch. Rend {nom: ce qui a été fait}.

    Un nom est une variable de VARIABLES (``PREFILL`` ou ``ACVRAM_PREFILL``),
    un backend (``cuda-fusionne``) ou un symbole de l'extension. Une variable
    de seuil (sans valeur torch) ou un nom inconnu sont des erreurs : un
    masque qui ne masque rien fabrique un « sans effet »."""
    from . import kernels
    from .kernels import backends
    par_nom = {v.nom: v for v in VARIABLES}
    par_nom.update({v.env: v for v in VARIABLES})
    fait = {}
    for nom in noms:
        if nom in par_nom:
            var = par_nom[nom]
            if var.torch is None:
                raise ValueError(f"{var.env} est un seuil, pas un chemin : rien à masquer")
            os.environ[var.env] = var.torch
            if var.nom == "DISABLE_KERNELS":
                # l'extension est mémoïsée : la variable seule ne suffit plus
                kernels._EXT, kernels._TRIED, kernels._ERROR = None, True, "masquée (regime.masquer)"
                backends._RESOLVED.clear()
            if var.lu_a is not None:
                mod, attr = var.lu_a
                m = importlib.import_module(mod)
                ancien = getattr(m, attr)
                nouveau = (var.torch == "1" if isinstance(ancien, bool)
                           else type(ancien)(var.torch))
                setattr(m, attr, nouveau)
            fait[nom] = f"{var.env}={var.torch}"
        elif nom in backends.noms():
            backends.masquer(nom)
            fait[nom] = f"backend {nom} retiré"
        else:
            kernels.masquer_noyaux([nom])          # KeyError si inconnu
            fait[nom] = f"noyau {nom} caché"
    return fait
