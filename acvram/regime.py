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
    Variable("GROUPED_RPW", "1", None, None,
             "GEMV groupée des experts : lignes par warp (1 | 2 | 4) ; l'activation est étagée une fois par bloc de GW_WARPS×RPW lignes"),
    Variable("GROUPED_OLD", "", None, "1", "témoin : l'ancien noyau groupé à 4 lignes par bloc"),
    Variable("MOE_GEMV", "v1", ("acvram.engine.model", "_MOE_GEMV"), "v1",
             "GEMV groupée du décodage MoE : v1 (une passe de poids par paire expert-jeton) | v2 (paires triées par expert, poids lus une fois pour ≤ 4 jetons, sortie identique au bit)"),
    Variable("MULTI_PROJ", "0", ("acvram.engine.model", "_MULTI_PROJ"), "0",
             "témoin (palier 2 non ouvert, 0,88 To/s) : q/k/v et qkv/gate/α/β du GDN en un lancement, chacune avec son scaler"),
    Variable("NARROW_MIN_M", "2", ("acvram.kernels", "_NARROW_MIN")),
    Variable("NARROW_ROWS", "32", ("acvram.kernels", "_NARROW_ROWS")),
    Variable("NARROW_MLA", "1", None, "0"),
    Variable("SEUIL_FUSION", "256", ("acvram.engine.model", "SEUIL_FUSION")),
    Variable("PREFILL_GROUPED", "groupe", ("acvram.engine.model", "_PREFILL_GROUPED"), "groupe",
             "GEMM groupée du prefill MoE : groupe (B0 Triton bf16 persistant, défaut depuis poste7-b0-et-cause-lm4-17-09) | w4a16 (B1, NVFP4 lu dans la tuile, opt-in jusqu'au scellé) | grouped_mm (torch, ancien défaut, témoin) | bmm par seaux (réfuté 0edc3b9)"),
    Variable("PREFILL_A4", "off", ("acvram.engine.model", "_PREFILL_A4"), None,
             "porte qualité W4A4 du prefill MoE : fausse quantification NVFP4 des activations en torch — off | gateup (entrée de gate/up) | both (+ entrée de down) ; poste7-lecture-profils-coder-17-09"),
    Variable("SANS_FUSION", "", None, "1"),
    Variable("SANS_FUSION_BF16", "", None, "1"),
    Variable("TETE_LIEE", "int8", ("acvram.engine.loader", "_TETE_LIEE")),
    Variable("TETE_FP32_ENTREE", "", None, "1"),
    # --- MoE ------------------------------------------------------------
    Variable("MOE_MMA", "1", ("acvram.engine.model", "_MOE_MMA"), "0", "prefill W4A4 sur MMA FP4"),
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
    Variable("PA_ARM", "A", None, "A"),
    Variable("SCALER_SANS_CACHE", "", None, "1"),
)

# Variables ACVRAM_* qui ne choisissent PAS un chemin de calcul : le test
# `test_regime_noyaux.py` exige que toute autre variable lue soit dans VARIABLES.
HORS_REGIME = frozenset({
    "ACVRAM_MODELS_DIR", "ACVRAM_TRACEBACK", "ACVRAM_VERBOSE_BUILD",
    "ACVRAM_TRACE_CRENEAUX", "ACVRAM_TRACE_ENTREES", "ACVRAM_TRACE_PTRS",
    "ACVRAM_TRACE_ROUTAGE", "ACVRAM_TRACE_STEPS", "ACVRAM_TRACE_COUCHES", "ACVRAM_CHRONO_SYNC", "ACVRAM_SYNC_COUCHES",
    "ACVRAM_WARM_GRAPHS", "ACVRAM_WARM_SPEC", "ACVRAM_PLAN_FIGE", "ACVRAM_SANS_REPLAN",
    "ACVRAM_SANS_PRECHARGE", "ACVRAM_POOL_SYNC", "ACVRAM_PIPELINE", "ACVRAM_PREFILL_BATCH",
    "ACVRAM_SPECULATION_LOT_MAX", "ACVRAM_MTP", "ACVRAM_HYBRID_SLOTS", "ACVRAM_DENSE_SLOTS", "ACVRAM_DEQUANT_TRANCHE_MAX",
    "ACVRAM_DENSE_ETROIT_BN", "ACVRAM_DENSE_ETROIT_BK", "ACVRAM_DENSE_ETROIT_WARPS", "ACVRAM_DENSE_ETROIT_STAGES", "ACVRAM_INSTA_PAS",
    "ACVRAM_GRAPHES_TABLE", "ACVRAM_MLP_HOTE_CPU", "ACVRAM_KDA_CHUNK", "ACVRAM_MAMBA_CHUNK",
    # compilation, placement, parc, mémoire : pas des chemins de calcul
    "ACVRAM_ALLOC_EXTENSIBLE", "ACVRAM_ARCH_FAMILY", "ACVRAM_CUDA_HOME", "ACVRAM_GW_WARPS",
    "ACVRAM_KERNEL_CACHE", "ACVRAM_BANC_ACCEPTE_REPLAN", "ACVRAM_BUDGET_JETONS",
    "ACVRAM_EXIL_COUCHES", "ACVRAM_EXIL_EXPERTS_FRACTION", "ACVRAM_SEUIL_EXIL", "ACVRAM_REPIN",
    "ACVRAM_FOND_COOL", "ACVRAM_FOND_ZEN", "ACVRAM_GALERIE_DIR", "ACVRAM_MODELES", "ACVRAM_PARC",
    "ACVRAM_VERROU_GLOB",   # test seul (ajout GUI n°1) : chemin du glob, pas un chemin de calcul
    "ACVRAM_LISTE_CLE", "ACVRAM_LISTE_PROMUS", "ACVRAM_MAX_PROMUS", "ACVRAM_ORDRE_SAC",
    "ACVRAM_ORDRE_SAC_INVERSE", "ACVRAM_GRAPHES_MUETS", "ACVRAM_MAX_GRAPHS", "ACVRAM_INSTA_MAX",
    "ACVRAM_REGIME_MUET",
    # lues dans acvram_kernels.cu (getenv) : témoins A/B et réglages d'instrument,
    # jamais mesurés comme défaut — à monter dans VARIABLES le jour où l'un l'est
    "ACVRAM_INT8_GEMV_WARP", "ACVRAM_INT8_TRANCHE", "ACVRAM_PA_CHUNK", "ACVRAM_PA_ETAPE",
    "ACVRAM_PAGED_ALLOC", "ACVRAM_PA_SANS_COMPTEUR",
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


def regime_ligne() -> str:
    """Une ligne pour l'en-tête d'une mesure : ce qui diffère du défaut,
    puis extension et masques. « défaut » seul veut dire : tout au défaut."""
    r = regime_noyaux()
    parts = [f"{k}={v if v else repr('')}" for k, v in r["hors_defaut"].items()] or ["défaut"]
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
