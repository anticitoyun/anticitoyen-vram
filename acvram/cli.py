"""Ligne de commande acvram.

    acvram doctor                     cette machine est-elle prête, et pour quoi
    acvram detect                     quel matériel est présent
    acvram plan MODELE                où irait chaque couche
    acvram convert MODELE -o REP      quantifie, un format par GPU de destination
    acvram serve REP                  serveur compatible avec l'API OpenAI
    acvram eval REP [REP ...]         perplexité, pour classer les formats
    acvram bench REP                  mesure ce que le plan ne faisait qu'estimer
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from typing import Optional

# La version a UNE seule source, acvram/__init__.py. Elle etait ici en dur et
# a derive : le paquet installe le 10/09/2026 annoncait 0.5.0 par dpkg, 0.3.0
# par acvram.__version__ et 0.2.0 par `acvram --version` — trois copies, trois
# valeurs, et celle que l utilisateur voit etait la plus ancienne des trois.
from . import __version__            # noqa: E402  (source unique de verite)



# Réglages À LA DEMANDE qui changent la sortie : l'aide de `serve` les nomme avec leur prix mesuré, gain ET qualité.
_EPILOGUE_SERVE = """\
réglages à la demande (variables d'environnement, hors défaut, HORS BIT) :
  ACVRAM_I8C_FP8_PREFILL=cublas   (pièce 260 ; défaut bf16)
      int8 ré-encodés du fp8 (manifeste « origine: fp8 », ex. Qwen3.8-27B-unsloth-mixte-i8c, 233 tenseurs) servis
      au préfill en W8A8 int8 (activation int8 par jeton, cuBLASLt) au lieu de la déquant bf16.
      gain  : préfill par lot 8 × 78 −26,8 % (0,4105 → 0,3005 s), mur par lot −2,4 %.
      prix  : KL de décodage max 0,215 (9 × le seuil admis), argmax 94,1 % (témoin 99,6 %),
              PPL wiki-gptq +1,05 % (7,0273 → 7,1013). Non retenu au défaut : le gain au mur ne paie pas la qualité.
"""

# Variables d'environnement que le code lit REELLEMENT. Le 9/09/2026,
# `MAXTOK=65536` a ete pose dans l'environnement d'une mesure de perplexite
# que rien ne lisait : la mesure s'est arretee a 16 fenetres au lieu de 128,
# sans que rien ne le signale. Une variable posee qui ne va nulle part est
# une consigne silencieusement ignoree.
#
# Cette liste se met a jour avec le code ; une epreuve verifie qu'elle ne
# derive pas.
#
# Tu viens d'ecrire `os.environ.get("ACVRAM_...")` ailleurs dans le depot ?
# Ajoute le nom ici avant de committer -- cinq variables l'ont deja oublie
# le 10/09, la garde ne les a signalees qu'apres coup, jamais au moment ou
# elles ont ete ecrites.
VARIABLES_LUES = {
    "ACVRAM_ALLOC_EXTENSIBLE",
    "ACVRAM_ARCHS",              # pièce 241 : architectures imposées à la compilation (kernels/__init__.py)
    "ACVRAM_KERNELS_PRECOMPILES",  # pièce 240 : dossier des noyaux .so précompilés (kernels/__init__.py)
    "ACVRAM_CHAUFFE_CTX", "ACVRAM_TYPE",   # chauffe du contexte (runner.py), oubliees de la liste le 20/09 (rouge 21/09)
    "ACVRAM_IMAGES_DIR",                    # dossier d images du serveur (server/protocol.py, P2 multimodal)
    "ACVRAM_PREFILL_COMPACT",          # C15-prefill (aaf9f9c3), oubliees de la liste : test_la_liste_des_variables_lues_ne_derive_pas rouge sur main
    "ACVRAM_PREFILL_COMPACT_ITEMS",
    "ACVRAM_BANC_ACCEPTE_REPLAN",
    "ACVRAM_KV_PLAN_OVERRIDE",
    "ACVRAM_KERNEL_CACHE",
    "ACVRAM_FOND_COOL",
    "ACVRAM_FOND_ZEN",
    "ACVRAM_GALERIE_DIR",
    "ACVRAM_LOGITS_BF16",
    "ACVRAM_MARLIN_CACHE",
    "ACVRAM_MAX_GRAPHS",
    "ACVRAM_PARC",
    "ACVRAM_MODELES",
    "ACVRAM_SEUIL_EXIL",
    "ACVRAM_REPIN",
    "ACVRAM_REGIME_MUET",
    "ACVRAM_VERROU",
    "ACVRAM_NOM",
    "ACVRAM_MLA_DEBUG_ECART",
    "ACVRAM_MLA_EAGER_TORCH",
    "ACVRAM_MLA_LATENT_FP8",
    "ACVRAM_KV_FORMAT",
    # Repli 104 (1) (memory/kv_k8v4.py) : sous k8v4, les 16 premières positions de chaque séquence gardent
    # V en int8 par jeton dans une réserve (0 ou 16) ; la ligne de régime imprime `kv=k8v4+puits16`.
    "ACVRAM_KV_PUITS",
    # C5-b (memory/kv_canal.py) : clés int8 par canal, et la taille de la
    # réserve bf16 des blocs courants.
    "ACVRAM_KV_INT8_CANAL",
    "ACVRAM_KV_CANAL_RANGS",
    "ACVRAM_ARBRE_LIBRE",       # garde d'import (__init__.py) : contournement nommé, arbre ≠ cwd
    "ACVRAM_DEPAQUETAGE",       # kernels/marlin_port/__init__.py : auto | cuda | triton | torch
    "ACVRAM_GDN_PREFILL_LOT",   # engine/couches.py : lot au préfill GDN
    "ACVRAM_GDN_ETAT_EN_PLACE", # engine/gdn.py : 156 F4, état GDN mis à jour en place (défaut 1)
    "ACVRAM_GDN_AB",            # engine/gdn.py : 175, portes α‖β bf16 en un appel (auto défaut | separe témoin | concat | triton)
    "ACVRAM_GDN_CONV_FUSEE",    # engine/gdn.py : 156 F2, conv de décodage fusionnée (défaut 1)
    "ACVRAM_ADMISSION_FENETRE_MS",  # server/app.py : 179, fenêtre d'admission (défaut 5 ms, en rafale)
    "ACVRAM_ADMISSION_GUET",    # server/app.py : 269 b, porte de la fenêtre à 1 requête si une autre est entrée (opt-in, défaut 0)
    "ACVRAM_DEPAQ_PARTAGE",     # kernels : 172, poids déquantifié partagé par la boucle par séquence (défaut 1, au bit)
    "ACVRAM_GDN_PORTES_NOYAU",  # engine/gdn.py : 156 F1, portes dans le noyau fla (défaut 1, ± ulp)
    "ACVRAM_GDN_NORME_FUSEE",   # engine/gdn.py : 156 F3, norme gated Triton (défaut 1, ± ulp)
    "ACVRAM_GDN_AB_FLUX",       # engine/gdn.py : 194 b2, β‖α sur un second flux pendant qkv‖gate (défaut 1, au bit ; 0 témoin)
    "ACVRAM_GDN_Z_BF16",        # engine/gdn.py : 182, z sans cast fp32 au décodage (défaut 1, au bit)
    "ACVRAM_GDN_QKV_GATE",      # engine/gdn.py : 176, qkv‖gate INT8 en une pile au décodage (défaut 1, au bit)
    "ACVRAM_NORME_REGISTRES",   # engine/layers.py : 156 F6, RMSNorm en registres (défaut 1, au bit)
    "ACVRAM_GDN_RES_DIFFERE",   # engine/model.py : 156 F5, résidu différé des couches GDN (défaut 1)
    "ACVRAM_HFQUANT_PAR_GROUPE",  # quant/hfquant.py : dispatch par groupe du compressed-tensors mixed-precision
    "ACVRAM_PREFILL_GROUPED",
    "ACVRAM_PAGED_ATTN",
    "ACVRAM_NARROW_KERNEL",
    "ACVRAM_NARROW_TRITON_MIN_B",
    "ACVRAM_ROUTE_PREP",
    "ACVRAM_ROPE_KV",
    "ACVRAM_NORME_FUSEE",
    # Deux interrupteurs de diagnostic (poste7-kv-lm4-clos-17-09 § 1), lus par
    # memory/kv_lm4.py::actif / hors_puits, pas par le noyau ou le chargeur :
    # la passe de cause de poste3 (K seul, V seul, puits) decide seulement si
    # tq3+1 s ecrit apres le commit B de poste4.
    "ACVRAM_KV_LM4_SEUL",
    "ACVRAM_KV_LM4_PUITS",
    # Bras du banc a trois bras de l'attention paginee. Lu par le noyau, donc
    # il DOIT etre declare ici : la garde l'a signale comme inconnu, ce qui est
    # exactement son role — une variable posee qui ne va nulle part est une
    # consigne silencieusement ignoree.
    "ACVRAM_PA_ARM",
    "ACVRAM_PREFILL_BATCH",
    "ACVRAM_BUDGET_JETONS",
    # Echappement de mesure de la fusion, pour les QUATRE empileurs.
    # ACVRAM_SANS_FUSION_BF16 ne coupait que le chemin bf16 : le gain de la
    # fusion n'etait donc mesurable qu'en bf16, et c'est ainsi qu'un +2,60 %
    # mesure la ou 100 % des groupes fusionnent a ete transporte sur un int8
    # ou 7,8 % seulement fusionnent.
    "ACVRAM_SANS_FUSION",
    # Renverse le signe de l'ordre du glouton budgetaire, RIEN D'AUTRE. Sert a
    # eprouver si le critere (gain de SNR par octet) est bien oriente pour un
    # objectif de perplexite : si oui le bras inverse est nettement pire, si non
    # il est meilleur ou equivalent. Instrument de mesure, pas reglage.
    "ACVRAM_ORDRE_SAC_INVERSE",
    # Choisit la cle de tri du glouton budgetaire : `snr` (defaut, gain de
    # decibels par octet) ou `erreur` (erreur de sortie evitee par octet). La
    # seconde n'est PAS une transformation monotone de la premiere : elle
    # applique 10^(-snr/20) aux deux SNR AVANT la soustraction, donc elle
    # privilegie les tenseurs a faible SNR de base — verifie sur nos donnees,
    # correlation -0,66 entre SNR de base et deplacement de rang. Un mode
    # inconnu leve une erreur au lieu de retomber en silence sur le defaut.
    "ACVRAM_ORDRE_SAC",
    "ACVRAM_LISTE_PROMUS",
    "ACVRAM_LISTE_CLE",
    "ACVRAM_MAX_PROMUS",
    "ACVRAM_SCALER_SANS_CACHE",
    # ACVRAM_SRC_HASH n'est PAS une variable d'environnement : c'est une option
    # de compilation (-D) portant le sha du .cu. Elle figure ici parce que
    # l'epreuve la reconnait au motif ACVRAM_[A-Z0-9_]+ dans le source, et
    # qu'une liste incomplete fait echouer la garde — pas parce qu'on la lit.
    "ACVRAM_SRC_HASH",
    "ACVRAM_ARCH_FAMILY",
    "ACVRAM_COLLE_MOE",
    "ACVRAM_CUDA_HOME",
    "ACVRAM_DENSE_ETROIT_BK",
    "ACVRAM_DENSE_ETROIT_BN",
    "ACVRAM_DENSE_ETROIT_STAGES",
    "ACVRAM_DENSE_ETROIT_WARPS",
    "ACVRAM_DENSE_NVFP4",
    "ACVRAM_DENSE_NVFP4_MIN_M",
    "ACVRAM_DENSE_SLOTS",
    "ACVRAM_DEQUANT_TRANCHE_MAX",
    "ACVRAM_TRANCHE_COPIE_MIN",
    "ACVRAM_DISABLE_CPU_KERNELS",
    "ACVRAM_DISABLE_CUDA_GRAPHS",
    "ACVRAM_DISABLE_FP4_GEMM",
    "ACVRAM_DISABLE_KERNELS",
    "ACVRAM_DISABLE_PAGED_ATTN",
    "ACVRAM_DOUBLE_DISPOSITION_DIAG",
    "ACVRAM_DUMP_MOE",
    "ACVRAM_EXIL_COUCHES",
    "ACVRAM_EXIL_EXPERTS_FRACTION",
    "ACVRAM_FUSION_NVFP4",
    "ACVRAM_FUSION_PARTIELLE",
    "ACVRAM_GC_FREEZE",
    "ACVRAM_GDN",
    "ACVRAM_GEMV_LAYOUT",
    "ACVRAM_GEMV_SPLITK",
    "ACVRAM_GODETS_B",
    "ACVRAM_GRAPHES_MUETS",
    "ACVRAM_GRAPHES_TABLE",
    "ACVRAM_GROUPED_OLD",
    "ACVRAM_GROUPED_RPW",
    "ACVRAM_GROUPED_XREG",
    "ACVRAM_GRAPHS_EAGER",
    "ACVRAM_GW_WARPS",
    "ACVRAM_HYBRID_KERNELS",
    "ACVRAM_HYBRID_SLOTS",
    "ACVRAM_INSTA_MAX",
    "ACVRAM_INSTA_PAS",
    "ACVRAM_INT8_GEMV_MAX",
    "ACVRAM_INT8_GEMV_MAX_PARTAGE",   # pièce 243 : seuil GEMV→GEMM int8 sous B′ (16 défaut, 80 témoin, hors bit)
    "ACVRAM_INT8_GEMV_WARP",
    "ACVRAM_PA_SANS_COMPTEUR",
    "ACVRAM_PAGED_ALLOC",
    "ACVRAM_PA_ETAPE",
    "ACVRAM_PA_CHUNK",
    "ACVRAM_PA_GQA",            # kernels/acvram_kernels.cu : 182, attention paginée groupée GQA (défaut actif, au bit)
    "ACVRAM_INT8_TRANCHE",
    "ACVRAM_INT8_TRANCHE_PREFILL",
    "ACVRAM_KDA_CHUNK",
    "ACVRAM_MAMBA_CHUNK",
    "ACVRAM_MLA_BATCH",
    "ACVRAM_MLA_BUCKET",
    "ACVRAM_MLA_GLUE",
    "ACVRAM_MLA_NORME_NOYAU",
    "ACVRAM_MLA_PREP_NOYAU",
    "ACVRAM_MLA_BATCH_FUSION",
    "ACVRAM_MLA_PREP_GRILLE",
    "ACVRAM_MLA_UNE_PASSE",
    "ACVRAM_MLP_HOTE_CPU",
    "ACVRAM_MODELS_DIR",
    "ACVRAM_MOE_DECODE_MASQUES",
    "ACVRAM_MOE_AWQ_TEMOIN",
    "ACVRAM_MOE_DECODE_FUSED",
    "ACVRAM_MOE_DECODE_MMA",
    "ACVRAM_MOE_DECODE_MMA_BT",
    "ACVRAM_MOE_DECODE_MMA_MIN_T",
    "ACVRAM_MOE_FUSED_ATOMIQUE",
    "ACVRAM_MOE_FUSED_ETAGES",
    "ACVRAM_MOE_FUSED_TN",
    "ACVRAM_MOE_ROUTE_PACK",
    "ACVRAM_NARROW_GEMM",
    "ACVRAM_NARROW_MIN_M",
    "ACVRAM_NARROW_MLA",
    "ACVRAM_NARROW_NVFP4",
    "ACVRAM_NARROW_ROWS",
    "ACVRAM_QA_COMPTE",
    "ACVRAM_MOE_GEMM_MAX",
    "ACVRAM_MOE_GLUE_TORCH",
    "ACVRAM_MOE_GROUPED_MAX",
    "ACVRAM_MOE_MMA",
    "ACVRAM_MOE_MMA_BT",
    "ACVRAM_MOE_MMA_ETAGES",
    "ACVRAM_MOE_GEMV",
    "ACVRAM_MOE_MMA_KS",
    "ACVRAM_MTP",
    "ACVRAM_MULTI_PROJ",
    "ACVRAM_NVFP4_GEMV_MAX",
    "ACVRAM_PIPELINE",
    "ACVRAM_SAMPLER_LOT",       # sampler vectorisé en opt-in (665eeacc)
    "ACVRAM_SAMPLER_LENT", "ACVRAM_SAMPLER_GRAPHE",             # levier 1 (défaut graphe, témoin lent, ancien nom lu)
    "ACVRAM_RAPATRIEMENT_FLUX", "ACVRAM_RAPATRIEMENT_EPINGLE",  # levier 2 (défaut épinglé, témoin flux, ancien nom lu)
    "ACVRAM_CAPTURE_MEM_MIN_MIO", "ACVRAM_CAPTURE_DELAI_S",   # gardes d interblocage de capture (22/09)
    "ACVRAM_ECHO_TRANCHE", "ACVRAM_ETROITES_FORME",   # opt-in ± 1 ulp : facteur de programmes par SM du split-K étroit (22/09)
    "ACVRAM_ETROIT_CANAL",                            # pièce 195 : K entier par canal, hors bit, défaut 1 depuis le 25/09 (0 = témoin)
    "ACVRAM_JOURNAL_TENSEURS",  # journal par tenseur de convert (cf97a3a0) — oubliées de la liste le 22/09 (rouge sur main)
    "ACVRAM_PLAN_FIGE",
    "ACVRAM_POOL_SYNC",
    "ACVRAM_PREFILL",
    "ACVRAM_PREFILL_A4",
    "ACVRAM_PREFILL_A8",
    "ACVRAM_MLA_A8",
    "ACVRAM_MLA_CORE", "ACVRAM_MLA_CORE_VB", "ACVRAM_MLA_CORE_DECODE", "ACVRAM_MLA_CORE_MAX_CLES",
    # C13-c (flash), sondes niveau 2, C15-3d (glue compacte) — 20/09
    "ACVRAM_MLA_FLASH_OPERANDES", "ACVRAM_MLA_FLASH_TUILE",
    "ACVRAM_MLA_ECRIT_TORCH", "ACVRAM_MLA_PREP_TEMOIN", "ACVRAM_MLA_QABS_DEUX_MOITIES",
    "ACVRAM_GLUE_COMPACT", "ACVRAM_GLUE_COMPACT_ITEMS", "ACVRAM_ATTN_WARPS_COMPACT", "ACVRAM_ATTN_REDUC_DEROULEE", "ACVRAM_PROJ_MARLIN", "ACVRAM_PROJ_MARLIN_MIN_M", "ACVRAM_PROJ_MARLIN_MIN_NK", "ACVRAM_PROJ_MARLIN_MIN_N", "ACVRAM_PROJ_MARLIN_DOUBLES", "ACVRAM_PROJ_MARLIN_CAPACITE", "ACVRAM_MTP_ETAT", "ACVRAM_GEMV_MARLIN_V2", "ACVRAM_GEMV_MARLIN_TPB", "ACVRAM_GEMV_MARLIN_S", "ACVRAM_PROJ_MARLIN_PORTEE", "ACVRAM_ROUTAGE_TEMOIN",
    "ACVRAM_PILE_SANS_RENDU",
    "ACVRAM_CPUS",                                    # 0.6.31 : affinité (acvram/hote.py)                          # gemma (c48c2b2c) : témoin de _rendre_le_cache_apres_la_pile
    "ACVRAM_PREFILL_W8R",
    "ACVRAM_MARLIN_DISTINCT",
    "ACVRAM_PREFILL_A8_FMT",
    "ACVRAM_PPL_TRANCHE",
    "ACVRAM_GEMV_LAYOUT",
    "ACVRAM_DOUBLE_DISPOSITION_DIAG",
    "ACVRAM_DUMP_MOE",
    "ACVRAM_PREFILL_INT8",
    "ACVRAM_I8C_FP8_PREFILL",   # pièce 260 : int8 d'origine fp8 au préfill (bf16 défaut | cublas)
    "ACVRAM_I8C_COPIE",         # pièce 260x : copie signée xor (défaut, au bit) | int16 (témoin)
    "ACVRAM_PREFILL_DEQUANT",
    "ACVRAM_SANS_FUSION_BF16",
    "ACVRAM_SANS_PRECHARGE",
    "ACVRAM_SANS_REPLAN",
    "ACVRAM_SEUIL_FUSION",
    "ACVRAM_SPECULATION_LOT_MAX",
    "ACVRAM_SYNC_COUCHES",
    "ACVRAM_TETE_LIEE",
    "ACVRAM_TRACEBACK",
    "ACVRAM_TETE_FP32_ENTREE",
    "ACVRAM_TRACE_CRENEAUX",
    "ACVRAM_TRACE_ENTREES",
    "ACVRAM_TRACE_PTRS",
    "ACVRAM_TRACE_ROUTAGE",
    "ACVRAM_TRACE_ROUTAGE_PT",
    "ACVRAM_CHRONO_SYNC",
    "ACVRAM_TRACE_COUCHES",
    "ACVRAM_TRACE_STEPS",
    "ACVRAM_VERBOSE_BUILD",
    "ACVRAM_VERROU_GLOB",
    "ACVRAM_WARM_GRAPHS",
    "ACVRAM_WARM_SPEC",
    # exportee par outils/carte.sh (son PID) a ce qu'il lance ; lue par eco.py
    # pour ne pas refuser sa propre prise de la carte
    "ACVRAM_CARTE_TENUE", "ACVRAM_ECO", "ACVRAM_MOE_DECODE_MMA_MARLIN",
    # pièces 62-65 (23/09) : MoE sur tensor cores (défaut aux godets ≥ MIN_T), glue fusionnée, seuil
    "ACVRAM_MOE_TENSOR", "ACVRAM_MOE_TENSOR_FUSION", "ACVRAM_MOE_TENSOR_MIN_T",
    # pièce 123 (24/09) : tables AWQ des experts sur le chemin tensor, opt-in (hors défaut, FAUX au critère relatif)
    "ACVRAM_AWQ_TENSOR",
    # pièce 82 (23/09) : gate·up fusionnés (w13), opt-in
    "ACVRAM_MOE_W13", "ACVRAM_MARLIN_PAR_LIGNE",
}


def _avertir_variables_inconnues() -> None:
    """Signale toute ACVRAM_* posee que le code ne lit pas."""
    inconnues = sorted(v for v in os.environ
                       if v.startswith("ACVRAM_") and v not in VARIABLES_LUES)
    if inconnues:
        print(red("  variables ignorees (le code ne les lit nulle part) : "
                  + ", ".join(inconnues)), flush=True)

def _tty() -> bool:
    """Une barre de progression a sa place sur un terminal, pas dans un tube ni un journal."""
    return sys.stderr.isatty() and not os.environ.get("NO_COLOR")


def _progress(text: str) -> None:
    """Ligne d'avancement. Sur un terminal elle s'ecrase ; redirigee vers un
    fichier elle s'empile, faute de quoi une conversion lancee en tache de fond
    ne montre plus rien du tout."""
    if _tty():
        sys.stderr.write(f"\r{text[:100]:<100}")
    else:
        sys.stderr.write(text.strip() + "\n")
    sys.stderr.flush()


def _progress_done() -> None:
    if _tty():
        sys.stderr.write("\r" + " " * 100 + "\r")
        sys.stderr.flush()


def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


def _h(n: float) -> str:
    for unite in ("o", "Kio", "Mio", "Gio", "Tio"):
        if abs(n) < 1024 or unite == "Tio":
            return f"{int(n)} o" if unite == "o" else f"{n:.1f} {unite}"
        n /= 1024
    return f"{n:.1f} Tio"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_detect(args: argparse.Namespace) -> int:
    from .hardware.detect import detect_rig
    rig = detect_rig(args.profile)
    if args.json:
        print(rig.to_json())
        return 0

    print(bold("materiel"))
    print(f"  source        {rig.source}")
    print(f"  distribution  {rig.distro}")
    print(f"  noyau         {rig.kernel}")
    print(f"  processeur    {rig.cpu.model}")
    if rig.cpu.efficiency_cores:
        print(f"                {rig.cpu.performance_cores} coeurs P + "
              f"{rig.cpu.efficiency_cores} coeurs E, "
              f"epinglez les fils sur le cpuset {rig.cpu.p_core_cpuset}")
    print(f"  memoire vive  {_h(rig.host.total)} au total, "
          f"{_h(rig.host.available)} disponibles")
    print(f"  pilote / cuda {rig.driver_version or '?'} / {rig.cuda_version or '?'}")
    print()
    if not rig.gpus:
        print(yellow("  aucun GPU NVIDIA detecte"))
        return 0
    print(bold("cartes graphiques"))
    for g in rig.gpus:
        caps = g.caps
        print(f"  [{g.index}] {g.name}")
        print(f"       {_h(g.total_mem)} de VRAM, ~{g.vram_bandwidth_gbps:.0f} Go/s")
        print(f"       sm_{caps.sm} ({caps.arch_name})   "
              f"fp4={_yn(caps.fp4_tensor_core)} fp8={_yn(caps.fp8_tensor_core)} "
              f"int8={_yn(caps.int8_tensor_core)} bf16={_yn(caps.bf16)}")
        print(f"       PCIe gen{g.pcie_gen_cur or g.pcie_gen_max} "
              f"x{g.pcie_width_cur or g.pcie_width_max} "
              f"-> {g.host_link_gbps:.1f} Go/s vers l'hote")
        print(f"       {green('poids ' + caps.weight_format)}, cache KV {caps.kv_format}")
    if len(rig.gpus) > 1 and not any(rig.p2p_matrix[0][1:]):
        print()
        print(dim("  pas de pair-a-pair entre GPU (attendu sur GeForce) : les "
                  "tenseurs transitent par la memoire hote epinglee"))
    return 0


def _yn(b: bool) -> str:
    return green("oui") if b else dim("non")


def cmd_doctor(args: argparse.Namespace) -> int:
    from .hardware.detect import detect_rig
    ok = True
    print(bold("acvram doctor"))

    try:
        import torch
        print(f"  {green('ok')}    torch {torch.__version__}, "
              f"cuda {torch.version.cuda or 'cpu-only'}")
    except ImportError:
        print(f"  {red('ECHEC')} torch n'est pas installe")
        return 1

    rig = detect_rig()
    if not rig.gpus:
        print(f"  {yellow('alerte')} aucun peripherique CUDA ; acvram tournera sur processeur seul")
    for g in rig.gpus:
        caps = g.caps
        line = f"  {green('ok')}    [{g.index}] {g.name} sm_{caps.sm} -> {caps.weight_format}"
        print(line)
        if caps.sm >= 120:
            cuda = torch.version.cuda or "0.0"
            major, _, minor = cuda.partition(".")
            if (int(major or 0), int(minor or 0)) < (12, 8):
                ok = False
                print(f"  {red('ECHEC')} {g.name} est Blackwell (sm_{caps.sm}) mais "
                      f"torch est compile pour CUDA {cuda}. sm_120 exige 12.8+. "
                      f"Installez : pip install torch --index-url "
                      f"https://download.pytorch.org/whl/cu128")
        if g.pcie_width_cur and g.pcie_width_cur < g.pcie_width_max:
            print(f"  {yellow('alerte')} [{g.index}] le lien tourne en x"
                  f"{g.pcie_width_cur} au lieu de x{g.pcie_width_max} ; les couches "
                  f"transferees seront {g.pcie_width_max / g.pcie_width_cur:.0f}x plus lentes")

    from . import kernels
    info = kernels.build_info()
    if info["available"]:
        print(f"  {green('ok')}    noyaux CUDA fusionnes compiles pour "
              f"{', '.join(info['device_caps']) or 'n/a'}")
    else:
        print(f"  {yellow('alerte')} noyaux CUDA fusionnes indisponibles "
              f"({info['error']}) ; chemin de reference utilise")

    cpu = info["cpu"]
    if cpu["available"]:
        simd = "AVX2+FMA" if cpu["avx2"] else "scalaire"
        note = "" if cpu["avx2"] else "  (pas d'AVX2 sur ce processeur -- bien plus lent)"
        print(f"  {green('ok')}    noyaux processeur compiles, chemin {simd}{note}")
    else:
        print(f"  {yellow('alerte')} noyaux processeur indisponibles ({cpu['error']}) ; "
              f"l'etage hote sera lent")

    fp4 = info["fp4_tensorcore"]
    if fp4["available"]:
        print(f"  {green('ok')}    produit FP4 sur tensor cores via {fp4['impl']}")
    else:
        print(f"  {yellow('alerte')} pas de produit FP4 sur tensor cores : {fp4['reason']}")
        print(f"        le prefill retombe sur dequantification + cuBLAS ; le "
              f"decodage n'est pas affecte")

    from .kernels import backends as bk
    print(f"  {green('ok')}    backends retenus par (format, peripherique) :")
    for row in bk.table():
        print(f"           {row['device']:<8} {row['format']:<9} -> "
              f"{' > '.join(row['backends'])}")

    ok = _doctor_modules() and ok

    from .engine.gdn import gdn_available
    if gdn_available():
        print(f"  {green('ok')}    Gated DeltaNet (hybrides Qwen3-Next/kimi)")
    else:
        print(f"  {yellow('alerte')} Gated DeltaNet indisponible ; les modeles "
              f"hybrides Qwen3-Next/kimi seront refuses au chargement : "
              f"pip install -e '.[gdn]'")

    if rig.host.total and rig.host.total < 32 * 1024 ** 3:
        print(f"  {yellow('alerte')} {_h(rig.host.total)} de memoire vive limitent "
              f"l'etage hote")
    _doctor_eco(cuda_ok=torch.cuda.is_available())
    return 0 if ok else 1


# (module importe, paquet pip) : PIL s'importe ainsi mais s'installe « pillow ».
_MODULES_REQUIS = (("safetensors", "safetensors"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                   ("tokenizers", "tokenizers"), ("jinja2", "jinja2"))
# Extras de pyproject : optionnels, donc une ALERTE et non un ECHEC (273, decision chef du 26/09 — renverse le
# 21/09 « trou P3 » qui les mettait en ECHEC : l'utilisateur de .deb ou de pip sans multimodal ne doit pas voir un
# doctor rouge ; un modele multimodal sans vision echoue toujours au chargement, en clair). Le Flatpak embarque vision.
_EXTRAS = {"vision": (("transformers", "transformers"), ("PIL", "pillow"))}


def _doctor_modules() -> bool:
    """Lignes ok/ECHEC des modules requis, ok/alerte par extra ; False si un REQUIS manque."""
    ok = True
    for mod, paquet in _MODULES_REQUIS:
        try:
            __import__(mod)
            print(f"  {green('ok')}    {mod}")
        except ImportError:
            ok = False
            print(f"  {red('ECHEC')} {mod} est absent (pip install {paquet})")
    for extra, mods in _EXTRAS.items():
        absents = []
        for mod, paquet in mods:
            try:
                __import__(mod)
            except ImportError:
                absents.append(paquet)
        if absents:
            print(f"  {yellow('alerte')} {extra} indisponible ({', '.join(absents)} absent) : "
                  f"pip install 'acvram[{extra}]'")
        else:
            print(f"  {green('ok')}    {extra} ({', '.join(m for m, _ in mods)})")
    return ok


def _doctor_eco(cuda_ok: bool) -> None:
    """Éco par défaut (poste7-eco-2700-defaut-19-09 § 4) : le droit sudo ET
    l'effet — `-lgc` puis lecture puis `-rgc`, sous le verrou de carte seulement
    (jamais pendant la mesure d'un pair) ; sans carte ou carte tenue : le droit
    seul (`sudo -n -l`)."""
    from . import eco
    mode = eco.mode_demande()
    print(f"  {dim('eco')}   mode demande : {mode} (config {eco.CONFIG}"
          f"{', ACVRAM_ECO pose' if os.environ.get('ACVRAM_ECO') else ''})")
    try:
        r = subprocess.run(["sudo", "-n", "-l", "nvidia-smi"], capture_output=True, text=True, timeout=15)
        droit = r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        droit = False
    if not droit:
        print(f"  {yellow('alerte')} pas de droit sudo -n sur nvidia-smi : le service tournera a "
              f"l'horloge libre, eco={mode}(libre: refus sudo), aucune cellule publiable.\n"
              f"        ligne sudoers (visudo -f /etc/sudoers.d/acvram-nvidia-smi) :\n"
              f"        {os.environ.get('USER', 'utilisateur')} ALL=(root) NOPASSWD: "
              f"/usr/bin/nvidia-smi -i * -lgc *\\,*, /usr/bin/nvidia-smi -i * -rgc")
        return
    print(f"  {green('ok')}    sudo -n nvidia-smi autorise")
    if not cuda_ok or mode == "off":
        return
    index = eco.index_carte()
    if eco._tenue_par_un_pair(index) is not None:
        print(f"  {yellow('alerte')} carte {index} tenue par un pair : l'effet de -lgc n'est pas "
              f"verifie (relancer sous outils/carte.sh quand elle est libre)")
        return
    h = eco.Horloge(mode, index)
    etat = h.poser()
    lu = h.effectif
    h.rendre()
    if etat == "effectif":
        print(f"  {green('ok')}    -lgc {mode},{mode} pris (lu {lu} MHz), -rgc rendu")
    else:
        print(f"  {red('ECHEC')} -lgc {mode},{mode} : {etat} (lu {lu}) — eco demande ≠ effectif, "
              f"aucune cellule publiable")


def cmd_plan(args: argparse.Namespace) -> int:
    from .engine.config import load_model_spec
    from .hardware.detect import detect_rig
    from .memory.tiering import PlannerOptions, auto_plan
    rig = detect_rig(args.profile)
    spec = load_model_spec(args.model, args.name)
    opts = PlannerOptions(
        max_model_len=args.max_model_len,
        max_concurrent_seqs=args.max_seqs,
        kv_bits=args.kv_bits,
        host_fraction=args.host_fraction,
        group_size=args.group_size,
        allow_host_tier=not args.no_host,
        force_format=args.format,
        gpus=args.gpus,
        host_exec=args.host_exec,
        host_compute_gb_s=args.host_gb_s,
    )
    plan, trials = auto_plan(spec, rig, opts)
    if args.json:
        print(json.dumps({"spec": spec.to_dict(), "plan": plan.to_dict(),
                          "trials": trials}, indent=2))
        return 0
    print(bold(spec.summary()))
    print()
    print(plan.render())
    return 0


def _calib_source(use_awq: bool, calib_file: Optional[str]) -> dict:
    """Nom + sha256 du corpus de calibration réellement utilisé.

    poste7 (`poste7-corpus-16-09.md` § 8) : le manifeste ne portait jamais QUEL
    corpus avait servi à la calibration — le doute sur GLM -k48 (calibré
    sur wiki-gptq, le corpus d'ÉVAL ?) ne pouvait pas se trancher en lisant
    le manifeste seul. Nom + sha256 du fichier réellement utilisé, ou du
    corpus intégré (acvram/data/calibration-anglais.txt) quand aucun `--calib-file` n'est
    fourni, ou une absence explicite quand awq=False — jamais une absence
    ambiguë.
    """
    import hashlib
    if not use_awq:
        return {"fichier": None, "sha256": None, "note": "aucune (awq=False)"}
    if calib_file and os.path.isfile(calib_file):
        with open(calib_file, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        return {"fichier": os.path.abspath(calib_file), "sha256": sha, "note": None}
    from .quant.collect import default_calib_path
    chemin = default_calib_path()
    with open(chemin, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {"fichier": chemin, "sha256": sha,
            "note": "corpus integre acvram/data/calibration-anglais.txt (Gutenberg #1342), "
                    "aucun --calib-file fourni"}


def cmd_convert(args: argparse.Namespace) -> int:
    from .engine.config import load_model_spec
    from .hardware.detect import detect_rig
    from .memory.tiering import PlannerOptions, auto_plan
    from .quant.convert import ConversionOptions, convert_checkpoint

    rig = detect_rig(args.profile)
    spec = load_model_spec(args.model, args.name)
    if not args.out:
        # Les chemins de la machine de developpement etaient codes ici en
        # dur, avec le nom d'utilisateur dedans : ils partaient tels quels
        # dans le paquet .deb, chez quiconque l'installe. Le repli se lit
        # desormais dans un fichier de configuration, absent par defaut.
        # ACVRAM_MODELS_DIR d'abord (heritage), sinon ACVRAM_MODELES partage
        # avec outils/racine_modeles.py et acvram/server/app.py.
        base = os.environ.get("ACVRAM_MODELS_DIR") or os.environ.get("ACVRAM_MODELES")
        if not base:
            conf = os.path.join(
                os.environ.get("XDG_CONFIG_HOME",
                               os.path.expanduser("~/.config")),
                "acvram", "modeles")
            try:
                with open(conf) as f:
                    for ligne in f:
                        ligne = ligne.strip()
                        if ligne and not ligne.startswith("#") and os.path.isdir(ligne):
                            base = ligne
                            break
            except OSError:
                pass
        if not base:
            print(red("aucun repertoire de sortie : passez -o, posez "
                      "ACVRAM_MODELS_DIR, ou ecrivez un chemin par ligne "
                      f"dans {conf}"))
            return 2
        args.out = os.path.join(base, spec.name.replace("/", "--"))
        print(f"  sortie : {args.out}")
    plan, _ = auto_plan(spec, rig, PlannerOptions(
        max_model_len=args.max_model_len, max_concurrent_seqs=args.max_seqs,
        group_size=args.group_size, force_format=args.format, gpus=args.gpus,
        host_exec=args.host_exec, host_compute_gb_s=args.host_gb_s))
    print(bold(spec.summary()))
    print()
    print(plan.render())
    print()
    # Drapeau structuré : chercher une sous-chaîne dans un message destiné à
    # l'utilisateur casse dès que ce message change de langue.
    if plan.overflowed and not args.force:
        print(red("conversion refusee : le modele ne tient pas. "
                  "Relancez avec --force pour ecrire les fragments malgre tout."))
        return 2
    from .quant.convert import refus_nvfp4_sans_gpu
    refus = refus_nvfp4_sans_gpu(plan, args.out, args.format)
    if refus:                                   # le nom ne peut plus mentir sur le format (30B « nvfp4-vision » int4_awq)
        print(red(refus))
        return 2

    # AWQ sans statistiques d'activation est inopérant : la recherche sur
    # grille n'a rien pour pondérer les canaux et se fixe sur une échelle plate.
    # Les statistiques sont donc relevées d'abord et, si c'est impossible, on le
    # dit et l'on retombe sur l'arrondi au plus proche, plutôt que d'annoncer une
    # mise à l'échelle qui n'a jamais eu lieu.
    stats = None
    use_awq = not args.no_awq
    from .quant.exl3 import is_exl3
    from .quant.gguf import is_gguf
    if use_awq and is_exl3(args.model):
        print(yellow("  source EXL3 : arrondi au plus proche, sans AWQ"))
        use_awq = False
    if use_awq and is_gguf(args.model):
        # Le collecteur de statistiques lit des safetensors ; et re-calibrer des
        # poids deja quantifies par llama.cpp apporterait peu de toute facon.
        print(yellow("  source GGUF : arrondi au plus proche, sans AWQ"))
        use_awq = False
    if use_awq:
        from .quant.collect import collect_activation_stats, load_calib_ids
        from .server.chat import load_tokenizer
        try:
            tokenizer = load_tokenizer(args.model)
            # Pièce 45 : `--corpus-jetons` fixe le TOTAL et dérive le nombre
            # de séquences — c est le chiffre que le rapport d observations
            # rend (« jetons_pour_p10 »), pas un nombre de séquences.
            n_seqs = args.calib_seqs
            if args.corpus_jetons:
                n_seqs = max(1, -(-int(args.corpus_jetons) // args.calib_len))
            calib = load_calib_ids(tokenizer, args.calib_file, n_seqs,
                                   args.calib_len, spec.vocab_size,
                                   gabarit=bool(getattr(args, "calib_gabarit", False)))
            calib_reel = (len(calib), sum(len(c) for c in calib))
            print(f"  calibration sur {calib_reel[0]} sequences "
                  f"({calib_reel[1]} jetons) ...")

            def cprog(done: int, total: int) -> None:
                _progress(f"  calibration couche {done}/{total}")

            stats = collect_activation_stats(
                args.model, spec, calib, device=args.calib_device,
                progress=cprog, obs_min=args.obs_min)
            _progress_done()
            obs = stats.pop("__observations__", None)
            print(f"  statistiques relevees pour {len(stats)} tenseurs")
            if obs and obs.get("experts"):
                # Pièce 45 : trois populations séparées. Les experts que le
                # corpus n atteint pas (< seuil de routage) ne comptent PAS
                # dans le facteur : plus de jetons ne les corrige pas.
                print(f"  observations par expert : {obs.get('atteints', 0)}/{obs['experts']} atteints "
                      f"(≥ {obs.get('seuil_expert_route')} jetons) — p10 {obs.get('p10_atteints')}, "
                      f"médiane {obs.get('mediane_atteints')}, max {obs.get('maximum')} ; "
                      f"non atteints {obs.get('jamais_routes', 0)} jamais + "
                      f"{obs.get('quasi_jamais_routes', 0)} quasi jamais "
                      f"({obs.get('part_non_atteints', 0):.1%})", flush=True)
                if obs.get("renvoi"):
                    print(f"  {obs['renvoi']}", flush=True)
                if not obs["suffisant"] and args.obs_min:
                    # Le refus porte sur le p10 des ATTEINTS, seul chiffre
                    # qu un corpus plus grand peut corriger (25(a) : l échelle
                    # AWQ n est stable qu à ≥ 512 observations).
                    f = obs.get("facteur_corpus_pour_atteindre")
                    if f:
                        combien = f"il faudrait ×{f} de jetons"
                        if obs.get("jetons_pour_p10"):
                            combien += f" ({obs['jetons_pour_p10']} au lieu de {obs['corpus_jetons']}"
                            if obs.get("minutes_pour_p10"):
                                combien += f", ~{obs['minutes_pour_p10']} min de collecte à ce rythme"
                            combien += ")"
                        print(red(f"  REFUS : p10 des experts atteints = {obs['p10_atteints']} < "
                                  f"{obs['obs_min_demande']} — {combien}"), flush=True)
                        if obs.get("jetons_pour_p10"):
                            print(f"  (relancer avec --corpus-jetons {obs['jetons_pour_p10']})", flush=True)
                    else:
                        print(red(f"  REFUS : {obs.get('raison', 'aucun expert atteint par ce corpus')}"), flush=True)
                    print("  (ou --obs-min 0 pour convertir quand même, en le disant dans la fiche ; "
                          "les experts NON ATTEINTS relèvent du routage, pas du volume — pièce 27)", flush=True)
                    return 2
        except Exception as exc:                      # noqa: BLE001
            print(yellow(f"  calibration indisponible ({exc}) ; "
                         f"repli sur l'arrondi au plus proche"))
            use_awq = False
            stats = None
    if not use_awq:
        calib_reel = (0, 0)

    calib_source = _calib_source(use_awq, args.calib_file)
    calib_source["gabarit"] = bool(getattr(args, "calib_gabarit", False))   # pièce 55 : le manifeste le dit

    # poste7-diff-octet-a-octet-retire-17-09, REGLES §4 : sans le commit du
    # convertisseur, deux manifestes du meme nom peuvent venir de deux
    # regimes de code differents et rien ne les distingue (motif exact du
    # 17/09 : bogue de calibration corrige entre deux convertis nommes
    # pareil). Refus a la CLI (le point d'usage reel), pas dans
    # convert_checkpoint (les tests construisent des jouets sans extraction
    # git, et n'ont pas besoin de cette provenance).
    from .quant.convert import _convertisseur_commit
    if _convertisseur_commit() is None:
        print(red("conversion refusee : le commit du convertisseur est "
                  "introuvable (pas une extraction git ?) -- sans lui, ce "
                  "converti ne pourra jamais etre distingue d'un autre du "
                  "meme nom produit par une version differente du code."))
        return 2
    if use_awq and not calib_source.get("sha256"):
        print(red("conversion refusee : le sha256 du corpus de calibration "
                  "n'a pas pu etre calcule -- sans lui, un converti AWQ ne "
                  "porte pas ce qui l'a produit."))
        return 2

    opts = ConversionOptions(
        repli_experts=args.repli_experts,
        out_dir=args.out, awq=use_awq, use_hadamard=args.hadamard,
        group_size=args.group_size, n_grid=args.grid,
        lm_head_format=args.lm_head_format, dry_run=args.dry_run,
        attn_qkvo_int8_canal=args.attn_qkvo_int8_canal,
        gdn_int8_canal=args.gdn_int8_canal,
        q3n_table=(tuple(float(v) for v in args.q3n_table.split(","))
                   if args.q3n_table else None),
        mixed_precision=args.mixed_precision, snr_floor=args.snr_floor,
        max_promotions=args.max_promotions,
        promotion_classes=tuple(c.strip() for c in args.promotion_classes.split(",") if c.strip()),
        autoriser_grossissement=args.autoriser_grossissement,
        quant_device=args.quant_device, bits_budget_gib=args.bits_budget,
        garder_grille=args.grille_erreurs,
        echelle_nvfp4=args.echelle,
        promotion_cout_max_mib=args.promotion_cout_max,
        format_impose=args.format, mesurer_kld=args.mesurer_kld,
        alpha_commun_gate_up=args.alpha_commun_gate_up,
        alpha_commun_experts=args.alpha_commun_experts,
        alpha_commun_qkv=args.alpha_commun_qkv,
        hadamard_experts=args.hadamard_experts,
        passage_direct=args.passage_direct,
        sans_vision=args.sans_vision,
        calib_source=calib_source,
        # ce que load_calib_ids a REELLEMENT rendu (poste7, poste7-calibration-
        # verdict-17-09 : le manifeste portait les defauts de classe 16/128,
        # jamais poses ni lus) ; 0/0 sans calibration
        calib_seqs=calib_reel[0], calib_tokens=calib_reel[1])

    last = [0.0]

    def progress(name: str, n: int, _: int) -> None:
        now = time.time()
        if now - last[0] < 0.5:
            return
        last[0] = now
        _progress(f"  {n} tenseurs quantifies  {name}")

    report = convert_checkpoint(args.model, plan, opts, spec=spec, stats=stats,
                                progress=progress)
    _progress_done()
    print(report.render())
    sortie = args.out
    if report.tenseurs_replies and not args.dry_run:
        # poste7-awq-relu2-garde-repli-17-09, REGLES §4 : le regime (combien
        # de tenseurs sont repartis a l'identite plutot qu'AWQ) dans le NOM,
        # pas seulement dans le manifeste -- un dossier au meme nom que le
        # converti "propre" laisserait croire aux deux regimes interchangeables.
        renomme = f"{args.out.rstrip('/')}-repli{len(report.tenseurs_replies)}"
        if not os.path.exists(renomme):
            os.rename(args.out, renomme)
            sortie = renomme
            print(f"  renomme : {bold(renomme)} "
                 f"({len(report.tenseurs_replies)} tenseur(s) replies)")
    if not args.dry_run:
        print()
        print(f"  servez-le avec : {bold(f'acvram serve {sortie}')}")
    return 0


def repli_speculatif(model) -> tuple[str, "str | None"]:
    """``--speculative auto`` : (mode, repli). ``mtp`` si le modèle porte une tête, sinon ``ngram`` avec le repli
    nommé ``mtp absent : <raison>`` (raison posée par le chargeur, `model.mtp_raison`)."""
    if getattr(model, "mtp", None) is not None:
        return "mtp", None
    return "ngram", "mtp absent : " + (getattr(model, "mtp_raison", "") or "raison inconnue")


def cmd_serve(args: argparse.Namespace) -> int:
    import torch
    import uvicorn

    from .engine.loader import load_model
    from .engine.runner import Engine
    from .server.app import create_app
    from .server.chat import load_tokenizer

    print(f"chargement de {args.model} ...")
    t0 = time.time()
    loaded = load_model(args.model, dtype=torch.bfloat16 if not args.fp16
                        else torch.float16,
                        max_model_len=args.max_model_len,
                        max_concurrent_seqs=args.max_batch,
                        device_override=args.device)
    tokenizer = load_tokenizer(args.model)
    speculator = None
    repli = None
    # Pièce 283 (poste5 277a-bis puis 277fix, poste5-277 9fdea0a23 ; élargie, ordre chef) :
    # ngram (défaut jusqu'ici) émettait des jetons hors de la distribution du modèle (277a-bis,
    # hybride GDN : 13 à 24 logits sous le premier choix, 4/5 invites) — cause trouvée par
    # poste5 : le pipeline n'était pas vidé au passage du décodage simple au spéculatif,
    # jetons répétés. Le bogue touche TOUT modèle servi avec ngram, dense compris — pas
    # seulement les hybrides. Défaut = none pour TOUS les alias ; ngram reste servable sur
    # demande explicite, avec un avertissement — le correctif n'est pas forcément livré dans
    # cette version (le test Coder de la 277fix n'est pas tranché), le message ne le suppose pas.
    demande_explicitement = args.speculative is not None
    if args.speculative is None:
        args.speculative = "none"
    if args.speculative == "auto":
        # La tete du modele si elle existe, le n-gramme sinon — repli NOMMÉ (pièce 105 : les têtes MTP de
        # Qwen3.8 étaient converties mais jamais chargées, et rien ne le disait).
        args.speculative, repli = repli_speculatif(loaded.model)
    if demande_explicitement and args.speculative == "ngram":
        print(red("  AVERTISSEMENT : ngram — bogue 277 (jetons répétés au passage simple → spéculatif) ; "
                  "sortie possiblement différente de --speculative none ; qualification en cours."))
    if args.speculative == "ngram":
        from .engine.speculative import NGramProposer
        speculator = NGramProposer()
        speculator.repli = repli
    elif args.speculative == "draft":
        if not args.draft_model:
            print(red("--speculative draft exige --draft-model"))
            return 2
        from .engine.speculative import DraftModelProposer
        draft = load_model(args.draft_model, dtype=torch.bfloat16,
                           device_override=args.draft_device)
        speculator = DraftModelProposer(draft, max_model_len=args.max_model_len)
        print(f"  modele brouillon : {args.draft_model} "
              f"({_h(draft.model.nbytes)})")
    elif args.speculative == "mtp":
        if getattr(loaded.model, "mtp", None) is None:
            print(red("--speculative mtp : ce modele n'a pas de tete nextn"))
            return 2
        from .engine.speculative import MTPProposer
        speculator = MTPProposer(loaded.model, max_model_len=args.max_model_len)
        print("  brouillon : tete de prediction multi-jetons du modele")

    engine = Engine(loaded, tokenizer, max_batch_size=args.max_batch,
                    max_model_len=args.max_model_len,
                    enable_prefix_cache=not args.no_prefix_cache,
                    speculator=speculator, spec_k=args.spec_k,
                    enable_cuda_graphs=not args.no_cuda_graphs,
                    host_kv_gib=args.host_kv_gib)
    # Ordre imposé (chef 21/09) : plan → clamp → capture → chauffe. Le contexte demandé se PROUVE au chargement
    # (poste7-3b-lanceur-contexte-20-09 (ii)) par un prefill plein dans le régime servi, graphes masqués ; le contexte
    # tenu clampe max_model_len, PUIS les graphes se capturent à ce tenu (GLM k48 : OOM à la capture au ctx demandé).
    # Refus nommé (rc 2) seulement sous --ctx-strict ou < 4096, jamais un 500 CUDA OOM à la requête.
    from .engine.runner import ContexteNonTenu
    try:
        tenu, n = engine.demarrer_service(strict=bool(getattr(args, "ctx_strict", False)),
                                          warm_max_len=int(os.environ.get("ACVRAM_WARM_GRAPHS", "2048")))
    except ContexteNonTenu as exc:
        print(red(f"  {exc}"))
        return 2
    print("  contexte      : " + ("non vérifié (ACVRAM_CHAUFFE_CTX=0)" if tenu is None else
                                 f"{tenu} jetons tenus (chauffe)" + (f", clampé (demandé {args.max_model_len})"
                                                                     if tenu < args.max_model_len else "")))
    if engine.graphs is not None:
        print(f"  graphes CUDA : actifs (decodage), {n} godets capturés d'avance (au contexte {engine.max_model_len})")
    # Le tas est énorme après le chargement (manifeste, tokenizer, modules) :
    # une collecte de génération 2 le parcourt entier — plus de 100 ms toutes
    # les quelques dizaines de pas. Geler ces objets les sort du parcours.
    if os.environ.get("ACVRAM_GC_FREEZE", "1") != "0":
        import gc
        gc.collect()
        gc.freeze()
        # et l'on espace les collectes : le pas de décodage crée peu d'objets
        # cycliques, inutile de balayer toutes les 700 allocations
        gc.set_threshold(50000, 20, 20)
    print(f"  charge en {time.time() - t0:.1f} s, "
          f"{_h(loaded.model.nbytes)} de poids")
    print(f"  blocs KV : {engine.allocator.num_blocks} "
          f"({engine.allocator.num_blocks * 16} jetons par couche)")
    print(f"  cache prefixe : {'desactive' if args.no_prefix_cache else 'actif'}")
    print(f"  speculation   : {args.speculative}"
          f"{'' if args.speculative == 'none' else f', k={args.spec_k}, '
                                                   f'lot_max={engine._garde_spec.lot_max}'}")
    if repli:
        # pièce 117 : le repli de `--speculative auto` était posé sur
        # `speculator.repli` (pièce 105) mais jamais imprimé ici — seule la
        # ligne de régime (`_speculation_texte`, engine/runner.py) le
        # portait. Même texte, jamais une seconde formulation du repli.
        from .engine.runner import _speculation_texte
        print(f" {_speculation_texte(engine.regime()['speculation']).strip()}")
    if tokenizer:
        print(f"  gabarit chat  : {tokenizer.template_source}")
    else:
        print(yellow("  aucun tokenizer.json ; les points d'entree /v1 qui "
                     "prennent du texte echoueront"))

    if args.regime:
        # Apres warm_graphs (si graphes actifs) : `piles_ok` a alors vu
        # passer au moins un pas GPU par couche MoE, sinon "?" partout et
        # le verdict ne dirait rien de plus que la ligne du chargement.
        ligne = engine.regime_ligne()
        print(f"  {ligne}")
        return 0 if "DÉGRADÉ" not in ligne else 1

    name = args.served_name or os.path.basename(os.path.abspath(args.model))
    app = create_app(engine, tokenizer, name,
                     {"model_path": args.model, "version": __version__})
    print()
    print(f"  {bold('OpenAI API')}  http://{args.host}:{args.port}/v1")
    print(f"  {dim('models')}      curl http://{args.host}:{args.port}/v1/models")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch

    from .evaluate import perplexity, render

    # Le cadrage se declare TOUJOURS, avec le corpus et la fenetre : une
    # perplexite ne se compare qu'a une autre prise au meme cadrage, et rien
    # dans le nombre publie ne dit lequel a servi. Imprime avant la mesure,
    # il part dans le journal meme si la sortie est redirigee.
    # Un chiffre qui peut sortir SEUL sera compare a tort. Deux defauts du
    # 9/09 en sont la preuve : min_context a 0 sans avertissement, et un plan
    # degrade rendant un nombre d'allure normale. Dans les deux cas le chiffre
    # voyageait sans ses conditions. La parade generique n'est pas de garder
    # chaque cas, c'est que la configuration EFFECTIVE sorte a cote du
    # resultat — cadrage, corpus et son sha, VRAM libre au chargement.
    _avertir_variables_inconnues()
    sha = "?"
    try:
        import hashlib
        h = hashlib.sha256()
        with open(args.corpus, "rb") as fh:
            for bloc in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloc)
        sha = h.hexdigest()[:12]
    except OSError:
        pass
    print(f"  cadrage : min_context={args.min_context} window={args.window} "
          f"stride={args.stride} max_tokens={args.max_tokens}", flush=True)
    print(f"  budget  : 1 séquence × {args.window + 16} jetons (cache KV et réserve de préfill "
          f"dimensionnés pour l'évaluation, pas pour un serveur — 22/09)", flush=True)
    print(f"  corpus  : {os.path.basename(args.corpus)} sha256:{sha}", flush=True)
    if torch.cuda.is_available():
        libre, total = torch.cuda.mem_get_info()
        print(f"  carte   : {libre / 2**30:.2f} Gio libres sur "
              f"{total / 2**30:.2f}", flush=True)
    if not args.min_context:
        # La garde de `evaluate` n'avertit que si le corpus est plus court que
        # la fenetre — jamais sur wiki.test.raw (1,29 Mo). Sans ce message,
        # oublier le cadrage donne un chiffre faux d'un facteur proche de 2
        # (9,525 contre 7,233 au protocole) sans le moindre signe.
        # Precision du 10/09 : il y a DEUX protocoles, et ce message n'en
        # nommait qu'un. Dire « le protocole » envoie chercher un defaut la ou
        # il n'y en a pas — meme famille que la divergence corpus reperee le
        # meme jour entre docs/BARRIERE-QUALITE-PROTOCOLE.md et la chaine de
        # l'etalon. Un avertissement doit nommer CE QU'IL INVALIDE.
        print(red("  min_context=0 : les premieres positions sont notees avec "
                  "un contexte quasi vide."), flush=True)
        print(red("    non comparable a la barriere de qualite de noyau, qui "
                  "impose --min-context 256 --window 512 --stride 512 sur "
                  "wiki.test.raw ;"), flush=True)
        print(red("    COMPARABLE en revanche a l'etalon exterieur "
                  "transformers/GPTQ, qui note des segments disjoints de 2048 "
                  "sans contexte reporte — c'est meme le seul cadrage qui lui "
                  "corresponde (--window 2048 --stride 2048 --min-context 0 "
                  "sur wiki-gptq.txt, reference 5,4141)."), flush=True)
    # Les modeles se chargeaient l'un apres l'autre dans le MEME processus sans
    # que le precedent soit libere. Le 9/09/2026, une barriere de qualite a
    # mesure un nvfp4 puis charge un bf16 par-dessus :
    #   plan reajuste : 32 MLP de plus en RAM hote (15,9 Gio pour 18,1 libres)
    #   OutOfMemoryError : 111,88 MiB libres sur 31,36 Gio
    # Le second modele est donc mesure EN REGIME DEGRADE, ou pas du tout — et
    # une perplexite prise sur un plan degrade n'est comparable a rien.
    #
    # Liberer entre deux ne suffit pas a garantir un plan identique : le
    # cache de l'allocateur et la fragmentation survivent. Un modele par
    # PROCESSUS reste la seule mesure propre, et c'est ce que dit
    # l'avertissement.
    if len(args.models) > 1:
        print(red(f"  {len(args.models)} modeles dans un seul processus : le "
                  f"plan du second depend de ce que le premier a laisse. "
                  f"Pour une mesure comparable, un appel par modele."),
              flush=True)
    results = []
    for i, path in enumerate(args.models):
        def prog(done: int, total: int, _p: str = path) -> None:
            _progress(f"  {os.path.basename(_p)}: window {done}/{total}")
        results.append(perplexity(
            path, args.corpus, window=args.window, stride=args.stride,
            max_tokens=args.max_tokens, device=args.device, progress=prog,
            min_context=args.min_context))
        _progress_done()
        if i + 1 < len(args.models):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    results.sort(key=lambda r: r.perplexity)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    print(render(results))
    if len(results) > 1:
        print()
        print(f"  meilleur : {bold(results[0].model)} a {results[0].perplexity:.3f}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmarks
    return run_benchmarks(args)


def cmd_profiles(args: argparse.Namespace) -> int:
    from .hardware.profiles import list_profiles
    for name, desc in list_profiles().items():
        print(f"  {bold(name)}\n      {desc}")
    return 0


def cmd_eco(args: argparse.Namespace) -> int:
    """Mode eco d'horloge (poste7-e1-eco-tenu-19-09 § 2, poste7-eco-2700-defaut-
    19-09 § 3) : `etat` lit ; 2700 | 2100 | off ECRIT config.json (le service
    le lit a son prochain demarrage) puis applique tout de suite quand la
    carte est libre ; `off` s'applique aussi sous un serveur qui tourne
    (l'utilisateur rend l'horloge), en le nommant."""
    from . import eco
    carte = eco.index_carte() if args.carte is None else args.carte
    if args.mode == "etat":
        h = eco.lire_sous_charge(carte)                    # jamais au repos : verrouillée oisive = 225
        print(f"  carte {carte} : horloge={bold(eco.etiquette_horloge(h))} sous charge légère "
              f"(lectures {h.get('lectures')}, stable={h.get('stable')}) ; config : eco={eco.mode_demande({})}")
        print("  " + json.dumps(h, ensure_ascii=False))
        return 0
    chemin = eco.ecrire_config({"eco": args.mode})
    print(f"  {chemin} : \"eco\": \"{args.mode}\" (lu par le prochain `acvram serve`)")
    pair = eco._tenue_par_un_pair(carte)
    if pair is not None and args.mode != "off":
        print(f"  carte {carte} tenue par PID {pair} : le reglage {args.mode} s'appliquera a son "
              f"prochain demarrage (un -lgc pendant la manche d'un autre change son regime)")
        return 0
    if pair is not None:
        print(f"  carte {carte} tenue par PID {pair} : -rgc applique quand meme (demande de l'utilisateur), "
              f"le serveur en cours passe a l'horloge libre — sa ligne de regime le dira")
        h = eco.Horloge("2700", carte); h.posee = True          # rendre() n'agit que sur une horloge posee
        return 0 if h.rendre() else 3
    return eco.regler(args.mode, carte)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acvram",
        description="anticitoyen VRAM/RAM — inference etagee, quantifiee par GPU\n"
                    "Soutenir : buymeacoffee.com/anticitoyen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--version", action="version",
                   version=f"acvram {__version__}\nSoutenir : buymeacoffee.com/anticitoyen")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="rapporte le materiel local")
    d.add_argument("--profile", help="utilise un profil declare au lieu de sonder")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_detect)

    doc = sub.add_parser("doctor", help="verifie que cette machine peut faire tourner acvram")
    doc.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("profiles", help="liste les profils materiels declares")
    pr.set_defaults(func=cmd_profiles)

    def add_plan_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("model", help="repertoire de modele HF (avec config.json)")
        sp.add_argument("--name", help="remplace le nom du modele")
        sp.add_argument("--profile", help="planifie pour un profil declare")
        sp.add_argument("--max-model-len", type=int, default=8192)
        sp.add_argument("--max-seqs", type=int, default=8,
                        help="sequences simultanees que le cache KV doit tenir")
        sp.add_argument("--group-size", type=int, default=128)
        sp.add_argument("--format", help="impose un seul format de poids partout")
        sp.add_argument("--gpus", default="auto",
                        help="auto (le debit tranche), all, ou des indices "
                             "comme 0,1")
        sp.add_argument("--host-exec", choices=["auto", "stream", "cpu"],
                        default="auto",
                        help="comment sont calcules les poids en RAM : copies "
                             "vers le GPU, ou sur place par le processeur")
        sp.add_argument("--host-gb-s", type=float, default=70.0,
                        help="bande passante DDR mesuree ; acvram bench la rapporte")

    pl = sub.add_parser("plan", help="montre ou serait placee chaque couche")
    add_plan_args(pl)
    pl.add_argument("--kv-bits", type=int, default=8)
    pl.add_argument("--host-fraction", type=float, default=0.85)
    pl.add_argument("--no-host", action="store_true",
                    help="refuse d'utiliser la memoire vive comme etage")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_plan)

    cv = sub.add_parser("convert", help="quantifie un point de controle en fragments acvram")
    add_plan_args(cv)
    cv.add_argument("-o", "--out",
                    help="repertoire de sortie (defaut : "
                         "$ACVRAM_MODELS_DIR/<nom>, sinon models_acvram/<nom> "
                         "sur le SSD 2TO_2023_980PRO1, sinon sur le HDD "
                         "4TO_SATACMR_2022)")
    cv.add_argument("--no-awq", action="store_true",
                    help="simple arrondi au plus proche, sans mise a l'echelle AWQ")
    cv.add_argument("--hadamard", choices=["auto", "always", "never"],
                    default="auto")
    cv.add_argument("--echelle", choices=["max6", "4sur6"], default="max6",
                    help="echelle de bloc nvfp4 : max6 (amax/6, classique) ou 4sur6 (par bloc, "
                         "amax/6 contre amax/4 au moindre MSE, arXiv 2512.02010 ; format et noyaux inchanges)")
    cv.add_argument("--grid", type=int, default=20,
                    help="finesse de la grille de recherche AWQ")
    cv.add_argument("--lm-head-format", help="format de la projection de sortie")
    cv.add_argument("--attn-qkvo-int8-canal", action="store_true",
                    help="q/k/v/o en int8 symetrique par canal (une echelle "
                         "par ligne, sans point-zero variable) au lieu du "
                         "groupe de 128 affine du reste du modele")
    cv.add_argument("--gdn-int8-canal", action="store_true",
                    help="153 : les cinq projections du GatedDeltaNet "
                         "(qkv/gate/alpha/beta/out) en int8 symetrique par "
                         "canal, meme regime que --attn-qkvo-int8-canal mais "
                         "drapeau distinct (les deux se cumulent)")
    cv.add_argument("--q3n-table", help=("niveaux q3n de ce modèle, huit "
                    "flottants séparés par des virgules (symétriques, bornes "
                    "±1) ; défaut : table de la spécification"))
    cv.add_argument("--dry-run", action="store_true",
                    help="rapporte tailles et erreurs sans ecrire de fragments")
    cv.add_argument("--force", action="store_true",
                    help="convertit meme si le modele ne tient pas")
    cv.add_argument("--calib-file", help="fichier texte de calibration (par defaut : "
                                         "acvram/data/calibration-anglais.txt, Gutenberg #1342)")
    cv.add_argument("--calib-seqs", type=int, default=32)
    cv.add_argument("--corpus-jetons", type=int, default=0, metavar="N",
                    help="pièce 45 : total de jetons de calibration (dérive --calib-seqs de --calib-len) ; "
                         "c'est le chiffre que le rapport d'observations rend quand il refuse "
                         "(« relancer avec --corpus-jetons N ») ; 0 = utiliser --calib-seqs")
    cv.add_argument("--calib-len", type=int, default=512)
    cv.add_argument("--calib-gabarit", action="store_true",
                    help="pièce 55 : passe chaque tranche de calibration par le gabarit de conversation du modèle "
                         "(tour utilisateur / tour assistant) avant l encodage ; refuse si le modèle n a pas de gabarit")
    cv.add_argument("--calib-device", default="cuda:0",
                    help="appareil sur lequel executer les passes de calibration")
    cv.add_argument("--obs-min", type=int, default=512,
                    help="pièce 32 : observations minimales par expert pendant la calibration (25(a) : "
                         "l'échelle AWQ n'est stable qu'à partir de 512) ; sous le seuil la conversion REFUSE "
                         "en nommant le facteur de corpus manquant ; 0 = pas de contrôle")
    cv.add_argument("--repli-experts", default="identite", choices=("identite", "mediane_couche"),
                    help="experts MoE routés < 8 fois par le corpus : identite (défaut, aucune échelle) "
                         "ou mediane_couche (statistique = médiane des experts calibrés de la couche ; pièce 25)")
    cv.add_argument("--promotion-cout-max", type=float, default=0.0, metavar="MIO",
                    help="prix plafond d'une promotion, en Mio ajoutes "
                         "(0 = aucun) : ecarte les gros tenseurs, dont la "
                         "promotion coute des octets relus a chaque jeton")
    cv.add_argument("--grille-erreurs", action="store_true",
                    help="conserve l'erreur des 21 valeurs de la grille AWQ "
                         "par tenseur, au manifeste : sert a calculer le prix "
                         "d'un exposant commun a un groupe empilable, sans "
                         "reconvertir")
    cv.add_argument("--bits-budget", type=float, default=0.0,
                    help="budget total de poids en Gio : les promotions sont "
                         "choisies par gain de SNR par octet (sac a dos), au "
                         "lieu du plancher SNR fixe")
    cv.add_argument("--quant-device", default="auto",
                    help="appareil de la recherche AWQ et de la quantification "
                         "(auto, cpu, cuda:0 ...)")
    cv.add_argument("--mixed-precision", choices=["auto", "off"], default="auto",
                    help="promeut vers un format plus large les tenseurs mal quantifies")
    cv.add_argument("--autoriser-grossissement", action="store_true",
                    help="autorise une conversion plus grosse que sa source "
                         "(refusée par défaut depuis le 8/09/2026)")
    cv.add_argument("--max-promotions", type=float, default=0.15,
                    metavar="PART",
                    help="part maximale de tenseurs promus (0,15 par defaut). "
                         "Le message de saturation conseillait de relever "
                         "cette option, qui n'existait pas : le plafond etait "
                         "atteint sur 27 modeles du parc sur 110, et au-dela "
                         "c'est l'ordre de parcours qui decide a la place du "
                         "SNR (5333 inversions mesurees sur Agents-A1-4B). "
                         "1.0 ne borne plus rien")
    cv.add_argument("--promotion-classes", default="",
                    help="classes promouvables, suffixes separes par des virgules "
                         "(ex. q_proj,k_proj,lm_head) ; vide = toutes")
    cv.add_argument("--snr-floor", type=float, default=0.0,
                    help="SNR en sortie de couche (dB) sous lequel un tenseur est "
                         "promu ; 0 (defaut) ne promeut rien. Mesure sur un 27B : "
                         "25 dB coute 13,4 %% de memoire et 10,6 %% de debit pour "
                         "2,0 %% de perplexite")
    cv.add_argument("--mesurer-kld", action="store_true",
                    help="publie le KLD couche-par-couche (proxy softmax, "
                         "duck.ai 12/09) au manifeste, a cote du SNR. "
                         "N'AFFECTE AUCUNE DECISION : le convertisseur promeut "
                         "toujours sur le SNR. Sert au protocole A/B")
    cv.add_argument("--alpha-commun-experts", action="store_true",
                    help="le meme alpha AWQ commun, mais PAR EXPERT MoE : "
                         "gate_proj et up_proj d'un expert lisent la meme "
                         "entree, et la pile groupee EXIGE qu'ils partagent "
                         "leur echelle — sinon engine/moe.py ne fusionne pas "
                         "les tables ([E, K], torch.equal global) et la "
                         "disposition Marlin est refusee : experts_layout="
                         "naturel, 8,62 ms/pas contre 6,7 sur le Coder-30B "
                         "(revue/poste1-disposition-naturel-qkv-22-09.md). "
                         "Opt-in tant que la mesure n'est pas faite ; coute "
                         "une seconde recherche AWQ par paire d'experts.")
    cv.add_argument("--alpha-commun-qkv", action="store_true",
                    help="pièce 100 B (23/09) : un SEUL alpha AWQ pour q_proj, "
                         "k_proj et v_proj d'une couche (même entrée), sinon "
                         "_scaler_commun refuse la pile qkv nvfp4 et q, k, v "
                         "partent en trois GEMM (+0,9 ms/pas, pièce 42). "
                         "o_proj non concerné. Opt-in.")
    cv.add_argument("--alpha-commun-gate-up", action="store_true",
                    help="item A7 (audit poste7, 14/09) : un SEUL alpha AWQ "
                         "pour chaque paire gate_proj/up_proj admissible, au "
                         "lieu d'un alpha independant par tenseur — "
                         "necessaire pour que le moteur fusionne la paire "
                         "(_scaler_commun refuse deux scalers differents). "
                         "Defaut faux : ne change pas la conversion sans "
                         "mesure (revue/prediction-a7-alpha-commun-gateup-"
                         "14-09.md)")
    cv.add_argument("--sans-vision", action="store_true",
                    help="source multimodale : n'ecrit pas la tour de vision, alias texte seul "
                         "(manifeste vision=non)")
    cv.add_argument("--passage-direct", action="store_true",
                    help="source deja NVFP4 (modelopt, compressed-tensors "
                         "nvfp4-pack-quantized) : copie ses poids 4 bits tels "
                         "quels — sans dequantifier, sans recherche AWQ, sans "
                         "promotion ni rotation — pour servir exactement les "
                         "poids de vLLM ; les couches en clair suivent le plan")
    cv.add_argument("--hadamard-experts", action="store_true",
                    help="tourne les poids d'experts (gate/up/down) en "
                         "Hadamard bloc H_512 avant quantification, SANS "
                         "echelle AWQ sur ces tenseurs — a l'essai contre le "
                         "defaut de bloc NVFP4 (E2M1, etendue 12:1) qu'aucune "
                         "echelle par canal ne corrige (poste7-hadamard-16-"
                         "09.md). Defaut faux")
    cv.set_defaults(func=cmd_convert)

    sv = sub.add_parser("serve", help="lance le serveur compatible OpenAI",
                        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOGUE_SERVE)
    sv.add_argument("model", help="repertoire de modele converti")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--max-model-len", type=int, default=8192)
    sv.add_argument("--max-batch", type=int, default=16)
    sv.add_argument("--served-name", help="nom annonce par /v1/models")
    sv.add_argument("--device", help="force toutes les couches sur un seul appareil")
    sv.add_argument("--fp16", action="store_true",
                    help="calcule en float16 au lieu de bfloat16")
    sv.add_argument("--log-level", default="info")
    sv.add_argument("--speculative", choices=["none", "ngram", "draft", "mtp", "auto"],
                    default=None,   # 283 : résolu après chargement (None = pas demandé) -- toujours "none"
                    help="defaut : none, pour TOUT alias -- pièce 277a-bis/277fix (poste5) : le "
                         "pipeline n'etait pas vide au passage du decodage simple au speculatif "
                         "(ngram), jetons repetes ; touche tout modele servi avec ngram, dense "
                         "compris (277a-bis : hybride GDN, jetons 13 a 24 logits sous le premier "
                         "choix, 4/5 invites). Demander ngram explicitement reste possible, avec "
                         "un avertissement au demarrage (sortie possiblement differente de none, "
                         "qualification du correctif en cours). "
                         "ngram ne coute rien et paie quand la sortie recopie "
                         "l'entree ; draft exige --draft-model ; mtp utilise "
                         "la tete nextn du modele charge si elle porte une "
                         "convention reconnue (Qwen3.5 et suivants, DeepSeek "
                         "-- voir noms_mtp dans engine/mtp.py), non rentable "
                         "en l'etat (docs/ARCHITECTURE.md) ; auto choisit mtp "
                         "si la tete est reconnue, sinon retombe sur ngram "
                         "avec un repli NOMME pose sur speculator.repli "
                         "(repli_speculatif, cli.py) -- rien ne l'imprime "
                         "encore au demarrage. N'importe quel propositeur "
                         "reste soumis a la garde de lot "
                         "ACVRAM_SPECULATION_LOT_MAX (defaut 2) : au-dela, "
                         "la carte est deja pleine a largeur 1 par sequence "
                         "et verifier plus large coute plus qu'il ne rend "
                         "(mesure 14/09, revue/verdict-cout-verification-"
                         "ngram-b12-14-09.md, -49,9 pourcent de debit a b=12) -- "
                         "voir GardeSpeculation dans engine/speculative.py")
    sv.add_argument("--draft-model", help="repertoire converti d'un petit modele "
                                          "charge de proposer des jetons")
    sv.add_argument("--draft-device", help="appareil du modele brouillon "
                                           "(par defaut : le GPU le plus oisif)")
    sv.add_argument("--spec-k", type=int, default=4,
                    help="jetons proposes par etape")
    sv.add_argument("--regime", action="store_true",
                    help="charge, imprime le regime (graphes/exil/piles/"
                         "cartes/chemin MoE) et quitte -- code 1 si degrade, "
                         "sans lancer le serveur")
    sv.add_argument("--host-kv-gib", type=float, default=8.0,
                    help="etage hote du cache KV en Gio (0 = desactive) : les "
                         "prefixes evinces de la VRAM descendent en RAM et "
                         "remontent au reemploi au lieu d'etre recalcules")
    sv.add_argument("--no-cuda-graphs", action="store_true",
                    help="rejoue chaque pas de decodage en eager plutot qu'en "
                         "graphe CUDA capture")
    sv.add_argument("--ctx-strict", action="store_true",
                    help="refuser (rc 2) un contexte non tenu par la chauffe au lieu de le clamper")
    sv.add_argument("--no-prefix-cache", action="store_true",
                    help="desactive la reutilisation du KV entre requetes")
    sv.set_defaults(func=cmd_serve)

    ev = sub.add_parser("eval", help="perplexite d'un ou plusieurs modeles convertis")
    ev.add_argument("models", nargs="+", help="repertoires de modeles convertis")
    ev.add_argument("--corpus", help="fichier texte servant a l'evaluation")
    ev.add_argument("--window", type=int, default=512)
    ev.add_argument("--stride", type=int, default=256)
    ev.add_argument("--max-tokens", type=int, default=8192)
    # DEFAUT 0 CONSERVE, mais il ne passe plus en silence. Une perplexite a
    # min_context 0 n'est PAS comparable a une perplexite cadree : sur
    # wiki.test.raw la table du protocole donne 9,525 contre 7,233, un facteur
    # proche de 2. Le seul garde-fou existant n'avertit que si le corpus est
    # plus court que la fenetre — ce qui n'arrive jamais sur ce corpus de
    # 1,29 Mo. La barriere reposait donc sur la memoire de l'operateur.
    ev.add_argument("--min-context", type=int, default=0,
                    help="n'note que les positions ayant au moins tant de "
                         "jetons de contexte (0 = tout, NON COMPARABLE a une "
                         "mesure cadree ; le protocole impose 256)")
    ev.add_argument("--device", help="impose un appareil")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval)

    bn = sub.add_parser("bench", help="mesure noyaux, bande passante et debit")
    bn.add_argument("model", nargs="?", help="repertoire de modele converti")
    bn.add_argument("--topology-out",
                    help="ou ecrire la topologie mesuree "
                         "(defaut ~/.config/acvram/acvram-topology.json)")
    bn.add_argument("--what", default="all",
                    choices=["all", "kernels", "bandwidth", "topology", "decode"])
    bn.add_argument("--json", action="store_true")
    bn.set_defaults(func=cmd_bench)

    ec = sub.add_parser("eco", help="mode eco d'horloge : verrouille l'horloge SM de la "
                                    "carte (sudo -n nvidia-smi -lgc), la libere, ou la lit")
    ec.add_argument("mode", choices=["2700", "2100", "off", "etat"],
                    help="2700 (defaut du service) | 2100 : ecrit config.json et verrouille "
                         "l'horloge SM a cette frequence (MHz) si la carte est libre ; "
                         "off : ecrit config.json et libere (-rgc), meme sous un serveur ; "
                         "etat : lit l'horloge sans rien changer. Le service pose son eco "
                         "lui-meme au chargement et le rend a l'arret ; la ligne de regime "
                         "porte eco=<demande>(<effectif>)")
    ec.add_argument("--carte", type=int, default=None,
                    help="index nvidia-smi de la carte (defaut : premier index de "
                         "CUDA_VISIBLE_DEVICES, sinon 0)")
    ec.set_defaults(func=cmd_eco)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrompu", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(red(f"introuvable : {exc}"), file=sys.stderr)
        return 2
    except Exception as exc:                          # noqa: BLE001
        print(red(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        if os.environ.get("ACVRAM_TRACEBACK"):
            raise
        if getattr(args, "func", None) is cmd_serve:
            # un serveur qui meurt au chargement n a que son journal pour le dire : `str(e)` seul y laissait
            # « size of tensor a (256) must match b (257) » sans site (P3 (4) 30B, poste2 21/09) — la pile s imprime
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
