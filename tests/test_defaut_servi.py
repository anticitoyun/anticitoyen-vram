"""Le défaut servi est un contrat par version (poste7-tests-rapides-cloture-20-09 § 4) : la ligne de régime du
défaut nu, à sec, et les défauts des variables qui ont changé de valeur au cours d'un .deb sont gelés ici, par
version. Un défaut qui dérive sans changement de version casse ce test ; un changement de version sans mise à
jour de la table le casse aussi — c'est voulu : chaque .deb réécrit sa ligne, avec ses bras (REGLES § 3)."""
from __future__ import annotations

import os
import subprocess
import sys

import acvram
from acvram import regime

# version → (défauts nommés, fin de la ligne de régime à sec). Ajouter une ligne par .deb qui change un défaut ;
# ne jamais modifier une ligne existante (elle est l'histoire du défaut servi).
DEFAUTS_PAR_VERSION = {
    "0.6.30": (
        {
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8",            # C15-3d (0.6.24)
            "PREFILL_COMPACT": "1",                                     # C15-prefill (0.6.30)
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048",            # règle des 2 048 clés (RESTE, 5afe017e)
            "MLA_GLUE": "1",                                            # =2 opt-in tant que 2a-bis n'est pas tenu
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin",
            "PILE_SANS_RENDU": "",                                      # cache des piles rendu (c48c2b2c, 0.6.29)
        },
        "mla_core=tf32(≤2048 clés) glue=compact(8) prefill_glue=compact",
    ),
    "0.6.31": (                                                          # 0.6.30 + réglages hôte (acvram/hote.py)
        {
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "1",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "",                                                  # aucune affinité par défaut (hote=thp,omp8)
            "MLA_PREP_GRILLE": "1",                                      # C14-b geste 3 (M1 bis) : prep regrillé au bit
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille glue=compact(8) prefill_glue=compact",   # hote=thp,omp8 est AVANT mla_core
    ),
    "0.6.32": (                                                          # 0.6.31 + MLA_GLUE=2 servi (M3 2a-bis tenu 4/4, 20/09)
        {
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048",
            "MLA_GLUE": "2",                                             # b=1 par decode_static_batch_complet ; 1 = témoin
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            # ROUTEUR_FUSE (pièce 3, route_x) : à ajouter ici SI M4 tenu, avant le .deb — un seul .deb 0.6.32
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
    ),
    "0.6.33": (                                                          # 0.6.32 + vision Gemma 4 servie (poste7 17 h 06, 20/09)
        {
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            # Aucune variable nouvelle : la tour de vision se sert quand le manifeste la déclare
            # (`vision=bf16(eager)` dans la ligne AVEC modèle, regime.py:454), rien dans le défaut nu ;
            # KV int8 gardé avec image (bande ≤ +1 %, verdict-decode-pas-31b-kv-20-09). Qwen3-VL : code
            # présent, aucun alias servi ni publié (P3 (1) non joué) — 0.6.34.
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
    ),
    "0.6.34": (                                                          # 0.6.33 + porte de famille du masque image, chauffe de contexte, lanceur paquet, Agent OS Open WebUI (20/09 soir)
        {
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            # Aucune variable nouvelle non plus : masque_images= (famille du manifeste), ctx_tenu= (chauffe au chargement,
            # opt-out ACVRAM_CHAUFFE_CTX=0 hors service) et source= (lanceur) sont des lectures, pas des défauts ;
            # candidat NON installé tant que P3 (1) (b″) et S2 0 × 500 ne sont pas tenus (poste2). La tour de vision se sert quand le manifeste la déclare
            # (`vision=bf16(eager)` dans la ligne AVEC modèle, regime.py:454), rien dans le défaut nu ;
            # KV int8 gardé avec image (bande ≤ +1 %, verdict-decode-pas-31b-kv-20-09). Qwen3-VL : code
            # présent, aucun alias servi ni publié (P3 (1) non joué) — 0.6.34.
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
    ),
    "0.6.35": (                                                          # 0.6.34 + levier 1 (échantillonnage dans le graphe, opt-out ACVRAM_SAMPLER_LENT=1),
        {                                                                # levier 2 (rapatriement épinglé, opt-out ACVRAM_RAPATRIEMENT_FLUX=1), verrou partagé,
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",    # garde de capture (mémoire minimale, délai) — 22/09
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            "SAMPLER_LENT": "0", "RAPATRIEMENT_FLUX": "0", "ETROITES_FORME": "",
            "CAPTURE_MEM_MIN_MIO": "1024", "CAPTURE_DELAI_S": "120",
            # sampler=graphe et rapatriement=epingle n'apparaissent que dans la ligne AVEC modèle (b ≥ 2, carte) ; le défaut nu
            # ne change pas de fin de ligne. Split-K étroites retiré (RÉFUTÉ b08a3d34) ; ETROITES_FORME vide = 4,3 au bit.
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
    ),
    "0.6.36": (                                                          # 0.6.35 + MoE sur tensor cores par défaut aux godets ≥ 2 (pièces 62-65, 23/09 :
        {                                                                # port marlin_moe_wna16 vLLM 0.29, glue fusionnée reproductible ; A5 poste2 +12,2 % t/s)
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            "SAMPLER_LENT": "0", "RAPATRIEMENT_FLUX": "0", "ETROITES_FORME": "",
            "CAPTURE_MEM_MIN_MIO": "1024", "CAPTURE_DELAI_S": "120",
            "MOE_TENSOR": "1", "MOE_TENSOR_FUSION": "1", "MOE_TENSOR_MIN_T": "8",   # =0 témoins ; glue A4 (FUSION=0) jamais servie ; godets 8/12/16
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
    ),
    "0.6.37": (                                                          # 0.6.36 + GEMV_SPLITK=1 défaut (S auto, pièce 70, 23/09 :
        {                                                                # +7,09 % b=1, KL 5/5, revue/poste2-piece67-23-09.md)
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            "SAMPLER_LENT": "0", "RAPATRIEMENT_FLUX": "0", "ETROITES_FORME": "",
            "CAPTURE_MEM_MIN_MIO": "1024", "CAPTURE_DELAI_S": "120",
            "MOE_TENSOR": "1", "MOE_TENSOR_FUSION": "1", "MOE_TENSOR_MIN_T": "8",
            "GEMV_SPLITK": "1",                                          # S auto défaut (0 témoin) — pièce 70
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
    ),
    "0.6.38": (                                                          # 0.6.37 + livraison 0.6.38 (23/09) : w13 au décodage
        {                                                                # par défaut (pièce 82 ter), MAX_GRAPHS 16→64 (pièce 85),
                                                                           # réduction d'attention déroulée (pièce 92)
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            "SAMPLER_LENT": "0", "RAPATRIEMENT_FLUX": "0", "ETROITES_FORME": "",
            "CAPTURE_MEM_MIN_MIO": "1024", "CAPTURE_DELAI_S": "120",
            "MOE_TENSOR": "1", "MOE_TENSOR_FUSION": "1", "MOE_TENSOR_MIN_T": "8",
            "GEMV_SPLITK": "1",
            "MOE_W13": "1",                                              # w13 au décodage, préfill séparé — pièce 82 ter
            "MAX_GRAPHS": "64",                                          # plafond relevé, sans éviction — pièce 85
            "ATTN_REDUC_DEROULEE": "1",                                  # réduction déroulée, au bit (−1,9 %) — pièce 92
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
    ),
    "0.7.0": (                                                           # 0.6.38 + tout ce qui a changé de défaut du 23/09 au 26/09 (pièce 232 :
        {                                                                # chaque valeur relue dans regime.py, pièce en commentaire)
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            "SAMPLER_LENT": "0", "RAPATRIEMENT_FLUX": "0", "ETROITES_FORME": "",
            "CAPTURE_MEM_MIN_MIO": "1024", "CAPTURE_DELAI_S": "120",
            "MOE_TENSOR": "1", "MOE_TENSOR_FUSION": "1", "MOE_TENSOR_MIN_T": "8",
            "GEMV_SPLITK": "1", "MOE_W13": "1", "MAX_GRAPHS": "64", "ATTN_REDUC_DEROULEE": "1",
            # --- changés ou apparus depuis 0.6.38 ---
            "GDN_AB": "auto",                                            # α/β GDN fusionnés dès 48 experts — 175/175b
            "GDN_AB_FLUX": "1",                                          # β‖α sur un second flux pendant qkv‖gate — 194 b2 (au bit)
            "GDN_QKV_GATE": "1",                                         # pile qkv‖gate 16384 × 5120 — 176
            "GDN_CONV_FUSEE": "1", "GDN_RES_DIFFERE": "1", "GDN_PORTES_NOYAU": "1",   # GDN : conv fusée, résidu différé, portes noyau
            "GDN_NORME_FUSEE": "1", "GDN_Z_BF16": "1", "GDN_ETAT_EN_PLACE": "1",       # GDN : norme fusée, z bf16, état en place (au bit)
            "GDN_PREFILL_LOT": "0",                                      # LOT du préfill GDN : 0 = d'avant (164 étape 0 : levier à requalifier)
            "NORME_REGISTRES": "1",                                      # rmsnorm en registres (au bit)
            "ETROIT_CANAL": "1",                                         # étroit int8 K entier par canal (table) — 195 b (décision déléguée)
            "INT8_TRANCHE": "",                                          # vide = 6 (défaut depuis la 187, au bit ; 16 = témoin d'avant)
            "INT8_TRANCHE_PREFILL": "",                                  # 187 : découpage à N > 16, vide = d'avant
            "MARLIN_PAR_LIGNE": "0",                                     # 209 À LA DEMANDE — 232 b : 229 (débit soutenu −15,3 %) contre 226 (salve +12,8 %), écart non expliqué ; la release garde l'ancien comportement
            "ADMISSION_FENETRE_MS": "5",                                 # fenêtre d'admission du service — 179
            "TRANCHE_COPIE_MIN": str(2**30),                             # copie int8 transitoire par tranche ≥ 1 Gio — 201 (3)
            "DEPAQ_PARTAGE": "1",                                        # .so du port Marlin par empreinte des sources — 161
            "DEPAQUETAGE": "auto",                                       # dépaquetage Marlin → bf16 : cuda (v2) — 147 L3'
            "PROJ_MARLIN": "1", "PROJ_MARLIN_MIN_M": "2", "PROJ_MARLIN_MIN_NK": "1024",   # projections denses sur la disposition Marlin — 147/155/160
            "PROJ_MARLIN_MIN_N": "2048", "PROJ_MARLIN_PORTEE": "denses", "PROJ_MARLIN_CAPACITE": "65536", "PROJ_MARLIN_DOUBLES": "",
            "GEMV_MARLIN_V2": "1", "GEMV_MARLIN_TPB": "0", "GEMV_MARLIN_S": "0",          # GEMV dense Marlin v2 (134/147), TPB et S automatiques
            "AWQ_TENSOR": "0",                                           # échelle AWQ dans le chemin tensor : opt-in (123 non fusionnée)
            "PA_GQA": "1",                                               # attention paginée : K/V lus une fois par groupe GQA
            "MTP_ETAT": "brut",                                          # état MTP brut — 201 (planificateur vision/MTP)
            "INT8_GEMV_MAX": "80",                                       # seuil GEMV int8 hors portée de partage (inchangé)
            "INT8_GEMV_MAX_PARTAGE": "16",                               # GEMV→GEMM int8 sous B′ à 16 — 243 (hors bit, KL tenue, +9,70 % servi)
            # Marges KV (201/212 : `loader._KV_MARGE_MIN` 1 536 Mio, `_KV_MARGE_MIN_GDN` 3 072 Mio) sont des constantes du loader,
            # pas des variables de régime : gelées par tests/test_marge_graphes_212.py.
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",   # inchangée (relevée à sec le 26/09)
    ),
    "0.7.1": (                                                           # 0.7.0 + pièces 260/260x (272 : relues dans regime.py)
        {                                                                # chaque valeur relue dans regime.py, pièce en commentaire)
            "GLUE_COMPACT": "1", "ATTN_WARPS_COMPACT": "8", "PREFILL_COMPACT": "1",
            "MLA_CORE": "tf32", "MLA_CORE_MAX_CLES": "2048", "MLA_GLUE": "2",
            "MARLIN_DISTINCT": "0", "MOE_DECODE_MMA": "1", "MOE_DECODE_MMA_MARLIN": "0",
            "KV_INT8_CANAL": "0", "GODETS_B": "1", "GEMV_LAYOUT": "marlin", "PILE_SANS_RENDU": "",
            "CPUS": "", "MLA_PREP_GRILLE": "1",
            "SAMPLER_LENT": "0", "RAPATRIEMENT_FLUX": "0", "ETROITES_FORME": "",
            "CAPTURE_MEM_MIN_MIO": "1024", "CAPTURE_DELAI_S": "120",
            "MOE_TENSOR": "1", "MOE_TENSOR_FUSION": "1", "MOE_TENSOR_MIN_T": "8",
            "GEMV_SPLITK": "1", "MOE_W13": "1", "MAX_GRAPHS": "64", "ATTN_REDUC_DEROULEE": "1",
            # --- changés ou apparus depuis 0.6.38 ---
            "GDN_AB": "auto",                                            # α/β GDN fusionnés dès 48 experts — 175/175b
            "GDN_AB_FLUX": "1",                                          # β‖α sur un second flux pendant qkv‖gate — 194 b2 (au bit)
            "GDN_QKV_GATE": "1",                                         # pile qkv‖gate 16384 × 5120 — 176
            "GDN_CONV_FUSEE": "1", "GDN_RES_DIFFERE": "1", "GDN_PORTES_NOYAU": "1",   # GDN : conv fusée, résidu différé, portes noyau
            "GDN_NORME_FUSEE": "1", "GDN_Z_BF16": "1", "GDN_ETAT_EN_PLACE": "1",       # GDN : norme fusée, z bf16, état en place (au bit)
            "GDN_PREFILL_LOT": "0",                                      # LOT du préfill GDN : 0 = d'avant (164 étape 0 : levier à requalifier)
            "NORME_REGISTRES": "1",                                      # rmsnorm en registres (au bit)
            "ETROIT_CANAL": "1",                                         # étroit int8 K entier par canal (table) — 195 b (décision déléguée)
            "INT8_TRANCHE": "",                                          # vide = 6 (défaut depuis la 187, au bit ; 16 = témoin d'avant)
            "INT8_TRANCHE_PREFILL": "",                                  # 187 : découpage à N > 16, vide = d'avant
            "MARLIN_PAR_LIGNE": "0",                                     # 209 À LA DEMANDE — 232 b : 229 (débit soutenu −15,3 %) contre 226 (salve +12,8 %), écart non expliqué ; la release garde l'ancien comportement
            "ADMISSION_FENETRE_MS": "5",                                 # fenêtre d'admission du service — 179
            "TRANCHE_COPIE_MIN": str(2**30),                             # copie int8 transitoire par tranche ≥ 1 Gio — 201 (3)
            "DEPAQ_PARTAGE": "1",                                        # .so du port Marlin par empreinte des sources — 161
            "DEPAQUETAGE": "auto",                                       # dépaquetage Marlin → bf16 : cuda (v2) — 147 L3'
            "PROJ_MARLIN": "1", "PROJ_MARLIN_MIN_M": "2", "PROJ_MARLIN_MIN_NK": "1024",   # projections denses sur la disposition Marlin — 147/155/160
            "PROJ_MARLIN_MIN_N": "2048", "PROJ_MARLIN_PORTEE": "denses", "PROJ_MARLIN_CAPACITE": "65536", "PROJ_MARLIN_DOUBLES": "",
            "GEMV_MARLIN_V2": "1", "GEMV_MARLIN_TPB": "0", "GEMV_MARLIN_S": "0",          # GEMV dense Marlin v2 (134/147), TPB et S automatiques
            "AWQ_TENSOR": "0",                                           # échelle AWQ dans le chemin tensor : opt-in (123 non fusionnée)
            "PA_GQA": "1",                                               # attention paginée : K/V lus une fois par groupe GQA
            "MTP_ETAT": "brut",                                          # état MTP brut — 201 (planificateur vision/MTP)
            "INT8_GEMV_MAX": "80",                                       # seuil GEMV int8 hors portée de partage (inchangé)
            "INT8_GEMV_MAX_PARTAGE": "16",
            "I8C_FP8_PREFILL": "bf16",                                   # int8 ré-encodés du fp8 au préfill : bf16 ; cublas À LA DEMANDE — 260
            "I8C_COPIE": "xor",                                          # copie signée q − 128 par xor, au bit — 260x                               # GEMV→GEMM int8 sous B′ à 16 — 243 (hors bit, KL tenue, +9,70 % servi)
            # Marges KV (201/212 : `loader._KV_MARGE_MIN` 1 536 Mio, `_KV_MARGE_MIN_GDN` 3 072 Mio) sont des constantes du loader,
            # pas des variables de régime : gelées par tests/test_marge_graphes_212.py.
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",   # inchangée (relevée à sec le 26/09)
    ),
}


def _var(nom):
    return next(v for v in regime.VARIABLES if v.nom == nom)


def test_la_version_servie_a_sa_ligne_gelee():
    assert acvram.__version__ in DEFAUTS_PAR_VERSION, (
        f"version {acvram.__version__} sans ligne gelée : ajouter son entrée (défauts + fin de ligne) avec le .deb")


def test_les_defauts_nommes_sont_ceux_de_la_version():
    attendus, _ = DEFAUTS_PAR_VERSION[acvram.__version__]
    ecarts = {n: (_var(n).defaut, d) for n, d in attendus.items() if _var(n).defaut != d}
    assert not ecarts, f"défaut dérivé sans changement de version (mesuré, attendu) : {ecarts}"


def test_la_ligne_de_regime_du_defaut_nu_a_sec():
    """Sous-processus sans aucune variable ACVRAM_* et sans carte : la ligne commence par « défaut » (rien hors
    défaut) et finit par les trois régimes que 0.6.30 sert."""
    _, fin = DEFAUTS_PAR_VERSION[acvram.__version__]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "from acvram import regime; print(regime.regime_ligne())"],
                         env=env, capture_output=True, text=True, timeout=180)
    ligne = out.stdout.strip().splitlines()[-1]
    assert ligne.startswith("[régime] défaut "), ligne
    assert ligne.endswith(fin), f"fin de ligne : {ligne!r} ≠ …{fin!r}"
