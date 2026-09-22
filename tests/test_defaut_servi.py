"""Le défaut servi est un contrat par version (sage-tests-rapides-cloture-20-09 § 4) : la ligne de régime du
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
    "0.6.33": (                                                          # 0.6.32 + vision Gemma 4 servie (sage 17 h 06, 20/09)
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
            # candidat NON installé tant que P3 (1) (b″) et S2 0 × 500 ne sont pas tenus (Manon). La tour de vision se sert quand le manifeste la déclare
            # (`vision=bf16(eager)` dans la ligne AVEC modèle, regime.py:454), rien dans le défaut nu ;
            # KV int8 gardé avec image (bande ≤ +1 %, verdict-decode-pas-31b-kv-20-09). Qwen3-VL : code
            # présent, aucun alias servi ni publié (P3 (1) non joué) — 0.6.34.
        },
        "mla_core=tf32(≤2048 clés) mla_prep=grille mla_glue=2 glue=compact(8) prefill_glue=compact",
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
