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
    Variable("I8C_FP8_PREFILL", "bf16", ("acvram.kernels", "_I8C_FP8_PREFILL"), None,
             "pièce 260 (opt-in, hors bit) : int8 ré-encodés du fp8 au préfill : bf16 (défaut, déquant de la 139) | cublas (W8A8 int8, A8 par jeton fusionnée, copie signée transitoire de la 201)"),
    Variable("I8C_COPIE", "xor", ("acvram.kernels", "_I8C_COPIE"), None,
             "pièce 260x (au bit) : copie signée q − 128 du chemin cublas (poids transitoires de la 201) : xor (défaut, un noyau, 2 o/poids) | int16 (témoin : l'aller-retour int16 d'avant)"),
    Variable("COLLE_MOE", "torch", ("acvram.engine.moe", "_COLLE_MOE"), "torch",
             "colle du préfill MoE : torch (argsort + bincount + _tuiles) | triton (P0 : tri + histogramme et grille en deux lancements, mêmes tenseurs)"),
    Variable("NVFP4_GEMV_MAX", "32", ("acvram.kernels", "_NVFP4_GEMV_MAX")),
    Variable("INT8_GEMV_MAX", "80", ("acvram.kernels", "_INT8_GEMV_MAX")),
    Variable("INT8_GEMV_MAX_PARTAGE", "16", ("acvram.kernels", "_INT8_GEMV_MAX_PARTAGE"), None, "pièce 243 (hors bit, KL tenue ; service mixte b=8 +9,70 % t/s) : seuil GEMV→GEMM int8 dans une portée depaquetage_partage (boucle par séquence GDN au préfill) ; 16 (défaut depuis 0.7.0) | 80 (témoin : sortie d'avant)"),
    Variable("DEPAQUETAGE", "auto", ("acvram.kernels.marlin_port", "_DEPAQUETAGE"), "torch",
             "pièce 147 (poste6, 24/09) : noyau du dépaquetage Marlin → bf16 au préfill de la disposition unique (PROJ_MARLIN=1 "
             "seulement) : auto (cuda si l'extension l'a, sinon triton) | cuda (lignes entières, au débit de nvfp4_dequant) | "
             "triton (tuile par programme, 24/09 matin) | torch (juge) — tous au bit entre eux ; le défaut (PROJ_MARLIN=0) n'y passe jamais"),
    Variable("NARROW_GEMM", "0", ("acvram.kernels", "_NARROW_GEMM"), "0"),
    Variable("NARROW_KERNEL", "mixte", ("acvram.kernels", "_NARROW_KERNEL"), None,
             "linéaires INT8 à b ≤ 16 : mixte (défaut, Triton dès b≥NARROW_TRITON_MIN_B, verdict-coder-c-mixte-17-09) | cuda | triton | tete"),
    Variable("NARROW_TRITON_MIN_B", "2", ("acvram.kernels", "_NARROW_TRITON_MIN_B")),
    Variable("NARROW_NVFP4", "0", ("acvram.kernels", "_NARROW_NVFP4"), "0"),
    Variable("DENSE_NVFP4", "triton", ("acvram.kernels", "_DENSE_NVFP4"), "gemv",
             "linéaires NVFP4 denses (et tête) à DENSE_NVFP4_MIN_M ≤ b ≤ 32 : triton (gemm_dense_etroit, poids lus une fois par pas, défaut depuis verdict-gemm-dense-palier1-situ-17-09) | gemv (témoin, poids relus par séquence)"),
    Variable("DENSE_NVFP4_MIN_M", "4", ("acvram.kernels", "_DENSE_NVFP4_MIN_M")),
    Variable("PROJ_MARLIN", "1", ("acvram.kernels", "_PROJ_MARLIN"), "0",
             "pièce 101 (23/09, opt-in) : 1 = linéaires NVFP4 denses par le Marlin porté de vLLM 0.29 (marlin_port, échelle "
             "globale par colonne pour q/k/v empilés) aux godets ≥ PROJ_MARLIN_MIN_M ; pièce 129 (24/09) : disposition préparée "
             "AU CHARGEMENT, mixte (rôles PROJ_MARLIN_DOUBLES en deux dispositions, M = 1 par nvfp4_gemv ; les autres en Marlin "
             "SEUL, M = 1 par nvfp4_gemv_marlin), mémoire prouvée au chargement (refus nommé) | 0 repli naturel ; "
             "pièce 156 : DÉFAUT (unique + v2 + TPB par forme, portée denses)"),
    Variable("PROJ_MARLIN_MIN_M", "2", ("acvram.kernels", "_PROJ_MARLIN_MIN_M")),
    Variable("PROJ_MARLIN_MIN_NK", "1024", ("acvram.kernels", "_PROJ_MARLIN_MIN_NK")),
    Variable("PROJ_MARLIN_MIN_N", "2048", ("acvram.kernels", "_PROJ_MARLIN_MIN_N"), None,
             "pièce 129 : N minimal d'un poids pris par la disposition Marlin (k/v à N = 1 024 plus lents en Marlin)"),
    Variable("PROJ_MARLIN_DOUBLES", "", None, None,
             "pièce 129 (A) : rôles gardés en DEUX dispositions (naturelle à M = 1, Marlin à M ≥ 2) ; les autres poids "
             "éligibles passent en Marlin SEUL (naturelle libérée) — revue/verdict-129-1-gemv-marlin-m1-24-09"),
    Variable("GEMV_MARLIN_V2", "1", ("acvram.kernels", "_GEMV_MARLIN_V2"), None,
             "pièce 130 (opt-in) : GEMV Marlin v2 à M = 1 sous la disposition Marlin seule (tuiles de colonnes par bloc, x en "
             "global, down en un lancement) | 0 : v1 (x en mémoire partagée, K ≤ 11 264 en deux moitiés)"),
    Variable("GEMV_MARLIN_TPB", "0", ("acvram.kernels", "_GEMV_MARLIN_TPB"), None,
             "pièce 130 : tuiles de 64 colonnes par bloc (1, 2, 4) ; 0 = par forme (2 si N ≥ 49 152, sinon 1 — 142 24B)"),
    Variable("GEMV_MARLIN_S", "0", ("acvram.kernels", "_GEMV_MARLIN_S"), None, "pièce 130 : split-K forcé ; 0 = règle de v1"),
    Variable("PROJ_MARLIN_PORTEE", "denses", ("acvram.kernels", "_PROJ_MARLIN_PORTEE"), None,
             "pièce 142 : global (tout poids dense éligible, linéaires hors experts des MoE compris) | denses (défaut "
             "depuis la 156 : un modèle à MoEBlock garde son chemin, les MoE ne sont pas mesurés)"),
    Variable("PROJ_MARLIN_CAPACITE", "65536", None, None,
             "pièce 129 : capacité KV minimale (jetons) exigée au chargement sous PROJ_MARLIN=1, sinon refus nommé ; non "
             "posée : séquences × longueur demandées par le chargement, sinon 65 536"),
    Variable("AWQ_TENSOR", "0", ("acvram.engine.moe", "_AWQ_TENSOR"), None,
             "pièce 123 (24/09, opt-in, jamais posé par un lanceur) : 1 = experts à tables AWQ d'activation sur le chemin "
             "tensor (échelle fondue dans moe_aligner_petit, xs) au lieu du repli GEMV nommé ; hors défaut : 123-quater "
             "FAUX au critère relatif (écart S1b/i8c 1,38-1,52 > 1,25 à b=12) | 0 défaut"),
    # lues dans acvram_kernels.cu (getenv, figées au premier lancement : un
    # PROCESSUS par valeur — poste7-gemv-experts-rpw-18-09)
    Variable("GROUPED_RPW", "4", None, None,
             "GEMV groupée des experts : lignes par warp (4 défaut depuis poste7-rpw-defaut-18-09, Coder b=12 1 262 t/s nu ; 1 | 2 témoins) ; l'activation est étagée une fois par bloc de GW_WARPS×RPW lignes"),
    Variable("GROUPED_OLD", "", None, "1", "témoin : l'ancien noyau groupé à 4 lignes par bloc"),
    Variable("GROUPED_XREG", "down", None, "0",
             "GEMV groupée des experts, K ≤ 2048 : down (défaut depuis verdict-gemv-experts-xreg-down-18-09 : ABAB × 0,944 nu, Coder b=12 1 307 t/s) = x en registres par tranche sur la projection down seule | 1 = gate/up aussi (témoin réfuté : 96 registres, 2 blocs/SM, +19 %) | 0 = x relu en shared (témoin) ; sortie identique au bit dans tous les cas"),
    Variable("GEMV_SPLITK", "1", None, "0",
             "GEMV Marlin (b) : split-K par lot aux petits lots (gate/up b=1 : 96 → 384 blocs, réduction du dernier bloc, déterministe) ; 1 défaut depuis 0.6.37 = S auto (+7,09 % b=1, KL 5/5, revue/poste2-piece67-23-09.md) | 0 = noyau d'avant (témoin) | ≥ 2 = S forcé (8 : moins bon que l'auto)"),
    Variable("GEMV_LAYOUT", "marlin", ("acvram.engine.moe", "_GEMV_LAYOUT"), "marlin",
             "P1 disposition UNIQUE (forme (b)), DÉFAUT depuis l'adoption du 18/09 : marlin = pile Marlin seule (préfill GEMM classe Marlin ET GEMV du décodage relisant les tuiles 16 k × 64 n ; la pile NVFP4 est rendue après le repack, experts_layout=marlin ; va avec PREFILL_GROUPED=marlin, sinon refus à l import ; scellé ≤ 0,97 × GEMV à b=1 et b=12, fp32 par ligne) | naturel = pile NVFP4 seule (témoin, avec PREFILL_GROUPED=groupe ; b=1 366 t/s contre 351 en marlin)"),
    Variable("MOE_TENSOR", "1", ("acvram.engine.moe", "_MOE_TENSOR"), "0",
             "pièce 65 (23/09) : GEMM groupée Marlin sur tensor cores aux godets ≥ 2 (défaut, A5 +12,2 % t/s b=12) | 0 témoin GEMV scalaire"),
    Variable("MOE_TENSOR_MIN_T", "8", ("acvram.engine.moe", "_MOE_TENSOR_MIN_T"), None,
             "pièce 65 : godet (jetons) minimal du chemin tensor — 8 mesuré (b=4 : GEMV +9,5 % plus rapide ; b=8 : tensor −4,3 %) ; 1, 2, 4 restent en GEMV"),
    Variable("MOE_TENSOR_FUSION", "1", ("acvram.engine.moe", "_MOE_TENSOR_FUSION"), "0",
             "pièce 63 : glue fusionnée, reproductible (défaut) | 0 témoin glue A4 (aligneur vLLM par atomiques, NON reproductible) — jamais servi"),
    Variable("MOE_W13", "1", ("acvram.engine.moe", "_MOE_W13"), "0",
             "pièces 82/82 ter (23/09) : 1 défaut = gate·up en une pile Marlin w13 (disposition unique, gate/up rendues, experts_layout=marlin-w13) — GEMM w13 au seul décodage tensor, sortie au 2⁻⁷ ; préfill (deux GEMM de largeur N lues dans w13) et GEMV au bit | 0 témoin gate et up séparées"),
    Variable("MARLIN_PAR_LIGNE", "0", ("acvram.engine.moe", "_MARLIN_PAR_LIGNE"), "1",
             "pièce 209 (25/09), À LA DEMANDE depuis la 232 b (26/09, chef) : 1 = piles d experts à sous-normales (Coder-30B couches 0, 1, 2, 4, 157) préparées en Marlin avec un facteur par ligne d expert (échelle globale par (expert, colonne), exact au bit des poids) et servies par le tensor / GEMV Marlin — qkvo-i8c +5,8 % (209 c), Coder nvfp4 pur +12,8 % en salve unique (226) mais −15,3 % / +25,4 % J en débit soutenu (229) : écart entre protocoles non expliqué | 0 défaut : préparation d avant, ces piles refusées (naturel, decode_mma) — ligne marlin(N/M) refus=[…]"),
    Variable("MARLIN_DISTINCT", "0", ("acvram.engine.moe", "_MARLIN_DISTINCT"), "0",
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
    Variable("PIPELINE", "1", None, "0", "lot préparé pendant le rejeu (runner) : 1 défaut depuis 0.6.34 (ids au bit b=1 et b=12) | 0 témoin"),
    Variable("SAMPLER_LOT", "0", None, "1", "échantillonnage vectorisé sur le lot (sampler) : 0 défaut = ancienne boucle par ligne (verdict eea064fe : le lot ralentit b=12 de 2,2 %) | 1 opt-in"),
    Variable("SAMPLER_LENT", "0", None, "1", "levier 1 (poste1-levier-1-conception-21-09, défaut depuis le verdict poste4 d145bf0d) : 0 défaut = glouton capturé dans le graphe (graphs.echantillon_glouton_dans, sampler=graphe si pipeline et graphes) | 1 témoin = ancien chemin hôte (_sample_lent hors graphe)"),
    Variable("SAMPLER_GRAPHE", "", None, None, "REMPLACÉE par SAMPLER_LENT (22/09) : encore lue avec avertissement, =0 vaut SAMPLER_LENT=1"),
    Variable("RAPATRIEMENT_FLUX", "0", None, "1", "levier 2 (poste1-levier-2-conception-22-09, défaut depuis le verdict poste4 22/09) : 0 défaut = ids/logprobs rapatriés par tampon hôte épinglé à double parité, _consommer n enfile rien (rapatriement=epingle si pipeline, graphes, sampler=graphe) | 1 témoin = .tolist() sur le flux"),
    Variable("RAPATRIEMENT_EPINGLE", "", None, None, "REMPLACÉE par RAPATRIEMENT_FLUX (22/09) : encore lue avec avertissement, =0 vaut RAPATRIEMENT_FLUX=1"),
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
    Variable("PILE_SANS_RENDU", "", None, "1",
             "1 = témoin : le cache de l'allocateur n'est pas rendu après chaque pile d'experts construite (bras A de gemma-capture-20-09 : première capture en OOM, Triton en OOM)"),
    Variable("MLP_HOTE_CPU", "", None, None, "1 = MLP exilé calculé sur le processeur"),
    Variable("KDA_CHUNK", "1", None, None, "taille de bloc KDA"),
    Variable("MAMBA_CHUNK", "1", None, None, "taille de bloc Mamba2"),
    Variable("ALLOC_EXTENSIBLE", "", None, None, "1 = allocateur CUDA à segments extensibles"),
    Variable("BUDGET_JETONS", "0", None, None, "budget de jetons du prefill (0 = illimité)"),
    Variable("EXIL_COUCHES", "", None, None, "couches exilées forcées"),
    Variable("EXIL_EXPERTS_FRACTION", "", None, None, "fraction d'experts exilés forcée"),
    Variable("SEUIL_EXIL", "0.20", None, None, "seuil d'exil du planificateur"),
    Variable("REPIN", "64", None, None, "période (pas) du ré-épinglage des experts"),
    Variable("MAX_GRAPHS", "64", None, None, "graphes CUDA gardés (au-delà : eager, aucune éviction)"),
    Variable("INSTA_MAX", "3", None, None, "relevés instantanés gardés"),
    Variable("KV_PLAN_OVERRIDE", "", None, None, "budget KV forcé (plan)"),
    Variable("LISTE_CLE", "", None, None, "clé de la liste de promotion"),
    Variable("LISTE_PROMUS", "", None, None, "tenseurs promus en int8 forcés"),
    Variable("MAX_PROMUS", "0", None, None, "plafond de tenseurs promus (0 = illimité)"),
    Variable("ORDRE_SAC", "snr", None, None, "ordre du sac à dos de placement (snr | …)"),
    Variable("ORDRE_SAC_INVERSE", "", None, None, "1 = sac à dos inversé"),
    Variable("PAGED_ALLOC", "", None, None, "allocateur paginé"),
    Variable("PA_CHUNK", "", None, None, "bloc de l'attention paginée"),
    Variable("PA_GQA", "1", None, "0",
             "pièce 182 : attention paginée int8 groupée GQA (un bloc par tête kv, K/V lus une fois pour ses G têtes), "
             "au bit du noyau d'origine ; lue par le lanceur CUDA à chaque appel ; 0 = noyau d'origine (témoin)"),
    Variable("PA_ETAPE", "", None, None, "étape de l'attention paginée"),
    Variable("PA_SANS_COMPTEUR", "", None, None, "attention paginée sans compteur"),
    Variable("INT8_GEMV_WARP", "", None, None, "warps du GEMV int8 (lu dans le .cu)"),
    Variable("INT8_TRANCHE", "", None, None, "découpage du GEMV int8 à N ≤ 16 : 4/6/8/10/12/16 (vide = 6, DÉFAUT depuis la "
             "pièce 187, au bit ; 16 = témoin d'avant, 12 = témoin du 14/09 ; lu dans le .cu, `ext.int8_tranches()`)"),
    Variable("INT8_TRANCHE_PREFILL", "", None, None, "pièce 187 : découpage du GEMV int8 à N > 16 (préfill ≤ INT8_GEMV_MAX), "
             "4/6/8/10/12/16 (vide = INT8_TRANCHE) ; au bit ; lu dans le .cu"),
    Variable("ECO", "", None, None,
             "poste7-eco-2700-defaut-19-09 § 1 : mode éco d'horloge DEMANDÉ par un instrument pour ses bras A/B (2700 | 2100 | off), jamais pour le service — le service lit config.json (\"eco\": \"2700\" par défaut) ; l'effectif est sur la ligne de régime (eco=<demandé>(<effectif>)) et un instrument ne publie pas si demandé ≠ effectif"),
    Variable("DOUBLE_DISPOSITION_DIAG", "0", None, "0",
             "diagnostic seulement (bissection du biais GEMV (b), poste7-p1-situ-verdict-18-09) : 1 = les deux dispositions gardées, préfill {groupe|marlin} × décodage {naturel|marlin} sur les mêmes piles ; jamais un régime servi"),
    Variable("MOE_GEMV", "v1", ("acvram.engine.moe", "_MOE_GEMV"), "v1",
             "GEMV groupée du décodage MoE : v1 (une passe de poids par paire expert-jeton) | v2 (paires triées par expert, poids lus une fois pour ≤ 4 jetons, sortie identique au bit)"),
    Variable("MULTI_PROJ", "0", ("acvram.engine.attention", "_MULTI_PROJ"), "0",
             "témoin (palier 2 non ouvert, 0,88 To/s) : q/k/v et qkv/gate/α/β du GDN en un lancement, chacune avec son scaler"),
    Variable("NARROW_MIN_M", "2", ("acvram.kernels", "_NARROW_MIN")),
    Variable("NARROW_ROWS", "32", ("acvram.kernels", "_NARROW_ROWS")),
    Variable("NARROW_MLA", "1", None, "0"),
    Variable("SEUIL_FUSION", "256", ("acvram.engine.attention", "SEUIL_FUSION")),
    Variable("PREFILL_GROUPED", "marlin", ("acvram.engine.moe", "_PREFILL_GROUPED"), "marlin",
             "GEMM groupée du prefill MoE : marlin (P1, classe Marlin sur la disposition unique, défaut depuis l'adoption du 18/09 : Coder 15 987 j/s, GLM 5 502) | groupe (B0 Triton bf16 persistant, défaut du 17/09 au 18/09, témoin) | w4a16 (B1, NVFP4 lu dans la tuile, opt-in jusqu'au scellé) | grouped_mm (torch, ancien défaut, témoin) | bmm par seaux (réfuté 0edc3b9)"),
    Variable("PREFILL_A4", "off", ("acvram.engine.moe", "_PREFILL_A4"), None,
             "porte qualité W4A4 du prefill MoE : fausse quantification NVFP4 des activations en torch — off | gateup (entrée de gate/up) | both (+ entrée de down) ; poste7-lecture-profils-coder-17-09"),
    Variable("PREFILL_A8", "off", ("acvram.engine.moe", "_PREFILL_A8"), None,
             "porte qualité W4A8 du prefill MoE (poste7-w4a4-clos-w4a8-porte-19-09) : fausse quantification des activations en torch — off | gateup | both ; format par PREFILL_A8_FMT ; exclusive de PREFILL_A4 ; scellé ratio − 1,0155 ≤ 0,004"),
    Variable("PREFILL_W8R", "0", ("acvram.engine.moe", "_PREFILL_W8R"), "0",
             "porte qualité W8r (poste7-poursuite-chantiers-19-09) : experts déquantifiés par _pile_bf16 re-arrondis en int8 symétrique par ligne — 1 = porte ; ne s'applique qu'aux chemins PREFILL_GROUPED=grouped_mm|bmm (pile naturelle, GEMV_LAYOUT=naturel) ; aucun noyau"),
    Variable("PREFILL_A8_FMT", "int8", ("acvram.engine.moe", "_PREFILL_A8_FMT"), None,
             "format de la porte A8 : int8 (par jeton, amax/127, l'arrondi de quantifier_a8 au bit) | e4m3 (E4M3 bloc 16, témoin mxf8f6f4)"),
    Variable("SANS_FUSION", "", None, "1"),
    Variable("SANS_FUSION_BF16", "", None, "1"),
    Variable("TETE_LIEE", "int8", ("acvram.engine.loader", "_TETE_LIEE")),
    Variable("TETE_FP32_ENTREE", "", None, "1"),
    # --- MoE ------------------------------------------------------------
    Variable("MOE_MMA", "1", ("acvram.engine.moe", "_MOE_MMA"), "0", "prefill W4A4 sur MMA FP4"),
    Variable("MOE_DECODE_MMA_MARLIN", "0", ("acvram.engine.moe", "_MOE_DECODE_MMA_MARLIN"), "0",
             "C17 (chantier-c17-mma2-lit-marlin-19-09) : sous la disposition unique Marlin, le décodage MoE par la MMA groupée (MOE_DECODE_MMA, t ≥ MIN_T) lit les TUILES MARLIN au lieu de laisser la GEMV Marlin ; 0 = jamais (défaut jusqu'au scellé) ; Mesure 1-ter sous 2 700 : ×0,89 à 45 distincts, ×1,23 à 8"),
    Variable("MOE_MMA_BT", "64", ("acvram.engine.moe", "_MOE_MMA_BT")),
    Variable("MOE_MMA_ETAGES", "4", ("acvram.engine.moe", "_MOE_MMA_ETAGES")),
    Variable("MOE_MMA_KS", "128", ("acvram.engine.moe", "_MOE_MMA_KS")),
    Variable("MOE_DECODE_MMA", "1", ("acvram.engine.moe", "_MOE_DECODE_MMA"), "0"),
    Variable("MOE_DECODE_MMA_BT", "16", ("acvram.engine.moe", "_MOE_DECODE_MMA_BT")),
    Variable("MOE_DECODE_MMA_MIN_T", "5", ("acvram.engine.moe", "_MOE_DECODE_MMA_MIN_T")),
    Variable("NORME_FUSEE", "0", ("acvram.engine.couches", "_NORME_FUSEE"), "0",
             "poste F (3b) : norme d'entrée dans le GEMV int8 q/k/v — RÉFUTÉ 6dbb1bb (+0,31 ms/pas, norme recalculée par bloc), témoin"),
    Variable("ROPE_KV", "0", ("acvram.engine.attention", "_ROPE_KV"), "0",
             "poste F (3a) : normes + RoPE + kv_write int8 en un noyau Triton — RÉFUTÉ a3f1b7e (corrompt sous graphe / codes ≠ kv_write_int8), témoin"),
    Variable("ECHO_TRANCHE", "256", None, None,
             "pièce 36 : positions par tranche quand `echo` demande les logprobs de l invite (Engine.logprobs_invite) — borne la pointe de VRAM des logits [tranche, vocab] fp32"),
    Variable("ETROITES_FORME", "", None, None,
             "pièce 35 : forme du noyau étroit int8 `W,S` (warps, étages) — vide = 4,3 (défaut) ; la sortie est AU BIT quelle que soit la forme (ni l ordre des sommes en K ni celui des tranches ne changent), seule l occupation change ; ligne etroites=serie|w{W}s{S}"),
    Variable("ETROIT_CANAL", "1", None, None,
             "pièce 195 (AU DÉFAUT depuis le 25/09, décision déléguée par l utilisateur ; HORS BIT « ± 1 ulp » : ordre des sommes fp32) : linéaires int8 symétriques par canal à 2 ≤ b ≤ 16 servies par `_etroit_canal_kernel` (K entier par programme, géométrie NInfer, table GEOMETRIE_CANAL par forme) — 1 = défaut ; 0 = TÉMOIN NOMMÉ (noyau d avant : vue g128, tranches) ; `BN,BK,W,S` = géométrie imposée ; ligne etroites=…+canal(table|temoin|géométrie) ; mixte b=8 +4,01 % / −3,76 % J, KL ≤ 2 × témoin (revue/poste6-piece195-verdict-25-09.md)"),
    Variable("CAPTURE_MEM_MIN_MIO", "1024", None, None,
             "garde d interblocage de capture (22/09) : mémoire libre minimale (Mio) sous laquelle une capture de graphe est refusée (eager) — une allocation manquante DANS la capture attend sans fin"),
    Variable("CAPTURE_DELAI_S", "120", None, None,
             "garde d interblocage de capture : au-delà de ce délai, alerte + pile de tous les fils au journal et abandon des captures suivantes (ligne graphes=off abandon(...))"),
    Variable("GLUE_COMPACT", "1", ("acvram.kernels", "_GLUE_COMPACT"), "0",
             "C15-3d (chantier-c15-niveau3-coder-20-09) : DÉFAUT 1 depuis le 20/09 (verdict-c15-niveau3-coder-19-09 addendum 05 h 15 : Coder b=12 servi sous éco 2 700 +11,5 % t/s, J 0,966 × le témoin, capture 5/5, experts égaux ≤ 1,2 × le taux du témoin A ON/OFF) ; 1 = 641 lancements/pas au lieu de 1 169 : logits du routeur par le MÊME cuBLAS que le témoin (routage au bit) + sélection top-k en un noyau à 1 warp, glue KV/q/valid au bit (kv_write_int8 par pas, q sans copie), GEMM étroit int8 réduit par son dernier programme (4 → 1 nœud), attention paginée réduite par son dernier programme (2 → 1, ATTN_WARPS_COMPACT) | 0 = témoin (le chemin d'avant)"),
    Variable("GLUE_COMPACT_ITEMS", "", ("acvram.kernels", "_GLUE_COMPACT_ITEMS"), None,
             "C15-3b (bissection) : sous GLUE_COMPACT=1, liste des fusions prises, séparées par des virgules — routeur | attn | etroit | kv ; vide = toutes ; une fusion absente suit le témoin"),
    Variable("PREFILL_COMPACT", "1", ("acvram.kernels", "_PREFILL_COMPACT"), "0",
             "C15-prefill (chantier-c15-prefill-20-09) : glue du préfill eager réduite par fusions AU BIT — 1 = épilogue des GEMM int8 cuBLAS en un noyau Triton (f32(acc)·s_x·s_w → bf16, gemm_w8a8.epilogue_i8c), A8 par jeton quantifiée une fois pour q/k/v, résidu différé (x + y absorbé par add_norm de la couche suivante), permutations MoE sans second tri ni conversions | 0 = témoin (le chemin d'avant) ; DÉFAUT 1 depuis le 20/09 (verdict-c15-prefill-19-09, poste2 bcd7a73b sur 93b4e586 : PPL au bit sur 3 tranches, capture 4/4, prefill Coder servi 22 748 / 22 683 contre 17 609 / 16 981 j/s = × 1,29-1,34, noyaux 89,37 contre 101,67 ms — le « ≤ 83 » du scellé initial reposait sur une prémisse fausse, dit par poste7) ; sans norm (rmsnorm un warp par ligne : 3,82 ms contre 2,12 au bloc, opt-in ITEMS=…,norm)"),
    Variable("PREFILL_COMPACT_ITEMS", "", ("acvram.kernels", "_PREFILL_COMPACT_ITEMS"), None,
             "C15-prefill (bissection) : sous PREFILL_COMPACT=1, liste des fusions prises — epilogue | a8 | residu | permut | attn (SDPA enable_gqa sans copie ×n_rep de K/V, sortie sans tampon à une séquence) | norm (rmsnorm un warp par ligne, même ordre de somme : HORS DÉFAUT, 3,82 ms contre 2,12 pour le bloc le 20/09, opt-in seulement) ; vide = le défaut (toutes sauf norm) ; une fusion absente suit le témoin"),
    Variable("ROUTE_PREP", "2", ("acvram.engine.moe", "_ROUTE_PREP"), "0",
             "poste F : 2 = moe_route + route_prep fusionnés (F2, défaut, verdict-f2-topk-17-09) | 1 = route_prep seul (F1) | 0 = torch"),
    Variable("MOE_DECODE_FUSED", "0", ("acvram.engine.moe", "_MOE_DECODE_FUSED"), "0"),
    Variable("MOE_FUSED_TN", "64", ("acvram.engine.moe", "_MOE_FUSED_TN")),
    Variable("MOE_FUSED_ATOMIQUE", "0", ("acvram.engine.moe", "_MOE_FUSED_ATOMIQUE"), "0"),
    Variable("MOE_FUSED_ETAGES", "3", ("acvram.engine.moe", "_MOE_FUSED_ETAGES")),
    Variable("MOE_ROUTE_PACK", "1", ("acvram.engine.moe", "_MOE_ROUTE_PACK"), "0"),
    Variable("MOE_GEMM_MAX", "48", ("acvram.engine.moe", "_MOE_GEMM_MAX")),
    Variable("MOE_GROUPED_MAX", "32", ("acvram.engine.moe", "_MOE_GROUPED_MAX")),
    Variable("MOE_GLUE_TORCH", "", None, "1", "moe_act + moe_reduce_trie en torch"),
    Variable("PREFILL_DEQUANT", "", None, "1", "experts routés : déquant + _grouped_mm au lieu des GEMM 4 bits"),
    Variable("MOE_DECODE_MASQUES", "", None, "1"),
    Variable("MOE_AWQ_TEMOIN", "0", ("acvram.engine.moe", "_MOE_AWQ_TEMOIN")),
    Variable("QA_COMPTE", "0", ("acvram.engine.moe", "_QA_COMPTE")),
    # --- MLA ------------------------------------------------------------
    Variable("MLA_BATCH", "2", ("acvram.engine.couches", "_MLA_BATCH"), "0"),
    Variable("MLA_BUCKET", "128", ("acvram.engine.mla", "MLA_BUCKET")),
    Variable("MLA_UNE_PASSE", "1", ("acvram.engine.mla", "_MLA_UNE_PASSE"), "0"),
    Variable("MLA_PREP_NOYAU", "1", ("acvram.engine.mla", "_MLA_PREP_NOYAU"), "0"),
    Variable("MLA_PREP_GRILLE", "1", ("acvram.engine.mla", "_MLA_PREP_GRILLE"), "0",
             "C14-b geste (3) (chantier-c14b-19-09 § Fait le 20/09) : 1 = mla_prep_batch regrillé (tuile k_b en shared, "
             "groupes de 4 créneaux : 492 blocs à b=12 au lieu de 172), AU BIT avec 0 (grille d'avant, temoin=True, "
             "prep_faux 0/799 sur carte) ; ne porte que le temps : DÉFAUT 1 en 0.6.31 sur M1 bis (scellé (a) ≤ 8 µs/couche ET pas b=12 non perdu) ; "
             "ligne mla_prep=grille|temoin ; indépendante de MLA_BATCH_FUSION"),
    Variable("MLA_BATCH_FUSION", "0", ("acvram.engine.mla", "_MLA_BATCH_FUSION"), "0",
             "C14-b (chantier-c14b-19-09, poste7-fiches-c5b-c13c-c14b-20-09 § 3) : 1 = au décodage par lot "
             "(decode_static_batch_complet) le combine de mla_decode_1p rend y = v_b·o_lat en bf16 "
             "(mla_1p_combine_vb_kernel) : l'einsum fp32 'hvr,bhr->bhv' (gemmSN_TN cuBLAS, 11,5 µs/couche à M=12) "
             "et sa conversion bf16 disparaissent ; même arithmétique à l'ordre des sommes près (± ulp fp32, pas au bit) ; "
             "pris seulement sous MLA_CORE_DECODE=fp32 ; scellé : ppl-decode-kv lot 12 ± 0,002 ET capture 5/5 ET pas b=12 "
             "GLM −0,7 ms → défaut | 0 = témoin (einsum)"),
    Variable("MLA_ECRIT_TORCH", "", None, "1",
             "sonde C15 niveau 2 (diagnostic) : 1 = à b=1 sous MLA_GLUE=2, l'écriture du latent passe par _ecrit_ligne (torch) au lieu de mla_ecrit_latent — isole ce noyau sous rejeu de graphe"),
    Variable("MLA_QABS_DEUX_MOITIES", "0", ("acvram.engine.mla", "_MLA_QABS_DEUX_MOITIES"), "0",
             "sonde (β) niveau 2 (diagnostic) : 1 = le chemin =1 calcule q_abs en deux moitiés fp32 puis arrondit (autre ordre de somme, ≤ 1 ulp, sans noyau) — PPL 8 192 + 512 : +1-3 % → le modèle est instable à la marge ; 0 = témoin"),
    Variable("MLA_PREP_TEMOIN", "0", ("acvram.engine.mla", "_MLA_PREP_TEMOIN"), "0",
             "sonde C15 niveau 2 (diagnostic, scratchpad/c15-temoin-20-09) : 1 = clones capturés des entrées/sorties de mla_prep_batch et de q_eff avant l'attention, lus après rejeu par temoin.py ; 0 = témoin"),
    Variable("MLA_GLUE", "2", ("acvram.engine.mla", "_MLA_GLUE"), "0",
             "C15 (chantier-c15-19-09) : DÉFAUT 2 depuis 0.6.32 (M3 2a-bis tenu 4/4 le 20/09 : ratio < 1 sur 7 520 lignes, −611 nœuds, pas b=1 −0,819 ms, J −4,4 %, PPL non établi pire ; 1 = témoin) ; DÉFAUT 1 du 20/09 matin (verdict-c15 : −459 lancements/pas, jetons identiques, GLM b=1 −0,43 ms) ; 1 = glue torch du décodage MLA b=1 retirée au bit (v_b fp32 une fois, cat kvp, stack RoPE, demi-tables cos/sin, résidu différé add_norm des couches MLA : −6 lancements/couche ; MoE : tok int64 servi, eid converti une fois, x[tok] une fois, tok_g = seq : −3 Marlin / −6 distincte) | 2 = en plus b=1 par decode_static_batch_complet (mla_prep_batch, numérique du lot, ≤ 1 ulp) | 0 = témoin"),
    Variable("MLA_CORE", "tf32", ("acvram.engine.mla", "_MLA_CORE"), "tf32",
             "C13 (poste7-c13a-defaut-19-09 § 1 + addendum) : précision des deux premiers produits du cœur d'attention MLA au PRÉFILL (scores, o_lat), softmax et noyau mla_1p en fp32 inchangés — tf32 DÉFAUT (C13-a tenu : 7 191 j/s, ΔPPL géo +0,00066) | fp32 (référence de qualité) | bf16 (C13-b : entrées bf16, acc fp32 ; scellé contre fp32 : prefill GLM ≥ 9 000 j/s ET ΔPPL géo ≤ +0,002 → défaut) | flash (C13-c forme 1, chantier-c13c-19-09 : noyau Triton causal fusionné kernels/attn_mla_causal.py, fp32 plein, un lancement par couche, scores jamais en HBM, moitié masquée sautée ; sortie = fp32 ± 8 ulp/ligne, aucune porte de PPL, toutes longueurs ; scellé cœur ≤ 110 ms ET prefill GLM ≥ 9 000 j/s ; sans Triton : repli fp32 nommé)"),
    Variable("MLA_FLASH_OPERANDES", "tf32", ("acvram.engine.mla", "_FLASH_OPERANDES"), None,
             "C13-c forme 2 : précision des opérandes du cœur flash (tf32 défaut, ≤ 2 048 clés vues ; fp32 = forme 1, 0,88 × cuBLAS mais structure ×26 ; tf32x3 sonde) — la ligne de régime nomme mla_core=flash(<operandes>,<tuile>)"),
    Variable("MLA_FLASH_TUILE", "", None, None,
             "C13-c diagnostic : BM,BN,warps,stages de la tuile du cœur flash (défaut : par la shared de la carte — sm_120 32,64,4,1 ; ≥ 200 Ko 64,64,8,2) ; la ligne de régime nomme la tuile"),
    Variable("MLA_CORE_VB", "0", ("acvram.engine.mla", "_MLA_CORE_VB"), "0",
             "C13 (poste7-c13a-defaut § 2) : 1 = le 3e produit du préfill y = v_b·o_lat (8ae21997) suit MLA_CORE ; 0 = fp32 ; scellé prefill ≥ 7 450 j/s ET ΔPPL géo ≤ +0,001 contre 2 produits → défaut 1"),
    Variable("MLA_CORE_MAX_CLES", "2048", ("acvram.engine.mla", "_MLA_CORE_MAX_CLES"), "2048",
             "diagnostic (prise à 36 tranches, poste2 20/09) : seuil de la règle des clés vues au préfill — au-delà, fp32 pour le morceau ; 0 = règle neutralisée (tf32/bf16 à toutes longueurs, ligne mla_core=tf32(sans règle des clés)) ; 2048 = défaut servi"),
    Variable("MLA_CORE_DECODE", "fp32", ("acvram.engine.mla", "_MLA_CORE_DECODE"), "fp32",
             "C13 niveau 2 (poste7-c13a-defaut § 2) : régime du cœur au DÉCODAGE (y = v_b·o_lat, sgemm fp32 1,5 ms/pas à b=12), indépendant de MLA_CORE — fp32 défaut | tf32 | bf16 ; scellé sgemm ≤ 0,6 ms ET ppl-decode-kv 3 tranches ± 0,001 ET capture {1,2,8,12,16} 5/5 → défaut"),
    Variable("MLA_A8", "off", ("acvram.engine.mla", "_MLA_A8"), None,
             "porte qualité FP8-MLA (poste7-cloture-23h59-19-09) : fausse quantification torch de l'ENTRÉE de q_b, kv_a et o — off | e4m3 (E4M3 bloc 16, format de la MMA mxf8f6f4) | int8 (par jeton, témoin) ; aucun noyau"),
    Variable("MLA_LATENT_FP8", "0", ("acvram.engine.mla", "_MLA_LATENT_FP8"), "0"),
    Variable("KV_LM4_SEUL", "", None, None, "diagnostic lm4 (kvcache.write) : lm4 sur k ou v seulement, int8 ailleurs"),
    Variable("KV_LM4_PUITS", "", None, None, "diagnostic lm4 : positions < N gardées int8 ; 0 = contrôle (lm4 partout par le diagnostic)"),
    Variable("MTP_ETAT", "brut", None, None,
             "pièce 105 : état caché lu par la tête MTP — brut (avant la norme finale, DeepSeek, défaut) | norme (après) | "
             "auto (selon la convention de la tête : qwen35 → norme, deepseek → brut)"),
    Variable("KV_FORMAT", "", ("acvram.memory.tiering", "_KV_FORMAT"), None,
             "cache KV des paliers carte : vide = capacités (int8) | k8v4 (pièce 104 : K int8 par jeton, V int4 "
             "par groupe de 32 canaux, −22 % d'octets, opt-in jugé KL 5/5 ≤ 0,74 et ppl-decode-kv 8 k ≤ +0,30 %) "
             "| lm4 4 bits par rotation | lm3, lm2 témoins"),
    Variable("KV_INT8_CANAL", "0", ("acvram.memory.kv_canal", "ACTIF"), "0",
             "C5-b (chantier-c5b-19-09) : 1 = clés int8 à échelle E4M3 par canal et par tête sur chaque bloc de 16 "
             "(bloc courant en bf16 dans une réserve, quantifié à sa fermeture ; V par jeton ; lecture par le noyau "
             "CUDA paginé variante CANAL, q ⊙ s_bloc) ; 0 = par jeton (défaut jusqu'au scellé : ΔPPL 3 × 2 048 ≤ +0,002 "
             "contre bf16 ET pas b=12 ≥ défaut − 1 % ET capture 5/5) ; ligne de régime kv=int8-canal16"),
    Variable("KV_CANAL_RANGS", "64", ("acvram.memory.kv_canal", "RANGS"), None,
             "C5-b : lignes bf16 de la réserve des blocs courants par couche (16 Kio chacune pour Coder) ; "
             "épuisée → le bloc repart par jeton"),
    Variable("MLA_NORME_NOYAU", "1", None, "0"),
    Variable("MLA_EAGER_TORCH", "", None, "1"),
    Variable("MLA_DEBUG_ECART", "", None),
    Variable("HYBRID_KERNELS", "1", None, "0", "KDA / GDN : noyaux hybrides ou torch"),
    Variable("GDN", "fla", ("acvram.engine.gdn", "_GDN_VOIE"), "torch",
             "Gated DeltaNet : fla (noyaux Triton de flash-linear-attention, prefill par blocs et décodage du lot en un lancement) | torch (référence transformers, séquence par séquence) ; 0 = refus des hybrides"),
    Variable("GDN_ETAT_EN_PLACE", "1", ("acvram.engine.gdn", "_GDN_ETAT_EN_PLACE"), "0",
             "pièce 156 F4 (DÉFAUT depuis 156 c, au bit ; 0 = témoin) : 1 = au décodage du lot, la récurrence fla écrit son état final dans le "
             "tampon statique (h0 = ht) au lieu d'une allocation suivie d'une copie de 25 Mo par couche (b=8, Qwen3.8)"),
    Variable("GDN_AB", "auto", ("acvram.engine.gdn", "_GDN_AB"), "separe",
             "pièce 175 (poste6, 25/09 ; DÉFAUT auto depuis 175 b) : portes α et β bf16 des couches GDN (alias mixte) : separe (témoin, deux F.linear) | "
             "concat (un F.linear sur β‖α) | triton (GEMM étroite fp32 déterministe, M ≤ 16) | auto (M = 1 concat, 2-8 triton, "
             "au-delà les deux appels : AU BIT des deux appels à chaque M, test_gdn_ab_175) ; inerte sur des α/β nvfp4"),
    Variable("GDN_AB_FLUX", "1", ("acvram.engine.gdn", "_GDN_AB_FLUX"), "0",
             "pièce 194 b2 (poste1, 25/09 ; DÉFAUT depuis le verdict 194 b2, au bit ; mixte b=8 +2,2 %) : 1 = β‖α (GDN_AB) sur un second flux pendant "
             "qkv‖gate, jointure avant la récurrence (M ≤ 16) ; ligne de régime « abflux » ; 0 = série (témoin)"),
    Variable("GDN_CONV_FUSEE", "1", ("acvram.engine.gdn", "_GDN_CONV_FUSEE"), "0",
             "pièce 156 F2 (DÉFAUT depuis 156 c, au bit ; 0 = témoin) : 1 = au décodage du lot, cast de qkv, état de conv, conv depthwise, "
             "silu et découpe q/k/v en un noyau Triton (gdn_conv.py), q/k sans répétition des têtes"),
    Variable("GDN_RES_DIFFERE", "1", ("acvram.engine.model", "_GDN_RES_DIFFERE"), "0",
             "pièce 156 F5 (DÉFAUT depuis 156 c, au bit ; 0 = témoin) : 1 = le résidu différé (add_norm, C15) admis aux couches Gated DeltaNet "
             "non MLA ; deux additions bf16 de moins par couche GDN"),
    Variable("GDN_PORTES_NOYAU", "1", ("acvram.engine.gdn", "_GDN_PORTES_NOYAU"), "0",
             "pièce 156 F1 (DÉFAUT depuis 156 d, ± ulp, KL contre témoins sur Qwen3.8 et Qwen3.5-35B ; 0 = témoin) : 1 = au décodage du lot (voie F4), softplus, exp et sigmoid des portes dans "
             "le noyau fla (A_log, dt_bias, APPLY_BETA_SIGMOID) au lieu de six noyaux torch par couche"),
    Variable("GDN_NORME_FUSEE", "1", ("acvram.engine.gdn", "_GDN_NORME_FUSEE"), "0",
             "pièce 156 F3 (DÉFAUT depuis 156 d, ± ulp, même KL ; 0 = témoin) : 1 = norme gated de la sortie GDN en un noyau Triton (gdn_norme.py), "
             "sortie bf16 ; lot, b=1 et préfill"),
    Variable("GDN_Z_BF16", "1", ("acvram.engine.gdn", "_GDN_Z_BF16"), "0",
             "pièce 182 (au bit) : 1 = z de la norme GDN rendu bf16 par _projections, lu en place (vue de la pile) par "
             "la norme F3 ; 0 = cast fp32 d'avant (témoin)"),
    Variable("GDN_QKV_GATE", "1", None, "0",
             "pièce 176 : 1 = GDN qkv‖gate INT8 en UNE pile au décodage (M ≤ 16), chaque segment gardant sa partition K "
             "(gemm_etroit._etroit_segments_kernel, au bit des deux appels) ; lue à la fusion ; 0 = deux appels (témoin)"),
    Variable("NORME_REGISTRES", "1", ("acvram.engine.layers", "_NORME_REGISTRES"), "0",
             "pièce 156 F6 (DÉFAUT depuis 156 d, au bit ; 0 = témoin) : 1 = RMSNorm hors préfill par rmsnorm_bf16_reg "
             "(ligne en registres, même découpe et même ordre de somme que rmsnorm_bf16) ; Qwen3.8 −0,30 ms/pas à b=8"),
    Variable("ADMISSION_FENETRE_MS", "5", None, None,
             "pièce 179 (DÉFAUT 5 depuis 179 b ; 0 = coupé) : fenêtre d'admission du serveur en ms — moteur vide et ≥ 2 "
             "requêtes en file, attendre que la file cesse de grossir avant le pas (préfill groupé d'une rafale) ; une "
             "requête seule n'attend pas"),
    Variable("DEPAQ_PARTAGE", "1", ("acvram.kernels", "_DEPAQ_PARTAGE"), "0",
             "pièce 172 (DÉFAUT, au bit ; 0 = témoin) : au préfill de plusieurs séquences, la boucle par séquence d'une "
             "couche à récurrence linéaire déquantifie chaque poids NVFP4 UNE fois (GEMM toujours par séquence)"),
    Variable("GDN_PREFILL_LOT", "0", ("acvram.engine.couches", "_GDN_PREFILL_LOT"), "0",
             "pièce 150 bis (opt-in) : 1 = au préfill de plusieurs séquences, projections Gated DeltaNet du lot en un "
             "appel (couches.py, forward_lot), convolution et règle delta par séquence ; autre M, donc pas au bit : KL"),
    # --- autres chemins de calcul ----------------------------------------
    Variable("FUSION_NVFP4", "1", None, "0", "témoin de mesure : fusion gate/up NVFP4"),
    Variable("FUSION_PARTIELLE", "0", None, "0"),
    Variable("LOGITS_BF16", "", None, "", "1 : tête en bf16 (précision-de-sortie-invisible-à-la-PPL)"),
    Variable("GRAPHS_EAGER", "", None, "1"),
    Variable("CPUS", "", None, "",
             "0.6.31 (acvram/hote.py) : affinité CPU du processus (`0-15`, `0-3,8`), posée par os.sched_setaffinity à l'import et à la construction d'Engine ; vide = aucune affinité (défaut) ; la ligne porte hote=…,cpus<plages relues>"),
    Variable("ATTN_WARPS_COMPACT", "8", ("acvram.kernels.attn_paginee", "WARPS_COMPACT"), "4",
             "C15-3d bis : warps du noyau d attention paginée fusionné (GLUE_COMPACT=1) ; DÉFAUT 8 (poste2 05 h 15, ABAB : B8 1 417 t·s⁻¹ · 0,2073 J = +11,5 % · 0,966 × A ; B4 1 386 · 0,2116 : 4 warps ne rend rien en W, 369 = 369, et perd 2 %) ; 4 = bras"),
    Variable("ATTN_REDUC_DEROULEE", "1", ("acvram.kernels.attn_paginee", "REDUC_DEROULEE"), "0",
             "pièce 92 (23/09) : réduction des tranches de l attention fusionnée déroulée (mêmes sommes, même ordre : au bit, 11/11 cellules du banc L2 froid + test) ; −1,9 % sur le lot b=12, −8,7 % à b=1 ctx 2 048 | 0 témoin boucle série"),
    Variable("ROUTAGE_TEMOIN", "0", ("acvram.engine.moe", "_ROUTAGE_TEMOIN"), "0",
             "diagnostic C15-3d : 1 = chaque couche MoE copie topi dans un tampon persistant (lisible sous graphes, equiv-b12.py « experts égaux ») ; 0 = témoin"),
    Variable("GODETS_B", "1", ("acvram.engine.graphs", "_GODETS_B"), "0",
             "clé de graphe CUDA, dimension b : 1 = lot arrondi au godet (puissances de deux, plafond HYBRID_SLOTS ; en place depuis le 11/09) | 0 = lot exact, témoin de mesure du chantier C4 (revue/chantier-c4-19-09) ; même sortie dans les deux cas"),
    Variable("TRANCHE_COPIE_MIN", str(2**30), ("acvram.kernels", "_TRANCHE_COPIE_MIN"), "0",
             "pièce 201 : octets de la copie fp32 entière d'un poids nvfp4 au-delà desquels les replis « naturel » et "
             "_marlin_seul tranchent (la tête d'un vocabulaire étendu, PPL) ; 0 = témoin (tranche comme la 153, change la "
             "sortie du préfill servi)"),
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
    "ACVRAM_JOURNAL_TENSEURS",                                                    # journal de conversion (cf97a3a0) : observation
    "ACVRAM_KERNELS_PRECOMPILES",                                                # 240 : dossier d'un .so précompilé — un chemin ; le .so servi est nommé par son empreinte
    "ACVRAM_ARCHS",                                                              # 070a (241) : architectures de compilation des noyaux précompilés, sans effet sur le calcul servi
    "ACVRAM_ARBRE_LIBRE",                                                         # garde d'import (a86fa1dd) : quel arbre est importé, aucun chemin de calcul
    "ACVRAM_TRACE_CRENEAUX", "ACVRAM_TRACE_ENTREES", "ACVRAM_TRACE_PTRS",
    "ACVRAM_TRACE_ROUTAGE", "ACVRAM_TRACE_ROUTAGE_PT", "ACVRAM_TRACE_STEPS", "ACVRAM_TRACE_COUCHES", "ACVRAM_CHRONO_SYNC", "ACVRAM_SYNC_COUCHES",
    
    "ACVRAM_PPL_TRANCHE",
    
    
    
    # compilation, placement, parc, mémoire : pas des chemins de calcul
    "ACVRAM_ARCH_FAMILY", "ACVRAM_CUDA_HOME", "ACVRAM_GW_WARPS",
    "ACVRAM_KERNEL_CACHE", "ACVRAM_BANC_ACCEPTE_REPLAN", 
    
    "ACVRAM_FOND_COOL", "ACVRAM_FOND_ZEN", "ACVRAM_GALERIE_DIR", "ACVRAM_MODELES", "ACVRAM_PARC",
    "ACVRAM_VERROU_GLOB",   # test seul (ajout GUI n°1) : chemin du glob, pas un chemin de calcul
    
    
    
    "ACVRAM_DUMP_MOE",   # dossier de recopie des tampons MoE (diagnostic a2711bc) : n'aiguille aucun calcul
    # chauffe du contexte (poste7-3b-lanceur-contexte (ii), s2-k48 § 2) : CHAUFFE_CTX=0 = opt-out de la preuve, TYPE
    # = intention du lanceur (carte.sh mesure/service) qui l ignore ; le régime servi qui en résulte est déjà sur la
    # ligne (`ctx_tenu=N(demandé M)` | non-verifie), aucun chemin de calcul n en dépend
    "ACVRAM_CHAUFFE_CTX", "ACVRAM_TYPE",
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


# Modèle chargé : « vision » de son acvram_manifest.json ("oui" | "non"), ou None
# tant qu'aucun chargeur ne l'a déclaré (ligne construite à sec, sans modèle).
_VISION_CHARGEE: Optional[str] = None
_I8C_PREFILL_BF16 = 0          # pièce 139 : poids int8 « origine: fp8 » du modèle chargé, servis en déquant bf16 au préfill
# M-RoPE (Qwen3-VL) : (section, entrelacé) du modèle chargé (manifeste `mrope_section` /
# `mrope_interleaved`, pièce (a)) — le mot `mrope=[24,20,20](interleaved)` ; None sinon.
_MROPE_CHARGE: Optional[tuple[list[int], bool]] = None


# Deepstack (Qwen3-VL, poste7-go-qwen3vl-parallele-20-09 § 2) : nombre de niveaux du modèle chargé,
# None quand rien ne le dit — le mot `deepstack=N` n'apparaît que déclaré.
_DEEPSTACK_CHARGE: Optional[int] = None

# Piece 133 (chef, meme classe que 127) : "architecture" du manifeste ("llama" | "moe"), None tant
# qu'aucun modele n'est charge (ligne construite a sec par un banc de noyaux GEMV, ou GEMV_LAYOUT reste
# pertinent). ACVRAM_GEMV_LAYOUT ne vit que dans moe.py et ne regle RIEN sur un modele sans MoEBlock
# (Qwen3.8-27B dense, entre autres) -- la ligne le disait quand meme, sans effet, ce qui a trompe une
# lecture de verdict (poste2, piece 102bis).
_ARCHITECTURE_CHARGEE: Optional[str] = None


def declarer_modele_charge(manifest: Optional[dict]) -> None:
    """Le chargeur déclare le manifeste du modèle qu'il vient de charger ; la
    ligne de régime en tire `vision=bf16(eager)` (manifeste `vision: oui`,
    contrat poste7-go-multimodal-organisation-20-09 § 2) ou `vision=off`.
    None : plus de modèle chargé, le mot disparaît. `mrope=[24,20,20](interleaved)`
    quand le manifeste porte `mrope_section` / `mrope_interleaved`. Le manifeste dit aussi le
    deepstack : `deepstack: 3` (nombre) ou `deepstack: oui` avec
    `deepstack_niveaux` (sinon 3, les `deepstack_visual_indexes` par défaut de
    Qwen3-VL) ; absent ou « non » : pas de mot."""
    global _VISION_CHARGEE, _ARCHITECTURE_CHARGEE, _I8C_PREFILL_BF16
    _VISION_CHARGEE = None if manifest is None else str(manifest.get("vision", "non"))
    _I8C_PREFILL_BF16 = 0 if manifest is None else sum(
        1 for e in (manifest.get("tensors") or {}).values()
        if isinstance(e, dict) and e.get("origine") == "fp8" and e.get("format") == "int8")
    _ARCHITECTURE_CHARGEE = None if manifest is None else manifest.get("architecture")
    if _VISION_CHARGEE != "oui":                          # pas de tour pour ce modèle : la précédente ne survit pas
        try:
            from .engine.vision import oublier_la_tour
            oublier_la_tour()
        except Exception:                                 # noqa: BLE001
            pass
    declarer_deepstack(_deepstack_du_manifeste(manifest))
    declarer_mrope(_mrope_du_manifeste(manifest))


def _deepstack_du_manifeste(manifest: Optional[dict]) -> Optional[int]:
    if not manifest:
        return None
    decl = manifest.get("deepstack")
    if decl is None or decl is False:
        return None
    if isinstance(decl, bool) or (isinstance(decl, str) and decl.strip().lower() in ("oui", "yes", "true")):
        n = manifest.get("deepstack_niveaux")
        idx = manifest.get("deepstack_visual_indexes")
        return int(n) if n is not None else (len(idx) if idx else 3)
    if isinstance(decl, (list, tuple)):
        return len(decl) or None
    if isinstance(decl, int) or (isinstance(decl, str) and decl.strip().isdigit()):
        return int(decl) or None
    return None


def declarer_deepstack(n: Optional[int]) -> None:
    """`deepstack=n` sur la ligne (n ≥ 1) ; None ou 0 : le mot disparaît."""
    global _DEEPSTACK_CHARGE
    _DEEPSTACK_CHARGE = int(n) if n else None


def deepstack_texte() -> Optional[str]:
    return f"deepstack={_DEEPSTACK_CHARGE}" if _DEEPSTACK_CHARGE else None


def _mrope_du_manifeste(manifest: Optional[dict]) -> Optional[tuple[list[int], bool]]:
    """(section, entrelacé) du manifeste : `mrope_section` / `mrope_interleaved`
    écrits par la conversion (quant/convert._manifeste_multimodal), sinon le
    rope_scaling du spec (`model`) ; None quand rien ne le dit."""
    if not manifest:
        return None
    section = manifest.get("mrope_section")
    entrelace = manifest.get("mrope_interleaved")
    if section is None:
        rs = (manifest.get("model") or {}).get("rope_scaling") or {}
        if not isinstance(rs, dict) or rs.get("mrope_section") is None:
            return None
        section, entrelace = rs["mrope_section"], rs.get("mrope_interleaved", False)
    return [int(x) for x in section], bool(entrelace)


def declarer_mrope(mrope: Optional[tuple[list[int], bool]]) -> None:
    """`mrope=[24,20,20](interleaved)` sur la ligne ; None : le mot disparaît."""
    global _MROPE_CHARGE
    _MROPE_CHARGE = None if mrope is None else (list(mrope[0]), bool(mrope[1]))


def mrope_texte() -> Optional[str]:
    """`mrope=[t,h,w](interleaved|blocs)` quand le modèle chargé a des positions à
    trois axes (engine/mrope), None sinon — un modèle texte n'ajoute aucun mot."""
    if _MROPE_CHARGE is None:
        return None
    section, entrelace = _MROPE_CHARGE
    return f"mrope=[{','.join(str(x) for x in section)}]({'interleaved' if entrelace else 'blocs'})"


def prefill_i8c_texte() -> Optional[str]:
    """Pièce 139 : ``prefill_int8=bf16(origine fp8 ×N)`` quand le modèle chargé a des poids int8 ré-encodés du fp8 —
    servis W8A16 au préfill (déquant bf16), pas W8A8 cublas ; None sinon (la ligne du défaut ne bouge pas)."""
    if not _I8C_PREFILL_BF16:
        return None
    from . import kernels                                   # 260 : l'opt-in cublas les sert en W8A8, nommé
    mode = "cublas" if kernels._I8C_FP8_PREFILL == "cublas" else "bf16"
    return f"prefill_int8={mode}(origine fp8 ×{_I8C_PREFILL_BF16})"


def vision_texte() -> Optional[str]:
    """`vision=bf16(eager)` | `vision=off` | None (aucun modèle déclaré : rien à
    ajouter, la fin de ligne du défaut nu reste celle que fige test_defaut_servi)."""
    if _VISION_CHARGEE is None:
        return None
    return "vision=bf16(eager)" if _VISION_CHARGEE == "oui" else "vision=off"   # la ligne préfère la tour réelle (engine/vision) quand elle est chargée


def glue_texte() -> str:
    """C15-3d (défaut le 20/09) : la glue compacte est nommée sur la ligne, défaut compris —
    ``glue=compact(8)`` (attention fusionnée à 8 warps), ``glue=compact(8,items=attn,kv)``
    en bissection, ``glue=temoin`` sous GLUE_COMPACT=0 : la cellule 0.6.24 (poste7 05 h 40)
    portait une ligne qui ne disait pas le régime servi."""
    try:
        from . import kernels as _k
        from .kernels import attn_paginee as _ap
        if not getattr(_k, "_GLUE_COMPACT", 0):
            return "glue=temoin"
        items = getattr(_k, "_GLUE_COMPACT_ITEMS", None) or ""
        # pièce 92 : réduction déroulée par défaut, au bit de la boucle série — seul le témoin est nommé
        serie = "" if getattr(_ap, "REDUC_DEROULEE", True) else ",reduc=serie"
        return f"glue=compact({_ap.WARPS_COMPACT}" + (f",items={items}" if items else "") + serie + ")"
    except Exception as exc:                              # noqa: BLE001
        return f"glue=?({type(exc).__name__})"


def prefill_glue_texte() -> str:
    """C15-prefill : la glue du préfill est nommée sur la ligne, défaut compris —
    ``prefill_glue=temoin`` (PREFILL_COMPACT=0), ``prefill_glue=compact`` (défaut depuis le 20/09),
    ``prefill_glue=compact(items=epilogue,a8)`` en bissection. Un chiffre de
    préfill sans cette étiquette ne dit pas quel chemin l'a produit."""
    try:
        from . import kernels as _k
        if not getattr(_k, "_PREFILL_COMPACT", 0):
            return "prefill_glue=temoin"
        items = getattr(_k, "_PREFILL_COMPACT_ITEMS", None) or ""
        return "prefill_glue=compact" + (f"(items={items})" if items else "")
    except Exception as exc:                              # noqa: BLE001
        return f"prefill_glue=?({type(exc).__name__})"


def regime_ligne() -> str:
    """Une ligne pour l'en-tête d'une mesure : ce qui diffère du défaut,
    puis extension et masques. « défaut » seul veut dire : tout au défaut."""
    r = regime_noyaux()
    parts = [f"{k}={v if v else repr('')}" for k, v in r["hors_defaut"].items()
             if k != "ACVRAM_GEMV_LAYOUT"] or ["défaut"]
    # la disposition lue par le GEMV des experts est toujours nommée (P1
    # disposition unique, poste7-p1-disposition-unique-18-09) : marlin | naturel —
    # SAUF sur un modèle chargé sans MoEBlock (pièce 133) : la variable n'y règle
    # rien (moe.py ne la lit que dans du code MoE), la nommer comme active
    # aurait trompé une lecture (poste2, pièce 102bis, Qwen3.8-27B dense).
    _gv = "ACVRAM_GEMV_LAYOUT=" + str(r["variables"].get("ACVRAM_GEMV_LAYOUT", "?"))
    if _ARCHITECTURE_CHARGEE is not None and _ARCHITECTURE_CHARGEE != "moe":
        _gv += "(inerte:modèle dense)"
    parts.append(_gv)
    # split-K toujours nommé, défaut 1 depuis 0.6.37 (pièce 70) : S auto ou repli
    # si le noyau manque (sans GPU, rien ne lève — pièce 70, 23/09)
    try:
        from . import kernels as _kernels
        _sk_mode = int(os.environ.get("ACVRAM_GEMV_SPLITK", "1"))
        _ext = _kernels.get_extension()
        if _ext is not None and hasattr(_ext, "nvfp4_gemv_marlin_splitk"):
            _sk = ("S(auto)" if _sk_mode == 1 else ("témoin(0)" if _sk_mode == 0 else f"forcé({_sk_mode})"))
            parts.append("gemv_splitk=" + _sk)
        else:
            parts.append("gemv_splitk=repli(noyau absent)")
    except Exception as exc:                                     # noqa: BLE001
        parts.append(f"gemv_splitk=?({type(exc).__name__})")
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
    # 0.6.31 : l'hôte effectif (THP, fils OpenMP, affinité) — un chiffre d'hôte sans cette étiquette compare
    # deux régimes sans le savoir (poste7 : OMP est le premier suspect si le prefill perd > 3 %)
    try:
        from .hote import hote_texte
        parts.append(hote_texte())
    except Exception as exc:                              # noqa: BLE001
        parts.append(f"hote=?({type(exc).__name__})")
    # L'horloge SM verrouillée (`nvidia-smi -lgc`, mode éco, poste7-e1-eco-tenu-
    # 19-09 § 2) est root et hors processus : aucun défaut ACVRAM_* ne bouge,
    # et un chiffre éco passerait pour un chiffre défaut. Nommée seulement
    # quand une carte est visible : à sec la ligne ne change pas.
    try:
        from .engine import mla as _mla
        if _mla.regime_coeur_texte():                     # tf32/bf16(≤2048 clés) | flash(fp32) | flash(repli fp32: …)
            parts.append(_mla.regime_coeur_texte())
        parts.append(_mla.regime_prep_texte())            # mla_prep=grille|temoin (C14-b geste 3, défaut 1 en 0.6.31)
        parts.append(_mla.regime_glue_texte())            # mla_glue=2|1(temoin)|0 (C15 2a-bis, défaut 2 en 0.6.32)
    except Exception:                                     # noqa: BLE001
        pass
    # Multimodal : la tour de vision (eager bf16, hors graphes, engine/vision) nommée avec la version
    # relevée dès qu'une est chargée ; sinon « vision=off » si un modèle est DÉCLARÉ (manifeste lu par
    # le moteur, declarer_modele_charge) ; rien sans modèle — la fin de ligne du défaut nu
    # (tests/test_defaut_servi.py) ne bouge pas. Contrat § 2 : vision=bf16(eager)|off.
    try:
        from .engine.vision import regime_texte as _vision_texte
        tour = _vision_texte()
    except Exception:                                     # noqa: BLE001
        tour = ""
    if tour:
        parts.append(tour)
    elif _VISION_CHARGEE == "oui":
        parts.append("vision=declaree(tour absente)")     # manifeste oui, tour non chargée : jamais « bf16(eager) » sans tour
    elif vision_texte():
        parts.append("vision=off")
    if mrope_texte():
        parts.append(mrope_texte())                       # mrope=[24,20,20](interleaved) : M-RoPE Qwen3-VL (engine/mrope)
    if deepstack_texte():
        parts.append(deepstack_texte())
    if prefill_i8c_texte():
        parts.append(prefill_i8c_texte())                   # deepstack=3 : seulement déclaré (manifeste / config de la tour)
    # C5-b : le format des clés est nommé dès qu'il n'est plus celui d'aujourd'hui
    # (int8 par jeton) — la variable dit ce qui est DEMANDÉ, cette étiquette ce
    # que le cache int8 fait de ses clés.
    try:
        from .memory import kv_canal as _kvc
        if _kvc.ACTIF:
            parts.append("kv=int8-canal16")
    except Exception:                                     # noqa: BLE001
        pass
    parts.append(glue_texte())
    parts.append(prefill_glue_texte())
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
