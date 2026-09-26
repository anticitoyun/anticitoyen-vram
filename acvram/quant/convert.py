"""Convertit un point de contrôle Hugging Face en fragments acvram.

Le résultat n'est pas un modèle quantifié mais un modèle *par classe
d'appareil* : un tenseur destiné à la RTX 5090 est écrit en NVFP4, le même
tenseur destiné à la RTX 3080 Ti est écrit en INT4. C'est le plan de placement
qui tranche, si bien que la conversion et le chargement s'accordent par
construction — il n'existe aucune vérification à l'exécution du type « ce GPU
sait-il lire ce format ? » que l'on puisse rater.

La calibration, quand elle est activée, procède couche par couche à la manière
d'AWQ : ne tenir qu'un seul bloc en bf16, y faire passer les états cachés de
calibration pour relever les magnitudes d'activation par canal, quantifier ce
bloc, le libérer, passer au suivant. Le pic de mémoire est d'un bloc et non du
modèle entier, ce qui rend simplement possible la calibration d'un modèle de
70 milliards de paramètres sur cette machine.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch

from ..engine.config import ModelSpec, load_model_spec
from ..memory.tiering import Plan
from . import formats
from .nvfp4 import NVFP4Tensor
from .calibrate import (ActStats, ChannelScaler, alpha_commun_gate_up,
                        quantize_with_calibration)

__all__ = ["ConversionOptions", "ConversionReport", "convert_checkpoint",
           "TensorRouter"]

SHARD_TARGET_BYTES = 4 * 1024 ** 3


@dataclass
class ConversionOptions:
    out_dir: str
    calibrate: bool = False
    # Sequences et jetons REELLEMENT lus par load_calib_ids, poses par cli.py ;
    # 0/0 = aucune calibration. Jamais un defaut qui ressemble a une mesure
    # (16/128 sont restes trois semaines au manifeste sans qu'aucune passe ne
    # les ait produits — poste7-calibration-verdict-17-09).
    calib_tokens: int = 0
    calib_seqs: int = 0
    use_hadamard: str = "auto"        # auto | always | never
    # Conserve l'erreur des 21 valeurs de la grille AWQ pour chaque tenseur, au
    # lieu du seul minimum. Sert a calculer le prix d'un exposant COMMUN a un
    # groupe empilable : `gate` et `up` lisent la meme entree mais chacun
    # choisit son exposant, et `_scaler_commun` refuse la fusion par
    # `torch.equal` — 5 empilements sur 64 sur Llama-2-7b-int8, pour +0,19 % la
    # ou une couverture complete vaudrait +2,43 %. Coute ~170 flottants par
    # tenseur au manifeste, rien a l'execution ; hors campagne, laisser a faux.
    garder_grille: bool = False
    awq: bool = True
    group_size: int = 128
    # Q1 (chef 21/09, arXiv 2512.02010) : règle d'échelle de bloc nvfp4, max6 | 4sur6 — posée sur le module
    # `quant.nvfp4` au début de la conversion, donc lue par la recherche AWQ ET la quantification finale
    echelle_nvfp4: str = "max6"
    keep_sensitive_16bit: bool = True  # normalisations, routeur, plongements
    lm_head_format: Optional[str] = None
    # poste7-p2-qkvo-int8-canal-18-09 : q/k/v/o restent int8, mais symetrique
    # PAR CANAL (une echelle par ligne de sortie, sans point-zero variable)
    # au lieu du groupe de 128 affine du reste du modele -- format cible de
    # torch._int_mm/cuBLASLt. Le reste de la conversion (awq=False, group_
    # size=128 ailleurs, mixed_precision, snr_floor) reste inchange.
    attn_qkvo_int8_canal: bool = False
    # 153 (chef 24/09) : meme regime, mais pour les cinq projections de
    # poids du GatedDeltaNet (qkv/gate/alpha/beta/out) -- drapeau distinct,
    # ne change pas le sens de attn_qkvo_int8_canal ci-dessus.
    gdn_int8_canal: bool = False
    # Table de niveaux q3n de CE modèle (huit flottants, symétrique, bornes
    # ±1) — écrite dans chaque entrée q3n du manifeste. None : TABLE_Q3N de
    # la spécification. Les niveaux s'ajustent par modèle (Lloyd-Max sur
    # échantillon stratifié) ; voir docs/FORMAT-3BITS.md du 8/09 au soir.
    q3n_table: Optional[tuple] = None
    n_grid: int = 20
    device: str = "cuda:0"
    # Appareil sur lequel se fait la recherche AWQ et la quantification. Elle
    # est dominee par des produits matriciels sur la grille de recherche : un
    # GPU la rend une dizaine de fois plus rapide qu'un i9. "auto" prend le
    # premier GPU disponible, "cpu" force l'ancien chemin.
    quant_device: str = "auto"
    # Journal une ligne par tenseur sur stderr (heure, rang, nom, forme,
    # secondes) : None = automatique, actif quand la quantification tourne
    # sur le processeur (poste2 21/09 : 31B `--quant-device cpu`, 96 min par
    # shard et RIEN d incrémental — `progress` ne parle que tous les 25
    # tenseurs, soit des dizaines de minutes à ce rythme) ou sous
    # ACVRAM_JOURNAL_TENSEURS=1 ; False le tait, True le force.
    journal_tenseurs: Optional[bool] = None
    # Pièce 25 (b) : ce qu un expert MoE reçoit quand le corpus de calibration
    # l a routé moins de MIN_ECHANTILLONS_AWQ fois — "identite" (défaut, le
    # repli d avant : aucune échelle, arrondi au plus proche) ou
    # "mediane_couche" (opt-in : statistique = médiane par canal des mean_abs
    # des experts calibrés de la même couche et projection, puis la même
    # recherche AWQ que les autres). 30B-VL 21/09 : 2 487/18 432 experts sans
    # stats, P3 (3) à 12,6 % contre 3 % — c est le levier à mesurer.
    repli_experts: str = "identite"
    dry_run: bool = False
    mixed_precision: str = "auto"     # auto | off
    # dB de rapport signal/bruit en sortie de couche sous lequel un tenseur est
    # promu d'un barreau. Zéro, donc rien : le décodage est limité par la bande
    # passante, et sur un 27B les promotions coûtaient 13,4 % de mémoire et
    # 10,6 % de débit pour 2,0 % de perplexité. Le mécanisme reste entier,
    # `--snr-floor 25` rétablit l'ancien comportement.
    snr_floor: float = 0.0
    # Autorise une conversion qui produit un modèle PLUS GROS que sa source.
    # Refusée par défaut depuis le 8/09/2026 : convertir un GGUF de 3,4 bits
    # par poids vers NVFP4 (4,5) a fait grossir Qwen3-Coder-Next d'un tiers,
    # créé 14 Gio d'exil en RAM hôte et coûté un facteur quinze au décodage.
    autoriser_grossissement: bool = False
    # Format reclame explicitement en ligne de commande (`--format int8`), par
    # opposition au format nominal choisi par la politique de placement. La
    # distinction commande le comportement de la garde anti-grossissement : une
    # politique peut choisir a la place de l'utilisateur, elle ne doit pas
    # ecraser son choix explicite. Le 8/09/2026, un temoin demande en int8 est
    # sorti avec ses MLP en q3n a 3,25 bits — SNR de 14,9 dB contre 44 pour le
    # reste du modele — et le dossier s'appelait « temoin-int8 ». La bascule
    # etait annoncee a l'ecran, dans un journal detache que personne n'a lu, et
    # la mesure qui en est sortie a fait chercher un biais d'instrument pendant
    # une demi-journee.
    format_impose: Optional[str] = None
    max_promotions: float = 0.15      # part maximale de tenseurs promus
    # Classes de tenseurs PROMOUVABLES (suffixes de nom, ex. ("q_proj", "k_proj")) ;
    # vide = toutes. Bead 1aj décodage (poste7, 14/09) : sur Coder-30B tous les
    # q/k/v/o et le lm_head sont promus int8 (0,93 + 0,32 Go par jeton à b=1, +18 %
    # d'octets sur llama.cpp) ; tout retirer (A6, snr_floor=0) coûte +8,79 % de PPL.
    # Le test par projection convertit en ne laissant promouvoir qu'une classe à
    # la fois pour trouver laquelle refuse le W4.
    promotion_classes: tuple = ()
    # Prix plafond d'une promotion, en mébioctets ajoutés (0 = pas de plafond).
    # Le quota ci-dessus compte des tenseurs ; or une porte de 0,1 Mio et une
    # projection MLP de 39 Mio gagnent le même nombre de décibels en montant
    # d'un barreau. À quota par tenseurs, l'ordre de rencontre décide, et les
    # projections épuisent le budget avant que les portes soient vues. Un
    # plafond de prix trie par ce qui compte vraiment : les octets relus à
    # chaque jeton.
    promotion_cout_max_mib: float = 0.0
    # Budget d'octets pour l'affectation par sac à dos (0 = mécanisme classique
    # de plancher SNR). Avec un budget, chaque tenseur promouvable est mesuré
    # dans les deux formats, puis les promotions sont choisies par gain de SNR
    # par octet dépensé, jusqu'à épuisement : « le meilleur modèle qui tient
    # dans N gibioctets », au lieu d'un seuil arbitraire.
    bits_budget_gib: float = 0.0
    # Mesure le KLD couche-par-couche (proxy softmax du meme vecteur de
    # sortie que out_snr_db) et le publie dans le manifeste, en plus du SNR.
    # N'AFFECTE AUCUNE DECISION : le convertisseur continue de promouvoir sur
    # `out_snr_db`. Sert au protocole A/B (revue/duck-poste2-12-09.md) qui
    # compare, a budget d'octets EGAL, si un classement par KLD aurait promu
    # d'autres tenseurs — sans jamais changer ce qui se convertit aujourd'hui.
    mesurer_kld: bool = False
    # Item A7 de l'audit poste7 (14/09) : `gate_proj` et `up_proj` lisent la
    # meme entree mais chacun cherchait son propre alpha AWQ, si bien que
    # `_scaler_commun` (layers.py) refusait presque toujours de porter un
    # scaler unique sur la paire empilee — 5 fusions recuperees sur 64 sur
    # Llama-2-7b-int8 (revue/alpha-partage-recuperer-59-fusions.md). Ce flag
    # cherche UN alpha commun par paire (`calibrate.alpha_commun_gate_up`),
    # identique par construction. Defaut a faux : ne change pas la
    # conversion par defaut sans mesure (revue/prediction-a7-alpha-commun-
    # gateup-14-09.md).
    alpha_commun_gate_up: bool = False
    # 22/09 (`poste1-disposition-naturel-qkv-22-09`) : le MÊME alpha commun,
    # mais PAR EXPERT MoE — les experts en étaient exclus (« leur pile groupée
    # obéit à une autre règle »), or c'est l'inverse : la pile groupée EXIGE
    # que gate et up partagent leur échelle, sinon `_try_build_stacks` ne
    # fusionne pas les tables (`engine/moe.py:336-337`, `torch.equal` global
    # sur [E, K]) et `_construire_marlin` refuse la disposition Marlin
    # (`moe.py:446`) : Qwen3-Coder-30B-A3B-nvfp4-qkv-22-09 charge en
    # `experts_layout=naturel`, 8,62 ms/pas contre 6,7. Opt-in tant que la
    # mesure n'est pas faite : coût = une seconde recherche AWQ par paire
    # d'experts à la conversion.
    alpha_commun_experts: bool = False
    # 23/09 (pièce 100 B, poste6) : le même alpha commun pour q/k/v d'une
    # couche d'attention — ils lisent la même entrée, et `stack_nvfp4_linears`
    # (layers.py, `_scaler_commun`) refuse la pile qkv dès que la calibration
    # AWQ leur donne trois échelles d'activation distinctes (pièce 42 :
    # q, k, v servis en trois GEMM, +0,9 ms/pas). o_proj n'est pas concerné
    # (autre entrée). Opt-in ; prix : une recherche AWQ de plus par couche.
    alpha_commun_qkv: bool = False
    # poste7 (`poste7-hadamard-16-09.md`, 16/09) : la métrique W4A4 des experts
    # (`quantize_activation_nvfp4`, cb2784b) RÉFUTÉE plus mauvaise (1,0229)
    # que sans elle (1,0183) — le défaut n'est pas l'alpha choisi mais le
    # bloc de 16 lui-même (E2M1, étendue 12:1) sous UN canal aberrant, qu'AUCUNE
    # échelle par canal ne peut corriger puisqu'elle s'applique à tout le
    # canal, pas à un bloc isolé. Rotation de Walsh-Hadamard bloc-diagonale
    # H_512 sur les poids d'experts avant quantification (gate/up K=2048,
    # down K=1536, les deux divisibles par 512) : étale l'aberration sur
    # 512 valeurs au lieu d'une seule dans chaque bloc de 16, SANS échelle
    # AWQ (`use_awq=False` sur ces tenseurs — la rotation est le mécanisme
    # à l'essai, pas un supplément à l'échelle). Défaut faux : n'affecte
    # aucune conversion existante sans le demander explicitement.
    hadamard_experts: bool = False
    # Passage DIRECT d'une source déjà NVFP4 (modelopt, compressed-tensors
    # nvfp4-pack-quantized) : ses poids 4 bits sont copiés tels quels — ni
    # déquantification, ni recherche AWQ, ni promotion, ni rotation — pour
    # que le converti serve exactement les poids de vLLM (poste7-convertisseur-
    # formats-16-09 § 3.1). Les couches gardées en clair par la source
    # suivent le plan comme avant.
    passage_direct: bool = False
    # Pièce 139 : alias TEXTE depuis une source VL — la tour (VISION_PREFIXES) n'est pas écrite, manifeste
    # vision=non. Pour une famille dont le masque de plage image n'est pas connu (Qwen3_5, vision.py:268).
    sans_vision: bool = False
    # poste7 (`poste7-corpus-16-09.md` § 8) : nom + sha256 du fichier de
    # calibration reellement utilise (ou du corpus integre, ou une absence
    # explicite si awq=False) -- calcule par cli.py, porte au manifeste via
    # `asdict(opts)` (`options.calib_source`). Ce champ ne pilote AUCUNE
    # decision de conversion, il ne fait que documenter ce qui a servi :
    # le doute sur GLM -k48 (calibre sur le corpus d'eval wiki-gptq ?) ne
    # pouvait pas se trancher en lisant le manifeste seul.
    calib_source: Optional[dict] = None


@dataclass
class ConversionReport:
    model: str = ""
    tensors: int = 0
    in_bytes: int = 0
    out_bytes: int = 0
    per_format: dict[str, int] = field(default_factory=dict)
    worst_layers: list[dict] = field(default_factory=list)
    promotions: list[dict] = field(default_factory=list)
    mean_out_snr_db: float = 0.0
    seconds: float = 0.0
    # Combien d'experts, dans les couches MoE, n'avaient AUCUNE statistique
    # d'activation a la calibration (jamais routes, ou trop peu, pendant le
    # corpus de calibration) : une mesure du corpus. Ces experts recoivent
    # une echelle identite EXPLICITE dans le manifeste (jamais une absence
    # ambigue) -- voir la boucle principale et revue/poste7-glm-awq-
    # pile-15-09.md (poste7 a6a7436 : echelle AWQ par expert gardee dans la
    # pile, portee cote loader par poste4).
    experts_sans_stats: int = 0
    # Pièce 25 (b) : experts sans stats ayant reçu, à la place de l identité,
    # une statistique de repli = médiane par canal des `mean_abs` des experts
    # de la même couche et même projection (opt-in `repli_experts`).
    experts_repli_mediane: int = 0
    experts_sans_stats_noms: list = field(default_factory=list)
    # poste7-awq-relu2-garde-repli-17-09 : tenseurs dont l'AWQ a ete rejete
    # (ratio de norme hors bornes) et repliés à l'identité — vide si
    # aucun. `cmd_convert` (cli.py) lit ce champ pour ajouter `-repliN` au
    # nom du dossier de sortie (REGLES §4 : le régime dans le nom).
    tenseurs_replies: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.in_bytes / max(1, self.out_bytes)

    def render(self) -> str:
        lines = [f"converti : {self.model}", ""]
        lines.append(f"  tenseurs         {self.tensors}")
        lines.append(f"  entree           {_h(self.in_bytes)}")
        lines.append(f"  sortie           {_h(self.out_bytes)}  "
                     f"(x{self.ratio:.2f} plus petit)")
        for fmt, n in sorted(self.per_format.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {fmt:<12} {_h(n)}")
        lines.append(f"  SNR sortie moyen {self.mean_out_snr_db:.1f} dB")
        if self.promotions:
            lines.append(f"  promus           {len(self.promotions)} tenseurs vers un "
                         f"format plus large (sous le plancher de SNR) :")
            for p in self.promotions[:5]:
                lines.append(f"    {p['name']:<46} {p['from']} -> {p['to']}  "
                             f"{p['before']:.1f} -> {p['after']:.1f} dB")
            if len(self.promotions) > 5:
                lines.append(f"    ... et {len(self.promotions) - 5} autres")
        if self.worst_layers:
            lines.append("  pires tenseurs :")
            for w in self.worst_layers[:5]:
                lines.append(f"    {w['name']:<52} {w['out_snr_db']:6.1f} dB")
        lines.append(f"  duree            {self.seconds:.1f} s")
        return "\n".join(lines)


def _h(n: float) -> str:
    for unite in ("o", "Kio", "Mio", "Gio", "Tio"):
        if abs(n) < 1024 or unite == "Tio":
            return f"{int(n)} o" if unite == "o" else f"{n:.1f} {unite}"
        n /= 1024
    return f"{n:.1f} Tio"


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


# Échelles de promotion. Un tenseur qui se quantifie mal monte d'un barreau
# plutôt que d'entraîner tout le modèle vers un format plus large : dépenser
# 8 bits sur les quelques pour cent de tenseurs qui en ont besoin coûte une
# fraction de bit par poids sur l'ensemble.
# q3n promeut vers int8 comme nvfp4 : sans cette entrée, la reconversion
# « à filet égal » du 8/09 (snr_floor 25) a rendu un manifeste STRICTEMENT
# identique au sans-filet — zéro promotion, en silence, options.snr_floor
# pourtant à 25. Un filet qui ignore un format doit le dire, pas se taire.
PROMOTE = {"int4_awq": "int8", "nvfp4": "int8", "q3n": "int8", "int8": "bf16"}

# Largeur nominale d'un format, bits par poids échelles comprises. Sert à
# chiffrer le prix d'une promotion avant de la calculer : la mesurer d'abord
# reviendrait à quantifier deux fois tous les tenseurs du modèle.
#
# Ce chiffre-ci est celui qui DECIDE : il alimente `bpw_cible`, donc le budget
# mémoire, donc l'exil d'une couche — qui coûte 61 à 70 % du débit. Il était
# écrit en dur dans un dictionnaire, TROISIEME copie de la même grandeur après
# `FormatSpec.bpw` et `FormatSpec.bits_per_weight()`, et il en divergeait :
# int8 8,25 contre 8,1875 réels à groupe 128, int4_awq 4,25 contre 4,15625.
# Surtout, la valeur en dur ignorait `--group-size` : à groupe 32 l'int8 réel
# vaut 8,75 et la constante 8,25 SOUS-estimait le budget de 5,7 % (401 Mio sur
# 6,74e9 poids) — le sens dangereux, celui qui fait croire qu'un modèle tient.
# Une seule source désormais : quant/formats.py, à la taille de groupe réelle.
def bpw_nominal(fmt: str, group_size: int = 128) -> float:
    try:
        return formats.bits_per_weight(fmt, group_size=group_size)
    except KeyError:
        return 16.0


def cout_promotion_mib(numel: int, base: str, cible: str,
                       group_size: int = 128) -> float:
    """Mébioctets qu'ajoute le passage de ``base`` à ``cible``."""
    ecart = bpw_nominal(cible, group_size) - bpw_nominal(base, group_size)
    return numel * ecart / 8 / 1048576

SENSITIVE_SUFFIXES = (
    "layernorm.weight", "norm.weight", "_norm.weight",
    "conv1d.weight", "a_log.weight", "dt_bias.weight",
    "conv1d_q.weight", "conv1d_k.weight", "conv1d_v.weight",
    "conv.conv.weight",           # noyau court LFM2 [d, L]
    "mamba.conv1d.weight", "mamba.conv1d.bias", "mamba.A.weight",
    "mamba.D.weight", "mamba.dt_bias.weight", "mamba.norm.weight",
    ".a.weight",                  # -exp(A_log) de KDA (kimi-linear)
    "e_score_correction_bias",
    "layernorm.bias", "norm.bias",
    "router_scale.weight", "per_expert_scale.weight",
    "k_b_proj.weight", "v_b_proj.weight",   # absorptions MLA [H, r, d] : 3D, petits
    "mlp.gate.weight",            # routeur MoE : minuscule et décisif
    "shared_expert_gate.weight",  # porte de l'expert partagé : 1 ligne
    "embed_tokens.weight",
)


# Tour visuelle (contrat poste7-go-multimodal-organisation-20-09 § 2, pièce (a)) :
# Gemma 4 range l'encodeur sous model.vision_tower.* et son projecteur sous
# model.embed_vision.*, Qwen3-VL sous model.visual.*. Ces tenseurs sont GARDÉS
# EN BF16 SOUS LEUR NOM SOURCE, jamais quantifiés (même règle que les
# projections MLA, REGLES § 9) : le plan de placement ne connaît que les
# couches texte, et `model.vision_tower.*.layers.N.*` porte un indice de
# couche qui n'est pas le sien. Un alias sans ces tenseurs = texte seul.
VISION_PREFIXES = ("model.vision_tower.", "model.embed_vision.", "model.visual.",
                   "model.vision_embedder.")     # gemma4_unified (12B) : embedder de patches, sans SigLIP (20/09)
# Audio (gemma4_unified : model.embed_audio.*, model.audio_tower.*) : NON servi — écarté, mais nommé
# (journal + manifeste audio: "non servi"), jamais en silence.
AUDIO_PREFIXES = ("model.embed_audio.", "model.audio_tower.")
_AUDIO_ECARTES: list[str] = []


def est_tenseur_audio(name: str) -> bool:
    return name.startswith(AUDIO_PREFIXES)


def est_tenseur_vision(name: str) -> bool:
    return name.startswith(VISION_PREFIXES)


# Qwen3-VL (contrat poste7-go-qwen3vl-parallele-20-09 § 2) : les fusions deepstack
# (model.visual.deepstack_merger_list.N.*, une par indice de
# vision_config.deepstack_visual_indexes) sont des tenseurs de la tour : gardés
# bf16 sous leur nom source par VISION_PREFIXES, et NOMMÉS au manifeste
# (deepstack: oui + la liste d'indices de la config), comme M-RoPE
# (text rope_scaling.mrope_section / mrope_interleaved), pour que le chargeur
# ne devine rien.
DEEPSTACK_PREFIXES = ("model.visual.deepstack_merger_list.",)


def est_tenseur_deepstack(name: str) -> bool:
    return name.startswith(DEEPSTACK_PREFIXES)


def _manifeste_multimodal(manifest: dict, spec) -> None:
    """deepstack et M-RoPE au manifeste, lus de la config source (spec.raw,
    text_config dépliée) et des tenseurs GARDÉS : jamais devinés. Config et
    tenseurs doivent se répondre — une fusion deepstack sans indice, ou un
    indice sans fusion, est un converti muet servi faux : refus nommé."""
    raw = getattr(spec, "raw", None) or {}
    vc = raw.get("vision_config") or {}
    indices_tenseurs = sorted({int(k.split(".")[3]) for k in manifest["tensors"]
                               if est_tenseur_deepstack(k)})
    indices_config = vc.get("deepstack_visual_indexes")
    manifest["deepstack"] = "oui" if indices_tenseurs else "non"
    if indices_config is not None:
        manifest["deepstack_visual_indexes"] = list(indices_config)
    if bool(indices_tenseurs) != bool(indices_config) or (
            indices_tenseurs and indices_tenseurs != list(range(len(indices_config)))):
        raise ValueError(
            f"deepstack incohérent : fusions gardées {indices_tenseurs} contre "
            f"vision_config.deepstack_visual_indexes={indices_config} (une fusion "
            f"model.visual.deepstack_merger_list.N par indice de la config)")
    rs = raw.get("rope_scaling") or raw.get("rope_parameters") or {}
    if isinstance(rs, dict) and rs.get("mrope_section") is not None:
        manifest["mrope_section"] = [int(x) for x in rs["mrope_section"]]
        manifest["mrope_interleaved"] = bool(rs.get("mrope_interleaved", False))


def _est_projection_attn(name: str) -> bool:
    """poste7-p2-qkvo-int8-canal-18-09, generalise le 18/09 pour GLM (MLA) :
    q/k/v/o couvre l'attention GQA de Coder, mais GLM nomme ses projections
    q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj -- aucun suffixe
    fixe ne les couvre tous. Toute pondération sous `.self_attn.` qui n'est
    PAS une norme (q_a_layernorm, kv_a_layernorm) est une projection ;
    k_b_proj/v_b_proj y passent aussi mais restent bf16 (SENSITIVE_SUFFIXES,
    absorptions MLA 3D) donc jamais gênés par le filtre fmt=="int8" en aval."""
    return (".self_attn." in name and name.endswith(".weight")
           and "norm" not in name)


def _est_projection_gdn(name: str) -> bool:
    """153 (chef 24/09) : les cinq projections de poids du GatedDeltaNet,
    noms acvram post-renommage (_QWEN35_RENOMMAGE) -- qkv/gate/alpha/beta/out.
    Exclus explicitement : conv1d.weight (conv depthwise, pas une projection
    dense), a_log.weight/dt_bias.weight (vecteurs [num_v_heads], pas des
    matrices), et toute norme."""
    return (".linear_attn." in name and name.endswith(".weight")
           and "norm" not in name
           and not name.endswith(("conv1d.weight", "a_log.weight", "dt_bias.weight")))


def _bilan_attn_int8(opts, tensors: dict) -> dict:
    """Règle 6 (23/09, pièce 55) : `--attn-qkvo-int8-canal` ne rend « par canal »
    qu'une projection d'attention DÉJÀ promue int8 (`snr_floor`, classes de
    promotion, format imposé) — l. 1758 et 1916 : `and fmt == "int8"`. Sur
    gemma-4-31B (o_proj 21,5 dB, snr_floor 0) aucune ne l'était et le manifeste
    disait quand même `attn_int8 = canal` : une étiquette prise pour une preuve,
    une conversion de 405 s jugée sur ce qu'elle n'avait pas fait. On compte, on
    étiquette selon les faits, et l'absence d'effet est écrite dans le manifeste
    ET imprimée — jamais tue."""
    attn = {k: v for k, v in tensors.items() if _est_projection_attn(k)}
    n_int8 = sum(1 for v in attn.values() if v.get("format") == "int8")
    n_quant = sum(1 for v in attn.values() if v.get("format") not in (None, "bf16", "fp16", "fp32"))
    if not getattr(opts, "attn_qkvo_int8_canal", False):
        return {"attn_int8": "groupe"}
    if n_int8 == 0:
        msg = (f"--attn-qkvo-int8-canal SANS EFFET : 0 projection d'attention en int8 sur {n_quant} "
               f"quantifiée(s) — l'int8 par canal ne s'applique qu'à une projection promue int8 ; "
               f"relancer avec --snr-floor > 0, ou --promotion-classes q_proj,k_proj,v_proj,o_proj "
               f"--max-promotions 1.0 --snr-floor 99 pour forcer l'attention en int8")
        print(f"[acvram] AVERTISSEMENT : {msg}", flush=True)
        return {"attn_int8": "groupe", "attn_int8_canal_demande": True, "avertissements": [msg]}
    return {"attn_int8": "canal", "attn_int8_canal_tenseurs": n_int8}


class TensorRouter:
    """Décide du format de stockage de chaque tenseur, à partir du plan de placement."""

    def __init__(self, spec: ModelSpec, plan: Plan, opts: ConversionOptions) -> None:
        self.spec = spec
        self.plan = plan
        self.opts = opts
        self._layer_fmt = {lp.index: lp.fmt for lp in plan.layers}

    def layer_index(self, name: str) -> Optional[int]:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    def format_for(self, name: str) -> str:
        """Le format d'un tenseur est celui de l'appareil où sa couche s'exécute."""
        if est_tenseur_vision(name):
            # jamais quantifiée, jamais fp16 même sous --format fp16 : le
            # contrat multimodal sert la tour en bf16 eager (VISION_PREFIXES)
            return "bf16"
        if name.endswith("e_score_correction_bias"):
            # fp32 INCONDITIONNEL, pas seulement "16 bits protégés" : ce biais
            # porte une grande valeur commune (~9 pour GLM-4.7-Flash) et une
            # correction fine par expert de l'ordre de 0,01-0,03 qui TRANCHE
            # le top-k entre experts quasi ex-aequo. Le pas bf16 a cette
            # magnitude (~0,03) est du meme ordre que la correction elle-meme :
            # arrondi en bf16, deux experts voisins peuvent echanger leur rang.
            # Mesure : equivalence CPU GLM-4.7-Flash, positions 1 et 3 sur 16
            # jetons synthetiques divergent encore meme apres avoir force TOUT
            # le reste (routeur, hidden states) en fp32 des deux cotes — seul
            # ce biais, quantifie en bf16 a la conversion, expliquait le reste
            # (revue/verdict-equivalence-glm-14-09.md).
            return "fp32"
        if self.spec.model_type == "nemotron_h" and (
                name.endswith(("mamba.in_proj.weight", "mamba.out_proj.weight"))
                or ".self_attn." in name):
            # poste7-hybrides-etape1-close-gemm-dense-17-09 : le checkpoint
            # officiel NVFP4 (models_vllm/.../hf_quant_config.json,
            # quant_algo=MIXED_PRECISION) laisse ces tenseurs hors de
            # `quantized_layers` (self_attn, les 6 couches full_attention) ou
            # les passe en FP8 (mamba.in_proj/out_proj) -- jamais en NVFP4.
            # acvram ne porte pas de format FP8 autonome ; bf16 est la
            # meilleure approximation disponible (plus fin que FP8, jamais
            # plus grossier). Notre conversion précédente quantifiait les
            # deux en NVFP4 comme le reste du modèle : PPL 1,0632 contre
            # 0,987 pour vLLM officiel.
            return "bf16"
        if self.opts.keep_sensitive_16bit and name.endswith(SENSITIVE_SUFFIXES):
            # LE FORMAT 16 BITS DEMANDE, PAS bf16 EN DUR. Cette regle protege
            # les tenseurs sensibles de la quantification : pour un modele en
            # nvfp4 ou int4, rendre bf16 est une PROMOTION et l'intention est
            # respectee. Mais quand `--format fp16` est demande, la meme ligne
            # DEGRADE — sept bits de mantisse au lieu de dix — et le fait en
            # silence sur les plongements, c'est-a-dire sur l'ENTREE du modele,
            # dont la troncature se propage dans toutes les couches.
            #
            # Constate le 10/09 en convertissant Llama-2-7b avec --format fp16
            # pour servir d'etalon : 257 tenseurs en fp16 et 66 en bf16, dont
            # model.embed_tokens.weight. Un etalon dont l'entree est tronquee
            # n'est plus l'original, et l'ecart serait allé au compte du moteur.
            return ("fp16" if self.opts.format_impose == "fp16" else "bf16")
        if ((".linear_attn." in name or ".self_attn." in name)
                and self._fmt_brut(name) == "q3n"):
            # Plancher int8 pour TOUTE l'attention et la tête de sortie quand
            # la cible est q3n. La conversion NVFP4 saine de Coder-Next les
            # avait toutes en int8 ; le plancher restreint à la seule GDN
            # (v0.4.93) n'a pas suffi : trace du 8/09, cosinus contre le
            # modèle sain à 0,99 sur les couches GDN puis 0,57 dès la
            # première attention pleine (couche 3, q/k/v/o en q3n) et bruit
            # ensuite. À 3,25 bits, l'erreur sur q/k traverse le softmax.
            return "int8"
        if name.endswith(".bias"):
            return "bf16"
        if name.startswith("lm_head"):
            fmt = (self.opts.lm_head_format
                   or self._layer_fmt.get(self.spec.num_layers - 1, "int4_awq"))
            # Même plancher que l'attention : la tête projette sur 151 936
            # classes, à 3,25 bits ses logits ne classent plus.
            return "int8" if fmt == "q3n" else fmt
        idx = self.layer_index(name)
        # La tête de prédiction multi-jetons est un bloc de transformeur de
        # plus : elle suit le format de la dernière couche, pas le bf16 des
        # tenseurs hors couches. Deux conventions HF pour la nommer : un
        # préfixe `.mtp.`/`model.mtp.` dédié, OU — GLM-4.7-Flash — le même
        # schéma `model.layers.N.*` que les couches réelles, avec N EGAL au
        # nombre de couches réelles (une de plus que le dernier indice
        # valide). Sans le second cas, `layer_index` rend cet indice hors
        # plan tel quel et le repli silencieux plus bas l'aurait quantifié
        # en int4_awq — le format decouvert le 15/09 sur
        # model.layers.47.mlp.experts.*.gate_proj.weight (47 couches
        # reelles, indices 0-46), qui n'a RIEN a voir avec une promotion
        # SNR par expert (verifie : aucun `promoted_from` sur ces
        # tenseurs, le manifeste les quantifiait directement en int4_awq).
        if ".mtp." in name or name.startswith("model.mtp.") or idx == self.spec.num_layers:
            return self._layer_fmt.get(self.spec.num_layers - 1, "int4_awq")
        if idx is None:
            return "bf16"
        if idx not in self._layer_fmt:
            # PLUS JAMAIS de repli silencieux : un tenseur hors plan doit
            # arrêter la conversion en nommant la couche, pas se faire
            # quantifier dans un format que personne n'a choisi pour lui.
            raise ValueError(
                f"« {name} » (couche {idx}) n'a pas de format planifié -- "
                f"le plan de placement ne connaît que {sorted(self._layer_fmt)} "
                f"couches ; refus plutôt qu'un repli int4_awq silencieux")
        return self._layer_fmt[idx]

    def _fmt_brut(self, name: str) -> str:
        idx = self.layer_index(name)
        return self._layer_fmt.get(idx, "int4_awq") if idx is not None else "bf16"

    def wants_hadamard(self, name: str, fmt: str) -> bool:
        mode = self.opts.use_hadamard
        if mode == "never":
            return False
        if mode == "always":
            return True
        # auto : une rotation mérite sa place quand le groupe d'échelle est
        # large. Les groupes de 128 de l'INT4 ne peuvent pas absorber un canal
        # aberrant isolé, si bien qu'étaler les valeurs extrêmes aide de façon
        # mesurable. Les blocs de 16 du NVFP4 portent déjà leur propre échelle
        # et la rotation n'apporte que peu, au prix d'une transformée en
        # n log n sur chaque activation.
        return fmt == "int4_awq" and not name.endswith(SENSITIVE_SUFFIXES)


def _precalculer_alpha_commun_gate_up(
    model_path: str, spec, router: "TensorRouter",
    stats: Optional[dict[str, ActStats]], opts: ConversionOptions,
    qdev: torch.device) -> dict[str, torch.Tensor]:
    """Pre-passe A7 : un alpha AWQ COMMUN par paire gate_proj/up_proj admissible
    (meme format des deux cotes, ni bf16/fp16, hors experts MoE — leur pile
    groupee obeit a une autre regle, celle qui a fait echouer A6, voir
    `revue/verdict-a6-int8-snrfloor0-14-09.md`).

    Relit gate_proj et up_proj une SECONDE fois depuis le disque : prix
    accepte pour ne pas toucher a la boucle de conversion principale, qui
    reste un passage tenseur par tenseur. Rend un alpha `None` (pas de
    scaler) comme un alpha : les deux valent une fusion, un alpha degenere a
    l'identite n'ayant rien a partager.
    """
    # Deux files, pas une : l ordre d arrivée gate → up n est PAS garanti.
    # Sur Qwen3-Coder-30B (22/09), une paire sur 6 144 — couche 28, expert 46 —
    # arrivait dans l ordre inverse ; elle sortait alors de l alpha commun sans
    # rien dire, gardait deux échelles distinctes (écart relatif 0,20) et
    # faisait refuser la disposition Marlin pour TOUTE sa couche
    # (`experts_layout=marlin(47/48)`). Une file par côté, appariement dès que
    # les deux sont là : le résultat ne dépend plus de l ordre du checkpoint.
    attente_gate: dict[str, torch.Tensor] = {}
    attente_up: dict[str, torch.Tensor] = {}
    resultat: dict[str, torch.Tensor] = {}
    experts_vus = 0
    for name, tensor in _adapt_hf(_iter_checkpoint(model_path), spec):
        est_expert = ".mlp.experts." in name
        # Les deux portes sont indépendantes : `--alpha-commun-experts` seul
        # ne touche pas les paires denses, et réciproquement.
        if tensor.dim() != 2 or (not opts.alpha_commun_experts if est_expert
                                 else not opts.alpha_commun_gate_up):
            continue
        if name.endswith("gate_proj.weight"):
            cle = name[: -len("gate_proj.weight")]
            autre = attente_up.pop(cle, None)
            if autre is None:
                attente_gate[cle] = tensor
                continue
            gate_tensor, tensor = tensor, autre
        elif name.endswith("up_proj.weight"):
            cle = name[: -len("up_proj.weight")]
            gate_tensor = attente_gate.pop(cle, None)
            if gate_tensor is None:
                attente_up[cle] = tensor
                continue
        else:
            continue
        gate_name, up_name = f"{cle}gate_proj.weight", f"{cle}up_proj.weight"
        fmt = router.format_for(gate_name)
        if fmt != router.format_for(up_name) or fmt in ("bf16", "fp16"):
            continue
        st = stats.get(gate_name) if stats else None
        gate_t = gate_tensor.to(qdev, dtype=torch.float32)
        up_t = tensor.to(qdev, dtype=torch.float32)
        st_dev = None if st is None else ActStats(
            st.mean_abs.to(qdev), None, st.n_samples)
        # Le même régime que la recherche par tenseur qui consommera l'alpha
        # (boucle principale : `quantize_activation_nvfp4=est_expert and
        # fmt == "nvfp4"`) — un alpha cherché en W4A16 puis servi en W4A4
        # n'est pas celui qu'on croit.
        a4 = ".mlp.experts." in gate_name and fmt == "nvfp4" and not opts.hadamard_experts
        try:
            scale, _ = alpha_commun_gate_up(
                [gate_t, up_t], st_dev, fmt, group_size=opts.group_size,
                use_hadamard=router.wants_hadamard(gate_name, fmt),
                n_grid=opts.n_grid, quantize_activation_nvfp4=a4)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            scale, _ = alpha_commun_gate_up(
                [gate_tensor.to(torch.float32), tensor.to(torch.float32)],
                st, fmt, group_size=opts.group_size,
                use_hadamard=router.wants_hadamard(gate_name, fmt),
                n_grid=opts.n_grid, quantize_activation_nvfp4=a4)
        if ".mlp.experts." in gate_name:
            experts_vus += 1
        if scale is not None:
            # DEUX tenseurs, pas deux références au même : sur processeur
            # `.cpu()` rend l'objet tel quel, et safetensors refuse d'écrire
            # deux clés qui partagent leur mémoire (rencontré à sec le 22/09
            # sur le MoE jouet ; invisible sur carte, où `.cpu()` copie).
            resultat[gate_name] = scale.detach().cpu().clone()
            resultat[up_name] = scale.detach().cpu().clone()
        elif ".mlp.experts." in gate_name:
            # Alpha effondré à l'identité : la boucle principale écrira une
            # échelle d'unité explicite pour les experts (l. ~1727), la même
            # des deux côtés — la table reste fusionnable.
            resultat[gate_name] = torch.ones(gate_tensor.shape[1], dtype=torch.float32)
            resultat[up_name] = torch.ones(gate_tensor.shape[1], dtype=torch.float32)
    resultat["__experts_paires__"] = experts_vus       # compte, retiré par l'appelant
    return resultat



def _precalculer_alpha_commun_qkv(
    model_path: str, spec, router: "TensorRouter",
    stats: Optional[dict[str, ActStats]], opts: ConversionOptions,
    qdev: torch.device) -> dict[str, torch.Tensor]:
    """Pièce 100 B : un alpha AWQ COMMUN par triplet q_proj/k_proj/v_proj
    (même format des trois côtés, ni bf16/fp16). Même mécanique que
    `_precalculer_alpha_commun_gate_up` : une file par projection, appariement
    dès que les trois sont là (l'ordre du checkpoint n'est pas garanti), une
    seconde lecture du disque acceptée. Rend, par nom de tenseur, l'échelle
    forcée ; rien pour un triplet dont l'alpha s'effondre à l'identité (les
    trois scalers identité se fusionnent déjà)."""
    attente: dict[str, dict[str, torch.Tensor]] = {}
    resultat: dict[str, torch.Tensor] = {}
    for name, tensor in _adapt_hf(_iter_checkpoint(model_path), spec):
        if tensor.dim() != 2:
            continue
        m = re.match(r"(.*\.self_attn\.)([qkv])_proj\.weight$", name)
        if not m:
            continue
        cle, proj = m.group(1), m.group(2)
        attente.setdefault(cle, {})[proj] = tensor
        if len(attente[cle]) < 3:
            continue
        trio = attente.pop(cle)
        noms = [f"{cle}{p}_proj.weight" for p in "qkv"]
        fmts = {router.format_for(n) for n in noms}
        if len(fmts) != 1 or fmts & {"bf16", "fp16"}:
            continue
        fmt = fmts.pop()
        st = next((stats.get(n) for n in noms if stats and stats.get(n) is not None), None)
        ws = [trio[p].to(qdev, dtype=torch.float32) for p in "qkv"]
        st_dev = None if st is None else ActStats(st.mean_abs.to(qdev), None, st.n_samples)
        try:
            scale, _ = alpha_commun_gate_up(
                ws, st_dev, fmt, group_size=opts.group_size,
                use_hadamard=router.wants_hadamard(noms[0], fmt), n_grid=opts.n_grid)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            scale, _ = alpha_commun_gate_up(
                [trio[p].to(torch.float32) for p in "qkv"], st, fmt,
                group_size=opts.group_size,
                use_hadamard=router.wants_hadamard(noms[0], fmt), n_grid=opts.n_grid)
        if scale is not None:
            for n in noms:
                resultat[n] = scale.detach().cpu().clone()
    return resultat


def _quantize_on(dev: torch.device, tensor: torch.Tensor, fmt: str,
                 st: Optional[ActStats], **kw):
    """Quantifie sur ``dev``, en retombant sur le processeur si la VRAM manque.

    La recherche AWQ garde plusieurs copies en float32 du tenseur ; sur un
    ``lm_head`` de 150 000 lignes cela depasse ce que laisse une carte deja
    occupee. Un tenseur trop gros n'est pas une erreur : il se quantifie plus
    lentement, ailleurs.
    """
    if dev.type != "cpu":
        try:
            st_dev = None if st is None else ActStats(
                st.mean_abs.to(dev),
                None if st.max_abs is None else st.max_abs.to(dev),
                st.n_samples)
            return quantize_with_calibration(
                tensor.to(torch.float32).to(dev), fmt, st_dev, **kw)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    return quantize_with_calibration(tensor.to(torch.float32), fmt, st, **kw)


_RE_EXPERT = re.compile(r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$")


def _experts_par_couche(noms: list) -> dict:
    """["model.layers.3.mlp.experts.7.gate_proj.weight", …] → {"3": {"gate_proj": [7, …]}}."""
    out: dict = {}
    for n in noms:
        m = _RE_EXPERT.match(n)
        if m:
            out.setdefault(m.group(1).rsplit(".", 1)[1], {}).setdefault(m.group(3), []).append(int(m.group(2)))
    return out


def _stats_repli_mediane(stats: dict, name: str, min_echantillons: int, cache: dict):
    """Pièce 25 (b) : pour un expert sans statistique, la médiane PAR CANAL
    des `mean_abs` des experts calibrés (n_samples >= min) de la même couche
    et projection — le profil d activation typique de la couche, que la
    recherche AWQ traite ensuite comme n importe quelle statistique. None
    s il y a moins de 8 experts calibrés dans la couche (une médiane de
    trois n est pas un profil). Mis en cache par (couche, projection)."""
    from .calibrate import ActStats
    m = _RE_EXPERT.match(name)
    if not m:
        return None
    cle = (m.group(1), m.group(3))
    if cle in cache:
        return cache[cle]
    prefixe = f"{m.group(1)}.mlp.experts."
    suffixe = f".{m.group(3)}.weight"
    voisins = [st for n, st in stats.items()
               if n.startswith(prefixe) and n.endswith(suffixe) and st is not None
               and st.n_samples >= min_echantillons]
    if len(voisins) < 8:
        cache[cle] = None
        return None
    pile = torch.stack([st.mean_abs.to(torch.float32) for st in voisins])
    repli = ActStats(mean_abs=pile.median(dim=0).values, max_abs=None, n_samples=min_echantillons)
    cache[cle] = repli
    return repli


def _journal_tenseurs(opts: ConversionOptions, qdev: torch.device):
    """Rend un écrivain `(nom, forme, rang)` appelé au DÉBUT de chaque tenseur —
    il journalise le PRÉCÉDENT avec sa durée (toutes les sorties de boucle
    confondues) — ou None. Le dernier tenseur se journalise par `(None, None,
    rang)` après la boucle. Une ligne = heure, rang, nom, forme, secondes ;
    un `tail -f` du journal suffit à savoir où en est une conversion."""
    actif = opts.journal_tenseurs
    if actif is None:
        actif = qdev.type == "cpu" or os.environ.get("ACVRAM_JOURNAL_TENSEURS") == "1"
    if not actif:
        return None
    etat = {"nom": None, "forme": None, "rang": 0, "t": time.perf_counter()}

    def ecrire(nom, forme, rang):
        maintenant = time.perf_counter()
        if etat["nom"] is not None:
            print(f"[convert] {time.strftime('%H:%M:%S')} #{etat['rang']:<6d} {etat['nom']} {list(etat['forme'])} "
                  f"{maintenant - etat['t']:.2f} s", file=sys.stderr, flush=True)
        etat.update(nom=nom, forme=forme, rang=rang, t=maintenant)
    return ecrire


def _resolve_quant_device(choice: str) -> torch.device:
    """Ou quantifier. ``auto`` prend un GPU s'il y en a un, sinon le processeur."""
    if choice not in ("auto", ""):
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# checkpoint reading
# --------------------------------------------------------------------------


# Familles HF dont les couches à récurrence portent d'autres noms que notre
# manifeste (celui-ci a été fixé sur le GGUF de Qwen3.5) : renommage, normes
# zéro-centrées remises en (1 + w), conv1d aplatie, tours visuelle et MTP
# ignorées. Les conversions GGUF n'y passent pas (déjà dans nos conventions).
_QWEN35_HF = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
              "qwen3_next")
_QWEN35_RENOMMAGE = {
    "linear_attn.in_proj_qkv": "linear_attn.qkv",
    "linear_attn.in_proj_z": "linear_attn.gate",
    "linear_attn.in_proj_a": "linear_attn.alpha",
    "linear_attn.in_proj_b": "linear_attn.beta",
    "linear_attn.out_proj": "linear_attn.out",
    "linear_attn.A_log": "linear_attn.a_log.weight",
    "linear_attn.dt_bias": "linear_attn.dt_bias.weight",
}
_NORMES_ZERO_CENTREES = ("input_layernorm.weight", "post_attention_layernorm.weight",
                         "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                         "model.norm.weight")


def _adapt_muse(source: Iterator[tuple[str, torch.Tensor]], spec
                ) -> Iterator[tuple[str, torch.Tensor]]:
    """Muse-Glimmer : normes centrées (+1), plongement normalisé par ligne,
    gate_proj fusionné par tête dans q_proj ([q_h | porte_h]), q_norm/k_norm
    synthétisés (sans poids, facteur qk_scale_factor replié dans q_norm)."""
    nh, hd = spec.num_attention_heads, spec.head_dim
    qk = float(spec.raw.get("qk_scale_factor") or 3.87)
    eps = float(spec.rms_norm_eps)
    en_attente: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in source:
        name = name.replace("model.language_model.", "model.")
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                          "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight")):
            yield name, (t.to(torch.float32) + 1.0).to(t.dtype)
            continue
        if name == "model.embed_tokens.weight":
            e = t.to(torch.float32)
            yield name, (e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + eps)).to(t.dtype)
            continue
        if name.endswith(("self_attn.q_proj.weight", "self_attn.gate_proj.weight")):
            pref = name.rsplit("self_attn.", 1)[0] + "self_attn."
            lot = en_attente.setdefault(pref, {})
            lot["q" if name.endswith("q_proj.weight") else "g"] = t
            if len(lot) == 2:
                q, g = lot.pop("q"), lot.pop("g"); del en_attente[pref]
                fused = torch.cat((q.view(nh, hd, -1), g.view(nh, hd, -1)), dim=1)
                yield pref + "q_proj.weight", fused.reshape(nh * 2 * hd, -1).contiguous()
                yield pref + "q_norm.weight", torch.full((hd,), qk, dtype=t.dtype)
                yield pref + "k_norm.weight", torch.ones(hd, dtype=t.dtype)
            continue
        yield name, t
    assert not en_attente, f"q_proj/gate_proj dépareillés : {list(en_attente)}"


_NEMOTRON_H_MAMBA_HEADS = ("in_proj", "out_proj", "conv1d", "A_log", "D", "dt_bias", "norm")


def _nemotron_h_rename(name: str, n_layers: int) -> Optional[str]:
    """Nom acvram (`model.layers.N.*`) pour un tenseur brut nemotron_h
    (`backbone.layers.N.mixer.*` -> `model.layers.N.*`), ou ``None`` si le
    tenseur est hors plan (MTP, couche au-delà de ``n_layers``).

    Factorisé pour être PARTAGÉ par `_adapt_hf` (flux principal) et
    `collect.py` (calibration AWQ) -- poste7-hybrides-etape1-close-gemm-dense-
    17-09 : la calibration cherchait `model.embed_tokens.weight` directement
    sur le point de contrôle brut (nommé `backbone.embeddings.weight`),
    échouait avec un message masquant le repli sur l'arrondi au plus proche.
    Une seule table de vérité, pas une deuxième copie qui aurait pu diverger
    (VARIABLES vs HORS_REGIME sur kv_lm4.py, même session, 17/09)."""
    if name.startswith("backbone.layers."):
        _, _, idx, rest = name.split(".", 3)
        if int(idx) >= n_layers:
            return None
        pre = f"model.layers.{idx}."
        if rest == "norm.weight":
            return pre + "input_layernorm.weight"
        if rest.startswith("mixer."):
            sub = rest[len("mixer."):]
            tete = sub.split(".")[0]
            if tete in _NEMOTRON_H_MAMBA_HEADS:
                if tete == "A_log":
                    return pre + "mamba.A.weight"
                if tete in ("D", "dt_bias"):
                    return pre + f"mamba.{tete}.weight"
                return pre + "mamba." + sub
            if tete in ("q_proj", "k_proj", "v_proj", "o_proj"):
                return pre + "self_attn." + sub
            return pre + "mlp." + sub.replace("shared_experts.", "shared_expert.")
        return pre + rest
    if name == "backbone.embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "backbone.norm_f.weight":
        return "model.norm.weight"
    if name.startswith("mtp."):
        return None
    return name


def _nemotron_h_valeur(dst: str, t: torch.Tensor) -> torch.Tensor:
    """Transforme la VALEUR d'un tenseur nemotron_h déjà renommé par
    `_nemotron_h_rename` : A = −exp(A_log) (convention GGUF ssm_a), conv1d
    [d, 1, L] -> [d, L], D/dt_bias aplatis à 1D. Les trois suffixes de
    destination sont uniques dans tout le modèle -- aucune autre couche n'y
    aboutit, donc les tester sur `dst` après renommage donne le même
    résultat que les tester sur le nom brut avant."""
    if dst.endswith("mamba.A.weight"):
        return -torch.exp(t.to(torch.float32))
    if dst.endswith("mamba.conv1d.weight"):
        return t.reshape(t.shape[0], -1)
    if dst.endswith(("mamba.D.weight", "mamba.dt_bias.weight")):
        return t.reshape(-1)
    return t


def _adapt_hf(source: Iterator[tuple[str, torch.Tensor]], spec
              ) -> Iterator[tuple[str, torch.Tensor]]:
    """Enrobages multimodaux HF : le modèle de langue vit sous
    model.language_model. et va aux adaptateurs par architecture ; la tour
    visuelle (VISION_PREFIXES) passe INTACTE, nom et valeurs, sans jamais
    traverser un renommage texte. Tour audio et `visual.` nu (Qwen2-VL)
    restent écartés : alias texte seul."""
    en_attente: list[tuple[str, torch.Tensor]] = []

    def _texte_seul(src):
        for n, t in src:
            if est_tenseur_vision(n):
                en_attente.append((n, t))
                continue
            if est_tenseur_audio(n):
                _AUDIO_ECARTES.append(n)        # refus nommé « audio non servi » au manifeste et au journal
                continue
            if n.startswith("visual."):
                continue
            yield n.replace("model.language_model.", "model."), t

    for item in _adapt_texte(_texte_seul(source), spec):
        while en_attente:
            yield en_attente.pop(0)
        yield item
    while en_attente:
        yield en_attente.pop(0)


_EXPERTS_GROUPES_HUB = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")


def _scinder_experts_groupes(source: Iterator[tuple[str, torch.Tensor]]
                             ) -> Iterator[tuple[str, torch.Tensor]]:
    """Disposition hub transformers ≥ 5 des experts MoE (Qwen3-VL-MoE, vérifiée
    sur l'en-tête safetensors officiel le 21/09 : `experts.gate_up_proj`
    [E, H, 2I], `experts.down_proj` [E, I, H], sans index ni `.weight`) →
    un `nn.Linear` par expert et par projection, `experts.{e}.gate_proj.weight`
    [I, H] · `up_proj.weight` [I, H] · `down_proj.weight` [H, I], gate PUIS up
    (le module fait `linear(x, w).chunk(2)`).

    C'est la forme que TOUT le reste du convertisseur connaît (pile groupée
    `_verifier_experts_homogenes`, stats AWQ par expert, contrôle final
    `attendus`) : sans cette scission, la reconversion 30B-VL du 21/09
    (`verdict-reconversion-30b-bloquee-21-09`) quantifiait trois blobs 3D par
    couche sous un nom que ni le manifeste ni le chargeur ne comprennent, et
    le contrôle final refusait — pour la bonne raison. On aligne la source sur
    le contrat, pas le contrôle sur la dérive."""
    for name, t in source:
        if not name.endswith(_EXPERTS_GROUPES_HUB) or t.dim() != 3:
            yield name, t
            continue
        prefixe = name[: name.rindex(".")]          # …mlp.experts
        for e in range(t.shape[0]):
            for proj in _projections_du_blob(name):
                yield f"{prefixe}.{e}.{proj}.weight", _expert_depuis_blob(proj, t[e])


def _projections_du_blob(nom_blob: str) -> tuple[str, ...]:
    """Projections par expert contenues dans un blob hub (`gate_up_proj` en
    porte deux, `down_proj` une)."""
    return ("gate_proj", "up_proj") if nom_blob.endswith("gate_up_proj") else ("down_proj",)


def _expert_depuis_blob(proj: str, tranche: torch.Tensor) -> torch.Tensor:
    """Le poids `nn.Linear` d'UN expert depuis sa tranche de blob hub :
    `gate_up_proj[e]` [H, 2I] → gate = colonnes [:I], up = [I:], transposées
    en [I, H] ; `down_proj[e]` [I, H] → [H, I]. Seule définition de l'ordre
    gate/up et de la transposition, partagée par le flux principal et la
    collecte de calibration (`collect.py`) : les deux DOIVENT lire le même
    poids sous le même nom, sinon les statistiques AWQ vont au mauvais tenseur."""
    if proj == "down_proj":
        return tranche.t().contiguous()
    inter = tranche.shape[1] // 2
    tr = tranche[:, :inter] if proj == "gate_proj" else tranche[:, inter:]
    return tr.t().contiguous()


def _adapt_texte(source: Iterator[tuple[str, torch.Tensor]], spec
                 ) -> Iterator[tuple[str, torch.Tensor]]:
    mt = str(getattr(spec, "model_type", "") or spec.raw.get("model_type", ""))
    source = _scinder_experts_groupes(source)
    if mt == "muse_glimmer":
        yield from _adapt_muse(source, spec)
        return
    if mt in ("gemma4", "gemma4_text"):
        # HF/EXL3 : Gemma4RMSNorm multiplie par w tel quel (pas de 1 + w,
        # contrairement à Gemma 3) et le convertisseur llama.cpp ne décale
        # rien non plus (norm_shift = 0) : les normes passent intactes
        for name, t in source:
            if name.endswith(".layer_scalar"):        # HF : sans suffixe .weight
                name += ".weight"
            yield name, t
        return
    if mt == "nemotron_h":
        # noms HF (backbone.layers.N.mixer.*) → noms acvram ; A = −exp(A_log)
        # (convention GGUF ssm_a), conv1d [d,1,L] → [d,L]. Les GGUF passent
        # ici sans être touchés (déjà nommés). Renommage/valeur factorisés
        # dans `_nemotron_h_rename`/`_nemotron_h_valeur`, partagés avec
        # `collect.py`.
        n_layers = int(spec.num_layers)
        ignores = 0
        # EXL3 rembourre les deux dimensions à un multiple de 128 (in_proj
        # 10304 → 10368, experts 1856 → 1920) : on rogne aux tailles du modèle
        inner = int(spec.mamba_num_heads) * int(spec.mamba_head_dim)
        n_in_proj = 2 * inner + 2 * int(spec.mamba_n_groups) * int(spec.mamba_state_size) \
            + int(spec.mamba_num_heads)
        moe_i = int(spec.moe_intermediate_size or spec.intermediate_size or 0)
        shared_i = int(spec.shared_expert_intermediate_size or 0)
        dense_i = int(spec.intermediate_size or 0)

        def rogner(t: torch.Tensor, rows: int = 0, cols: int = 0) -> torch.Tensor:
            if isinstance(t, NVFP4Tensor):
                raise NotImplementedError("passage direct NVFP4 : nemotron_h rogne ses tenseurs, "
                                          "non pris en charge sans requantification")
            if rows and t.shape[0] > rows:
                t = t[:rows]
            if cols and t.dim() == 2 and t.shape[1] > cols:
                t = t[:, :cols]
            return t.contiguous()

        for name, t in source:
            if name.endswith(".weight") and t.dim() == 2:
                if name.endswith("mixer.in_proj.weight"):
                    t = rogner(t, rows=n_in_proj)
                elif name.endswith("mixer.out_proj.weight"):
                    t = rogner(t, cols=inner)
                elif ".mixer.experts." in name:
                    t = rogner(t, rows=moe_i) if name.endswith("up_proj.weight") else rogner(t, cols=moe_i)
                elif ".mixer.shared_experts." in name:
                    t = rogner(t, rows=shared_i) if name.endswith("up_proj.weight") else rogner(t, cols=shared_i)
                elif name.endswith(("mixer.up_proj.weight", "mixer.down_proj.weight")):
                    t = rogner(t, rows=dense_i) if name.endswith("up_proj.weight") else rogner(t, cols=dense_i)
            dst = _nemotron_h_rename(name, n_layers)
            if dst is None:
                ignores += 1
                continue
            yield dst, _nemotron_h_valeur(dst, t)
        if ignores:
            print(f"  nemotron_h : {ignores} tenseurs MTP ignorés")
        return
    if mt in ("lfm2", "lfm2_moe"):
        # noms HF Lfm2 : operator_norm/ffn_norm, feed_forward.w1/w3/w2,
        # self_attn.out_proj, q/k_layernorm, embedding_norm, expert_bias
        renames = (("feed_forward.experts.", "mlp.experts."), ("feed_forward.gate.weight", "mlp.gate.weight"),
                   ("feed_forward.expert_bias", "mlp.gate.e_score_correction_bias"),
                   ("feed_forward.w1.", "mlp.gate_proj."), ("feed_forward.w3.", "mlp.up_proj."),
                   ("feed_forward.w2.", "mlp.down_proj."), ("operator_norm.", "input_layernorm."),
                   ("ffn_norm.", "post_attention_layernorm."), ("self_attn.out_proj.", "self_attn.o_proj."),
                   ("self_attn.q_layernorm.", "self_attn.q_norm."), ("self_attn.k_layernorm.", "self_attn.k_norm."),
                   ("model.embedding_norm.", "model.norm."))
        for name, t in source:
            for a, b in renames:
                name = name.replace(a, b)
            if ".mlp.experts." in name:
                name = name.replace(".w1.", ".gate_proj.").replace(".w3.", ".up_proj.").replace(".w2.", ".down_proj.")
            if name.endswith("conv.conv.weight") and t.dim() == 3:
                t = t.reshape(t.shape[0], -1)
            yield name, t
        return
    if getattr(spec, "est_mla", False):
        # HF : kv_b_proj [nh·(nope+v), rank] scindé en k_b [nh, rank, nope] et
        # v_b [nh, v, rank] (convention du convertisseur llama.cpp, attendue
        # par le loader) ; experts partagés sous mlp.shared_expert. Le
        # critère est `ModelSpec.est_mla` (kv_lora_rank > 0), pas une liste de
        # model_type : glm4_moe_lite (GLM-4.7-Flash) en manquait (bead
        # anticitoyen-vram-992, 14/09). nope != v (192/256 sur ce modèle,
        # jamais posé chez nous avant) : la scission ne suppose PAS nope==v,
        # testé explicitement (test_mla_detection.py).
        nh, nope, vd, rank = (int(spec.num_attention_heads), int(spec.qk_nope_head_dim),
                              int(spec.v_head_dim), int(spec.kv_lora_rank))
        for name, t in source:
            if name.endswith("self_attn.kv_b_proj.weight"):
                kv = t.reshape(nh, nope + vd, rank)
                base = name[: -len("kv_b_proj.weight")]
                yield base + "k_b_proj.weight", kv[:, :nope, :].transpose(1, 2).contiguous()
                yield base + "v_b_proj.weight", kv[:, nope:, :].contiguous()
            else:
                yield name.replace("mlp.shared_experts.", "mlp.shared_expert."), t
        return
    if mt == "ernie4_5_moe":
        nh, nkv = spec.num_attention_heads, spec.num_key_value_heads

        def _depermute(t: torch.Tensor, n_head: int) -> torch.Tensor:
            # lignes par tête [p0a, p0b, p1a, p1b, …] (RoPE entrelacé HF)
            # → [a…, b…] (moitiés, notre RotaryEmbedding)
            r, c = t.shape
            return t.reshape(n_head, r // n_head // 2, 2, c).transpose(1, 2).reshape(r, c)

        for name, t in source:
            if name.endswith("mlp.moe_statics.e_score_correction_bias"):
                yield name.replace("moe_statics.", "gate."), t.reshape(-1)
            elif ".mlp.shared_experts." in name:
                yield name.replace("mlp.shared_experts.", "mlp.shared_expert."), t
            elif name.endswith("self_attn.q_proj.weight"):
                yield name, _depermute(t, nh)
            elif name.endswith("self_attn.k_proj.weight"):
                yield name, _depermute(t, nkv)
            else:
                yield name, t
        return
    if mt == "starcoder2":
        for name, t in source:
            yield name.replace("mlp.c_fc.", "mlp.up_proj.").replace("mlp.c_proj.", "mlp.down_proj."), t
        return
    if mt not in _QWEN35_HF or spec.raw.get("gdn_a_log_negexp"):
        yield from source
        return
    for name, t in source:
        name = name.replace("model.language_model.", "model.")
        if name.startswith(("model.visual", "visual.")):
            continue
        for src, dst in _QWEN35_RENOMMAGE.items():
            if src in name:
                name = name.replace(src, dst)
                break
        if name.endswith("linear_attn.conv1d.weight") and t.dim() == 3:
            t = t.reshape(t.shape[0], t.shape[-1])
        if name.endswith(_NORMES_ZERO_CENTREES):
            t = t.to(torch.float32) + 1.0        # (1 + w) de la référence
        yield name, t


def _iter_checkpoint(path: str, direct_nvfp4: bool = False) -> Iterator[tuple[str, torch.Tensor]]:
    """Lit les tenseurs d'un point de contrôle safetensors ou GGUF."""
    from safetensors import safe_open

    from .gguf import GGUFFile, is_gguf
    if is_gguf(path):
        g = GGUFFile(path)
        g.check_executable()
        yield from g.iter_tensors()
        return

    from .exl3 import EXL3Checkpoint, is_exl3
    if is_exl3(path):
        yield from EXL3Checkpoint(path).iter_tensors()
        return

    from .hfquant import HFQuantCheckpoint, is_hfquant
    if is_hfquant(path):
        yield from HFQuantCheckpoint(path).iter_tensors(direct_nvfp4=direct_nvfp4)
        return

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        files = sorted(set(index["weight_map"].values()))
    else:
        files = [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"aucun fichier .safetensors dans {path}")

    for fn in files:
        full = os.path.join(path, fn)
        with safe_open(full, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                yield key, fh.get_tensor(key)


class ShardWriter:
    """Accumule des tenseurs et les vide en fragments safetensors d'environ 4 Gio."""

    def __init__(self, out_dir: str, target: int = SHARD_TARGET_BYTES) -> None:
        self.out_dir = out_dir
        self.target = target
        self._buf: dict[str, torch.Tensor] = {}
        self._bytes = 0
        self._shard = 0
        self.weight_map: dict[str, str] = {}
        os.makedirs(out_dir, exist_ok=True)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().cpu().contiguous()
        self._buf[key] = t
        self._bytes += t.numel() * t.element_size()
        if self._bytes >= self.target:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        from safetensors.torch import save_file

        name = f"acvram-{self._shard:05d}.safetensors"
        save_file(self._buf, os.path.join(self.out_dir, name))
        for key in self._buf:
            self.weight_map[key] = name
        self._buf.clear()
        self._bytes = 0
        self._shard += 1

    @property
    def total_bytes(self) -> int:
        total = 0
        for fn in os.listdir(self.out_dir):
            if fn.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(self.out_dir, fn))
        return total


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------


def octets_du_modele(model_path: str) -> int:
    """Octets des poids de la source : fichiers gguf ou safetensors du dossier."""
    import glob
    p = model_path if os.path.isdir(model_path) else os.path.dirname(model_path) or "."
    fichiers = [f for motif in ("*.gguf", "*.safetensors", "*.bin")
                for f in glob.glob(os.path.join(p, motif))]
    return sum(os.path.getsize(f) for f in fichiers)


def garde_grossissement(octets_source: int, total_params: int,
                        bpw_cible: float, autorise: bool) -> Optional[str]:
    """Refuse une conversion qui ferait grossir le modèle.

    Rend un message d'avertissement à journaliser si la conversion grossit
    mais est autorisée ; rend None si elle ne grossit pas ; lève sinon.
    Le critère n'est pas « le format est plus large » mais « le résultat sera
    plus gros que la source » — payer de la place pour des instructions
    natives est un bon échange tant qu'on en a (fiche 17 du relevé biblio) ;
    ici on chiffre précisément ce qui sera payé.
    """
    if not octets_source or not total_params:
        return None
    bpw_source = octets_source * 8 / total_params
    if bpw_cible <= bpw_source * 1.02:      # 2 % de jeu : en-têtes, échelles
        return None
    octets_cible = int(total_params * bpw_cible / 8)
    msg = (f"la conversion ferait GROSSIR le modèle : source "
           f"{octets_source / 2**30:.1f} Gio ({bpw_source:.2f} bits/poids), "
           f"cible {octets_cible / 2**30:.1f} Gio ({bpw_cible:.2f} bits/poids), "
           f"+{(octets_cible / octets_source - 1) * 100:.0f} %")
    if autorise:
        return msg + " — autorisée par --autoriser-grossissement"
    raise ValueError(
        msg + ". Refusée : le surplus serait exilé en RAM hôte et coûterait "
        "plus cher que les instructions natives ne rapportent (Coder-Next, "
        "7-8/09/2026 : facteur 15 au décodage). Issues : le format q3n, "
        "3,25 bits/poids, quantiles 3 bits (perplexité à vérifier) ; servir "
        "la source par un moteur GGUF ; ou --autoriser-grossissement en "
        "connaissance de cause.")

# Octets par poids REELS, echelles de bloc comprises : NVFP4 coute 4 bits de
# poids plus une echelle e4m3 par bloc de 16, soit 4,5 bits et non 4.
_OCTETS_PAR_POIDS = {"bf16": 2.0, "fp16": 2.0, "int8": 1.0625,
                     "nvfp4": 0.5625, "int4_awq": 0.5625, "q3n": 0.40625}


def _verifier_formats_declares(manifest: dict, weight_map: dict) -> None:
    """Le format declare doit correspondre a ce qui est REELLEMENT ecrit.

    Le 9/09/2026, `Ornith-1.5-35B` declarait `format: nvfp4` sur trois blobs
    d'experts groupes du module MTP qui n'ont jamais ete quantifies : leur cle
    physique est directe, sans `.qweight` ni `.block_scale`, et leur dtype reel
    est F16. L'ecart valait 1,158 Gio a lui seul — 5,85 % du modele — et tout
    calcul de taille fonde sur le manifeste s'en trouvait faux.

    Un format declare qui ne correspond pas au stockage est pire qu'un format
    absent : il fait croire qu'on sait. Cette garde AVERTIT sans bloquer — la
    conversion a reussi, seul le manifeste est inexact — mais elle nomme les
    tenseurs, ce qui suffit a ne plus les compter de travers.
    """
    quantifies = {"nvfp4", "int8", "int4_awq", "q3n"}
    suspects = []
    for nom, entree in manifest.get("tensors", {}).items():
        fmt = str(entree.get("format"))
        if fmt not in quantifies:
            continue
        # un tenseur quantifie s'ecrit en plusieurs morceaux ; une cle directe
        # signifie que le tenseur est passe tel quel
        morceaux = any(f"{nom}.{suffixe}" in weight_map
                       for suffixe in ("qweight", "block_scale", "scales"))
        if not morceaux and nom in weight_map:
            suspects.append((nom, fmt))
    if suspects:
        print(f"[acvram] {len(suspects)} tenseur(s) declares quantifies mais "
              f"ecrits en direct — le manifeste surestime leur compression :",
              flush=True)
        for nom, fmt in suspects[:6]:
            print(f"           {nom} (declare {fmt})", flush=True)


def _verifier_homogeneite_moe(tensors: dict, num_layers: int) -> None:
    """Refuse une couche MoE dont les experts n'ont pas TOUS le même format.

    poste7, 15/09 (`revue/poste7-glm-formats-mixtes-15-09.md` §2, précisée par
    `revue/poste7-glm-awq-pile-15-09.md`) : l'homogénéité de FORMAT d'une pile
    d'experts se décide à la CONVERSION, pas au chargement — un loader qui
    « dépile » ou ramène au format majoritaire fabriquerait un régime que
    personne n'a mesuré. Cause réelle trouvée le 15/09 sur GLM-4.7-Flash :
    format mélangé (2 944 nvfp4 + 64 int4_awq), bloc MTP mal routé
    (`TensorRouter.format_for`), corrigé le même soir.

    L'HOMOGÉNÉITÉ D'ÉCHELLE AWQ N'EST PLUS EXIGÉE ICI : poste7 (a6a7436) a
    tranché que l'échelle AWQ par expert reste légitimement hétérogène dans
    la pile — c'est au chargeur (poste4, échelle `[E,K]` portée après
    route+pack) de la consommer, pas à la conversion de l'uniformiser ou de
    la refuser. Une échelle absente reste toujours écrite comme une
    identité EXPLICITE dans le manifeste (voir la boucle principale),
    jamais comme une absence ambiguë.

    N'examine PAS la couche `num_layers` (le bloc MTP, jamais chargé par le
    moteur au décodage sans spéculation MTP — il peut porter un format
    différent sans effet sur l'inférence réelle).
    """
    import re
    from collections import Counter, defaultdict
    par_groupe: dict[tuple, dict[int, str]] = defaultdict(dict)
    for nom, entree in tensors.items():
        mo = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$", nom)
        if not mo:
            continue
        couche, expert, proj = int(mo.group(1)), int(mo.group(2)), mo.group(3)
        if couche == num_layers:
            continue
        par_groupe[(couche, proj)][expert] = str(entree.get("format"))
    for (couche, proj), formats_par_expert in par_groupe.items():
        distincts = set(formats_par_expert.values())
        if len(distincts) > 1:
            formats = Counter(formats_par_expert.values())
            exemples = sorted(formats_par_expert.items())[:3]
            raise ValueError(
                f"couche {couche}, projection {proj} : experts à formats "
                f"MÉLANGÉS ({dict(formats)}) — la pile groupée ne peut pas "
                f"se construire (exemples : {exemples}). Refus d'écrire ce "
                f"dossier plutôt que de laisser un chargeur le découvrir "
                f"plus tard.")


def _diagnostic_fusion(tensors: dict) -> dict:
    """Ce qui empeche chaque groupe q/k/v et gate/up de fusionner, et ce que
    coûterait de le lever. **Consigne, ne decide pas.**

    La fusion exige que toutes les projections d'un groupe partagent leur
    format et leur bloc de Hadamard. Le convertisseur choisit pourtant un
    format PAR TENSEUR — le plancher de SNR promeut sans regarder les voisins
    du groupe. D'ou des groupes melant nvfp4, int8 et int4_awq, refuses a la
    fusion pour cette seule raison.

    Uniformiser aurait un prix que le mot « gratuit » cachait : promouvoir un
    tenseur de nvfp4 vers int8 DOUBLE ses octets, et sur un decodage lie a la
    memoire ces octets se paient a chaque pas. Mesure le 9/09/2026 sur trois
    modeles : le marche est bon sur l'un (+43 us/pas) et mauvais sur les deux
    autres (-539 et -553 us/pas). **Le signe depend de la largeur, donc la
    decision est par groupe et jamais uniforme.**

    Ce champ n'applique aucune regle : il accumule les octets qu'une promotion
    coûterait, pour que la decision devienne possible le jour ou le gain de
    fusion sera MESURE sur un modele reel. Il manque aujourd'hui son autre
    terme — les microsecondes gagnees ne sont qu'une borne prise sur un seul
    modele, a une seule forme, sur une courbe dentelee. Figer ce terme dans le
    convertisseur le rendrait invisible et durable : un poids converti ne se
    relit pas pour savoir d'ou venait sa constante.
    """
    import collections
    import re

    groupes = collections.defaultdict(list)
    for nom in tensors:
        m = re.match(r"(.*\.layers\.\d+)\.(self_attn\.[qkv]|mlp\.(?:gate|up))_proj\.weight$",
                     nom)
        if m:
            groupes[(m.group(1), "qkv" if "attn" in m.group(2) else "gate_up")].append(nom)

    mixtes = []
    for (prefixe, genre), noms in sorted(groupes.items()):
        attendu = 3 if genre == "qkv" else 2
        if len(noms) != attendu:
            continue
        fmts = {str(tensors[n].get("format")) for n in noms}
        hads = {tensors[n].get("hadamard_block") or 0 for n in noms}
        if len(fmts) == 1 and len(hads) == 1:
            continue
        octets = 0
        if len(fmts) > 1:
            cible = max(fmts, key=lambda f: _OCTETS_PAR_POIDS.get(f, 2.0))
            for n in noms:
                cnt = 1
                for d in (tensors[n].get("shape") or []):
                    cnt *= d
                octets += cnt * (_OCTETS_PAR_POIDS.get(cible, 2.0)
                                 - _OCTETS_PAR_POIDS.get(str(tensors[n].get("format")), 2.0))
        mixtes.append({
            "groupe": f"{prefixe}.{genre}",
            "formats": sorted(fmts),
            "hadamard_blocks": sorted(hads),
            "octets_si_uniformise": int(octets),
        })

    total = sum(g["octets_si_uniformise"] for g in mixtes)
    return {
        "groupes_totaux": len(groupes),
        "groupes_non_fusionnables": len(mixtes),
        "octets_ajoutes_si_uniformise": total,
        # Le cout se calcule ; le GAIN ne l'est pas ici, et c'est voulu : il
        # demande une mesure de fusion sur un modele reel. Sans lui, aucun
        # arbitrage n'est possible et aucun n'est applique.
        "gain_microsecondes": None,
        "detail": mixtes[:64],
    }


def _convertisseur_commit() -> Optional[dict]:
    """Commit du convertisseur qui a écrit CE manifeste -- REGLES §4 : une
    mesure porte son régime dans son en-tête, pas dans un nom de fichier ou
    une mémoire. Un correctif du convertisseur (par exemple `0e4ef38`,
    calibration nemotron_h) change le régime des convertis qu'il produit ;
    sans ce champ, deux manifestes au même nom peuvent être de deux régimes
    et rien ne le distingue. `None` (pas un défaut deviné) si le paquet
    n'est pas une extraction git -- une installation figée n'a pas de
    commit à rapporter."""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # `.git` est un FICHIER (pas un dossier) dans un git WORKTREE -- `isdir`
    # aurait rendu None sur tous nos convertis produits depuis un worktree
    # (poste2, poste3, poste4, ...), la norme ici, pas l'exception.
    if not os.path.exists(os.path.join(repo, ".git")):
        return None
    try:
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        sale = bool(subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return None
    return {"commit": commit, "arbre_modifie": sale}


def _sha256_du_checkpoint(chemin: str) -> Optional[str]:
    """sha256 combiné des poids source (un hash par fichier, puis hash de la
    liste triée `nom:sha256`) -- poste7-diff-octet-a-octet-retire-17-09,
    REGLES §4 : « un converti est identifié par son sha256, jamais par ses
    options » vaut tout autant pour la SOURCE. Deux répertoires du même nom
    mais de contenu différent (source déplacée/remplacée en silence) ne se
    distinguent pas par le chemin ni par le compte d'octets seul (`_octets_
    du_checkpoint`) si les tailles coïncident par hasard.
    `None` si aucun fichier de poids trouvé — jamais une valeur devinée."""
    import hashlib
    fichiers = []
    for r, _, fs in os.walk(chemin):
        for f in fs:
            if f.endswith((".safetensors", ".bin", ".gguf")):
                fichiers.append(os.path.join(r, f))
    if not fichiers:
        return None
    par_fichier = []
    for f in sorted(fichiers):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for bloc in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloc)
        par_fichier.append(f"{os.path.basename(f)}:{h.hexdigest()}")
    return hashlib.sha256("\n".join(par_fichier).encode()).hexdigest()


def _octets_du_checkpoint(chemin: str) -> int:
    """Somme des poids du checkpoint source, pour reconnaitre une source
    renommee ou deplacee. La taille seule ne PROUVE pas l identite — deux
    modeles de meme architecture et meme format pesent pareil — mais elle
    suffit a signaler qu on n a pas la bonne."""
    total = 0
    for r, _, fs in os.walk(chemin):
        for f in fs:
            if f.endswith((".safetensors", ".bin", ".gguf")):
                try:
                    total += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
    return total


def refus_nvfp4_sans_gpu(plan: Any, out_dir: str, format_impose: Optional[str]) -> Optional[str]:
    """Garde (chef 21/09, `poste1-hypotheses-qvl-30b-21-09`) : un nom de sortie ou un format demandé qui dit
    `nvfp4` alors que le plan n a AUCUN palier GPU est refusé, nommément. Sans carte visible à la conversion,
    `memory/tiering.py:360-366` (`build_tiers`) n émet que le palier hôte et le format retombe sur `int4_awq`
    (tiering.py:397, 514, 542) : le 30B « -nvfp4-vision » du 20/09 était un int4_awq planifié processeur —
    TTFT 4,28 s, J ×4,4, P3 (3) 21,8 %. Rend le message de refus, None si tout va bien."""
    tiers = list(getattr(plan, "tiers", []) or [])
    if any(getattr(t, "kind", "") == "gpu" for t in tiers):
        return None
    dit_nvfp4 = "nvfp4" in os.path.basename(os.path.normpath(out_dir)).lower() or (format_impose == "nvfp4")
    if not dit_nvfp4:
        return None
    formats = sorted({getattr(t, "weight_format", "?") for t in tiers}) or ["aucun palier"]
    return (f"conversion refusée : « {os.path.basename(os.path.normpath(out_dir))} »"
            f"{' / --format nvfp4' if format_impose == 'nvfp4' else ''} annonce nvfp4 mais le plan n a aucun palier GPU "
            f"(paliers : {', '.join(formats)}) — sans carte visible, memory/tiering.py:360-366 build_tiers ne "
            f"planifie que l hôte et le format retombe sur int4_awq (tiering.py:397/514/542). Convertir carte "
            f"visible (outils/carte.sh env CUDA_VISIBLE_DEVICES=0 …) ou nommer la sortie par son vrai format.")


def _noter_echelle_nvfp4(manifest: dict, entry: dict, qt: Any) -> None:
    """Q1 : part des blocs ayant choisi amax/4 et part des blocs clampés (amax/6 = E4M3_MAX : candidats confondus),
    par tenseur (`entry["echelle"]`) et cumulées (`manifest["echelle_nvfp4"]`) — écrit AVANT toute mesure : si le
    scellé E est réfuté, la ligne existe déjà et nomme la cause. Contrôle indépendant : outils/part-amax4.py relit
    les codes stockés (bloc dont le plus grand code vaut 6 = amax/6, vaut 4 = amax/4)."""
    st = getattr(qt, "echelle_stats", None)
    if not st or not st.get("blocs"):
        return
    n = st["blocs"]
    entry["echelle"] = {"regle": st["echelle"], "blocs": n, "part_amax4": round(st["amax4"] / n, 4),
                        "part_clampes": round(st["clampes"] / n, 4)}
    tot = manifest.setdefault("echelle_nvfp4", {"regle": st["echelle"], "blocs": 0, "amax4": 0, "clampes": 0})
    tot["blocs"] += n; tot["amax4"] += st["amax4"]; tot["clampes"] += st["clampes"]
    tot["part_amax4"] = round(tot["amax4"] / tot["blocs"], 4)
    tot["part_clampes"] = round(tot["clampes"] / tot["blocs"], 4)


def convert_checkpoint(model_path: str, plan: Plan, opts: ConversionOptions,
                       spec: Optional[ModelSpec] = None,
                       stats: Optional[dict[str, ActStats]] = None,
                       progress: Optional[Callable[[str, int, int], None]] = None
                       ) -> ConversionReport:
    """Quantifie chaque tenseur dans le format qu'attend son appareil de destination."""
    t0 = time.time()
    from .nvfp4 import regler_echelle
    regler_echelle(opts.echelle_nvfp4)
    spec = spec or load_model_spec(model_path)
    bpw_cible = max((bpw_nominal(t.weight_format, opts.group_size)
                     for t in plan.tiers if t.kind == "gpu"), default=4.5)
    octets_src = octets_du_modele(model_path)
    n_params = getattr(spec, "total_params", 0) or 0
    bpw_cible_nominal = next((t.weight_format for t in plan.tiers
                              if t.kind == "gpu"), None)
    bascule_faite = False
    bpw_src = (octets_src * 8 / n_params) if (octets_src and n_params) else 0.0
    grossirait = bool(octets_src and n_params
                      and not opts.autoriser_grossissement
                      and bpw_cible > bpw_src * 1.02)
    if grossirait and opts.format_impose:
        # Choix explicite de l'utilisateur : on refuse, on n'arrange pas.
        raise ValueError(
            f"--format {opts.format_impose} ({bpw_cible:.2f} bits/poids) ferait "
            f"grossir une source a {bpw_src:.2f} bits/poids. Refuse plutot que "
            f"bascule en silence : relancez avec --autoriser-grossissement pour "
            f"l'obtenir vraiment, ou sans --format pour laisser la politique "
            f"choisir un format compact.")
    if grossirait:
        # Plutôt que refuser d'emblée : basculer les étages GPU sur le format
        # le plus compact du dépôt, q3n (3,25 bits/poids). Si même lui grossit
        # la source, la garde ci-dessous refusera avec les issues restantes.
        # La bascule s'annonce parce qu'elle change la qualité : la perplexité
        # de q3n n'est validée sur aucun vrai modèle à ce jour (8/09/2026).
        print(f"[acvram] source à {octets_src * 8 / n_params:.2f} bits/poids : "
              f"les étages GPU passent de leur format nominal "
              f"({bpw_cible:.2f} b/p) à q3n (3,25) pour ne pas grossir — "
              f"perplexité à vérifier par `acvram eval`", flush=True)
        gpus = {t.name for t in plan.tiers if t.kind == "gpu"}
        for t in plan.tiers:
            if t.kind == "gpu":
                t.weight_format = "q3n"
        # Le routeur lit le format PAR COUCHE (lp.fmt), pas celui des étages :
        # la première bascule, qui ne changeait que les étages, a produit un
        # dossier de 42 Go entièrement en nvfp4/int4 malgré son propre message
        # « les étages passent à q3n » — l'intention était annoncée, le
        # résultat non vérifié. D'où aussi la vérification d'issue en fin de
        # conversion, sur les octets réellement écrits.
        for lp in plan.layers:
            if getattr(lp, "fmt", None) and lp.exec_device in gpus | {"cpu"}:
                lp.fmt = "q3n"
        bpw_cible = bpw_nominal("q3n", opts.group_size)
        bascule_faite = True
        bpw_cible_nominal = "q3n"
    avert = garde_grossissement(octets_src, n_params,
                                bpw_cible, opts.autoriser_grossissement)
    if avert:
        print(f"[acvram] {avert}", flush=True)
    router = TensorRouter(spec, plan, opts)
    report = ConversionReport(model=spec.name)
    writer = ShardWriter(opts.out_dir)
    manifest: dict[str, Any] = {
        "acvram_version": 1,
        # poste7-expert-partage-cle-portee-17-09, REGLES §4 : un correctif du
        # convertisseur (ex. 0e4ef38) change le régime des convertis qu'il
        # produit -- ce champ distingue les manifestes d'avant/après.
        "convertisseur": _convertisseur_commit(),
        "model": spec.to_dict(),
        "plan": plan.to_dict(),
        "options": asdict(opts),
        # D ou vient ce modele. Le manifeste portait out_dir et jamais
        # l entree : le 9/09/2026, retrouver la source du temoin bf16 a
        # demande de comparer ses tenseurs bit a bit a un candidat, faute de
        # pouvoir la lire. Un couple a une variable exige la meme origine ;
        # sans cette cle, on ne peut pas garantir qu on mesure deux formats
        # plutot que deux modeles.
        "source": {
            "chemin": os.path.abspath(model_path),
            "nom": os.path.basename(os.path.abspath(model_path)),
            "octets": _octets_du_checkpoint(model_path),
            "sha256": _sha256_du_checkpoint(model_path),
        },
        # poste7-diff-octet-a-octet-retire-17-09 (REGLES §4) : la recherche AWQ
        # n'est PAS bit-exacte d'une execution a l'autre (reductions
        # flottantes multi-thread non associatives, mesure sur GLM -- 10,7 %
        # d'ecart de reconstruction sur un tenseur nvfp4 malgre un SNR
        # identique a 2 decimales). Aucune graine ne corrige cela : le
        # chemin de calibration/quantification n'a pas de generateur
        # aleatoire, donc rien a fixer -- l'ecart vient du threading, pas du
        # hasard. Champ honnête, pas fabriqué : documente l'ABSENCE d'un
        # levier plutôt que d'y mettre une valeur qui n'en contrôle rien.
        "graine": {
            "valeur": None,
            "note": "aucun generateur aleatoire dans la calibration/"
                    "quantification -- la non-determinisme observee vient "
                    "des reductions flottantes multi-thread, pas d'une graine.",
        },
        "avertissement_determinisme": (
            "AWQ non deterministe : un converti est identifie par son "
            "sha256 (poids ET manifeste), jamais par ses options -- deux "
            "conversions aux memes options peuvent differer de l'ordre de "
            "10 % sur la reconstruction d'un tenseur nvfp4, meme SNR "
            "rapporte."),
        # Ce qui a ete demande et ce qui est sorti, cote a cote et toujours,
        # meme quand ils coincident. Un manifeste qui ne porte que le resultat
        # laisse croire qu'il a ete voulu.
        "formats_nominaux": {
            "demande": opts.format_impose,
            "obtenu": bpw_cible_nominal,
            "bits_par_poids_source": round(bpw_src, 3) if bpw_src else None,
            "bits_par_poids_cible": round(bpw_cible, 3),
            "bascule_anti_grossissement": bool(bascule_faite),
        },
        # poste7-p2-qkvo-int8-canal-18-09 : nom du regime d'attention porte au
        # niveau du manifeste, pas seulement sur chaque tenseur -- "canal"
        # distingue ce converti de la pile classee (q/k/v/o groupe-128).
        "attn_int8": "canal" if opts.attn_qkvo_int8_canal else "groupe",
        "tensors": {},
    }

    qdev = _resolve_quant_device(opts.quant_device)

    snrs: list[float] = []
    per_layer: list[dict] = []
    keys = []
    # poste7-awq-experts-peu-routes-portee-17-09 : garde permanente, pas
    # seulement le seuil d'echantillons qui evite la CAUSE la plus
    # frequente -- verifie l'EFFET sur CHAQUE tenseur quantifie, quelle que
    # soit la cause d'une echelle mal reglee. Borne large (0,80-1,25) : une
    # quantification saine reste tres proche de 1 (mesure sur 340 experts
    # sains, 0,944-1,012) ; les six malades de Nemotron sortaient a
    # 0,29-0,69.
    RATIO_NORME_BAS, RATIO_NORME_HAUT = 0.80, 1.25
    # poste7-awq-plancher-median-faute-18-09 : le nombre qui relie directement
    # a la PPL est l'ETENDUE des echelles (max/min de `scaler.scale`), pas
    # seulement le ratio de norme en sortie -- 1,7e6x -> PPL 1,4301 ;
    # 6e5x -> 1,2634 ; ~630x sain. `_magnitude_avec_plancher_relatif` borne
    # cette etendue a 4096 PAR CONSTRUCTION ; ce plafond est une garde de
    # secours pour ce qui y echapperait quand meme (`forced_scale`).
    ETENDUE_MAX = 4096.0
    ratios_hors_bornes: list[tuple[str, float]] = []
    # poste7-awq-relu2-garde-repli-17-09, geste (b) : repli identite PAR
    # TENSEUR plutot que refus de toute la conversion, sous un plafond --
    # au-dela de la moitie des tenseurs repliee, ce n'est plus une
    # reparation ciblee mais une calibration cassee dans son ensemble.
    tenseurs_replies: list[tuple[str, float]] = []
    PLAFOND_REPLI = 0.50
    budget_candidats: list[dict] = []

    alpha_commun: dict[str, torch.Tensor] = {}
    if opts.alpha_commun_gate_up or opts.alpha_commun_experts:
        alpha_commun = _precalculer_alpha_commun_gate_up(
            model_path, spec, router, stats, opts, qdev)
        paires_experts = alpha_commun.pop("__experts_paires__", 0)
        print(f"[acvram] alpha commun gate/up : {len(alpha_commun) // 2} "
              f"paires fusionnees" + (f", dont {paires_experts} paires d'experts MoE"
                                      if opts.alpha_commun_experts else ""), flush=True)
    if opts.alpha_commun_qkv:
        commun_qkv = _precalculer_alpha_commun_qkv(model_path, spec, router, stats, opts, qdev)
        alpha_commun.update(commun_qkv)
        print(f"[acvram] alpha commun q/k/v : {len(commun_qkv) // 3} triplets partagent leur échelle", flush=True)
    # poste7 (gate!=up, main 7c3698d) : `_try_build_stacks` ne force plus
    # gate_proj == up_proj -- il fusionne quand la recherche les rend egaux,
    # garde une seconde ligne/quantification sinon. L'echelle forcee a
    # l'identite par paire (`_precalculer_alpha_commun_experts`, 908e926)
    # n'est donc plus necessaire pour les experts ; la recherche independante
    # par tenseur ci-dessous suffit (identite explicite quand elle s'effondre
    # malgre tout, ligne ~1220).

    from .hfquant import is_hfquant, par_groupe_actif
    source_quantifiee = opts.passage_direct and is_hfquant(model_path)
    # Pièce 139 : sous l'opt-in par groupe, hfquant rend les poids fp8 en fp32 exact (seuls tenseurs fp32 en 2-D
    # de cette source) ; ils sont ré-encodés en int8 symétrique PAR CANAL — même nombre d'octets que le fp8 servi
    # par NInfer (bf16 clair : 29,7 Go de poids par pas, ne tient pas sur la carte), +1 % d'erreur en quadrature.
    fp8_en_int8_canal = par_groupe_actif() and is_hfquant(model_path)
    vision_bytes = 0                     # Σ octets des tenseurs VISION_PREFIXES gardés
    from .gguf import is_gguf, mmproj_a_cote
    if is_gguf(model_path) and mmproj_a_cote(model_path):
        print(f"[acvram] refus nommé : {mmproj_a_cote(model_path)} ignoré, la voie "
              f"GGUF-mmproj est hors périmètre (contrat multimodal § 2 pièce (a), "
              f"seule la voie safetensors HF porte la tour) ; alias texte seul",
              flush=True)
    observations = (stats or {}).pop("__observations__", None) if isinstance(stats, dict) else None
    journal = _journal_tenseurs(opts, qdev)
    cache_repli: dict = {}
    vision_ecartee = 0
    for name, tensor in _adapt_hf(_iter_checkpoint(model_path, opts.passage_direct), spec):
        if opts.sans_vision and est_tenseur_vision(name):
            vision_ecartee += 1
            continue
        if journal:
            journal(name, tuple(tensor.shape), report.tensors)
        report.tensors += 1
        if progress and report.tensors % 25 == 0:
            progress(name, report.tensors, 0)
        if isinstance(tensor, NVFP4Tensor):
            # passage direct : les octets de la source, sans recherche ni
            # métrique (aucune référence bf16 à comparer) ; format imposé
            # nvfp4 quel que soit le plan, sinon le poids servi ne serait
            # plus celui de vLLM
            qt = tensor
            report.in_bytes += qt.nbytes
            sd = {k: v.cpu() for k, v in qt.state_dict(prefix=f"{name}.").items()}
            if not opts.dry_run:
                for k, v in sd.items():
                    writer.add(k, v)
            entry = {"format": "nvfp4", "shape": list(qt.shape), "keys": list(sd.keys()),
                     "group_size": opts.group_size, "hadamard_block": 0, "has_act_scale": False,
                     "bpw": round(float(qt.bits_per_weight), 3), "passage_direct": True}
            report.per_format["nvfp4"] = report.per_format.get("nvfp4", 0) + qt.nbytes
            manifest["tensors"][name] = entry
            keys.append(name)
            continue
        report.in_bytes += tensor.numel() * tensor.element_size()
        fmt = router.format_for(name)
        entry: dict[str, Any] = {"format": fmt, "shape": list(tensor.shape)}
        origine_fp8 = fp8_en_int8_canal and tensor.dim() == 2 and tensor.dtype == torch.float32
        if origine_fp8:
            fmt = "int8"
            entry = {"format": fmt, "shape": list(tensor.shape), "origine": "fp8"}
        elif (source_quantifiee and tensor.dim() == 2 and tensor.dtype in (torch.bfloat16, torch.float16)
                and fmt not in ("bf16", "fp16", "fp32")):
            # passage direct : une couche que la source a GARDÉE EN CLAIR
            # (ignore / exclude_modules : lm_head, plongements, routeurs,
            # attention MLA chez GadflyII) reste en clair — quantifier ici
            # servirait d'autres poids que vLLM (poste3, 17/09 : q_a/q_b/kv_a/
            # o_proj et lm_head passés en nvfp4 par le plan)
            fmt = "bf16" if tensor.dtype == torch.bfloat16 else "fp16"
            entry = {"format": fmt, "shape": list(tensor.shape), "passage_direct": "clair"}

        if fmt == "fp32" or fmt in ("bf16", "fp16") or tensor.dim() != 2:
            if fmt == "fp32":
                out = tensor.to(torch.float32)
            else:
                out = tensor.to(torch.bfloat16 if fmt == "bf16" else torch.float16)
            if not opts.dry_run:
                writer.add(f"{name}", out)
            entry["keys"] = [name]
            report.per_format[fmt] = report.per_format.get(fmt, 0) + \
                out.numel() * out.element_size()
            if est_tenseur_vision(name):
                vision_bytes += out.numel() * out.element_size()
            manifest["tensors"][name] = entry
            keys.append(name)
            continue

        st = stats.get(name) if stats else None
        # poste7-hybrides-etape1-close-gemm-dense-17-09 : Nemotron calibA PPL
        # 1,4301 vs 1,0304 sans calibration, degradation uniforme sur les 3
        # tranches. Diff tenseur par tenseur (a sec, disque) : 70 tenseurs
        # exclus (mamba.in_proj/out_proj, self_attn) byte-identiques entre
        # les deux convertis (pas la cause) ; expert partage sain (ratio de
        # norme 0,994 +/- 0,015) ; MAIS 6/346 experts.*.down_proj
        # echantillonnes ont un ratio de norme calibA/officielle de 0,29 a
        # 0,69 (SNR interne pourtant bon, 27-29 dB -- coherent avec la
        # metrique W_EFF, pas avec le poids d'origine une fois reechelonne).
        # Cause : `act_scale` de ces tenseurs s'etend sur 1,68e6x/2,4e5x
        # (0,0001 a 177, 0,008 a 19) contre ~630x pour un tenseur sain
        # (0,013 a 8,1) -- la recherche AWQ (`search_channel_scales`,
        # clamp PAR VALEUR a [1e-4,1e4], jamais sur l'ETENDUE) tire un motif
        # de salience extreme d'une statistique BRUITEE : un expert MoE peu
        # routé sur bras-A (prose anglaise, 32 sequences) voit une poignee
        # de jetons, et `mean_abs` par canal n'estime plus rien -- exactement
        # le regime que "trop peu sur le corpus" (message plus bas) NOMMAIT
        # deja sans jamais le TESTER : `experts_sans_stats` ne comptait que
        # n_samples == 0, jamais "trop peu". Sous ce seuil, l'identite (le
        # meme repli que "jamais routé") vaut mieux qu'une AWQ instable.
        MIN_ECHANTILLONS_AWQ = 8
        if st is not None and st.n_samples < MIN_ECHANTILLONS_AWQ:
            st = None
        est_expert = ".mlp.experts." in name
        if est_expert and st is None:
            report.experts_sans_stats += 1
            report.experts_sans_stats_noms.append(name)
            if opts.repli_experts == "mediane_couche" and stats:
                st = _stats_repli_mediane(stats, name, MIN_ECHANTILLONS_AWQ, cache_repli)
                if st is not None:
                    report.experts_repli_mediane += 1
        # poste7 (a6a7436, 15/09 ; gate!=up 7c3698d, 16/09) : l'echelle AWQ par
        # expert reste dans la pile -- le moteur (poste4) la porte cote
        # loader ([E,K], appliquee apres route+pack), gate_proj et up_proj
        # n'ont plus besoin d'etre egaux : recherche AWQ independante par
        # tenseur, comme pour les tenseurs denses. Quand la recherche
        # s'effondre malgre tout a l'identite (`scaler.scale is None`), on
        # pose une echelle identite EXPLICITE (torch.ones) : le manifeste ne
        # doit jamais melanger "echelle absente" et "echelle presente" par
        # ambiguite d'absence.
        #
        # poste7 (`poste7-organisation-16-09.md`, poste1 `verdict-glm-awq-int8-
        # 16-09.md` : q_a 0,982, kv_a 0,983, o_proj sans table -- retrait
        # sans risque, tous >= 0,9) : l'AWQ sur un tenseur int8 compense une
        # erreur ~16x plus petite qu'en 4 bits pour le meme cout de division
        # a chaque jeton (`layers.py:479-481`, mesure +2,06 ms/pas,
        # `verdict-glm-2ms-manifestes-16-09.md`) -- retiree des tenseurs
        # int8, gardee sur nvfp4 ou elle compense une vraie perte.
        #
        # poste7 (`poste7-glm-mma0-verdict-16-09.md` § 2/4) : sous la pile
        # groupee W4A4, l'activation est elle-meme quantifiee en NVFP4 apres
        # division par l'echelle (`nvfp4_quant_act`, model.py:1022) -- la
        # metrique de recherche doit voir ce meme chemin pour les experts,
        # sinon elle choisit un alpha qui elargit l'etendue intra-bloc de
        # l'activation sans le savoir (`quantize_activation_nvfp4`,
        # calibrate.py). RÉFUTÉ (`verdict-glm-k48-w4a4-prefill-16-09.md`,
        # 1,0229 pire que 1,0183 sans elle) : le defaut est le bloc E2M1 de
        # 16 lui-meme, pas l'alpha -- `hadamard_experts` (poste7, `poste7-
        # hadamard-16-09.md`) tourne le poids ET l'activation en H_512 AVANT
        # quantification, SANS echelle AWQ sur ces tenseurs (les deux
        # mecanismes ne se cumulent pas ici, l'un remplace l'autre a l'essai).
        experts_hadamard = est_expert and opts.hadamard_experts and fmt == "nvfp4"
        # poste7-p2-qkvo-int8-canal-18-09 : q/k/v/o seuls, et seulement si le
        # routeur les a places en int8 -- le reste du modele garde le groupe
        # de 128 affine (opts.group_size) sans y toucher.
        attn_canal = origine_fp8 or (opts.attn_qkvo_int8_canal and fmt == "int8"
                                     and _est_projection_attn(name)) or (
                                     opts.gdn_int8_canal and fmt == "int8"
                                     and _est_projection_gdn(name))
        group_size_tenseur = tensor.shape[1] if attn_canal else opts.group_size
        qt, scaler, metrics = _quantize_on(
            qdev, tensor, fmt, st,
            group_size=group_size_tenseur,
            use_hadamard=router.wants_hadamard(name, fmt) or experts_hadamard,
            use_awq=opts.awq and fmt == "nvfp4" and not experts_hadamard,
            n_grid=opts.n_grid, garder_grille=opts.garder_grille,
            table=opts.q3n_table if fmt == "q3n" else None,
            mesurer_kld=opts.mesurer_kld,
            forced_scale=alpha_commun.get(name),
            quantize_activation_nvfp4=est_expert and fmt == "nvfp4" and not experts_hadamard,
            hadamard_block=512 if experts_hadamard else None,
            symmetric=attn_canal,
        )
        if est_expert and scaler.scale is None and not experts_hadamard:
            scaler = ChannelScaler(
                torch.ones(tensor.shape[1], dtype=torch.float32),
                scaler.hadamard_block,
            )
        # poste7-awq-relu2-garde-repli-17-09 : geste (b), repli PAR TENSEUR au
        # lieu du refus global. Cause identifiee (pas "peu route" comme
        # d'abord suppose) : l'entree de `down_proj` chez nemotron_h est
        # ReLU²(up(x)) -- creuse par construction, les canaux jamais actives
        # sur le corpus s'ecrasent au plancher `clamp(min=1e-6)`
        # (`calibrate.py:302`) contre ~1 ailleurs, etendue 1,7e6x independante
        # du nombre d'echantillons. Coder/GLM (SiLU a porte) n'ont pas ce
        # motif -- 0 malade chez eux n'etait pas un hasard de corpus.
        # poste7-awq-plancher-median-faute-18-09, garde n°2 : l'ÉTENDUE des
        # échelles par canal (max/min de `scaler.scale`) est le nombre qui
        # relie directement la casse à la PPL (1,7e6× -> PPL 1,4301 ;
        # 6e5× -> 1,2634 ; ~630× sain) -- le ratio de norme est un SYMPTOME
        # en sortie, l'étendue est la CAUSE mesurable côté échelle. `_magni
        # tude_avec_plancher_relatif` borne cette étendue à 4096 PAR
        # CONSTRUCTION ; ce contrôle attrape tout ce qui y échapperait
        # quand même (ex. `forced_scale`, qui contourne la recherche).
        etendue = (scaler.scale.max() / scaler.scale.min().clamp(min=1e-12)).item() \
            if scaler.scale is not None else None
        if etendue is not None and etendue > ETENDUE_MAX:
            tenseurs_replies.append((name, etendue))
            qt, scaler, metrics = _quantize_on(
                qdev, tensor, fmt, None,
                group_size=group_size_tenseur,
                use_hadamard=router.wants_hadamard(name, fmt) or experts_hadamard,
                use_awq=False,
                n_grid=opts.n_grid, garder_grille=opts.garder_grille,
                table=opts.q3n_table if fmt == "q3n" else None,
                mesurer_kld=opts.mesurer_kld,
                hadamard_block=512 if experts_hadamard else None,
                symmetric=attn_canal,
            )
            if est_expert and scaler.scale is None:
                scaler = ChannelScaler(
                    torch.ones(tensor.shape[1], dtype=torch.float32),
                    scaler.hadamard_block)
        elif "ratio_norme" in metrics and st is not None:
            r = metrics["ratio_norme"]
            if not (RATIO_NORME_BAS <= r <= RATIO_NORME_HAUT):
                tenseurs_replies.append((name, r))
                qt, scaler, metrics = _quantize_on(
                    qdev, tensor, fmt, None,
                    group_size=group_size_tenseur,
                    use_hadamard=router.wants_hadamard(name, fmt) or experts_hadamard,
                    use_awq=False,
                    n_grid=opts.n_grid, garder_grille=opts.garder_grille,
                    table=opts.q3n_table if fmt == "q3n" else None,
                    mesurer_kld=opts.mesurer_kld,
                    hadamard_block=512 if experts_hadamard else None,
                    symmetric=attn_canal,
                )
                if est_expert and scaler.scale is None:
                    scaler = ChannelScaler(
                        torch.ones(tensor.shape[1], dtype=torch.float32),
                        scaler.hadamard_block)
        if fmt == "q3n":
            entry["block"] = qt.block
            entry["table"] = list(qt.table)
            # Le critère de choix de table n'est PAS le creux : c'est le taux
            # de zéros EXACTS de la source (contrôle bf16 du 8/09 : à creux
            # égal, seul un poids nul profite d'un niveau zéro — une source
            # bf16 en a 0 %, un GGUF à grille avec zéro ~21 %). Mesuré ici et
            # écrit à côté du choix, pour que la règle soit vérifiable.
            entry["taux_zeros_source"] = round(
                float((tensor == 0).float().mean()), 4)
            # Sceau : lie la table aux octets réellement écrits. Un manifeste
            # régénéré sans reconversion ferait lire d'anciens poids avec une
            # nouvelle table, silencieusement — le pire mode de défaillance.
            import hashlib as _h
            entry["sceau"] = _h.sha256(
                qt.qweight.flatten()[:64].cpu().numpy().tobytes()
                + repr(list(qt.table)).encode()).hexdigest()[:16]

        # Précision mixte, deux régimes : plancher SNR classique (défaut,
        # plafonné), ou budget global (bits_budget_gib > 0) où les deux
        # formats sont mesurés et la décision revient au sac à dos de fin de
        # passe. Les experts d'un bloc restent exclus dans les deux cas : le
        # chemin de décodage groupé exige leur pile homogène, et leur SNR
        # individuel pèse peu.
        candidat_budget = (opts.bits_budget_gib > 0 and not est_expert
                           and fmt in PROMOTE
                           and metrics["out_snr_db"] < 40.0)
        if candidat_budget:
            wider = PROMOTE[fmt]
            q2, s2, m2 = _quantize_on(
                qdev, tensor, wider, st,
                group_size=opts.group_size,
                use_hadamard=router.wants_hadamard(name, wider),
                use_awq=opts.awq and wider == "nvfp4", n_grid=opts.n_grid,
                garder_grille=opts.garder_grille,
                mesurer_kld=opts.mesurer_kld)
            if m2["out_snr_db"] > metrics["out_snr_db"] + 0.5:
                sd2 = q2.state_dict(prefix=f"{name}.")
                sd2.update(s2.state_dict(prefix=f"{name}."))
                sd2 = {k: v.cpu() for k, v in sd2.items()}
                base_octets = sum(v.numel() * v.element_size()
                                  for v in qt.state_dict().values())
                larges_octets = sum(v.numel() * v.element_size()
                                    for v in sd2.values())
                candidat = {
                    "name": name, "from": fmt, "to": wider,
                    "gain_db": m2["out_snr_db"] - metrics["out_snr_db"],
                    "cout": max(1, larges_octets - base_octets),
                    "sd": sd2, "metrics": m2,
                }
                # Protocole A/B KLD vs SNR (revue/protocole-ab-kld-vs-snr.md,
                # 13/09) : un gain de KLD est une REDUCTION de divergence, donc
                # base moins large — l'inverse du sens de `gain_db`, ou plus
                # large moins base. Absent si `opts.mesurer_kld` est faux ; le
                # mode de tri `kld` refuse alors de partir plutot que de
                # degenerer en silence sur une cle jamais mesuree.
                if "out_kld_bits" in metrics and "out_kld_bits" in m2:
                    candidat["gain_kld_bits"] = (
                        metrics["out_kld_bits"] - m2["out_kld_bits"])
                budget_candidats.append(candidat)
            else:
                candidat_budget = False
        elif (not est_expert
                and opts.mixed_precision != "off"
                and metrics["out_snr_db"] < opts.snr_floor
                and fmt in PROMOTE
                and (not opts.promotion_classes
                     or any(name.endswith(c + ".weight") or name.endswith(c)
                            for c in opts.promotion_classes))
                and (not opts.promotion_cout_max_mib
                     or cout_promotion_mib(tensor.numel(), fmt, PROMOTE[fmt],
                                           opts.group_size)
                     <= opts.promotion_cout_max_mib)
                and len(report.promotions) < opts.max_promotions * max(1, len(keys) + 1)):
            wider = PROMOTE[fmt]
            # poste7-p2-qkvo-int8-canal-18-09 : sur Qwen3-Coder, q/k/v/o partent
            # en nvfp4 (20,5-20,7 dB, sous le plancher) et n'ATTEIGNENT l'int8
            # QUE PAR CETTE PROMOTION -- `attn_canal`/`group_size_tenseur`
            # calcules plus haut valaient donc faux (fmt="nvfp4" a ce moment).
            # Refaits ici sur `wider`, le format candidat ; n'ecrasent les
            # variables du manifeste que si la promotion est ACCEPTEE
            # plus bas, sinon le tenseur reste nvfp4 et group_size=largeur
            # entiere y serait un mensonge de bilan.
            attn_canal_candidat = (opts.attn_qkvo_int8_canal and wider == "int8"
                                   and _est_projection_attn(name)) or (
                                   opts.gdn_int8_canal and wider == "int8"
                                   and _est_projection_gdn(name))
            group_size_candidat = tensor.shape[1] if attn_canal_candidat else opts.group_size
            q2, s2, m2 = _quantize_on(
                qdev, tensor, wider, st,
                group_size=group_size_candidat,
                use_hadamard=router.wants_hadamard(name, wider),
                use_awq=opts.awq and wider == "nvfp4", n_grid=opts.n_grid,
                garder_grille=opts.garder_grille,
                symmetric=attn_canal_candidat)
            if m2["out_snr_db"] > metrics["out_snr_db"] + 1.0:
                report.promotions.append({
                    "name": name, "from": fmt, "to": wider,
                    "before": round(metrics["out_snr_db"], 2),
                    "after": round(m2["out_snr_db"], 2)})
                fmt, qt, scaler, metrics = wider, q2, s2, m2
                attn_canal, group_size_tenseur = attn_canal_candidat, group_size_candidat
                entry["format"] = fmt
                entry["promoted_from"] = report.promotions[-1]["from"]
                # Le SNR d'AVANT, celui pris dans `promoted_from`. Sans lui,
                # `snr_db` d'un promu est celui du format d'arrivee, et le
                # comparer a celui d'un non-promu compare deux FORMATS en
                # croyant comparer deux merites : sur 36 modeles a plancher
                # actif, les promus sortaient a 44,4 dB de mediane contre 20,4
                # aux epargnes, regularite si nette qu'elle passait pour un
                # resultat — c'etait l'ecart int8/nvfp4, rien d'autre.
                # Regle qui en decoule : un SNR se publie TOUJOURS avec le
                # format dans lequel il a ete pris. Ici `snr_db` va avec
                # `format`, `snr_db_source` avec `promoted_from`.
                entry["snr_db_source"] = report.promotions[-1]["before"]
        # Le SNR de CHAQUE tenseur, promu ou non. Sans lui on ne peut pas
        # repondre a la question qui juge le quota : existe-t-il un tenseur
        # NON promu dont le SNR est pire que celui d'un promu ? Si oui, le
        # quota n'est pas un critere de qualite mais un ordre de parcours.
        if "out_snr_db" in metrics:
            entry["snr_db"] = round(float(metrics["out_snr_db"]), 3)
        # OPTION (opts.mesurer_kld), n'influence aucune decision : publiee au
        # manifeste pour le protocole A/B, a cote du SNR qui reste le critere.
        if "out_kld_bits" in metrics:
            entry["kld_bits"] = round(float(metrics["out_kld_bits"]), 6)
        _noter_echelle_nvfp4(manifest, entry, qt)
        sd = qt.state_dict(prefix=f"{name}.")
        sd.update(scaler.state_dict(prefix=f"{name}."))
        # Les fragments s'ecrivent depuis la memoire hote : on redescend ce que
        # la quantification a produit sur le GPU.
        sd = {k: v.cpu() for k, v in sd.items()}

        if candidat_budget and budget_candidats and \
                budget_candidats[-1]["name"] == name:
            # decision differee au sac a dos : rien n'est ecrit maintenant
            budget_candidats[-1].update({
                "sd_base": sd, "entry": entry, "fmt_base": fmt,
                "metrics_base": metrics, "nbytes_base": qt.nbytes,
                "hadamard_block": scaler.hadamard_block,
                "has_act_scale": scaler.scale is not None,
            })
            keys.append(name)
            continue

        snrs.append(metrics["out_snr_db"])
        if "ratio_norme" in metrics:
            r = metrics["ratio_norme"]
            if not (RATIO_NORME_BAS <= r <= RATIO_NORME_HAUT):
                ratios_hors_bornes.append((name, r))
        per_layer.append({"name": name, **{k: round(v, 2) if isinstance(v, float)
                                           else v for k, v in metrics.items()}})
        if not opts.dry_run:
            for k, v in sd.items():
                writer.add(k, v)
        entry.update({
            "keys": list(sd.keys()),
            "group_size": group_size_tenseur,
            **({"symmetrique": True} if attn_canal else {}),
            "hadamard_block": scaler.hadamard_block,
            "has_act_scale": scaler.scale is not None,
            "bpw": round(metrics["bpw"], 3),
            "out_snr_db": round(metrics["out_snr_db"], 2),
            **({"ratio_norme": round(metrics["ratio_norme"], 4)}
               if "ratio_norme" in metrics else {}),
            **({"rotation": "hadamard-512"} if experts_hadamard else {}),
            # L'echelle de sortie, sans laquelle le SNR ne se compare pas d'un
            # tenseur a l'autre. Publiee pour qu'une analyse posterieure au
            # manifeste puisse reconstituer l'erreur absolue sans reconvertir.
                **({"out_ref_norm": round(float(metrics["out_ref_norm"]), 6)}
                   if "out_ref_norm" in metrics else {}),
                **({"erreurs_grille": metrics["erreurs_grille"],
                    "alpha_retenu": metrics["alpha_retenu"]}
                   if "erreurs_grille" in metrics else {}),
        })
        report.per_format[fmt] = report.per_format.get(fmt, 0) + qt.nbytes
        manifest["tensors"][name] = entry
        keys.append(name)

    # ---- sac à dos : depenser le budget la ou chaque octet paie le plus ----
    if journal:
        journal(None, None, report.tensors)      # le dernier tenseur
    if budget_candidats:
        deja = sum(report.per_format.values()) + \
            sum(c["nbytes_base"] for c in budget_candidats)
        reste = opts.bits_budget_gib * 1024 ** 3 - deja
        # par gain de SNR par octet, decroissant — le glouton du sac a dos
        # fractionnaire, optimal a un tenseur pres
        # ORDRE DU GLOUTON, et un echappement pour l'EPROUVER.
        #
        # Le critere est le gain de SNR par octet. Or ce qui decide est la
        # perplexite par octet, et la courbe du quota du 10/09 montre que les
        # deux ne coincident pas sur la queue : les 27 tenseurs refuses en
        # dernier rendent 2,867 milli-PPL chacun contre 0,796 pour les 26
        # acceptes juste avant, soit 3,6 fois plus.
        #
        # ACVRAM_ORDRE_SAC_INVERSE renverse le signe, RIEN D'AUTRE : meme sac a
        # dos, meme budget, meme plancher, meme format. Si le critere est bien
        # oriente, le bras inverse doit etre nettement PIRE ; s'il est mal
        # oriente sur la queue, il sera meilleur ou equivalent. Un controle qui
        # ne peut pas confirmer l'hypothese par construction, puisque les deux
        # bras sortent du meme code a un signe pres.
        #
        # L'ordre inverse n'est PAS un candidat : c'est un instrument. Un ordre
        # optimal se cherchera ensuite, et l'ecart entre les deux bras donne la
        # borne de ce que l'ordre vaut, quel qu'il soit.
        # TROIS CLES, et le defaut ne change pas : un A/B est en cours de
        # jugement sur `snr`, et deplacer le defaut sous lui l'invaliderait.
        #
        # `snr`    : gain de decibels par octet — la cle historique.
        # `erreur` : erreur de sortie EVITEE par octet, 10^(-snr/20) applique
        #            aux DEUX SNR avant la soustraction. Ce n'est donc PAS une
        #            transformation monotone de la cle `snr` : a ecart de
        #            decibels egal, un tenseur a faible SNR de base evite dix
        #            fois plus d'erreur qu'un tenseur a fort SNR de base.
        #
        #                A   SNR 20 -> 30 dB   erreur 0,1000 -> 0,0316   gain 0,0684
        #                B   SNR 40 -> 50 dB   erreur 0,0100 -> 0,0032   gain 0,0068
        #
        #            La cle en decibels les classe ex aequo ; la cle en erreur
        #            place A dix fois devant B. Argument de chef, VERIFIE sur
        #            nos donnees : correlation -0,66 entre le SNR de base et le
        #            deplacement de rang, et 20,8 dB de SNR de base moyen pour
        #            les tenseurs qui montent contre 29,7 pour ceux qui
        #            descendent. Le reordonnancement est concentre en TETE du
        #            classement — 90 % de desaccord au top-10, 3 % au top-100 —
        #            c'est-a-dire la ou le glouton puise en premier.
        # `inverse`: le signe renverse, INSTRUMENT et non candidat.
        #
        # NI L'UNE NI L'AUTRE N'EST LA PERPLEXITE. La cle `erreur` est un
        # meilleur substitut, fonde, pas une mesure : trancher demanderait un
        # DL par tenseur sur une perte de calibration, une passe avant par
        # tenseur et par format.
        _mode = os.environ.get("ACVRAM_ORDRE_SAC", "snr").strip().lower()
        if os.environ.get("ACVRAM_ORDRE_SAC_INVERSE"):
            _mode = "inverse"
        # ACVRAM_LISTE_PROMUS : promouvoir EXACTEMENT une liste de noms, sans
        # ordre et sans budget. Ce n'est plus un sac a dos — c'est le seul
        # moyen de monter un bras qui isole un groupe de tenseurs choisi
        # ailleurs (bras X et Y du 10/09). Le fichier prime sur toute autre
        # valeur d'ACVRAM_ORDRE_SAC : deux consignes contradictoires doivent
        # lever, jamais laisser deviner laquelle a gagne.
        _liste_chemin = os.environ.get("ACVRAM_LISTE_PROMUS", "").strip()
        if _liste_chemin:
            _demande = os.environ.get("ACVRAM_ORDRE_SAC", "").strip().lower()
            if _demande and _demande != "liste":
                raise ValueError(
                    f"ACVRAM_LISTE_PROMUS={_liste_chemin!r} et "
                    f"ACVRAM_ORDRE_SAC={_demande!r} sont contradictoires : une "
                    f"liste explicite n'a pas d'ordre. Retirer l'une des deux.")
            _mode = "liste"
        elif _mode == "liste":
            raise ValueError(
                "ACVRAM_ORDRE_SAC=liste exige ACVRAM_LISTE_PROMUS=<fichier>. "
                "Sans liste, le mode ne promouvrait rien et le dossier "
                "ressemblerait a une conversion au format de base.")
        _genres_vus: dict[str, int] = {}
        if _mode not in ("snr", "erreur", "inverse", "absolu",
                         "base_croissant", "liste", "genre",
                         "cout_decroissant", "kld"):
            raise ValueError(
                f"ACVRAM_ORDRE_SAC={_mode!r} inconnu ; attendu snr, erreur, "
                f"absolu, base_croissant, genre, cout_decroissant, kld, liste "
                f"ou inverse. Un mode inconnu qui "
                f"retomberait en silence sur le defaut ferait mesurer autre "
                f"chose que ce qui est demande.")
        # Protocole A/B (13/09) : trier par KLD exige de l'avoir mesure. Sans
        # cette garde, `kld` degenererait en silence sur une cle absente —
        # exactement le defaut que ce bloc de gardes existe pour eviter sur
        # les autres modes.
        if _mode == "kld" and not opts.mesurer_kld:
            raise ValueError(
                "ACVRAM_ORDRE_SAC=kld exige ConversionOptions.mesurer_kld=True "
                "(--mesurer-kld) : sans lui, out_kld_bits n'est jamais calcule "
                "et le tri par KLD retomberait en silence sur un classement "
                "vide.")

        def _abs_err(m):
            """Erreur ABSOLUE en sortie de couche, en unites de sortie.

            `out_snr_db` et `out_rel_err` sont des RAPPORTS : le denominateur
            ||y_ref|| y disparait. C'est le bon chiffre pour juger un tenseur
            contre lui-meme, et le mauvais pour en classer deux l'un contre
            l'autre — ce que fait precisement ce sac a dos. Ce qui se propage
            jusqu'a la perte est l'erreur absolue ; deux tenseurs a 20 et 30 dB
            dont les sorties valent 1 et 100 portent 0,1 et 3,16 d'erreur, et
            le classement en decibels met le plus nuisible en second.

            Repli sur l'erreur relative quand `out_ref_norm` est absent — un
            manifeste produit avant l'ajout du champ. Le repli est SIGNALE par
            l'absence du champ, pas silencieux : `absolu` degenere alors en
            `erreur`, ce qui est exactement l'ancien comportement.
            """
            ech = m.get("out_ref_norm")
            if "out_abs_err" in m:
                return float(m["out_abs_err"])
            rel = 10.0 ** (-(m["out_snr_db"]) / 20.0)
            return rel * float(ech) if ech else rel

        # DEUX CLES DERIVEES DE L OBSERVATION, ET ELLES SONT INDISSOCIABLES.
        #
        # `genre` et `cout_decroissant` sont bâties POUR reproduire le bras B.
        # Elles ne peuvent donc pas servir a le confirmer : une clé taillée sur
        # un résultat le reproduit par construction. Ce qui les validerait est
        # un SECOND modèle — si « promouvoir down_proj puis lm_head » gagne
        # aussi sur un modèle que nous n'avons pas regardé, ce n'est plus une
        # description, c'est une loi. Ecrit AVANT de les construire, sur la
        # demande de chef.
        #
        # ET ELLES SONT CONFONDUES SUR LLAMA-2-7B, mesure le 10/09 :
        #
        #   genre        n   cout de promotion par tenseur
        #   lm_head      1   57,62 Mio
        #   down_proj   24   19,82 Mio
        #   gate/up      2   19,82 Mio
        #   q k v o     76    7,38 Mio     <- les quatre au meme cout
        #
        #   max de X = 7,38 Mio, min de Y = 19,82 Mio, facteur 2,69, AUCUN
        #   recouvrement.
        #
        # Sur ce modele, « le genre du tenseur » et « le tenseur le plus gros »
        # designent exactement le meme ensemble : les projections d attention
        # sont 4096x4096, le MLP est 11008x4096. On ne peut PAS trancher entre
        # les deux explications ici. Les deux cles existent donc en paire :
        # elles doivent rendre le MEME ensemble de promus sur Llama-2, et
        # divergerent sur un modele ou une projection d attention est aussi
        # grosse qu une projection de MLP (GQA, MLA, tete non liee). Leur
        # ecart, quand il apparaitra, sera la mesure du confondant.
        _GENRE_PRIORITE = {
            # plus petit = promu plus tot. `lm_head` sort dans les logits et
            # `down_proj` ecrit dans le residuel : leur erreur ne traverse
            # aucune normalisation qui l absorbe. HYPOTHESE de chef, pas
            # un fait — et elle predit que `o_proj`, qui ecrit AUSSI dans le
            # residuel, devrait suivre `down_proj`. Il est dans X, du cote qui
            # perd. La prediction est donc en attente d un bras o_proj contre
            # v_proj : meme cout, meme famille, position residuelle opposee.
            "lm_head": 0, "down_proj": 1, "gate_proj": 2, "up_proj": 2,
            "o_proj": 3, "v_proj": 4, "q_proj": 5, "k_proj": 5,
        }

        def _genre(nom):
            if nom.endswith("lm_head.weight") or nom == "lm_head.weight":
                return "lm_head"
            parties = nom.split(".")
            return parties[-2] if len(parties) > 2 else nom

        def _cle(c):
            if _mode == "genre":
                # rang du genre, puis cout decroissant a genre egal : un genre
                # inconnu part en DERNIER (99) au lieu de lever, parce que le
                # sac a dos doit rester utilisable sur une architecture que
                # cette table ne connait pas. Le manifeste porte le compte des
                # genres inconnus rencontres, sinon l ignorance serait muette.
                g = _genre(c["name"])
                _genres_vus[g] = _genres_vus.get(g, 0) + 1
                return (_GENRE_PRIORITE.get(g, 99), -c["cout"])
            if _mode == "cout_decroissant":
                # le plus gros d abord, sans un seul decibel. Jumelle de
                # `genre` sur Llama-2 par construction du modele.
                return -c["cout"]
            if _mode == "base_croissant":
                # BRAS TEMOIN, construit AVANT la mesure qu'il doit departager.
                #
                # Si une cle raffinee ameliore la perplexite, deux explications
                # restent en lice : la cle est un meilleur critere, ou bien
                # elle promeut simplement les tenseurs les plus mal quantifies.
                # Ce bras isole la seconde : il trie par SNR de base croissant
                # en IGNORANT le cout, donc il fait « les mal quantifies
                # d'abord » et rien de plus.
                #
                # Nomme par chef avant la mesure, et construit tout de suite
                # sur son insistance : un bras nomme mais non construit
                # s'expose a etre ecrit APRES avoir vu le resultat, et un bras
                # qui existe avant la mesure ne peut pas etre ajuste par elle.
                return c["metrics_base"]["out_snr_db"]
            if _mode == "absolu":
                gagne = (_abs_err(c["metrics_base"]) - _abs_err(c["metrics"]))
                return -gagne / c["cout"]
            if _mode == "erreur":
                gagne = (10.0 ** (-(c["metrics_base"]["out_snr_db"]) / 20.0)
                         - 10.0 ** (-(c["metrics"]["out_snr_db"]) / 20.0))
                return -gagne / c["cout"]
            if _mode == "kld":
                # Protocole A/B (revue/protocole-ab-kld-vs-snr.md) : meme
                # glouton, cle differente. `gain_kld_bits` est deja oriente
                # (base moins large, positif si le format large divergence
                # moins) ; absent seulement si un candidat a echappe a la
                # mesure KLD (mesurer_kld actif mais tenseur non recandidat au
                # tri budgetaire) — traite comme un gain nul, donc promu en
                # dernier plutot que de faire lever tout le tri sur un seul
                # candidat incomplet.
                return -c.get("gain_kld_bits", 0.0) / c["cout"]
            signe = 1.0 if _mode == "inverse" else -1.0
            return signe * c["gain_db"] / c["cout"]

        # LA LISTE EXPLICITE COURT-CIRCUITE L'ORDRE ET LE BUDGET.
        _liste_noms, _liste_sha = None, None
        if _mode == "liste":
            import hashlib
            with open(_liste_chemin, "rb") as fh:
                _brut = fh.read()
            _liste_sha = hashlib.sha256(_brut).hexdigest()
            _txt = _brut.decode("utf-8")
            try:
                _charge = json.loads(_txt)
            except json.JSONDecodeError:
                _charge = [l.strip() for l in _txt.splitlines() if l.strip()]
            if isinstance(_charge, dict):
                # un fichier de groupes porte plusieurs listes ; il faut dire
                # laquelle, sinon le choix serait fait par l'ordre des cles
                _cle_liste = os.environ.get("ACVRAM_LISTE_CLE", "").strip()
                if _cle_liste not in _charge:
                    raise ValueError(
                        f"{_liste_chemin} contient plusieurs listes "
                        f"({sorted(_charge)}) ; poser ACVRAM_LISTE_CLE pour "
                        f"dire laquelle. Sans elle le bras serait choisi par "
                        f"l'ordre des cles du fichier.")
                _charge = _charge[_cle_liste]
                _liste_sha = hashlib.sha256(
                    (_liste_sha + ":" + _cle_liste).encode()).hexdigest()
            _liste_noms = list(dict.fromkeys(_charge))
            if not _liste_noms:
                raise ValueError(f"{_liste_chemin} ne contient aucun nom.")
            # TROISIEME GARDE (chef) : une liste figee qui se desynchronise
            # du parc promouvrait moins que prevu EN SILENCE, et le compte de
            # promus ne le dirait pas — on attendrait 76 et on en aurait 74
            # sans savoir lesquels. Deux absences distinctes, deux messages :
            # absent du modele, ou present mais non promouvable.
            _dispo = {c["name"] for c in budget_candidats}
            _hors = [n for n in _liste_noms if n not in _dispo]
            if _hors:
                raise ValueError(
                    f"REFUS : {len(_hors)} des {len(_liste_noms)} noms de "
                    f"{_liste_chemin} ne sont pas des candidats promouvables "
                    f"de ce modele, dont {_hors[:3]}. Une liste qui ne "
                    f"s'applique plus au parc promouvrait moins que demande "
                    f"sans que le compte de promus le dise.")
            ordre = [c for c in budget_candidats if c["name"] in set(_liste_noms)]
        else:
            ordre = sorted(budget_candidats, key=_cle)
        # Le budget est un budget de DOSSIER : `deja` compte le plancher, c'est
        # a dire tout ce qui n'est pas promouvable (part 16 bits, echelles
        # d'activation) plus chaque candidat dans son format de base. Deux
        # issues muettes existaient, et toutes deux fabriquent un faux point de
        # courbe : un budget SOUS le plancher promeut zero tenseur et rend un
        # dossier identique a une conversion sans budget, et un budget qui
        # reste inemploye rend un dossier moins large que demande. Dans les
        # deux cas le dossier porte un budget dans son nom et une autre
        # grandeur dans ses octets. On l'annonce, et on l'ecrit au manifeste.
        plancher_octets = deja
        budget_octets = opts.bits_budget_gib * 1024 ** 3
        promus = set()
        # ACVRAM_MAX_PROMUS=N : plafond de COMPTE, budget ignore — promeut
        # exactement les N premiers de l'ordre. Sert l'experience « compte
        # egal » (11/09) : a 149 fixe, chaque ordre choisit SES 149, ce qui
        # isole « quels tenseurs » de « combien ». Absent ou 0 -> remplissage
        # au budget, comportement normal.
        _max_promus = int(os.environ.get("ACVRAM_MAX_PROMUS", "0") or "0")
        if _max_promus > 0:
            if _max_promus > len(ordre):
                raise ValueError(
                    f"ACVRAM_MAX_PROMUS={_max_promus} > {len(ordre)} candidats.")
            for c in ordre[:_max_promus]:
                promus.add(c["name"])
                reste -= c["cout"]
        else:
            for c in ordre:
                if _mode == "liste":
                    # budget IGNORE : definition du mode. Cout retire pour que
                    # depense_gib reste vrai ; peut passer negatif (annonce plus bas).
                    promus.add(c["name"])
                    reste -= c["cout"]
                elif c["cout"] <= reste:
                    promus.add(c["name"])
                    reste -= c["cout"]
        cout_total = sum(c["cout"] for c in budget_candidats)
        if _mode == "liste":
            pass  # ni plancher ni budget : les deux messages seraient faux
        elif reste < 0:
            print(f"[acvram] budget de {opts.bits_budget_gib:.3f} Gio SOUS le "
                  f"plancher de {plancher_octets / 1024 ** 3:.3f} Gio : aucune "
                  f"promotion possible, le dossier sortira au format de base. "
                  f"Le plus petit budget qui promeut quelque chose est "
                  f"{(plancher_octets + min(c['cout'] for c in budget_candidats)) / 1024 ** 3:.3f} Gio.",
                  flush=True)
        elif promus and reste > 0 and len(promus) < len(budget_candidats):
            print(f"[acvram] budget non epuise : {reste / 2 ** 20:.1f} Mio "
                  f"inemployes, aucun candidat restant ne tient dedans "
                  f"({len(promus)}/{len(budget_candidats)} promus).", flush=True)
        manifest["budget"] = {
            # Le sens de l'ordre est ECRIT au manifeste : un dossier produit
            # par le bras inverse doit etre reconnaissable sans son journal.
            # Un mode absent de cette table levait un KeyError APRES toute la
            # conversion — des minutes de calcul perdues sur une faute de
            # frappe, et pire : le mode `absolu` ajoute plus haut n'y figurait
            # pas. La table doit couvrir exactement les modes acceptes par la
            # garde de _mode ; un repli explicite vaut mieux qu'une exception
            # tardive, et il porte le nom du mode pour rester lisible.
            "ordre_glouton": {
                "base_croissant": "snr_de_base_croissant_sans_cout",
                "snr": "snr_par_octet_decroissant",
                "erreur": "erreur_evitee_par_octet_decroissante",
                "absolu": "erreur_absolue_evitee_par_octet_decroissante",
                "inverse": "snr_par_octet_croissant",
                "liste": "liste_explicite_sans_ordre_ni_budget",
                "genre": "genre_du_tenseur_DERIVE_de_l_observation",
                "cout_decroissant": "cout_decroissant_sans_decibel",
                "kld": "kld_couche_par_octet_decroissant",
            }.get(_mode, f"mode_{_mode}_sans_description"),
            "demande_gib": opts.bits_budget_gib,
            "plancher_gib": round(plancher_octets / 1024 ** 3, 4),
            "plafond_gib": round((plancher_octets + cout_total) / 1024 ** 3, 4),
            "depense_gib": round((budget_octets - max(reste, 0)) / 1024 ** 3, 4)
                           if reste >= 0 else round(plancher_octets / 1024 ** 3, 4),
            "restant_mio": round(max(reste, 0) / 2 ** 20, 1),
            "sous_le_plancher": reste < 0,
            "promus": len(promus),
            "candidats": len(budget_candidats),
            "max_promus_impose": _max_promus or None,
            # TEMOIN D'ORDRE. Sans lui, l'ordre du glouton n'est pas
            # observable depuis le dossier : le seul effet visible est
            # l'ensemble des promus, et cet ensemble ne suffit pas a le
            # reconstituer — le glouton SAUTE un candidat trop gros pour le
            # reste du budget et en prend un moins cher ensuite, donc les
            # promus ne sont pas un prefixe de l'ordre. Un test qui verifiait
            # « aucun promu de priorite pire qu'un recale » echouait pour
            # cette raison, sur un tri pourtant correct.
            "ordre_20_premiers": [c["name"] for c in ordre[:20]],
            "cout_20_premiers_mio": [round(c["cout"] / 2 ** 20, 2)
                                     for c in ordre[:20]],
        }
        if _mode == "genre":
            inconnus = {g: n for g, n in _genres_vus.items()
                        if g not in _GENRE_PRIORITE}
            manifest["budget"].update({
                "genre_derive_de_l_observation": True,
                "genres_vus": dict(sorted(_genres_vus.items())),
                "genres_inconnus": inconnus,
            })
            if inconnus:
                print(f"[acvram] ordre `genre` : {sum(inconnus.values())} "
                      f"tenseurs de genre inconnu promus en dernier "
                      f"({sorted(inconnus)}). La table de priorite a ete "
                      f"ecrite sur Llama-2 ; sur cette architecture elle est "
                      f"incomplete.", flush=True)
        if _mode == "liste":
            # SANS LE SHA, DEUX BRAS NOMMES X NE SONT PAS COMPARABLES. Le
            # dossier doit porter la trace de ce qui l'a produit : nous avons
            # perdu une manche le 10/09 parce qu'un dossier ne la portait pas.
            manifest["budget"].update({
                "liste_chemin": _liste_chemin,
                "liste_cle": os.environ.get("ACVRAM_LISTE_CLE", "") or None,
                "liste_sha256": _liste_sha,
                "liste_noms": len(_liste_noms),
                "budget_ignore": True,
            })
            if len(promus) != len(_liste_noms):
                raise AssertionError(
                    f"{len(promus)} promus pour {len(_liste_noms)} noms "
                    f"demandes — la garde de liste aurait du lever avant.")
            print(f"[acvram] mode liste : {len(promus)} tenseurs promus, "
                  f"budget ignore, liste sha256 {_liste_sha[:12]}", flush=True)
        for c in budget_candidats:
            name = c["name"]
            large = name in promus
            sd = c["sd"] if large else c["sd_base"]
            met = c["metrics"] if large else c["metrics_base"]
            fmt = c["to"] if large else c["fmt_base"]
            entry = c["entry"]
            snrs.append(met["out_snr_db"])
            if "ratio_norme" in met:
                r = met["ratio_norme"]
                if not (RATIO_NORME_BAS <= r <= RATIO_NORME_HAUT):
                    ratios_hors_bornes.append((name, r))
            per_layer.append({"name": name,
                              **{k: round(v, 2) if isinstance(v, float) else v
                                 for k, v in met.items()
                                 if k != "erreurs_grille"}})
            if not opts.dry_run:
                for k, v in sd.items():
                    writer.add(k, v)
            entry.update({
                "format": fmt,
                "keys": list(sd.keys()),
                "group_size": opts.group_size,
                "hadamard_block": c["hadamard_block"],
                "has_act_scale": c["has_act_scale"],
                "bpw": round(met["bpw"], 3),
                "out_snr_db": round(met["out_snr_db"], 2),
                **({"ratio_norme": round(met["ratio_norme"], 4)}
                   if "ratio_norme" in met else {}),
                # DEUXIEME SITE D'ECRITURE, et c'est celui que prend le SAC A
                # DOS. `out_ref_norm` n'etait ajoute qu'au site 1065, sur le
                # chemin du plancher SNR — donc une conversion BUDGETAIRE ne
                # portait jamais l'echelle. Constate le 10/09 : 0 tenseur sur
                # 323 dans un dossier converti APRES l'ajout du champ, et
                # c'est le garde de sortie du bras qui l'a dit, pas une
                # relecture. Meme classe de defaut que `erreurs_grille`, qu'il
                # avait fallu ajouter aux DEUX sites.
                **({"out_ref_norm": round(float(met["out_ref_norm"]), 6)}
                   if "out_ref_norm" in met else {}),
                **({"erreurs_grille": met["erreurs_grille"],
                    "alpha_retenu": met["alpha_retenu"]}
                   if "erreurs_grille" in met else {}),
            })
            octets = sum(v.numel() * v.element_size() for v in sd.values())
            report.per_format[fmt] = report.per_format.get(fmt, 0) + octets
            manifest["tensors"][name] = entry
            if large:
                report.promotions.append({
                    "name": name, "from": c["from"], "to": c["to"],
                    "before": round(c["metrics_base"]["out_snr_db"], 2),
                    "after": round(met["out_snr_db"], 2)})

    # Un point de contrôle dont l'architecture n'est pas vraiment comprise
    # (noms de tenseurs non traduits) produirait un modèle mutilé qui échoue
    # au chargement — ou pire, qui répond du charabia. Vérifier que chaque
    # couche attendue par la spécification a bien ses projections.
    attendus = []
    for i in range(spec.num_layers):
        if spec.layer_types and spec.layer_types[i] == "conv":
            attendus.append(f"model.layers.{i}.conv.in_proj.weight")
        elif spec.layer_types and spec.layer_types[i] == "parallel":
            attendus.append(f"model.layers.{i}.mamba.in_proj.weight")
            attendus.append(f"model.layers.{i}.self_attn.q_proj.weight")
            attendus.append(f"model.layers.{i}.mlp.gate_proj.weight")
            continue
        elif spec.layer_types and spec.layer_types[i] in ("mamba", "mlp", "moe"):
            attendus.append(f"model.layers.{i}." + {
                "mamba": "mamba.in_proj.weight", "mlp": "mlp.up_proj.weight",
                "moe": "mlp.experts.0.up_proj.weight"}[spec.layer_types[i]])
            continue
        elif spec.layer_types and spec.layer_types[i] == "linear_attention":
            if spec.model_type == "kimi_linear":
                attendus.append(f"model.layers.{i}.linear_attn.q_proj.weight")
                attendus.append(f"model.layers.{i}.linear_attn.conv1d_q.weight")
            else:
                attendus.append(f"model.layers.{i}.linear_attn.qkv.weight")
                attendus.append(f"model.layers.{i}.linear_attn.conv1d.weight")
        elif spec.q_lora_rank:
            attendus.append(f"model.layers.{i}.self_attn.q_a_proj.weight")
        else:
            attendus.append(f"model.layers.{i}.self_attn.q_proj.weight")
        if spec.model_type == "nemotron_h":
            continue                        # couche d'attention seule, sans MLP
        if not spec.mlp_gated:
            attendus.append(f"model.layers.{i}.mlp.up_proj.weight")
            continue
        if spec.is_moe and i >= spec.first_k_dense_replace:
            attendus.append(f"model.layers.{i}.mlp.experts.0.gate_proj.weight")
        else:
            attendus.append(f"model.layers.{i}.mlp.gate_proj.weight")
    manquants = [n for n in attendus if n not in manifest["tensors"]]
    if manquants:
        raise ValueError(
            f"conversion incomplète : {len(manquants)} tenseurs attendus "
            f"absents (premier : {manquants[0]}). L'architecture de la source "
            f"n'est probablement pas prise en charge — rien n'est écrit.")

    # MEME denominateur que la condition l.966 — `keys`, pas `attendus`.
    # Les deux listes ne recensent pas la meme chose et le plafond calcule sur
    # la mauvaise donnerait un seuil de saturation faux.
    plafond = opts.max_promotions * max(1, len(keys) + 1)
    if report.promotions and len(report.promotions) >= plafond - 1:
        # QUOTA SATURE. A partir de cet instant, ce n'est plus le SNR qui
        # decide d'une promotion mais l'ORDRE DE PARCOURS du checkpoint : deux
        # tenseurs de SNR identique recoivent des sorts opposes selon leur
        # position. Mesure le 9/09/2026 : 27 modeles du parc sur 110 saturent
        # a l'unite pres, et sur l'un d'eux AUCUN des 48 groupes q/k/v n'a ses
        # trois membres promus quand 31 en ont exactement un — la signature
        # d'un regulateur de debit, pas d'une difficulte de couche.
        print(f"[acvram] quota de promotions SATURE : {len(report.promotions)} "
              f"sur un plafond de {plafond:.0f}. Au-dela du plafond, l'ordre de "
              f"parcours a decide a la place du SNR — les promotions ne sont "
              f"plus triees par besoin. Relever --max-promotions ou trier en "
              f"deux passes.", flush=True)
        manifest["quota_promotions_sature"] = True

    if observations:
        manifest["observations_experts"] = observations          # pièce 32 : ce que le corpus a réellement vu
    if report.experts_sans_stats:
        manifest["experts_sans_stats"] = report.experts_sans_stats
        # la LISTE (pièce 25 (c) : l instrument KL cible ces experts), compacte
        # par couche et projection : {"3": {"gate_proj": [7, 12], ...}}
        manifest["experts_sans_stats_liste"] = _experts_par_couche(report.experts_sans_stats_noms)
        manifest["experts_repli"] = opts.repli_experts
        if report.experts_repli_mediane:
            manifest["experts_repli_mediane"] = report.experts_repli_mediane
        print(f"[acvram] {report.experts_sans_stats} expert(s) sans statistique "
             f"d'activation a la calibration (jamais routes ou trop peu sur le "
             f"corpus) — mesure du corpus de calibration. Chacun recoit une "
             f"echelle identite EXPLICITE dans le manifeste (jamais une "
             f"absence ambigue).",
             flush=True)

    ratios_tous = [e["ratio_norme"] for e in manifest["tensors"].values()
                  if "ratio_norme" in e]
    if ratios_tous:
        manifest["ratio_norme_min"] = round(min(ratios_tous), 4)
        manifest["ratio_norme_max"] = round(max(ratios_tous), 4)
    if tenseurs_replies:
        # Plafond AVANT le repli, sur le total de tenseurs qui ont eu une
        # recherche AWQ (denominateur `ratios_tous`, pas seulement les
        # experts) : au-dela de la moitie, ce n'est plus une reparation
        # ciblee, la calibration entiere est a refaire.
        part = len(tenseurs_replies) / max(1, len(ratios_tous))
        if part > PLAFOND_REPLI:
            pires = sorted(tenseurs_replies, key=lambda x: abs(x[1] - 1.0), reverse=True)
            detail = "; ".join(f"{n} : {r:.3f}" for n, r in pires[:10])
            raise ValueError(
                f"conversion refusée : {len(tenseurs_replies)}/{len(ratios_tous)} "
                f"tenseurs ({part:.0%}) hors des bornes de ratio de norme "
                f"[{RATIO_NORME_BAS},{RATIO_NORME_HAUT}] même après repli à "
                f"l'identité — au-delà de {PLAFOND_REPLI:.0%}, ce n'est plus "
                f"une réparation ciblée, la calibration est à refaire "
                f"entièrement. Pires : {detail}. "
                f"poste7-awq-relu2-garde-repli-17-09.")
        report.tenseurs_replies = [n for n, _ in tenseurs_replies]
        manifest["tenseurs_replies_identite"] = {
            "nombre": len(tenseurs_replies),
            "part": round(part, 4),
            "noms": report.tenseurs_replies,
        }
        print(f"[acvram] {len(tenseurs_replies)}/{len(ratios_tous)} tenseur(s) "
             f"repliés à l'identité (ratio de norme hors [{RATIO_NORME_BAS},"
             f"{RATIO_NORME_HAUT}] avec AWQ, cause : entrée creuse ReLU² sur "
             f"nemotron_h — poste7-awq-relu2-garde-repli-17-09).", flush=True)
    if ratios_hors_bornes:
        pires = sorted(ratios_hors_bornes, key=lambda x: abs(x[1] - 1.0), reverse=True)
        detail = "; ".join(f"{n} : {r:.3f}" for n, r in pires[:10])
        raise ValueError(
            f"conversion refusée : {len(ratios_hors_bornes)} tenseur(s) "
            f"hors des bornes de ratio de norme [{RATIO_NORME_BAS},"
            f"{RATIO_NORME_HAUT}] — {detail}. Une quantification saine reste "
            f"proche de 1 (mesure : 0,944-1,012 sur 340 experts sains) ; "
            f"poste7-awq-experts-peu-routes-portee-17-09, motif Nemotron "
            f"17/09 (échelle AWQ mal réglée par une statistique bruitée).")

    _verifier_homogeneite_moe(manifest["tensors"], spec.num_layers)
    manifest.update(_bilan_attn_int8(opts, manifest["tensors"]))     # règle 6 : l'étiquette suit les faits
    manifest["diagnostic_fusion"] = _diagnostic_fusion(manifest["tensors"])
    manifest["vision_bytes"] = vision_bytes
    manifest["vision"] = "oui" if vision_bytes else "non"
    if vision_ecartee:
        manifest["vision_ecartee"] = vision_ecartee
        print(f"[acvram] --sans-vision : {vision_ecartee} tenseur(s) de la tour écartés, alias texte seul", flush=True)
    if vision_bytes:
        _manifeste_multimodal(manifest, spec)
    if _AUDIO_ECARTES:
        manifest["audio"] = "non servi"
        print(f"[acvram] audio non servi : {len(_AUDIO_ECARTES)} tenseur(s) de la tour audio écartés "
              f"(ex. {_AUDIO_ECARTES[0]}) — alias texte + vision", flush=True)
        _AUDIO_ECARTES.clear()
    if not opts.dry_run:
        _verifier_formats_declares(manifest, writer.weight_map)

    if not opts.dry_run:
        writer.flush()
        manifest["weight_map"] = writer.weight_map
        with open(os.path.join(opts.out_dir, "acvram_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        _copy_tokenizer(model_path, opts.out_dir)
        from .gguf import GGUFFile, is_gguf
        if is_gguf(model_path) and not opts.dry_run:
            GGUFFile(model_path).export_sidecars(opts.out_dir)
        from .exl3 import EXL3Checkpoint, is_exl3
        if is_exl3(model_path) and not opts.dry_run:
            EXL3Checkpoint(model_path).export_sidecars(opts.out_dir)
        report.out_bytes = writer.total_bytes
    else:
        report.out_bytes = sum(report.per_format.values())

    report.mean_out_snr_db = sum(snrs) / len(snrs) if snrs else 0.0
    report.worst_layers = sorted(per_layer, key=lambda d: d["out_snr_db"])
    report.seconds = time.time() - t0
    if (report.in_bytes and report.out_bytes > report.in_bytes * 1.02
            and not opts.autoriser_grossissement):
        raise ValueError(
            f"issue de conversion : {report.out_bytes / 2**30:.1f} Gio écrits "
            f"pour {report.in_bytes / 2**30:.1f} Gio de source — la garde "
            f"d'intention a été contournée quelque part ; le dossier est "
            f"conservé pour inspection mais NE DOIT PAS être servi.")
    return report


_TEKKEN_CHAT = (
    "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'system' %}"
    "[SYSTEM_PROMPT]{{ m['content'] }}[/SYSTEM_PROMPT]{% elif m['role'] == 'user' %}"
    "[INST]{{ m['content'] }}[/INST]{% else %}{{ m['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}")


def _copy_tokenizer(src: str, dst: str) -> None:
    import shutil
    for fn in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
               "special_tokens_map.json", "generation_config.json", "config.json",
               "chat_template.jinja",
               # Qwen3-VL (2B, 20/09) : le gabarit vit dans chat_template.json (format transformers ≤ 4) ; sans lui
               # AutoProcessor.apply_chat_template refuse (« does not have a chat template ») et l'API image ne rend rien
               "chat_template.json", "added_tokens.json", "merges.txt", "vocab.json",
               # multimodal : le processeur d'images accompagne la tour
               "processor_config.json", "preprocessor_config.json",
               "video_preprocessor_config.json"):
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, fn))
    tekken = os.path.join(src, "tekken.json")
    if os.path.isfile(tekken) and not os.path.isfile(os.path.join(src, "tokenizer.json")):
        # Mistral « tekken » seul (Devstral, Small 3.x) : tokenizer HF
        # reconstruit, gabarit v7-tekken ([SYSTEM_PROMPT]/[INST])
        from transformers.integrations.mistral import convert_tekken_tokenizer
        tok = convert_tekken_tokenizer(tekken)
        tok.chat_template = _TEKKEN_CHAT
        tok.save_pretrained(dst)
        try:                                    # regex de pré-tokenisation corrigée
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(dst, fix_mistral_regex=True).save_pretrained(dst)
        except TypeError:
            pass
        print("  tokenizer reconstruit depuis tekken.json")
