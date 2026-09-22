"""Planificateur de placement : quelle couche vit où, et dans quel format.

La machine offre trois étages de mémoire aux caractéristiques très éloignées :

    étage         taille   bande passante   format
    -----------   ------   --------------   ---------
    RTX 5090       32 Go     ~1790 Go/s     NVFP4  (4,50 bits/poids)
    RTX 3080 Ti    12 Go      ~912 Go/s     INT4   (4,16 bits/poids)
    DDR5 hôte      96 Go     ~70 Go/s en local, mais limitée à la vitesse
                             du lien PCIe dès qu'on transfère

Décoder un seul jeton est limité par la mémoire : le temps de produire un jeton
est essentiellement le temps de lire une fois chaque poids actif. Le travail du
planificateur est donc de minimiser le temps de lecture total sous la contrainte
de capacité de chaque étage, et l'asymétrie décisive est qu'une couche
*transférée* est limitée par son lien PCIe et non par la bande passante de la
VRAM — environ 55 Go/s sur un port Gen5 x16, et seulement 7 Go/s sur un port
Gen4 x4 relié au chipset.

Deux faits structurels guident chaque décision :

* Une couche à mélange d'experts creux ne lit que ``top_k`` experts par jeton.
  Un modèle MoE de 235 milliards de paramètres dont 8 experts sur 128 sont
  actifs lit environ 5 % de ses poids : la mémoire vive devient alors un
  emplacement parfaitement raisonnable pour les experts, et c'est ce qui rend
  96 Go de DDR5 plus précieux qu'il n'y paraît.
* Les cartes GeForce n'ont pas de NVLink et NVIDIA y désactive le pair-à-pair
  PCIe : un passage d'un GPU à l'autre transite donc par la mémoire hôte
  épinglée. Ce passage ne transporte qu'un vecteur d'état caché par jeton,
  quelques kilooctets, donc il est bon marché — mais cela impose que le pipeline
  ne franchisse qu'une seule fois la frontière entre les cartes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..engine.config import ModelSpec
from ..hardware.detect import Rig
from ..quant.formats import bits_per_weight
from . import kv_lm4

# Format du cache KV des paliers carte : vide = celui des capacités détectées
# (int8 sur Ampere+). `lm4` : 4 bits par rotation (kv_lm4), témoins lm3/lm2.
_KV_FORMAT = os.environ.get("ACVRAM_KV_FORMAT", "").lower()
if _KV_FORMAT and _KV_FORMAT not in ("int8", "fp8_e4m3", "fp16", "bf16", *kv_lm4.FORMATS):
    raise ValueError(f"ACVRAM_KV_FORMAT={_KV_FORMAT!r} : attendu int8, fp8_e4m3, fp16, bf16, lm4, lm3 ou lm2")


__all__ = ["Tier", "LayerPlacement", "Plan", "PlannerOptions", "plan_placement"]

MB = 1024 ** 2
GB = 1024 ** 3

# Surcoût par GPU que l'on ne peut pas allouer : contexte CUDA, tampons de
# travail de cuBLAS et cuDNN, comptabilité propre de l'allocateur, et ce que
# retient le serveur d'affichage.
CUDA_CONTEXT_RESERVE = 800 * MB
FRAGMENTATION_MARGIN = 0.03

# Débits denses de plaque signalétique, utilisés seulement pour classer les
# options de prefill. Ce sont des estimations, pas des mesures : `acvram bench`
# les remplace par de vrais nombres.
TFLOPS = {
    120: {"nvfp4": 838.0, "fp8": 419.0, "bf16": 209.0, "fp16": 209.0},
    89:  {"fp8": 660.0, "bf16": 330.0, "fp16": 330.0},
    86:  {"int8": 136.0, "fp16": 68.0, "bf16": 68.0},
    80:  {"int8": 125.0, "fp16": 62.0, "bf16": 62.0},
}


@dataclass
class Tier:
    name: str                    # "cuda:0", "cuda:1", "cpu"
    kind: str                    # "gpu" | "host"
    device_index: int
    capacity: int                # usable bytes for weights, after reserves
    weight_format: str
    kv_format: str
    read_bandwidth: float        # GB/s when the weights are resident here
    link_bandwidth: float        # GB/s host -> this device
    sm: int = 0
    # Limite de puissance de la carte, en watts, telle que le pilote la
    # declare. Zero quand elle est inconnue : le calcul d'energie s'abstient
    # alors plutot que de supposer.
    power_limit_w: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayerPlacement:
    """Où vit un bloc de transformeur.

    L'attention et le MLP sont placés indépendamment. Cette distinction est tout
    l'intérêt pour un MoE creux : le bloc d'attention d'une couche de
    Qwen3-235B compte environ 71 millions de paramètres quand ses 128 experts en
    comptent 2,4 milliards. Épingler l'attention en VRAM ne coûte donc presque
    rien et garde le chemin critique de la latence hors du bus PCIe, tandis que
    les experts — dont 8 seulement sont lus par jeton — résident en mémoire vive.
    """

    index: int
    exec_device: str             # where the maths runs
    attn_storage: str            # exec_device or "cpu"
    mlp_storage: str             # exec_device or "cpu"
    fmt: str
    attn_bytes: int
    mlp_bytes: int
    mlp_active_bytes: int        # what one token actually reads from the MLP
    is_moe: bool = False
    cached_expert_fraction: float = 0.0
    mlp_exec: str = "gpu"        # "gpu" (stream the weights in) or "cpu"
    # Placement PAR EXPERT (bead anticitoyen-vram-pds, Sage §4, revue/
    # sage-cache-experts-13-09.md) : la couche entière comme unité de
    # placement (mlp_storage ci-dessus) reste le comportement par défaut ;
    # ces deux champs sont le premier pas vers un grain plus fin, ADDITIFS et
    # sans effet tant qu'ils sont None — aucun code existant ne les lit.
    #   experts_residents : combien des experts de cette couche vivent en
    #     VRAM (les autres, en RAM épinglée zéro-copie) — None = pas encore
    #     décidé à ce grain, `cached_expert_fraction` reste la seule borne.
    #   taux_succes : le h_pin/h_lru MESURÉ pour ce nombre de résidents
    #     (`memory/trace_routage.taux_de_succes_pin`) — None = non mesuré, et
    #     alors le plan garde `experts_residents / n_experts` comme borne
    #     HAUTE (un routage uniforme ne fait pas mieux), jamais une valeur
    #     inventée qui se ferait passer pour une mesure.
    experts_residents: Optional[int] = None
    taux_succes: Optional[float] = None

    @property
    def total_bytes(self) -> int:
        return self.attn_bytes + self.mlp_bytes

    @property
    def streamed(self) -> bool:
        return self.attn_storage == "cpu" or self.mlp_storage == "cpu"

    @property
    def resident_bytes(self) -> int:
        n = 0
        if self.attn_storage != "cpu":
            n += self.attn_bytes
        if self.mlp_storage != "cpu":
            n += self.mlp_bytes
        return n

    def to_dict(self) -> dict:
        d = asdict(self)
        d["streamed"] = self.streamed
        return d


@dataclass
class PlannerOptions:
    max_model_len: int = 8192
    max_concurrent_seqs: int = 8
    kv_bits: int = 8
    reserve_per_gpu: int = CUDA_CONTEXT_RESERVE
    allow_host_tier: bool = True
    host_fraction: float = 0.85           # of MemAvailable we are willing to use
    kv_vram_fraction: float = 0.35        # ceiling on VRAM spent on KV cache
    expert_cache_fraction: float = 0.5    # of leftover VRAM, for hot MoE experts
    force_format: Optional[str] = None
    group_size: int = 128
    pin_attention: bool = True            # keep attention off the host tier
    gpus: Optional[str] = None            # "auto" | "all" | "0" | "0,1"
    host_exec: str = "auto"               # auto | stream | cpu
    # Coût fixe d'un transfert d'expert vers la carte, en microsecondes.
    #
    # NON MESURÉ. Zéro par défaut : le modèle reste alors celui d'avant, faux
    # mais connu — il est optimiste d'un facteur 2,4 à 3,4 avec un plancher
    # d'environ 3 ms par jeton qu'aucun terme en octets ne peut produire.
    # Une constante inventée le rendrait faux ET crédible, ce qui est pire.
    #
    # Pour la mesurer : chronométrer un transfert d'expert sur trois tailles
    # écartées d'un facteur quatre. L'ordonnée à l'origine est cette valeur ;
    # la pente doit retrouver la bande passante du lien. Si l'ordonnée est
    # nulle, le plancher vient d'ailleurs — réveil de carte ou synchronisation
    # par couche — et ce terme n'est pas le bon.
    transfer_fixed_us: float = 0.0
    # --- second objectif : les joules -------------------------------------
    #
    # Le planificateur optimise des SECONDES. L'objectif pose est double :
    # vitesse ET jetons par kilojoule. Un plan peut etre plus rapide et plus
    # cher en energie — c'est le cas du chemin processeur, mesure le
    # 8 septembre 2026 a 183 s de temps processeur pour 200 jetons contre 25 s
    # par le bus, a debit egal a 5 % pres. Sept fois plus, et invisible pour le
    # compteur de la carte, qui ne voit que ce qu'elle consomme elle-meme.
    #
    # Les trois valeurs ci-dessous sont a ZERO : sans elles le calcul d'energie
    # s'abstient et rend None, ce qui est honnete. Une valeur inventee rendrait
    # un chiffre de joules credible et faux, et le planificateur choisirait
    # dessus.
    #
    # RESERVE SUR LA MESURE QUI LES MOTIVE, a lever avant de les renseigner :
    # les 183 s sont utime+stime du processus, donc du temps processeur
    # reellement consomme — mais une ATTENTE ACTIVE y compte en plein alors
    # qu'elle consomme peu. Si le chemin processeur attend le bus en tournant,
    # une part de ces 183 s n'est pas de l'energie. Comparer utime et stime
    # separement, ou passer au wattmetre de prise, tranche la question. Tant
    # qu'elle ne l'est pas, on ne sait pas si le terme doit compter des
    # secondes de calcul ou des octets deplaces.
    watts_processeur: float = 0.0     # a la charge, tous cœurs occupes
    watts_carte_au_repos: float = 0.0  # carte alimentee, sans calcul
    # Part de la limite de puissance reellement atteinte pendant un decodage.
    # Le decodage est borne par la memoire, pas par le calcul : une carte n'y
    # tire jamais sa limite. Zero = inconnu, donc pas de calcul.
    fraction_puissance_decodage: float = 0.0
    host_compute_gb_s: float = 70.0       # DDR5 streaming reads, measured by bench
    # Débit RÉEL d'un produit de matrices déquantifiant sur processeur. Ce
    # n'est PAS le débit de lecture ci-dessus : un GEMM NVFP4 doit déballer de
    # l'E2M1 sans instruction native et reconstruire les échelles par bloc,
    # ce que ne mesure aucune lecture séquentielle. None = jamais mesuré, et
    # dans ce cas on refuse de choisir le chemin processeur (voir _placer).
    host_gemm_gb_s: Optional[float] = None


@dataclass
class Plan:
    model: str
    tiers: list[Tier] = field(default_factory=list)
    layers: list[LayerPlacement] = field(default_factory=list)
    embed_device: str = "cpu"
    lm_head_device: str = "cuda:0"
    kv_bytes_per_token: int = 0
    kv_budget: dict[str, int] = field(default_factory=dict)
    kv_max_tokens: int = 0
    # Etat recurrent des couches lineaires : sur la carte, une copie par
    # sequence, et budgete nulle part avant le 9/09/2026.
    etat_recurrent_bytes: int = 0
    expert_cache_bytes: dict[str, int] = field(default_factory=dict)
    stage_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Drapeau structuré plutôt qu'une recherche dans le texte des
    # avertissements : la recherche d'une sous-chaîne anglaise dans un message
    # destiné à l'utilisateur a silencieusement cessé de fonctionner le jour où
    # ce message est passé en français.
    overflowed: bool = False
    est_decode_tok_s: float = 0.0
    est_prefill_tok_s: float = 0.0
    est_bytes_per_token: int = 0
    # Second objectif. None quand les constantes d'energie ne sont pas
    # renseignees : le planificateur ne doit pas croire savoir.
    est_joules_par_jeton: Optional[float] = None
    est_jetons_par_kj: Optional[float] = None
    total_weight_bytes: int = 0
    bytes_per_tier: dict[str, int] = field(default_factory=dict)
    # Le nombre de séquences pour lequel `kv_max_tokens` a été dimensionné
    # (`opts.max_concurrent_seqs`, ligne 447 ci-dessous) — porté sur le plan
    # pour que `regime_ligne()` et `certifie-b12` puissent dire CONTRE QUOI le
    # budget a été calculé, sans quoi un lot réel différent de ce chiffre
    # tronque des séquences en silence (`loader.py::_replanifier` ne le
    # passait pas, trouvé par Laure le 17/09 sur un banc b=12 planifié pour 8).
    kv_planned_seqs: int = 0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "tiers": [t.to_dict() for t in self.tiers],
            "layers": [l.to_dict() for l in self.layers],
            "embed_device": self.embed_device,
            "lm_head_device": self.lm_head_device,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_budget": self.kv_budget,
            "kv_max_tokens": self.kv_max_tokens,
            "kv_planned_seqs": self.kv_planned_seqs,
            "etat_recurrent_bytes": self.etat_recurrent_bytes,
            "expert_cache_bytes": self.expert_cache_bytes,
            "stage_ranges": {k: list(v) for k, v in self.stage_ranges.items()},
            "total_weight_bytes": self.total_weight_bytes,
            "bytes_per_tier": self.bytes_per_tier,
            "est_decode_tok_s": round(self.est_decode_tok_s, 2),
            "est_prefill_tok_s": round(self.est_prefill_tok_s, 1),
            "est_bytes_per_token": self.est_bytes_per_token,
            "est_joules_par_jeton": (round(self.est_joules_par_jeton, 3)
                                     if self.est_joules_par_jeton else None),
            "est_jetons_par_kj": (round(self.est_jetons_par_kj, 1)
                                  if self.est_jetons_par_kj else None),
            "overflowed": self.overflowed,
            "warnings": self.warnings,
        }

    def device_of_layer(self, i: int) -> str:
        return self.layers[i].exec_device

    def render(self) -> str:
        lines = [f"plan de placement pour {self.model}", ""]
        w = max([len(t.name) for t in self.tiers] + [6])
        lines.append(f"  {'etage':<{w}}  {'format':<9} {'capacite':>10} "
                     f"{'poids':>10} {'KV':>10}  {'tranche':<14}")
        for t in self.tiers:
            used = self.bytes_per_tier.get(t.name, 0)
            kv = self.kv_budget.get(t.name, 0)
            rng = self.stage_ranges.get(t.name)
            stage = f"couches {rng[0]}-{rng[1]}" if rng else "-"
            lines.append(f"  {t.name:<{w}}  {t.weight_format:<9} {_h(t.capacity):>10} "
                         f"{_h(used):>10} {_h(kv):>10}  {stage:<14}")
        lines.append("")
        host_attn = [l.index for l in self.layers if l.attn_storage == "cpu"]
        host_mlp = [l.index for l in self.layers if l.mlp_storage == "cpu"]
        lines.append(f"  poids au total     {_h(self.total_weight_bytes)}")
        lines.append(f"  lu par jeton       {_h(self.est_bytes_per_token)}")
        lines.append(f"  KV par jeton       {_h(self.kv_bytes_per_token)}"
                     f"  -> {self.kv_max_tokens:,} jetons en cache")
        lines.append(f"  plongements        {self.embed_device}")
        lines.append(f"  lm_head            {self.lm_head_device}")
        if host_mlp:
            lines.append(f"  MLP en RAM hote    {_compact_ranges(host_mlp)}")
            on_cpu = [l.index for l in self.layers if l.mlp_exec == "cpu"]
            if on_cpu:
                lines.append(f"    calcule sur CPU  {_compact_ranges(on_cpu)}"
                             f"  (la DDR5 est plus large que le lien PCIe)")
            streamed = [l.index for l in self.layers
                        if l.mlp_storage == "cpu" and l.mlp_exec == "gpu"]
            if streamed:
                lines.append(f"    transfere au GPU {_compact_ranges(streamed)}")
        if host_attn:
            lines.append(f"  attention en RAM   {_compact_ranges(host_attn)}")
        for dev, b in self.expert_cache_bytes.items():
            if b:
                lines.append(f"  cache d'experts    {_h(b)} sur {dev}")
        lines.append("")
        lines.append(f"  decodage estime    {self.est_decode_tok_s:.1f} jetons/s  (lot de 1)")
        lines.append(f"  prefill estime     {self.est_prefill_tok_s:,.0f} jetons/s")
        for warn in self.warnings:
            lines.append(f"  ! {warn}")
        return "\n".join(lines)


def _h(n: float) -> str:
    for unite in ("o", "Kio", "Mio", "Gio", "Tio"):
        if abs(n) < 1024 or unite == "Tio":
            return f"{int(n)} o" if unite == "o" else f"{n:.1f} {unite}"
        n /= 1024
    return f"{n:.1f} Tio"


def _compact_ranges(nums: list[int]) -> str:
    if not nums:
        return "-"
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


# --------------------------------------------------------------------------
# tier construction
# --------------------------------------------------------------------------


def build_tiers(rig: Rig, opts: PlannerOptions) -> list[Tier]:
    tiers: list[Tier] = []
    for g in sorted(rig.gpus, key=lambda x: (-x.vram_bandwidth_gbps, x.index)):
        caps = g.caps
        fmt = opts.force_format or (caps.weight_format if caps else "int4_awq")
        kvf = _KV_FORMAT or (caps.kv_format if caps else "int8")
        # Planifier sur la mémoire LIBRE, pas totale. Le 8/09/2026, le plan
        # recalculé de Qwen3-Coder-Next donnait 8 Gio de poids à la 3080 Ti
        # « de 10,9 Gio » alors que le llama-server permanent y tenait déjà
        # 5,1 Gio : OOM au chargement, deux cases de mesure perdues. Le
        # chargeur bornait bien le cache KV sur la VRAM libre, mais après que
        # le plan avait déjà distribué les poids.
        usable = int((g.free_mem - opts.reserve_per_gpu) * (1 - FRAGMENTATION_MARGIN))
        tiers.append(Tier(
            name=f"cuda:{g.index}", kind="gpu", device_index=g.index,
            capacity=max(0, usable), weight_format=fmt, kv_format=kvf,
            read_bandwidth=g.vram_bandwidth_gbps,
            link_bandwidth=g.host_link_gbps,
            sm=caps.sm if caps else 0,
            power_limit_w=getattr(g, "power_limit_w", 0.0) or 0.0,
        ))
    if opts.allow_host_tier:
        avail = rig.host.available or rig.host.total
        tiers.append(Tier(
            name="cpu", kind="host", device_index=-1,
            capacity=int(avail * opts.host_fraction),
            # `force_format` D'ABORD, comme le tiers GPU (ligne ~350) : sans
            # carte visible (CUDA_VISIBLE_DEVICES="", defaut de toute session
            # depuis carte.sh/guet.sh du 14/09), `tiers` est VIDE ici — l'ancien
            # repli `tiers[0].weight_format if tiers else "int4_awq"` retombait
            # alors TOUJOURS sur int4_awq, ignorant silencieusement --format.
            # Ce n'etait pas visible avant que CUDA_VISIBLE_DEVICES="" devienne
            # le defaut : sans carte visible, toute conversion --format bf16
            # sans GPU quantifiait les poids malgre la demande explicite.
            weight_format=opts.force_format or
            (tiers[0].weight_format if tiers else "int4_awq"),
            kv_format=_KV_FORMAT or "fp16", read_bandwidth=70.0,
            link_bandwidth=max((t.link_bandwidth for t in tiers), default=25.0),
        ))
    return tiers


def _bytes(n_params: float, fmt: str, group_size: int) -> int:
    return int(n_params * bits_per_weight(fmt, group_size=group_size) / 8)


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------


def plan_placement(spec: ModelSpec, rig: Rig,
                   opts: Optional[PlannerOptions] = None) -> Plan:
    """Affecte chaque tenseur à un étage et estime ce qu'il en coûtera.

    L'ordre des décisions est délibéré. Le cache KV est dimensionné en premier
    parce qu'il croît avec le trafic, et qu'un modèle incapable de tenir son
    contexte est inutile même si ses poids logent parfaitement. Vient ensuite
    l'attention, parce qu'elle est petite et se trouve sur le chemin critique de
    la latence. Les poids des MLP et des experts se disputent ce qui reste, et
    le perdant part en mémoire vive, où le coût d'un défaut est un transfert
    PCIe plutôt qu'une erreur de mémoire saturée.
    """
    opts = opts or PlannerOptions()
    tiers = build_tiers(rig, opts)
    gpu_tiers = [t for t in tiers if t.kind == "gpu"]
    host_tier = next((t for t in tiers if t.kind == "host"), None)
    plan = Plan(model=spec.name, tiers=tiers)

    if not gpu_tiers:
        plan.warnings.append("aucun peripherique CUDA detecte ; plan pour processeur seul "
                             "(kv_max_tokens = 0 : aucune carte visible, pas un contexte ; "
                             "a sec, --profile planifie pour une machine declaree)")
        gpu_tiers = []

    remaining = {t.name: float(t.capacity) for t in tiers}

    # ---- 0. etat recurrent des couches lineaires -------------------------
    # AVANT le cache KV, parce qu'il n'est pas negociable : `kda.py` l'alloue
    # dans le forward, une copie par sequence, et rien ne l'en empeche si la
    # place manque. Le budget KV, lui, se borne. Ce poste n'etait budgete
    # NULLE PART : la surestimation du KV — qui comptait toutes les couches au
    # lieu des seules couches a cache — lui servait de provision de fait.
    # Corriger cette surestimation sans reserver ici deplacerait le defaut.
    # Sur les dix-sept Qwen3.x-27B du parc, 2304 Mio a seize sequences contre
    # 1854 Mio de budget KV entier : ils allouaient deja hors budget.
    inconnus = spec.types_de_couche_inconnus
    if inconnus:
        plan.warnings.append(
            f"types de couche non budgetes : {', '.join(inconnus)} — leur "
            f"etat eventuel n'est provisionne NULLE PART. Provisionner zero "
            f"en silence est le defaut corrige le 9/09/2026 sur mamba et conv.")
    etat_rec = spec.etat_recurrent_bytes(opts.max_concurrent_seqs)
    plan.etat_recurrent_bytes = etat_rec
    if gpu_tiers and etat_rec:
        pool_e = sum(remaining[t.name] for t in gpu_tiers)
        for t in gpu_tiers:
            remaining[t.name] -= etat_rec * remaining[t.name] / max(1.0, pool_e)

    # ---- 1. cache KV -----------------------------------------------------
    kv_bits = opts.kv_bits
    if gpu_tiers and gpu_tiers[0].kv_format in kv_lm4.FORMATS:
        kv_bits = kv_lm4.bits(gpu_tiers[0].kv_format)    # le budget suit le format
    kv_per_tok = spec.kv_bytes_per_token(kv_bits)
    plan.kv_bytes_per_token = kv_per_tok
    plan.kv_planned_seqs = opts.max_concurrent_seqs
    if gpu_tiers and kv_per_tok:
        wanted = kv_per_tok * opts.max_model_len * opts.max_concurrent_seqs
        pool = sum(remaining[t.name] for t in gpu_tiers)
        target = min(wanted, pool * opts.kv_vram_fraction)
        for t in gpu_tiers:
            share = target * remaining[t.name] / max(1.0, pool)
            plan.kv_budget[t.name] = int(share)
            remaining[t.name] -= share
        plan.kv_max_tokens = int(sum(plan.kv_budget.values()) // max(1, kv_per_tok))
        if plan.kv_max_tokens < opts.max_model_len:
            plan.warnings.append(
                f"le budget KV tient {plan.kv_max_tokens:,} jetons, moins qu'un "
                f"contexte complet de {opts.max_model_len:,} ; baissez "
                f"--max-model-len ou montez --kv-vram-fraction")

    # ---- 2. tranches du pipeline -----------------------------------------
    # Des plages contiguës, dimensionnées proportionnellement à la capacité de
    # chaque GPU une fois le cache KV retiré, pour que le pipeline ne franchisse
    # qu'une seule frontière entre cartes. Un franchissement est bon marché — un
    # état caché par la mémoire hôte épinglée — mais ne doit pas arriver à
    # chaque couche.
    n = spec.num_layers
    stage_of: list[str] = []
    if gpu_tiers:
        weights_pool = sum(max(0.0, remaining[t.name]) for t in gpu_tiers)
        cursor = 0
        for i, t in enumerate(gpu_tiers):
            if i == len(gpu_tiers) - 1:
                count = n - cursor
            else:
                frac = max(0.0, remaining[t.name]) / max(1.0, weights_pool)
                count = max(1, int(round(frac * n)))
                count = min(count, n - cursor - (len(gpu_tiers) - i - 1))
            if count > 0:
                plan.stage_ranges[t.name] = (cursor, cursor + count - 1)
            stage_of.extend([t.name] * count)
            cursor += count
        stage_of = stage_of[:n] + [gpu_tiers[-1].name] * max(0, n - len(stage_of))
    else:
        stage_of = ["cpu"] * n

    # ---- 3. attention : épinglée sur le GPU de sa tranche si elle y tient --
    placements: list[LayerPlacement] = []
    used = {t.name: 0.0 for t in tiers}
    for layer in spec.layers:
        dev = stage_of[layer.index]
        fmt = next((t.weight_format for t in tiers if t.name == dev), "int4_awq")
        a_bytes = _bytes(layer.attn_params + layer.norm_params, fmt, opts.group_size)
        m_bytes = _bytes(layer.mlp_params, fmt, opts.group_size)
        m_active = _bytes(layer.active_params - layer.attn_params - layer.norm_params,
                          fmt, opts.group_size)
        attn_storage = dev
        if opts.pin_attention and dev != "cpu" and remaining[dev] >= a_bytes:
            remaining[dev] -= a_bytes
            used[dev] += a_bytes
        elif host_tier is not None:
            attn_storage = "cpu"
            remaining["cpu"] -= a_bytes
            used["cpu"] += a_bytes
        else:
            remaining[dev] -= a_bytes
            used[dev] += a_bytes
        placements.append(LayerPlacement(
            index=layer.index, exec_device=dev, attn_storage=attn_storage,
            mlp_storage="pending", fmt=fmt, attn_bytes=a_bytes,
            mlp_bytes=m_bytes, mlp_active_bytes=m_active, is_moe=layer.is_moe))

    # ---- 4. plongements et lm_head ---------------------------------------
    # La table de plongements n'est qu'une collecte d'une ligne par jeton : la
    # laisser en mémoire vive coûte quelques kilooctets de trafic PCIe. lm_head
    # est en revanche un produit matriciel sur tout le vocabulaire à chaque
    # étape : il mérite sa place sur le GPU le plus rapide.
    embed_bytes = spec.embed_params * 2
    fastest = gpu_tiers[0].name if gpu_tiers else "cpu"
    head_fmt = gpu_tiers[0].weight_format if gpu_tiers else "int4_awq"
    # `lm_head_params` vaut ZERO quand la tete est liee — il n existe alors
    # aucun tenseur `lm_head`. Mais le chargeur fabrique quand meme une copie
    # quantifiee de la table d embedding pour la projection, et elle occupe la
    # carte : la compter ici est un rattrapage de comptage, pas une provision.
    head_bytes = (_bytes(spec.lm_head_params, head_fmt, opts.group_size)
                  + spec.tete_liee_bytes(opts.group_size))
    if gpu_tiers and remaining[fastest] > head_bytes:
        plan.lm_head_device = fastest
        remaining[fastest] -= head_bytes
        used[fastest] += head_bytes
    else:
        plan.lm_head_device = "cpu"
        used["cpu"] = used.get("cpu", 0.0) + head_bytes
    # L'etage hote EXISTE des que `allow_host_tier` est vrai — son defaut —,
    # meme quand rien n'y est exile. Tester son existence mettait donc la table
    # de plongements cote hote SYSTEMATIQUEMENT, y compris sur un modele qui
    # tient entierement sur la carte. Le releve du 9/09 sur
    # Qwen2.5-Coder-14B-bf16-pur : 0 couche exilee, et `embed_device: cpu`.
    #
    # Cout : le gather s'execute cote hote et le vecteur repart vers la carte,
    # soit DEUX traversees PCIe par jeton (latence, pas volume : la table ne
    # fournit qu'une ligne de 10 Kio). Et face a llama.cpp qui met la table sur
    # la carte avec `-ngl 999`, c'est une asymetrie de placement qui se lit
    # comme une difference de moteur.
    #
    # On teste desormais LA PLACE, comme `lm_head_device` deux lignes plus haut
    # — les deux champs decidaient la meme chose par deux logiques opposees.
    if gpu_tiers and remaining[fastest] > embed_bytes:
        plan.embed_device = fastest
    else:
        plan.embed_device = "cpu"
    if plan.embed_device != "cpu":
        remaining[plan.embed_device] -= embed_bytes
        used[plan.embed_device] += embed_bytes
    else:
        used["cpu"] += embed_bytes

    # ---- 5. MLP et experts : remplir la VRAM d'avant en arrière -----------
    for lp in placements:
        dev = lp.exec_device
        if dev != "cpu" and remaining.get(dev, 0) >= lp.mlp_bytes:
            remaining[dev] -= lp.mlp_bytes
            used[dev] += lp.mlp_bytes
            lp.mlp_storage = dev
        elif host_tier is not None and remaining["cpu"] >= lp.mlp_bytes:
            remaining["cpu"] -= lp.mlp_bytes
            used["cpu"] += lp.mlp_bytes
            lp.mlp_storage = "cpu"
        else:
            lp.mlp_storage = "cpu"
            used["cpu"] += lp.mlp_bytes
            remaining["cpu"] -= lp.mlp_bytes

    if host_tier is not None and remaining["cpu"] < 0:
        plan.overflowed = True
        plan.warnings.append(
            f"le modele deborde tous les etages de {_h(-remaining['cpu'])} ; "
            f"prenez un modele plus petit, ou ajoutez --kv-vram-fraction 0.15")

    # ---- 5b. comment sont calculés les poids résidant en mémoire vive -----
    # Une couche laissée en RAM peut être copiée vers le GPU ou calculée sur
    # place. Les deux chemins sont limités par la mémoire et lisent les mêmes
    # octets : le plus rapide est donc simplement celui dont le bus est le plus
    # large. Le PCIe 5.0 x16 donne environ 54 Go/s, la DDR5-6000 en double canal
    # environ 70 Go/s en lecture séquentielle. Calculer sur place laisse en
    # outre le GPU libre au lieu de le faire attendre une copie.
    for lp in placements:
        if lp.mlp_storage != "cpu":
            lp.mlp_exec = "gpu"
            continue
        if opts.host_exec == "stream":
            lp.mlp_exec = "gpu"
        elif opts.host_exec == "cpu":
            lp.mlp_exec = "cpu"
        else:
            link = next((t.link_bandwidth for t in tiers
                         if t.name == lp.exec_device), 25.0)
            taux = opts.host_gemm_gb_s
            if taux is None:
                # Jamais mesuré. On ne choisit pas un chemin dont on ignore le
                # coût : le transfert vers le GPU est borné par le bus, donc
                # prévisible, alors que le calcul hôte ne l'est pas. Comparer
                # ici un débit de lecture DDR à une largeur de bus revenait à
                # croire le processeur plus rapide qu'une carte graphique.
                lp.mlp_exec = "gpu"
            else:
                lp.mlp_exec = "cpu" if taux > link else "gpu"

    # ---- 6. cache d'experts fréquents -------------------------------------
    # La VRAM qui subsiste devient un cache LRU pour les experts restés en
    # mémoire vive. Le routage n'est pas uniforme en pratique : un cache
    # contenant un dixième des experts sert donc sensiblement plus qu'un dixième
    # des lectures.
    host_moe = [lp for lp in placements if lp.is_moe and lp.mlp_storage == "cpu"]
    if host_moe and gpu_tiers:
        for t in gpu_tiers:
            spare = max(0.0, remaining[t.name]) * opts.expert_cache_fraction
            if spare > 64 * MB:
                plan.expert_cache_bytes[t.name] = int(spare)
                remaining[t.name] -= spare
        cache_total = sum(plan.expert_cache_bytes.values())
        host_expert_bytes = sum(lp.mlp_bytes for lp in host_moe)
        if host_expert_bytes:
            raw = cache_total / host_expert_bytes
            # Rapport de CAPACITE, sans majoration.
            #
            # Il portait un facteur 1,3 au titre d'un « biais de routage » : les
            # experts frequents seraient touches plus souvent que leur part.
            # C'est plausible et ce n'est pas mesure — aucun taux de succes reel
            # n'a jamais ete releve sur ce parc. Un facteur invente qui majore
            # rend le plan optimiste : il annonce moins d'octets traversant le
            # lien qu'il n'en passera.
            #
            # Sans mesure, on prend la BORNE HAUTE du cout, c'est-a-dire la
            # borne basse du taux de succes. Un plan choisi sur une borne haute
            # est au pire trop prudent ; choisi sur une estimation majoree, il
            # est au mieux chanceux.
            #
            # Ce que cela ne change PAS : le choix du plan. cached_expert_fraction
            # n'entre que dans _estimate, donc dans decode_tok_s, donc dans le
            # troisieme critere de _rang — celui qui ne departage que les plans
            # SANS exil, ou cette fraction vaut zero. La correction porte sur le
            # debit annonce, pas sur la decision.
            hit = min(1.0, raw)
            for lp in host_moe:
                lp.cached_expert_fraction = hit

    plan.layers = placements
    plan.bytes_per_tier = {k: int(v) for k, v in used.items()}
    plan.total_weight_bytes = int(sum(used.values()))
    _estimate(spec, plan, tiers, opts)
    return plan


def _estimate(spec: ModelSpec, plan: Plan, tiers: list[Tier],
              opts: PlannerOptions) -> None:
    by_name = {t.name: t for t in tiers}
    fastest_link = max((t.link_bandwidth for t in tiers if t.kind == "gpu"),
                       default=25.0)
    LAUNCH = 15e-6

    seconds = 0.0
    bytes_read = 0
    for lp in plan.layers:
        t = by_name.get(lp.exec_device)
        if t is None or t.kind != "gpu":
            # Exécution sur processeur : limitée par la bande passante DDR.
            seconds += (lp.attn_bytes + lp.mlp_active_bytes) / (70.0 * 1e9)
            bytes_read += lp.attn_bytes + lp.mlp_active_bytes
            continue

        # attention
        if lp.attn_storage == "cpu":
            seconds += lp.attn_bytes / (min(t.link_bandwidth, fastest_link) * 1e9)
        else:
            seconds += lp.attn_bytes / (t.read_bandwidth * 1e9)
        bytes_read += lp.attn_bytes

        # MLP et experts
        active = lp.mlp_active_bytes
        if lp.mlp_storage == "cpu" and lp.mlp_exec == "cpu":
            # Calculé sur place : limité par la bande passante DDR, plus un
            # état caché qui traverse le bus dans chaque sens — quelques
            # kilooctets, donc du bruit.
            # Chiffrer ce GEMM avec un débit de lecture était une erreur de
            # grandeur, pas de calibrage : le coût est dans le déballage.
            taux = opts.host_gemm_gb_s or opts.host_compute_gb_s
            seconds += active / (taux * 1e9)
            bytes_read += active
        elif lp.mlp_storage == "cpu":
            from_cache = active * lp.cached_expert_fraction
            from_host = active - from_cache
            t_copy = from_host / (t.link_bandwidth * 1e9)
            t_math = active / (t.read_bandwidth * 1e9)
            # Cout FIXE par transfert d'expert, en plus des octets.
            #
            # Une copie d'expert n'est pas un flux continu : c'est une suite de
            # copies discretes, chacune payant sa latence de bus et sa
            # synchronisation. Le modele n'avait que des termes en octets, et
            # aucun terme en octets ne peut produire un plancher constant.
            #
            # Ce qui a mis ce terme en evidence : le modele est optimiste d'un
            # facteur 2,4 a 3,4 sur soixante et un modeles mesures une fois
            # chacun, avec un plancher d'environ 3 ms par jeton et une pente
            # d'environ 2,5 ms par doublement de taille. Un plancher constant
            # demande un terme constant ; une pente logarithmique suit le
            # NOMBRE de blocs transferes, pas leur volume.
            #
            # LE COUT FIXE EXISTE MAIS IL NE DOMINE PAS, et c'est une mesure
            # qui l'a etabli contre la lecture qui avait propose ce terme.
            #
            # Diviser par trois le nombre de copies, a volume inchange, fait
            # passer une couche de 3,11 a 2,88 ms. Si les latences fixes
            # dominaient, on serait tombe vers 1,04 ; la lecture predisait 2,07.
            # De ces deux points : latences fixes 0,345 ms, soit ONZE POUR CENT
            # du total ; le reste, 2,765 ms, correspond a 6,5 Go/s sur un bus
            # mesure a 18,7.
            #
            # Donc l'essentiel du cout n'est ni la latence par transfert ni le
            # debit nominal. Il reste a nommer : debit effectif reellement plus
            # bas, ou synchronisation par expert que le nombre de copies ne
            # change pas.
            #
            # transfer_fixed_us reste a ZERO malgre ces deux points. Ils
            # viennent d'un seul modele et d'une seule couche ; en tirer une
            # constante generale serait refaire l'erreur qu'on vient de
            # corriger deux fois sur le nombre de copies.
            #
            # LE NOMBRE DE TRANSFERTS, corrige deux fois par la mesure.
            #
            # Un expert route coute NEUF copies sur l'arbre actuel : trois
            # projections — porte, montee, descente — et trois tenseurs par
            # projection, un poids NVFP4 etant fait de qweight, des echelles de
            # bloc et de l'echelle globale, copies un par un. Dix experts
            # routes font donc quatre-vingt-dix copies par couche exilee.
            #
            # Deux versions fausses avant celle-ci : des blocs de 4 Mo, soit
            # cinq copies (facteur dix-huit trop peu), puis trois copies par
            # expert (facteur trois). Chaque correction venait d'un relevé, pas
            # d'un raisonnement.
            #
            # Ce nombre TOMBE A TROIS quand les trois tenseurs d'un poids
            # voyagent dans un seul tampon epingle. Il n'est donc PAS ecrit ici
            # : NVFP4Tensor.transferts_par_poids() le compte sur la classe
            # elle-meme, et suivra cette optimisation sans qu'on y revienne.
            # Deux valeurs ecrites en dur ont deja ete fausses, d'un facteur
            # dix-huit puis trois.
            from ..quant.nvfp4 import NVFP4Tensor
            PROJECTIONS = 3          # porte, montee, descente
            TENSEURS_PAR_EXPERT = PROJECTIONS * NVFP4Tensor.transferts_par_poids()
            couche = spec.layers[lp.index]
            actifs = getattr(couche, "n_experts_active", 0) or 0
            if actifs:
                n_transferts = actifs * TENSEURS_PAR_EXPERT
            else:
                # Couche dense exilee : une copie par tenseur de MLP.
                n_transferts = TENSEURS_PAR_EXPERT
            seconds += max(t_copy, t_math) + n_transferts * opts.transfer_fixed_us * 1e-6
            bytes_read += int(from_host)
        else:
            seconds += active / (t.read_bandwidth * 1e9)
            bytes_read += active
        seconds += LAUNCH

    crossings = sum(1 for a, b in zip(plan.layers, plan.layers[1:])
                    if a.exec_device != b.exec_device)
    seconds += crossings * 60e-6
    plan.est_bytes_per_token = bytes_read
    plan.est_decode_tok_s = 1.0 / seconds if seconds > 0 else 0.0

    # --- second objectif : les joules par jeton ---------------------------
    #
    # Trois sources, et le calcul S'ABSTIENT si l'une des constantes manque.
    # Rendre None quand on ne sait pas vaut mieux qu'un chiffre credible :
    # celui-ci servirait a choisir un plan.
    #
    #   1. les cartes, pendant la duree du jeton. Le decodage est borne par la
    #      memoire, pas par le calcul : une carte n'y tire pas sa limite de
    #      puissance, d'ou fraction_puissance_decodage. La valeur au repos
    #      compte pour les cartes qui ne portent rien mais restent alimentees ;
    #   2. le processeur, pour la part de la couche qui s'execute dessus. C'est
    #      le terme que le modele en secondes ne voit pas : il compte la meme
    #      duree qu'un transfert par le bus, alors qu'il occupe des cœurs ;
    #   3. rien d'autre. La memoire vive, les ventilateurs et la carte mere ne
    #      sont pas modelises, et le chiffre est donc une BORNE BASSE.
    if (opts.watts_processeur > 0 and opts.fraction_puissance_decodage > 0
            and seconds > 0):
        j_cartes = 0.0
        for t in tiers:
            if t.kind != "gpu" or t.power_limit_w <= 0:
                continue
            porte = any(lp.exec_device == t.name for lp in plan.layers)
            if porte:
                j_cartes += (t.power_limit_w * opts.fraction_puissance_decodage
                             * seconds)
            elif opts.watts_carte_au_repos > 0:
                j_cartes += opts.watts_carte_au_repos * seconds
        # Le temps processeur n'est pas la duree du jeton : c'est la somme des
        # tranches ou des cœurs travaillent vraiment, et elle peut depasser la
        # duree si plusieurs cœurs tournent.
        sec_proc = 0.0
        for lp in plan.layers:
            t = by_name.get(lp.exec_device)
            if t is None or t.kind != "gpu":
                sec_proc += (lp.attn_bytes + lp.mlp_active_bytes) / (70.0 * 1e9)
            elif lp.mlp_storage == "cpu" and lp.mlp_exec == "cpu":
                taux = opts.host_gemm_gb_s or opts.host_compute_gb_s
                sec_proc += lp.mlp_active_bytes / (taux * 1e9)
        j_proc = sec_proc * opts.watts_processeur
        total = j_cartes + j_proc
        if total > 0:
            plan.est_joules_par_jeton = total
            plan.est_jetons_par_kj = 1000.0 / total

    # prefill : limité par le calcul, et les poids d'une couche transférée sont
    # lus une fois pour tout le lot au lieu d'une fois par jeton
    pf = 0.0
    for lp in plan.layers:
        t = by_name.get(lp.exec_device)
        if t is None or t.kind != "gpu":
            continue
        layer = spec.layers[lp.index]
        table = TFLOPS.get(t.sm, TFLOPS[86])
        peak = table.get(lp.fmt) or table.get("bf16") or 60.0
        pf += (2.0 * layer.active_params) / (peak * 1e12 * 0.55)
    plan.est_prefill_tok_s = 1.0 / pf if pf > 0 else 0.0


# --------------------------------------------------------------------------
# recherche externe
# --------------------------------------------------------------------------


def auto_plan(spec: ModelSpec, rig: Rig,
              opts: Optional[PlannerOptions] = None,
              verbose: bool = False) -> tuple[Plan, list[dict]]:
    """Explore le petit espace des configurations sensées et retient la meilleure.

    ``plan_placement`` répond à « ces réglages étant donnés, où va chaque
    chose ». Il ne peut pas répondre aux deux questions qui décident réellement
    du débit :

    * Faut-il seulement utiliser le second GPU ? Étendre à la 3080 Ti un modèle
      qui tient déjà sur la 5090 rend le décodage mono-flux *plus lent*, parce
      que les tranches s'exécutent en série et que la moitié d'entre elles lit
      désormais à 912 Go/s au lieu de 1790. Un second GPU ne mérite sa place que
      lorsqu'il évite de renvoyer des poids en mémoire vive.
    * Quelle part de VRAM donner au cache KV ? Chaque gigaoctet donné au cache
      est un gigaoctet de poids repoussé sur le bus PCIe, et lire un poids par
      le PCIe coûte environ trente fois ce qu'il coûte depuis la VRAM. Passé le
      point où un contexte complet tient, davantage de cache ne vaut presque
      rien pour un flux unique.

    On répond aux deux en essayant la poignée de combinaisons et en les classant
    sur le débit de décodage estimé, en rejetant tout ce qui ne peut pas tenir un
    contexte complet ou qui déborde la machine.
    """
    base = opts or PlannerOptions()
    n_gpus = len([g for g in rig.gpus])
    candidates: list[tuple[Plan, dict]] = []
    trials: list[dict] = []

    kv_fractions = [0.06, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55]
    gpu_counts = _gpu_counts(base.gpus, n_gpus)

    for used_gpus in gpu_counts:
        sub = _subset_rig(rig, used_gpus)
        for kvf in kv_fractions:
            o = PlannerOptions(**{**base.__dict__, "kv_vram_fraction": kvf})
            p = plan_placement(spec, sub, o)
            overflow = p.overflowed
            ctx_ok = p.kv_max_tokens >= base.max_model_len
            host_bytes = p.bytes_per_tier.get("cpu", 0)
            rec = {
                "gpus": used_gpus, "kv_fraction": kvf,
                "decode_tok_s": round(p.est_decode_tok_s, 2),
                "kv_tokens": p.kv_max_tokens,
                "host_bytes": host_bytes,
                "feasible": (not overflow) and ctx_ok,
                "overflow": overflow, "context_ok": ctx_ok,
            }
            trials.append(rec)
            if rec["feasible"]:
                candidates.append((p, rec))

    if not candidates:
        # Aucune configuration ne satisfait toutes les contraintes. On dit
        # laquelle a cédé, et de combien, plutôt que de rendre en silence un
        # plan inexécutable.
        fits = [t for t in trials if not t["overflow"]]
        if fits:
            # La question n'est plus « ce contexte tient-il ? » (non) mais
            # « combien tiennent ? » (Sage sage-s2-k48-feu-vert-21-09, addendum) :
            # tout ce que la VRAM laisse après les poids va au cache, au pas de
            # 1024 jetons. Prendre la plus petite fraction (0,06) rendait 3 938
            # jetons au 31B vision pour 24 576 possibles ; sans carte visible
            # (CUDA_VISIBLE_DEVICES vide, sans --profile) le plan rend 0 et le dit.
            best = max(fits, key=lambda t: t["decode_tok_s"])
            sub = _subset_rig(rig, best["gpus"])
            p = _plan_kv_maximal(spec, sub, base, best["kv_fraction"])
            p.warnings.append(
                f"aucune configuration ne tient un contexte complet de "
                f"{base.max_model_len:,} jetons ; le cache tient "
                f"{p.kv_max_tokens:,} jetons (maximum apres les poids, pas de 1024)")
            return p, trials

        # Cela ne tient nulle part. Disons ce qu'il faudrait.
        p = plan_placement(spec, rig, base)
        p.kv_max_tokens, p.kv_budget = 0, {}     # les poids ne logent pas : aucun contexte, pas un budget fictif
        capacity = sum(t.capacity for t in p.tiers)
        short = p.total_weight_bytes - capacity
        need_bpw = capacity * 8 / max(1, spec.total_params)
        p.warnings.append(
            f"{spec.name} ne tient pas sur cette machine : {_h(p.total_weight_bytes)} "
            f"de poids contre {_h(capacity)} de capacite utilisable, il manque "
            f"{_h(max(0, short))}.")
        p.warnings.append(
            f"il faudrait {need_bpw:.2f} bits par poids ou moins ; les formats "
            f"disponibles sont {bits_per_weight('nvfp4'):.2f} (NVFP4) et "
            f"{bits_per_weight('int4_awq', group_size=base.group_size):.2f} (INT4). "
            f"Options : --host-fraction 0.95, un modele plus petit, ou plus de RAM.")
        return p, trials

    # Classement des configurations. Le débit estimé seul ne suffit pas : il
    # a déjà choisi, sur Qwen3-Coder-Next, de laisser une carte de 12 Gio
    # oisive pendant que 14 Gio de perceptrons partaient sur le processeur.
    # Un poids exilé en RAM vive est deux ordres de grandeur sous n'importe
    # quel GPU, quelle que soit la confiance qu'on accorde à l'estimation ;
    # on écarte donc d'abord l'exil, on optimise le débit ensuite.
    # On ne compte que les poids de couches : la table de plongements vit
    # toujours en RAM par construction, et n'est qu'une collecte d'une ligne
    # par jeton.
    def _exil(p) -> int:
        n = 0
        for lp in p.layers:
            if lp.mlp_storage == "cpu":
                n += lp.mlp_bytes
            if lp.attn_storage == "cpu":
                n += lp.attn_bytes
        return n

    def _rang(pr):
        p, rec = pr
        exil = _exil(p) / max(1, p.total_weight_bytes)
        if exil <= 0.0:
            return (1, 0.0, rec["decode_tok_s"])
        return (0, -exil, rec["decode_tok_s"])

    best_plan, best_rec = max(candidates, key=_rang)
    if best_rec["gpus"] < n_gpus:
        idle = [f"cuda:{g.index}" for g in
                sorted(rig.gpus, key=lambda g: -g.vram_bandwidth_gbps)[best_rec["gpus"]:]]
        exile = _exil(best_plan)
        if exile > 0:
            best_plan.warnings.append(
                f"{', '.join(idle)} reste oisif alors que {_h(exile)} de poids "
                f"sont en RAM hote : aucune configuration testee ne faisait "
                f"mieux, mais ce plan est suspect. Forcez --gpus all et comparez.")
        else:
            best_plan.warnings.append(
                f"{', '.join(idle)} laisse oisif a dessein : le modele tient sans lui, "
                f"et ajouter une tranche plus lente au pipeline couterait du debit. "
                f"Servez-vous-en pour un second modele, ou forcez avec --gpus all.")
    if verbose:
        best_plan.warnings.append(
            f"retenu {best_rec['gpus']} GPU, kv_fraction={best_rec['kv_fraction']} "
            f"parmi {len(trials)} candidats")
    return best_plan, trials


KV_PAS_JETONS = 1024


def _plan_kv_maximal(spec: ModelSpec, rig: Rig, base: PlannerOptions, fraction_min: float) -> Plan:
    """Le plus grand cache KV qui laisse les poids loger : dichotomie sur ``kv_vram_fraction`` entre
    ``fraction_min`` (tient) et 0,98, puis ``kv_max_tokens`` arrondi au multiple inferieur de
    ``KV_PAS_JETONS`` et budget reduit d'autant ; aucun poids exile de plus que le plan a ``fraction_min``
    (libre ÷ KV/jeton, les poids restant ou ils logent). 0 seulement si rien ne tient a ``fraction_min``."""
    def essai(f: float) -> Plan:
        return plan_placement(spec, rig, PlannerOptions(**{**base.__dict__, "kv_vram_fraction": f}))
    lo, hi = fraction_min, 0.98
    p = essai(lo)
    if p.overflowed:
        return p
    hote = p.bytes_per_tier.get("cpu", 0)     # le cache ne doit exiler aucun poids de plus que le plan qui tient
    for _ in range(14):                       # resolution < 1e-4 sur la fraction
        mid = (lo + hi) / 2
        q = essai(mid)
        if q.overflowed or q.bytes_per_tier.get("cpu", 0) > hote:
            hi = mid
        else:
            lo, p = mid, q
    n = p.kv_max_tokens // KV_PAS_JETONS * KV_PAS_JETONS
    if 0 < n < p.kv_max_tokens:
        r = n / p.kv_max_tokens
        p.kv_budget = {k: int(v * r) for k, v in p.kv_budget.items()}
        p.kv_max_tokens = n
    return p


def _subset_rig(rig: Rig, n_gpus: int) -> Rig:
    """Une copie de la machine n'exposant que les ``n_gpus`` cartes les plus rapides."""
    return Rig(
        gpus=sorted(rig.gpus, key=lambda g: -g.vram_bandwidth_gbps)[:n_gpus],
        host=rig.host, cpu=rig.cpu, driver_version=rig.driver_version,
        cuda_version=rig.cuda_version, kernel=rig.kernel, distro=rig.distro,
        p2p_matrix=rig.p2p_matrix, source=rig.source,
    )


def _gpu_counts(spec: Optional[str], n_gpus: int) -> list[int]:
    """Quels nombres de GPU la recherche a le droit d'envisager.

    ``auto``, la valeur par défaut, essaie tous les nombres et laisse le débit
    trancher : c'est ce qui laisse une seconde carte plus lente oisive quand le
    modèle n'en a pas besoin. ``all`` impose toutes les cartes — utile lorsqu'on
    préfère la marge de VRAM pour un contexte plus long aux quelques derniers
    jetons par seconde.
    """
    if not n_gpus:
        return [0]
    if spec in (None, "", "auto"):
        return list(range(1, n_gpus + 1))
    if spec == "all":
        return [n_gpus]
    try:
        wanted = {int(x) for x in spec.replace(" ", "").split(",") if x != ""}
    except ValueError:
        raise ValueError(f"--gpus attend auto, all, ou des indices comme 0,1 ; "
                         f"reçu {spec!r}") from None
    if not wanted:
        return [n_gpus]
    return [min(n_gpus, max(wanted) + 1)]


# --- coût de l'exil : le geste 1 de ggrun (bead anticitoyen-vram-jt5) ---------
#
# `_reajuster_plan` (engine/loader.py) décide d'exiler un MLP en RAM hôte sur le
# seul axe des OCTETS : il descend des couches tant que les poids résidents
# dépassent la capacité de l'étage. Or le coût de l'exil est entièrement dans le
# TEMPS — un facteur 3 à 4 mesuré (docs/TEST-EXIL.md:97-102, +196 s pour douze
# couches), parce qu'une couche transférée est bornée par le lien PCIe (~26 Go/s
# gen4x16) et non par la VRAM en lecture (~1050 Go/s mesuré, pas le pic de
# plaque 1792 — `Gpu.vram_bandwidth_gbps`, docs/PLAFOND-MEMOIRE-5090.md) :
# près de deux ordres de grandeur par octet.
# Décider sur les octets ne voit pas cette falaise.
#
# Cette fonction la chiffre AVANT le chargement, sans carte, à partir des bandes
# DÉCLARÉES du plan (`Tier.link_bandwidth`, `Tier.read_bandwidth`) : pour chaque
# MLP exilé, un aller PCIe par jeton coûte `mlp_active_bytes / link_bandwidth` ;
# on rapporte la somme au plancher d'un pas de décodage SANS exil (chaque poids
# actif lu une fois en VRAM). Le rapport est le surcoût relatif de l'exil.
#
# Elle SIGNALE, elle ne refuse pas : un modèle qui ne tient qu'au prix de l'exil
# doit tout de même se charger — l'alternative est l'OOM, pas un plan plus
# rapide. Le signal rend le chiffre qui manquait à la décision sur les octets.
#
# Rend None — et non zéro — quand elle ne peut pas chiffrer honnêtement : aucun
# MLP exilé, ou une bande inconnue (règle : un échec est un résultat, pas un
# chiffre reconstruit). Un plan sans exil, ou dont le lien est aussi rapide que
# la VRAM, doit pouvoir rendre « pas de falaise » — c'est ce qui en fait un
# contrôle et non une alarme qui sonne toujours.

def estimer_cout_exil(plan: "Plan", seuil: float = 0.20) -> Optional[dict]:
    """Surcoût en temps de l'exil des MLP en RAM hôte, rapporté au pas résident.

    `seuil` est la part du pas de décodage au-delà de laquelle l'exil est une
    falaise (0,20 par défaut). Rend un dict chiffré, ou None si rien n'est
    exilé ou si une bande manque pour chiffrer.
    """
    tiers = {t.name: t for t in plan.tiers}
    t_transfert = 0.0        # secondes de PCIe ajoutées par jeton par l'exil
    t_pas_resident = 0.0     # plancher d'un pas SANS exil (tout lu en VRAM)
    octets_exiles = 0
    n_exiles = 0
    for l in plan.layers:
        t = tiers.get(l.exec_device)
        if t is None or t.kind != "gpu":
            continue          # une couche calculée sur l'hôte n'est pas la
                              # falaise du streaming : autre modèle de coût
        if t.read_bandwidth <= 0:
            return None       # bande VRAM inconnue : on ne chiffre pas
        t_pas_resident += (l.attn_bytes + l.mlp_active_bytes) / (t.read_bandwidth * 1e9)
        if l.mlp_storage == "cpu":
            if t.link_bandwidth <= 0:
                return None   # lien PCIe inconnu : on ne chiffre pas
            t_transfert += l.mlp_active_bytes / (t.link_bandwidth * 1e9)
            octets_exiles += l.mlp_active_bytes
            n_exiles += 1
    if n_exiles == 0 or t_pas_resident <= 0:
        return None
    ratio = t_transfert / t_pas_resident
    return {
        "n_couches_exilees": n_exiles,
        "octets_exiles": octets_exiles,
        "t_transfert_s": t_transfert,
        "t_pas_resident_s": t_pas_resident,
        "ratio": ratio,
        "seuil": seuil,
        "franchit_seuil": ratio > seuil,
    }
