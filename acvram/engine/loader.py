"""Assemble un modèle exécutable à partir des fragments acvram et d'un plan de
placement.

Le manifeste écrit par le convertisseur consigne, pour chaque tenseur, son
format et les clés qui le portent. Le chargement est donc mécanique : lire les
clés, reconstruire le conteneur quantifié, et le poser là où le plan l'indique —
résident sur un GPU, ou épinglé en mémoire hôte derrière un
:class:`~acvram.engine.layers.StreamedWeight`.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Optional

import torch

from ..memory import expert_usage
from ..memory.kvcache import BLOCK_SIZE, KVCacheConfig, PagedKVCache
from ..memory.table_adresses import construire_table
from ..memory.tiering import Plan
from ..quant.calibrate import ChannelScaler
from ..quant.formats import INT8Tensor, PlainTensor
from ..quant.int4 import INT4Tensor
from ..quant.nvfp4 import NVFP4Tensor
from .config import ModelSpec
from .layers import ExpertPool, QuantLinear, RMSNorm, RotaryEmbedding
from .model import (ACVRamModel, Attention, DecoderLayer, DecoderLayerGDN, MoEBlockGemma,
                    DecoderLayerGemma, DecoderLayerParallel, MLP, MLP2, MoEBlock)

__all__ = ["LoadedModel", "load_model"]


class _ShardReader:
    """Accès paresseux aux tenseurs d'un ensemble de fragments safetensors."""

    def __init__(self, path: str, weight_map: dict[str, str]) -> None:
        self.path = path
        self.weight_map = weight_map
        self._open: dict[str, Any] = {}

    def get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open
        fn = self.weight_map.get(key)
        if fn is None:
            raise KeyError(f"{key} est absent de la table du manifeste")
        if fn not in self._open:
            self._open[fn] = safe_open(os.path.join(self.path, fn),
                                       framework="pt", device="cpu")
        return self._open[fn].get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def close(self) -> None:
        self._open.clear()


class LoadedModel:
    def __init__(self, model: ACVRamModel, spec: ModelSpec, plan: Plan,
                 manifest: dict, path: str) -> None:
        self.model = model
        self.spec = spec
        self.plan = plan
        self.manifest = manifest
        self.path = path


_noyaux_signales = False


def _avertir_noyaux() -> None:
    """Dit franchement quand les noyaux CUDA manquent.

    Le repli sur les implémentations de référence divise le débit par un ordre
    de grandeur, et il ne se signalait que par un ``warnings.warn`` noyé dans
    la sortie du chargement. Une compilation qui échoue — un nvcc trop ancien,
    un en-tête absent — passait ainsi inaperçue pendant des semaines.
    """
    global _noyaux_signales
    if _noyaux_signales or not torch.cuda.is_available():
        return
    _noyaux_signales = True
    from ..kernels import build_info
    info = build_info()
    if info.get("available"):
        return
    raison = (info.get("error") or "raison inconnue").strip().splitlines()
    print("\n[acvram] ATTENTION : les noyaux CUDA ne sont PAS disponibles.",
          file=sys.stderr)
    print("[acvram] le moteur tourne sur les implementations de reference, "
          "environ dix fois plus lentes.", file=sys.stderr)
    print(f"[acvram] cause : {raison[0][:300]}", file=sys.stderr)
    print("[acvram] verifiez `python -m acvram doctor`.\n", file=sys.stderr)


def _build_quant(entry: dict, name: str, reader: _ShardReader,
                 group_size: int) -> Any:
    fmt = entry["format"]
    shape = tuple(entry["shape"])
    sd = {k.rsplit(".", 1)[-1]: reader.get(k) for k in entry["keys"]}
    if fmt == "nvfp4":
        return NVFP4Tensor(
            sd["qweight"], sd["block_scale"].view(torch.float8_e4m3fn),
            sd["global_scale"], shape, sd["qweight"].shape[-1] * 2)
    if fmt == "int4_awq":
        return INT4Tensor(sd["qweight"], sd["scales"], sd["zeros"],
                          entry.get("group_size", group_size), shape,
                          sd["qweight"].shape[-1] * 2)
    if fmt == "int8":
        return INT8Tensor(sd["qweight"], sd["scales"], sd["zeros"],
                          entry.get("group_size", group_size), shape)
    if fmt == "q3n":
        from ..quant.q3n import (BLOC_DEFAUT, TABLE_Q3N, Q3NTensor,
                                 valider_table_q3n)
        table = entry.get("table")
        table = valider_table_q3n(table) if table is not None else TABLE_Q3N
        sceau = entry.get("sceau")
        if sceau is not None:
            import hashlib as _h
            reel = _h.sha256(
                sd["qweight"].flatten()[:64].cpu().numpy().tobytes()
                + repr([float(v) for v in table]).encode()).hexdigest()[:16]
            if reel != sceau:
                raise ValueError(
                    f"{name} : le sceau table/poids ne correspond pas — "
                    "manifeste régénéré sans reconversion ? Le modèle ne "
                    "doit pas être servi avec cette table.")
        return Q3NTensor(sd["qweight"],
                         sd["block_scale"].view(torch.float8_e4m3fn),
                         sd["global_scale"],
                         entry.get("block", BLOC_DEFAUT), shape,
                         "q3n", table)
    if fmt in ("bf16", "fp16"):
        return PlainTensor(sd["weight"], shape, fmt)
    raise KeyError(f"unknown format {fmt!r} for {name}")


def _build_scaler(entry: dict, name: str, reader: _ShardReader) -> Optional[ChannelScaler]:
    block = entry.get("hadamard_block", 0)
    scale = None
    key = f"{name}.act_scale"
    if entry.get("has_act_scale") and reader.has(key):
        scale = reader.get(key)
    if scale is None and not block:
        return None
    return ChannelScaler(scale, block)


def _linear(name: str, manifest: dict, reader: _ShardReader,
            group_size: int) -> Optional[QuantLinear]:
    entry = manifest["tensors"].get(name)
    if entry is None:
        return None
    q = _build_quant(entry, name, reader, group_size)
    scaler = _build_scaler(entry, name, reader)
    bias_key = name.replace(".weight", ".bias")
    bias = reader.get(bias_key) if reader.has(bias_key) else None
    return QuantLinear(q, bias, scaler, entry["shape"][0], entry["shape"][1])


def indice_origine(architectures) -> str:
    """Ce que l'architecture declaree dit de la convention de `a_log`.

    `a_log` porte soit log(A) (sources HF), soit deja -A (convertisseur GGUF).
    **Le signe ne departage rien** : sur les modeles reels, log(A) et -A
    tombent tous deux dans les negatifs, et la plage [-5,6, -1,1] est justement
    ce que l'inversion produit a partir de [-0,34, -0,004]. La transformation
    est quasi involutive sur cette plage : aucun controle de vraisemblance ne
    peut trancher, et un seuil donnerait une fausse assurance.

    Ce qui tranche est l'ORIGINE. Une architecture HF authentique
    (`Qwen3_5ForConditionalGeneration`, `...ForCausalLM` d'une famille connue)
    porte log(A). Un dossier issu d'un GGUF porte l'architecture que notre
    propre lecteur fabrique — `LlamaForCausalLM` — et la convention inverse.
    L'indice est rendu tel quel : il oriente, il ne decide pas.
    """
    archs = [str(a) for a in (architectures or [])]
    if any("ForConditionalGeneration" in a or "Qwen3_5" in a for a in archs):
        return "source HF probable, a_log = log(A), pas d'inversion"
    if archs == ["LlamaForCausalLM"]:
        return ("architecture generique, typique d'un dossier issu d'un GGUF "
                "— l'inversion est probablement necessaire")
    return "origine indeterminee"


def _kv_format(plan: Plan, device: str) -> str:
    """Format du cache KV d'un appareil : celui du palier du plan (écrit dans
    le manifeste à la conversion), ou `ACVRAM_KV_FORMAT` s'il est posé — le
    plan vient du manifeste, l'environnement doit donc s'appliquer ICI, pas
    seulement à la construction des paliers (tiering.build_tiers)."""
    from ..memory.tiering import _KV_FORMAT
    return _KV_FORMAT or next((t.kv_format for t in plan.tiers if t.name == device), "int8")


def load_model(path: str, plan: Optional[Plan] = None,
               dtype: torch.dtype = torch.bfloat16,
               max_model_len: Optional[int] = None,
               max_concurrent_seqs: Optional[int] = None,
               device_override: Optional[str] = None) -> LoadedModel:
    """Charge en mémoire un répertoire de modèle converti, placé selon le plan."""
    with open(os.path.join(path, "acvram_manifest.json"), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    spec = ModelSpec(**{k: v for k, v in manifest["model"].items()
                        if k in ModelSpec.__dataclass_fields__})
    # Éco par défaut (sage-eco-2700-defaut-19-09 § 1) : le processus qui charge
    # un moteur pose l'horloge SM (`-lgc 2700,2700`) AVANT le premier octet
    # chargé, une fois par processus ; rendue par `Engine.fermer`, atexit et
    # SIGTERM/SIGINT. Sans carte (CUDA_VISIBLE_DEVICES vide) ou sous
    # ACVRAM_ECO=off : rien n'est exécuté, l'état est nommé quand même.
    from .. import eco as _eco
    _eco.poser_pour_ce_processus()
    # `ModelSpec.to_dict()` ne serialise pas `raw`, et le manifeste ne porte
    # donc AUCUNE des cles brutes de la configuration. Or le chargeur en lit
    # certaines — `gdn_a_log_negexp` decide si `a_log` doit etre retransforme,
    # et sans lui la decroissance des couches recurrentes est calculee sur un
    # facteur deja transforme : **onze pour cent d'ecart au lieu d'un**, mesure
    # le 8/09/2026 contre llama.cpp sur la couche 0 (a_log charge a -0,3379,
    # la valeur du disque, la ou -1,085 etait attendu).
    #
    # Le piege qui m'a fait rétracter a tort ce diagnostic le matin meme :
    # `load_model_spec(dossier)` lit `config.json` et porte bien le drapeau,
    # tandis que `load_model` reconstruit le spec depuis le MANIFESTE. Verifier
    # l'un ne dit rien de l'autre. On complete donc `raw` depuis la
    # configuration, ce qui repare aussi les modeles deja convertis.
    # Les manifestes recents portent les cles utiles (CLES_BRUTES_UTILES) ; les
    # anciens non. On complete depuis la configuration, et l'on DIT quand on ne
    # peut pas — servir en silence un modele dont on ignore la convention est
    # exactement ce qui a coute la journee du 8/09.
    manquantes = [c for c in ModelSpec.CLES_BRUTES_UTILES
                  if c not in manifest["model"]]
    if manquantes:
        chemin_cfg = os.path.join(path, "config.json")
        if os.path.isfile(chemin_cfg):
            try:
                with open(chemin_cfg, "r", encoding="utf-8") as fh:
                    spec.raw = json.load(fh)
            except Exception as e:                       # pragma: no cover
                print(f"[acvram] config.json illisible ({e}) : les conventions "
                      f"{manquantes} sont inconnues, le modele peut etre servi "
                      f"faux sans erreur", flush=True)
        else:
            print(f"[acvram] ni le manifeste ni config.json ne portent "
                  f"{manquantes} : conventions inconnues, le modele peut etre "
                  f"servi faux sans erreur", flush=True)
    else:
        spec.raw = {c: manifest["model"][c]
                    for c in ModelSpec.CLES_BRUTES_UTILES
                    if c in manifest["model"]}
    if (spec.layer_types and "linear_attention" in spec.layer_types
            and "gdn_a_log_negexp" not in (spec.raw or {})
            and "gdn_a_log_negexp" not in manifest["model"]):
        # `a_log` porte soit log(A) (sources HF), soit deja -A (convertisseur
        # GGUF). **Le signe ne departage rien** : sur les modeles reels, log(A)
        # et -A tombent tous deux dans les negatifs, et la plage [-5,6, -1,1]
        # est justement ce que l'inversion produit a partir de [-0,34, -0,004].
        # Les deux lectures sont numeriquement plausibles ; aucun controle de
        # vraisemblance ne peut trancher.
        #
        # Ce qui tranche est la SOURCE : une architecture HF authentique
        # (Qwen3_5ForConditionalGeneration...) porte log(A) ; un dossier issu
        # d'un GGUF porte l'architecture que le lecteur fabrique
        # (LlamaForCausalLM) et la convention inverse. On le dit plutot que de
        # choisir en silence — un modele servi avec la mauvaise convention a
        # une decroissance jusqu'a cent fois trop forte et ne le signale pas.
        print(f"[acvram] couches recurrentes sans convention declaree pour "
              f"a_log ({indice_origine((spec.raw or {}).get('architectures'))})"
              f" : verifier une sortie de couche contre une reference avant de "
              f"servir ce modele", flush=True)
    if plan is None:
        plan = _plan_from_manifest(manifest, spec, max_model_len=max_model_len,
                                   max_concurrent_seqs=max_concurrent_seqs)
    _avertir_noyaux()
    reader = _ShardReader(path, manifest["weight_map"])
    group_size = manifest.get("options", {}).get("group_size", 128)

    def dev(name: str) -> torch.device:
        return torch.device(device_override or name)

    # plongements : une simple collecte, donc la RAM ne coûte qu'une petite copie par jeton
    embed = reader.get("model.embed_tokens.weight").to(dtype)
    embed_dev = dev(plan.embed_device) if plan.embed_device != "cpu" \
        else torch.device("cpu")
    embed = embed.to(embed_dev)
    if embed_dev.type == "cpu":
        # Sans copie, la table reste un mmap du fichier safetensors : chaque
        # jeton nouveau touche une page non chargée — une lecture disque de
        # 100 ms au milieu du décodage. Résidente en RAM épinglée, elle se
        # collecte en microsecondes et se copie sans étape intermédiaire.
        # pin_memory() exige CUDA ; sur processeur pur (tests), on garde la
        # copie sans épinglage — performance dégradée, justesse identique.
        embed = embed.contiguous().clone()
        if torch.cuda.is_available():
            embed = embed.pin_memory()

    rope_gemma = None
    rope = RotaryEmbedding(spec.rotary_dim or spec.head_dim,
                           spec.max_position_embeddings,
                           spec.rope_theta, spec.rope_scaling)

    # Profil de routage persistant (bead pds, point 1) : lu une seule fois
    # pour tout le modèle, `None` si aucun n'existe encore — cas normal au
    # premier chargement, pas une erreur (`expert_usage.charger`).
    _profil_usage = expert_usage.charger(
        os.path.join(path, expert_usage.NOM_PROFIL))
    # Tampons GPU des poids DENSES exilés (attention, MLP dense, expert
    # partagé) : UN pool par appareil, `_DENSE_SLOTS` jeux par forme de
    # tenseur, partagés par toutes les couches. Avant : chaque StreamedWeight
    # gardait DEUX copies privées de son poids sur la carte pour toute la vie
    # du process (layers.py StreamedWeight._ensure) — exiler une couche dense
    # DOUBLAIT son empreinte VRAM au lieu de la libérer : Llama-3.3-70B-nvfp4,
    # 52/80 couches exilées = 18,6 Gio épinglés à l'hôte et ~37 Gio réclamés à
    # la carte au premier préfill (OOM 448 Mio, Laure verdict-palier2-nemotron-
    # 17-09), quel que soit le budget. Avec le pool : ≤ _DENSE_SLOTS × la
    # taille de chaque forme distincte (≈ 1-2 Gio sur le 70B), compté dans la
    # réserve du plan (_reserve_prefill).
    _pools_denses: dict = {}

    def _pool_dense(device: torch.device):
        from .layers import ExpertPool
        if device.type != "cuda":
            return None
        cle = str(device)
        if cle not in _pools_denses:
            _pools_denses[cle] = ExpertPool(device, _DENSE_SLOTS, dense=True)
        return _pools_denses[cle]

    layers: list[DecoderLayer] = []
    caches: dict[int, PagedKVCache] = {}
    # Caches KV differes : voir la fusion des projections plus bas, qui a
    # besoin de place libre au moment ou elle concatene.
    a_allouer: list = []
    _borner_kv_avec_exil(plan, manifest, dev, spec, max_model_len,
                         reserve=_reserve_prefill(spec, max_model_len, manifest, plan))
    kv_blocks = _kv_blocks_per_device(plan, spec, max_model_len)

    # Arène épinglée pour les poids exilés, dimensionnée ICI parce que le plan
    # vient d'être arrêté : avant lui, la taille exilée n'est pas connue et le
    # pool serait un chiffre deviné.
    #
    # Pourquoi une arène plutôt que `pin_memory()` par poids : l'allocateur
    # hôte de PyTorch arrondit chaque allocation à la PUISSANCE DE 2 supérieure.
    # Nos tenseurs d'experts font 3,00 Mio (768 x 2048 en bf16) et sont donc
    # arrondis à 4 — mesuré le 8/09/2026, facteur 1,333 en régime asymptotique,
    # soit 46,01 Gio épinglés là où les poids en pèsent 33,76. Une arène dont
    # la taille est une puissance de 2 ne paie rien, et une vue prise dedans
    # est elle-même épinglée. Gain mesuré sur trois tailles : ~25 %.
    _attn_r, _mlp_r, _embed_r, _head_r = _octets_reels(manifest)
    _exiles = sum(_mlp_r.get(l.index, l.mlp_bytes)
                  for l in plan.layers if l.mlp_storage == "cpu")
    _exiles += sum(_attn_r.get(l.index, l.attn_bytes)
                   for l in plan.layers if l.attn_storage == "cpu")
    if _exiles > 0:
        from .layers import reserver_pool
        _pool_hote = reserver_pool(_exiles)
        if _pool_hote is not None:
            print(f"[acvram] arène épinglée : {_exiles / 2**30:.2f} Gio réservés "
                  f"pour les poids exilés", flush=True)

    for lp in plan.layers:
        i = lp.index
        p = f"model.layers.{i}."
        d = dev(lp.exec_device)
        streamed_attn = lp.attn_storage == "cpu"
        streamed_mlp = lp.mlp_storage == "cpu"

        def lin(suffix: str, streamed: bool) -> QuantLinear:
            m = _linear(p + suffix, manifest, reader, group_size)
            if m is None:
                raise KeyError(f"tenseur manquant {p + suffix}")
            return m.to_device(d, streamed=streamed,
                               pool=_pool_dense(d) if streamed else None)

        # Le MLP peut vivre et s'exécuter sur le processeur pendant que l'attention reste sur le GPU.
        mlp_on_cpu = (lp.mlp_storage == "cpu"
                      and getattr(lp, "mlp_exec", "gpu") == "cpu")
        mlp_dev = torch.device("cpu") if mlp_on_cpu else d
        streamed_mlp = streamed_mlp and not mlp_on_cpu

        def mlin(suffix: str) -> QuantLinear:
            m = _linear(p + suffix, manifest, reader, group_size)
            if m is None:
                raise KeyError(f"tenseur manquant {p + suffix}")
            return m.to_device(mlp_dev, streamed=streamed_mlp,
                               pool=_pool_dense(mlp_dev) if streamed_mlp else None)

        # Les experts d'une couche exilée partagent un pool de tampons GPU
        # dimensionné pour les experts routés d'un jeton, au lieu de deux
        # copies privées chacun (voir ExpertPool).
        # Deux fois les experts routés, plus marge : le chemin direct de
        # décodage lance TOUTES les copies d'une couche avant le premier
        # calcul, et gate et up partagent la même disposition — vingt copies
        # en vol pour dix experts. Avec topk+1 emplacements, la onzième copie
        # réécrivait un emplacement que le GEMV du jeton courant n'avait pas
        # encore lu (l'événement « libre » venait du jeton précédent) :
        # écart de 4e-2 au lieu de 5e-4 au test à sec du 8/09.
        # Le repli doit etre celui du bloc, jamais plus petit. Il valait 2 ici
        # et 8 la (`MoEBlock(..., spec.num_experts_per_tok or 8)`) : sur un
        # modele dont la configuration ne porte pas le nombre d'experts par
        # jeton, le bloc en routait huit et le pool n'avait que six
        # emplacements pour les seize copies d'entree. Depuis le 8/09/2026 le
        # pool refuse bruyamment d'ecraser un emplacement en vol au lieu de
        # rendre les octets d'un autre expert ; il faut encore qu'il en ait
        # assez.
        # Un placement PAR EXPERT (bead pds) peut vouloir des experts
        # streamés même sur une couche autrement résidente (`streamed_mlp`
        # faux) : le pool doit alors exister aussi, sinon `elin` recevrait
        # `streamed=True` sans tampon où copier.
        experts_par_couche = lp.experts_residents is not None
        pool = (ExpertPool(mlp_dev, 2 * (spec.num_experts_per_tok or 8) + 2)
                if streamed_mlp or experts_par_couche else None)

        def elin(suffix: str, streamed: Optional[bool] = None) -> QuantLinear:
            # `streamed=None` : comportement d'aujourd'hui, uniforme pour
            # toute la couche (`streamed_mlp`). Un appelant qui connaît un
            # placement PAR EXPERT (bead pds, point 1) passe `True`/`False`
            # explicitement — additif, aucun appel existant ne change.
            m = _linear(p + suffix, manifest, reader, group_size)
            if m is None:
                raise KeyError(f"tenseur manquant {p + suffix}")
            s = streamed_mlp if streamed is None else streamed
            return m.to_device(mlp_dev, streamed=s, pool=pool if s else None)

        def routeur(suffix: str) -> QuantLinear:
            # Le routeur pèse quelques mégaoctets et MoEBlock lit son poids
            # brut (`router.qweight.weight`) sans passer par le mécanisme de
            # transfert : un routeur « streamed » laissait ce poids sur le
            # processeur et faisait planter la première requête d'un
            # perceptron exilé (« mat2 is on cpu »). Il reste résident, là où
            # s'exécute le perceptron.
            m = _linear(p + suffix, manifest, reader, group_size)
            if m is None:
                raise KeyError(f"tenseur manquant {p + suffix}")
            return m.to_device(mlp_dev, streamed=False)

        def norm_opt(suffix: str) -> Optional[RMSNorm]:
            """RMSNorm facultative — absente des modeles sans QK-norm."""
            w = reader.get(p + suffix) if p + suffix in manifest["tensors"] else None
            return None if w is None else RMSNorm(w.to(dtype).to(d),
                                                  spec.rms_norm_eps)

        def faire_mlp() -> torch.nn.Module:
            if manifest["tensors"].get(p + "mlp.gate.weight") is None:
                return MLP(mlin("mlp.gate_proj.weight"),
                           mlin("mlp.up_proj.weight"),
                           mlin("mlp.down_proj.weight"))
            router = routeur("mlp.gate.weight")

            # Compte d'abord (aucun tenseur chargé : de simples clés sondées)
            # pour pouvoir décider un placement PAR EXPERT avant de construire
            # quoi que ce soit — bead pds, point 1.
            n_experts = 0
            while manifest["tensors"].get(
                    p + f"mlp.experts.{n_experts}.gate_proj.weight"):
                n_experts += 1

            residents = None
            if (lp.experts_residents is not None
                    and 0 < lp.experts_residents < n_experts):
                profil_couche = (_profil_usage or {}).get(i)
                residents, source = expert_usage.decider_residents(
                    profil_couche, n_experts, lp.experts_residents)
                print(f"[acvram] couche {i} : placement par expert, "
                      f"{lp.experts_residents}/{n_experts} résidents "
                      f"({source})", flush=True)

            experts = []
            for e in range(n_experts):
                # `residents is None` : comportement d'aujourd'hui, uniforme
                # (`elin` retombe sur `streamed_mlp`). Sinon, CHAQUE expert
                # reçoit une décision explicite — plus de repli implicite une
                # fois un placement par expert actif pour cette couche.
                s = None if residents is None else (e not in residents)
                experts.append(MLP(
                    elin(f"mlp.experts.{e}.gate_proj.weight", streamed=s),
                    elin(f"mlp.experts.{e}.up_proj.weight", streamed=s),
                    elin(f"mlp.experts.{e}.down_proj.weight", streamed=s)))
            shared = None
            shared_gate = None
            if manifest["tensors"].get(p + "mlp.shared_expert.gate_proj.weight"):
                shared = MLP(
                    mlin("mlp.shared_expert.gate_proj.weight"),
                    mlin("mlp.shared_expert.up_proj.weight"),
                    mlin("mlp.shared_expert.down_proj.weight"))
                if manifest["tensors"].get(p + "mlp.shared_expert_gate.weight"):
                    shared_gate = reader.get(
                        p + "mlp.shared_expert_gate.weight").to(dtype).to(mlp_dev)
            score_bias = None
            if manifest["tensors"].get(p + "mlp.gate.e_score_correction_bias"):
                score_bias = reader.get(
                    p + "mlp.gate.e_score_correction_bias").float().to(mlp_dev)
            bloc = MoEBlock(router, experts, spec.num_experts_per_tok or 2,
                            shared, shared_gate=shared_gate,
                            norm_topk_prob=bool(spec.raw.get("norm_topk_prob", True)),
                            scoring=spec.router_scoring,
                            score_bias=score_bias,
                            routed_scale=spec.routed_scaling_factor)
            if residents is not None:
                # `_pin_experts` : ce que `Engine.__init__` lit pour peupler
                # `_pin` (REPIN réel, `memory/repin.py`). La table d'adresses
                # n'existe que pour NVFP4 (contrat `memory/table_adresses.py`
                # — `qweight`/`block_scale` y sont spécifiques au format) :
                # un modèle INT4 obtient le placement par expert (et le repli
                # correct par `elin`/streamed) mais pas encore la table, que
                # le noyau de Laurine ne consomme de toute façon que pour
                # NVFP4 aujourd'hui.
                bloc._pin_experts = residents
                if all(isinstance(m.gate_proj.qweight, NVFP4Tensor)
                      for m in experts):
                    try:
                        bloc._table_qw = {}
                        bloc._table_bscale = {}
                        for nom in ("gate_proj", "up_proj", "down_proj"):
                            tq, tb = construire_table(experts, nom, device=d)
                            bloc._table_qw[nom] = tq
                            bloc._table_bscale[nom] = tb
                    except ValueError as exc:
                        # Une adresse nulle (règle 5 : verifier_table peut
                        # rendre faux) — un expert n'a encore aucune adresse
                        # réelle. Pas fatal : le chemin par expert existant
                        # reste correct sans table, seulement plus lent.
                        print(f"[acvram] couche {i} : table d'adresses non "
                              f"construite ({exc}) — repli sur le chemin par "
                              f"expert existant", flush=True)
            return bloc

        if spec.model_type in ("gemma4", "gemma4_text"):
            if rope_gemma is None:
                rope_gemma = (
                    RotaryEmbedding(spec.head_dim, spec.max_position_embeddings,
                                    spec.rope_theta_swa or 1e4, None, d, dtype),
                    RotaryEmbedding(spec.global_head_dim, spec.max_position_embeddings,
                                    spec.rope_theta, None, d, dtype,
                                    n_active=int(spec.partial_rotary_factor_full
                                                 * spec.global_head_dim / 2)))
            local = spec.layer_types[i] == "sliding_attention"
            hd = spec.head_dim if local else spec.global_head_dim
            nkv = spec.num_key_value_heads if local else spec.num_global_key_value_heads
            a_v = manifest["tensors"].get(p + "self_attn.v_proj.weight") is not None
            attn = Attention(
                spec,
                lin("self_attn.q_proj.weight", streamed_attn),
                lin("self_attn.k_proj.weight", streamed_attn),
                lin("self_attn.v_proj.weight", streamed_attn) if a_v else None,
                lin("self_attn.o_proj.weight", streamed_attn),
                rope_gemma[0] if local else rope_gemma[1],
                norm_opt("self_attn.q_norm.weight"),
                norm_opt("self_attn.k_norm.weight"),
                n_kv_heads=nkv, head_dim=hd, scale=1.0,
                v_norm_eps=spec.rms_norm_eps, k_eq_v=not a_v,
                window=spec.sliding_window if local else 0)
            mlp_g = MLP(mlin("mlp.gate_proj.weight"), mlin("mlp.up_proj.weight"),
                        mlin("mlp.down_proj.weight"), act=spec.mlp_activation)
            n4 = lambda suffix: RMSNorm(reader.get(p + suffix).to(dtype).to(d), spec.rms_norm_eps)
            out_scale = None
            if manifest["tensors"].get(p + "layer_scalar.weight") is not None:
                out_scale = reader.get(p + "layer_scalar.weight").to(torch.float32).to(d).reshape(-1)[0]
            moe = n1 = n2 = p2 = None
            if manifest["tensors"].get(p + "mlp.gate.weight") is not None:
                experts = []
                e = 0
                while manifest["tensors"].get(p + f"mlp.experts.{e}.gate_proj.weight"):
                    experts.append(MLP(elin(f"mlp.experts.{e}.gate_proj.weight"),
                                       elin(f"mlp.experts.{e}.up_proj.weight"),
                                       elin(f"mlp.experts.{e}.down_proj.weight"),
                                       act=spec.mlp_activation))
                    e += 1
                moe = MoEBlockGemma(
                    routeur("mlp.gate.weight"), experts, spec.num_experts_per_tok or 8,
                    reader.get(p + "mlp.router_scale.weight").to(torch.float32).to(mlp_dev),
                    reader.get(p + "mlp.per_expert_scale.weight").to(torch.float32).to(mlp_dev),
                    spec.rms_norm_eps)
                n1 = n4("post_feedforward_layernorm_1.weight")
                n2 = n4("post_feedforward_layernorm_2.weight")
                p2 = n4("pre_feedforward_layernorm_2.weight")
            layers.append(DecoderLayerGemma(
                i, attn, mlp_g, n4("input_layernorm.weight"),
                n4("post_attention_layernorm.weight"),
                n4("pre_feedforward_layernorm.weight"),
                n4("post_feedforward_layernorm.weight"), out_scale, d,
                moe=moe, post_ffn_norm_1=n1, post_ffn_norm_2=n2, pre_ffn_norm_2=p2))
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = _kv_format(plan, lp.exec_device)
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=nkv, head_dim=hd,
                    num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "muse_glimmer":
            local = spec.layer_types[i] == "sliding_attention"
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn), rope,
                             norm_opt("self_attn.q_norm.weight"),
                             norm_opt("self_attn.k_norm.weight"),
                             window=spec.sliding_window if local else 0,
                             output_gate=True)
            mlp_g = MLP(mlin("mlp.gate_proj.weight"), mlin("mlp.up_proj.weight"),
                        mlin("mlp.down_proj.weight"), act=spec.mlp_activation)
            eps_post = spec.post_norm_eps or spec.rms_norm_eps
            n4 = lambda suffix, e: RMSNorm(reader.get(p + suffix).to(dtype).to(d), e)
            layers.append(DecoderLayerGemma(
                i, attn, mlp_g, n4("input_layernorm.weight", spec.rms_norm_eps),
                n4("post_attention_layernorm.weight", eps_post),
                n4("pre_feedforward_layernorm.weight", spec.rms_norm_eps),
                n4("post_feedforward_layernorm.weight", eps_post), None, d))
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = _kv_format(plan, lp.exec_device)
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "starcoder2":
            from .layers import LayerNorm
            ln = lambda suffix: LayerNorm(
                reader.get(p + suffix + ".weight").to(dtype).to(d),
                reader.get(p + suffix + ".bias").to(dtype).to(d) if reader.has(p + suffix + ".bias") else None,
                spec.rms_norm_eps)
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn), rope,
                             window=spec.sliding_window)
            mlp_s = MLP2(mlin("mlp.up_proj.weight"), mlin("mlp.down_proj.weight"),
                         spec.mlp_activation)
            couche = DecoderLayer(i, attn, mlp_s, ln("input_layernorm"),
                                  ln("post_attention_layernorm"), d, mlp_dev)
            layers.append(couche)
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = _kv_format(plan, lp.exec_device)
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "falcon_h1":
            from .mamba2 import Mamba2Mixer
            petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
            cb = petit("mamba.conv1d.bias") if manifest["tensors"].get(p + "mamba.conv1d.bias") else None
            mamba = Mamba2Mixer(
                in_proj=lin("mamba.in_proj.weight", False), out_proj=lin("mamba.out_proj.weight", False),
                conv_weight=petit("mamba.conv1d.weight"), conv_bias=cb,
                dt_bias=petit("mamba.dt_bias.weight"), A=petit("mamba.A.weight"),
                D=petit("mamba.D.weight"), norm_weight=petit("mamba.norm.weight"),
                num_heads=spec.mamba_num_heads, head_dim=spec.mamba_head_dim,
                n_groups=spec.mamba_n_groups, state_size=spec.mamba_state_size,
                eps=spec.rms_norm_eps).to(d)
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn), rope)
            mlp_f = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(reader.get(p + "post_attention_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerParallel(i, attn, mamba, mlp_f, in_norm, post_norm, d))
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = _kv_format(plan, lp.exec_device)
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "nemotron_h":
            kind = spec.layer_types[i]
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            if kind == "mamba":
                from .mamba2 import Mamba2Mixer
                petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
                cb = petit("mamba.conv1d.bias") if manifest["tensors"].get(p + "mamba.conv1d.bias") else None
                bloc = Mamba2Mixer(
                    in_proj=lin("mamba.in_proj.weight", False),
                    out_proj=lin("mamba.out_proj.weight", False),
                    conv_weight=petit("mamba.conv1d.weight"), conv_bias=cb,
                    dt_bias=petit("mamba.dt_bias.weight"), A=petit("mamba.A.weight"),
                    D=petit("mamba.D.weight"), norm_weight=petit("mamba.norm.weight"),
                    num_heads=spec.mamba_num_heads, head_dim=spec.mamba_head_dim,
                    n_groups=spec.mamba_n_groups, state_size=spec.mamba_state_size,
                    eps=spec.rms_norm_eps).to(d)
                layers.append(DecoderLayerGDN(i, bloc, None, in_norm, None, d, mlp_device=mlp_dev))
                continue
            if kind in ("mlp", "moe"):
                if kind == "mlp":
                    mlp_n = MLP2(mlin("mlp.up_proj.weight"), mlin("mlp.down_proj.weight"), "relu2")
                else:
                    router = routeur("mlp.gate.weight"); experts = []; e = 0
                    while manifest["tensors"].get(p + f"mlp.experts.{e}.up_proj.weight"):
                        experts.append(MLP2(elin(f"mlp.experts.{e}.up_proj.weight"),
                                            elin(f"mlp.experts.{e}.down_proj.weight"), "relu2"))
                        e += 1
                    shared = None
                    if manifest["tensors"].get(p + "mlp.shared_expert.up_proj.weight"):
                        shared = MLP2(mlin("mlp.shared_expert.up_proj.weight"),
                                      mlin("mlp.shared_expert.down_proj.weight"), "relu2")
                    bias = None
                    if manifest["tensors"].get(p + "mlp.gate.e_score_correction_bias"):
                        # le routage vit avec les experts (RAM hôte si le plan
                        # les y a mis) : même appareil que les scores
                        bias = reader.get(p + "mlp.gate.e_score_correction_bias").float().to(mlp_dev)
                    mlp_n = MoEBlock(router, experts, spec.num_experts_per_tok or 2, shared,
                                     norm_topk_prob=bool(spec.raw.get("norm_topk_prob", True)),
                                     scoring=spec.router_scoring, score_bias=bias,
                                     routed_scale=spec.routed_scaling_factor)
                couche = DecoderLayer(i, None, mlp_n, in_norm, None, d, mlp_dev)
                layers.append(couche)
                continue
            # attention (sans RoPE), cache paginé
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn),
                             None if not spec.attention_rope else rope)
            couche = DecoderLayer(i, attn, None, in_norm, None, d, d)
            layers.append(couche)
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = _kv_format(plan, lp.exec_device)
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type in ("lfm2", "lfm2_moe") and spec.layer_types[i] == "conv":
            from .lfm2 import LFM2ShortConv
            bloc = LFM2ShortConv(
                in_proj=lin("conv.in_proj.weight", False),
                out_proj=lin("conv.out_proj.weight", False),
                conv_weight=reader.get(p + "conv.conv.weight").to(torch.float32).to(d),
                dim=spec.hidden_size).to(d)
            mlp_c = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(reader.get(p + "post_attention_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerGDN(i, bloc, mlp_c, in_norm, post_norm, d, mlp_device=mlp_dev))
            continue

        # `kimi_linear` (recurrence lineaire GDN, pas de KV compresse) et les
        # MLA (kv_lora_rank > 0, ModelSpec.est_mla) partagent ce chemin de
        # chargement, distingues plus bas par layer_types[i]. Le critere MLA
        # n'est plus une liste de model_type : glm4_moe_lite (GLM-4.7-Flash)
        # en manquait (bead anticitoyen-vram-992, 14/09).
        est_kimi = spec.model_type == "kimi_linear" or spec.est_mla
        if est_kimi:
            petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
            petit16 = lambda suffix: reader.get(p + suffix).to(dtype).to(d)
            if spec.layer_types[i] == "linear_attention":
                from .kda import KimiDeltaAttention
                bloc = KimiDeltaAttention(
                    q_proj=lin("linear_attn.q_proj.weight", False),
                    k_proj=lin("linear_attn.k_proj.weight", False),
                    v_proj=lin("linear_attn.v_proj.weight", False),
                    out_proj=lin("linear_attn.out_proj.weight", False),
                    f_a=lin("linear_attn.f_a.weight", False),
                    f_b=lin("linear_attn.f_b.weight", False),
                    g_a=lin("linear_attn.g_a.weight", False),
                    g_b=lin("linear_attn.g_b.weight", False),
                    beta=lin("linear_attn.beta.weight", False),
                    conv_q=petit("linear_attn.conv1d_q.weight"),
                    conv_k=petit("linear_attn.conv1d_k.weight"),
                    conv_v=petit("linear_attn.conv1d_v.weight"),
                    dt_bias=petit("linear_attn.dt_bias.weight"),
                    a=petit("linear_attn.a.weight"),
                    norm_weight=petit("linear_attn.norm.weight"),
                    num_heads=spec.linear_num_value_heads,
                    head_dim=spec.linear_value_head_dim,
                    eps=spec.rms_norm_eps).to(d)
                bloc.fuse_projections()
            else:
                from .mla import MLAttention
                q_lora = manifest["tensors"].get(p + "self_attn.q_a_proj.weight") is not None
                rope_mla = None
                if spec.mla_rope:
                    if "rope_mla_partage" not in dir():
                        rope_mla_partage = RotaryEmbedding(
                            spec.qk_rope_head_dim, spec.max_position_embeddings,
                            spec.rope_theta, spec.rope_scaling, d, dtype)
                    rope_mla = rope_mla_partage
                bloc = MLAttention(
                    q_proj=None if q_lora else lin("self_attn.q_proj.weight", False),
                    q_a_proj=lin("self_attn.q_a_proj.weight", False) if q_lora else None,
                    q_a_norm=petit16("self_attn.q_a_layernorm.weight") if q_lora else None,
                    q_b_proj=lin("self_attn.q_b_proj.weight", False) if q_lora else None,
                    rope=rope_mla,
                    kv_a_proj=lin("self_attn.kv_a_proj_with_mqa.weight", False),
                    o_proj=lin("self_attn.o_proj.weight", False),
                    kv_a_norm=petit16("self_attn.kv_a_layernorm.weight"),
                    k_b=petit16("self_attn.k_b_proj.weight"),
                    v_b=petit16("self_attn.v_b_proj.weight"),
                    num_heads=spec.num_attention_heads,
                    qk_nope=spec.qk_nope_head_dim,
                    qk_rope=spec.qk_rope_head_dim,
                    kv_lora_rank=spec.kv_lora_rank,
                    v_dim=spec.v_head_dim,
                    eps=spec.rms_norm_eps).to(d)
                rs = spec.rope_scaling or {}
                if str(rs.get("rope_type") or rs.get("type") or "") == "yarn":
                    # YaRN : la sortie d'attention est rescalée par mscale² (DeepSeek)
                    import math as _m
                    msc = float(rs.get("mscale_all_dim") or rs.get("mscale") or 0.0)
                    fac = float(rs.get("factor") or 1.0)
                    if msc and fac > 1.0:
                        bloc.scale = bloc.scale * (0.1 * msc * _m.log(fac) + 1.0) ** 2
                bloc.fuse_projections()
            mlp_kimi = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight"
                                         ).to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(
                reader.get(p + "post_attention_layernorm.weight"
                           ).to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerGDN(i, bloc, mlp_kimi,
                                          in_norm, post_norm, d, mlp_device=mlp_dev))
            continue

        est_gdn = bool(spec.layer_types) and \
            spec.layer_types[i] == "linear_attention"
        if est_gdn:
            from .gdn import GatedDeltaNet, gdn_available
            if not gdn_available():
                raise RuntimeError(
                    f"la couche {i} est à récurrence linéaire (Gated DeltaNet) "
                    f"mais `transformers` n'est pas installé avec le support "
                    f"nécessaire : pip install -e '.[gdn]'")
            petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
            gdn = GatedDeltaNet(
                qkv=lin("linear_attn.qkv.weight", False),
                gate=lin("linear_attn.gate.weight", False),
                alpha=lin("linear_attn.alpha.weight", False),
                beta=lin("linear_attn.beta.weight", False),
                out=lin("linear_attn.out.weight", False),
                conv_weight=petit("linear_attn.conv1d.weight"),
                dt_bias=petit("linear_attn.dt_bias.weight"),
                # le convertisseur GGUF stocke -exp(A_log) (drapeau
                # gdn_a_log_negexp) ; les sources HF portent A_log tel quel
                a_log=(torch.log(torch.clamp(
                    -petit("linear_attn.a_log.weight"), min=1e-12))
                       if spec.raw.get("gdn_a_log_negexp")
                       else petit("linear_attn.a_log.weight")),
                norm_weight=petit("linear_attn.norm.weight"),
                num_k_heads=spec.linear_num_key_heads,
                num_v_heads=spec.linear_num_value_heads,
                head_k_dim=spec.linear_key_head_dim,
                head_v_dim=spec.linear_value_head_dim,
                eps=spec.rms_norm_eps).to(d)
            mlp_gdn = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight"
                                         ).to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(
                reader.get(p + "post_attention_layernorm.weight"
                           ).to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerGDN(i, gdn, mlp_gdn, in_norm, post_norm, d, mlp_device=mlp_dev))
            continue

        attn = Attention(
            spec,
            lin("self_attn.q_proj.weight", streamed_attn),
            lin("self_attn.k_proj.weight", streamed_attn),
            lin("self_attn.v_proj.weight", streamed_attn),
            lin("self_attn.o_proj.weight", streamed_attn),
            rope,
            norm_opt("self_attn.q_norm.weight"),
            norm_opt("self_attn.k_norm.weight"),
            output_gate=spec.attn_output_gate)

        mlp: torch.nn.Module = faire_mlp()

        in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d),
                          spec.rms_norm_eps)
        post_norm = RMSNorm(
            reader.get(p + "post_attention_layernorm.weight").to(dtype).to(d),
            spec.rms_norm_eps)

        couche = DecoderLayer(i, attn, mlp, in_norm, post_norm, d, mlp_dev)
        couche.residual_multiplier = spec.residual_multiplier
        layers.append(couche)

        n_blocks = kv_blocks.get(lp.exec_device, 0)
        if n_blocks:
            kv_fmt = _kv_format(plan, lp.exec_device)
            a_allouer.append((i, KVCacheConfig(
                num_layers=1, num_kv_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim, num_blocks=n_blocks,
                dtype=kv_fmt, device=str(d))))

    # Projections empilées : gate/up des MLP (denses, experts partagés) et
    # q/k/v de l'attention — une GEMV au lieu de deux ou trois par couche.
    # Les experts d'un MoE en sont exclus : ils passent par le chemin groupé,
    # qui empile déjà les 128 experts, et les fusionner un à un doublerait
    # leurs poids sans rien accélérer.
    #
    # AVANT l'allocation des caches KV, et ce n'est pas cosmétique :
    # l'empilement alloue le tenseur concaténé avant de libérer les deux
    # sources, soit un pic de la taille d'une paire (283 Mio sur Qwen2.5-14B).
    # Le budget KV remplit la carte jusqu'à la marge — 119 Mio libres après
    # chargement — et l'allocateur devait purger son cache à chaque couche pour
    # trouver la place : il y parvenait, en laissant la trace
    # « memory allocation failed with OOM » à chaque paire, et en fragmentant.
    # Ici la fusion se fait pendant que le budget KV est encore libre.
    for layer in layers:
        experts = {id(e) for m in layer.modules() if isinstance(m, MoEBlock)
                   for e in m.experts}
        for m in layer.modules():
            if isinstance(m, MLP) and id(m) not in experts:
                m.fuse()
            elif isinstance(m, Attention):
                m.fuse()
            elif hasattr(m, "fuse") and type(m).__name__ == "GatedDeltaNet":
                m.fuse()

    # Chaque empilement alloue son tenseur concatene avant de liberer les deux
    # sources : 0,355 Gio de pic par fusion, 95 fois. Les blocs liberes restent
    # dans le cache de l'allocateur, a des tailles qui ne correspondent plus a
    # ce qu'on demandera ensuite -- 2,49 Gio reserves non alloues apres
    # chargement, et 0,64 Gio seulement de VRAM libre. Le banc, qui dimensionne
    # ses caches plus largement que le chargement nu, tombait alors en OOM sur
    # une demande de 2 Mio. Rendre ces blocs au pilote avant d'allouer les
    # caches KV recupere 0,92 Gio, sans rien changer a ce qui est alloue.
    if torch.cuda.is_available() and a_allouer:
        torch.cuda.empty_cache()

    for i, cfg in a_allouer:
        caches[i] = PagedKVCache(cfg)

    head_dev = dev(plan.lm_head_device) if plan.lm_head_device != "cpu" \
        else torch.device("cpu")
    if spec.model_type == "starcoder2":
        from .layers import LayerNorm
        norm = LayerNorm(reader.get("model.norm.weight").to(dtype).to(head_dev),
                         reader.get("model.norm.bias").to(dtype).to(head_dev)
                         if reader.has("model.norm.bias") else None, spec.rms_norm_eps)
    else:
        norm = RMSNorm(reader.get("model.norm.weight").to(dtype).to(head_dev),
                       spec.rms_norm_eps)
    if manifest["tensors"].get("lm_head.weight"):
        lm_head = _linear("lm_head.weight", manifest, reader,
                          group_size).to_device(head_dev)
    else:
        # Plongements partagés avec la sortie. La table sert deux fois et pas de
        # la même façon : à l'entrée c'est un *gather* d'une ligne, à la sortie
        # une projection qui relit la matrice entière **à chaque jeton**. Sur
        # Qwen3-4B ce sont 742 Mio, soit 24 % des octets lus par jeton pour une
        # seule couche — mesuré à 900 µs par passe sur la 3080 Ti, dix pour cent
        # du temps de décodage. La table reste en bf16 pour le gather ; la
        # projection en prend une copie quantifiée, qui coûte de la place mais
        # divise sa lecture par deux (int8) ou par trois et demi (nvfp4).
        lm_head = QuantLinear(_tete_liee(embed.to(head_dev)))
    tetes_mtp = _charger_mtp(manifest, reader, spec, plan, group_size, dtype,
                             head_dev, rope, kv_blocks)
    reader.close()

    model = ACVRamModel(spec, embed, layers, norm, lm_head, caches, dtype)
    if tetes_mtp:
        model.mtp = tetes_mtp[0]
        print(f"[acvram] tête de prédiction multi-jetons chargée "
              f"({len(tetes_mtp)} disponible(s))")
    return LoadedModel(model, spec, plan, manifest, path)


def _charger_mtp(manifest: dict, reader: "_ShardReader", spec: ModelSpec,
                 plan: Plan, group_size: int, dtype: torch.dtype,
                 device: torch.device, rope, kv_blocks: dict[str, int]) -> list:
    """Construit les têtes ``nextn`` que la conversion a conservées.

    Absentes de la plupart des modèles ; leur absence n'est pas une erreur. Une
    tête coûte un bloc de transformeur — sur un 27B de soixante-quatre couches,
    un soixante-quatrième du modèle — et sert de brouillon spéculatif.
    """
    from .mtp import MTPHead, cles_mtp

    indices = cles_mtp(manifest)
    if not indices or os.environ.get("ACVRAM_MTP") == "non":
        return []

    tetes = []
    for n in indices:
        p = f"model.mtp.{n}."

        def lin(suffix: str):
            m = _linear(p + suffix, manifest, reader, group_size)
            return None if m is None else m.to_device(device)

        def norme(suffix: str):
            cle = p + suffix
            if not reader.has(cle):
                return None
            return RMSNorm(reader.get(cle).to(dtype).to(device),
                           spec.rms_norm_eps)

        q, k, v, o = (lin("self_attn.q_proj.weight"), lin("self_attn.k_proj.weight"),
                      lin("self_attn.v_proj.weight"), lin("self_attn.o_proj.weight"))
        eh = lin("eh_proj.weight")
        in_norm, post_norm = norme("input_layernorm.weight"), norme("post_attention_layernorm.weight")
        enorm, hnorm = norme("enorm.weight"), norme("hnorm.weight")
        fin = norme("shared_head_norm.weight") or norme("norm.weight")
        gate, up, down = (lin("mlp.gate_proj.weight"), lin("mlp.up_proj.weight"),
                          lin("mlp.down_proj.weight"))
        if None in (q, k, v, o, eh, in_norm, post_norm, enorm, hnorm, fin,
                    gate, up, down):
            print(f"[acvram] tête MTP {n} incomplète, ignorée")
            continue

        attn = Attention(spec, q, k, v, o, rope,
                         q_norm=norme("self_attn.q_norm.weight"),
                         k_norm=norme("self_attn.k_norm.weight"),
                         output_gate=spec.attn_output_gate)
        mlp = MLP(gate, up, down, spec.mlp_activation)
        couche = DecoderLayer(spec.num_layers + n, attn, mlp, in_norm,
                              post_norm, device)
        n_blocks = max(64, kv_blocks.get(str(device), 512) // 16)
        cache = PagedKVCache(KVCacheConfig(
            num_layers=1, num_kv_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim, num_blocks=n_blocks,
            dtype=_kv_format(plan, str(device)), device=str(device)))
        tetes.append(MTPHead(couche, enorm, hnorm, eh, fin, cache, device))
    return tetes


# Format de la projection de sortie quand elle partage la table des
# plongements : « bf16 » ne quantifie rien, « int8 » ou « nvfp4 » prennent une
# copie quantifiée pour la sortie seule.
_TETE_LIEE = os.environ.get("ACVRAM_TETE_LIEE", "int8").lower()


def _par_tranches(w: torch.Tensor, quant, fmt: str, lignes: int = 8192) -> Any:
    """Quantifie un très gros tenseur par paquets de lignes.

    Les quantifieurs passent par une copie float32 du tenseur entier : sur une
    table de plongements de 152 000 lignes, ce sont 1,45 Gio d'un coup, qui ne
    tiennent pas toujours à côté du modèle déjà chargé. Les lignes sont
    indépendantes — l'échelle est par ligne et par groupe — donc les traiter par
    paquets donne bit pour bit le même résultat pour un pic borné.
    """
    from ..quant.formats import INT8Tensor
    from ..quant.nvfp4 import NVFP4Tensor
    morceaux = [quant(w[i:i + lignes]) for i in range(0, w.shape[0], lignes)]
    if len(morceaux) == 1:
        return morceaux[0]
    if fmt == "int8":
        return INT8Tensor(
            torch.cat([m.qweight for m in morceaux]),
            torch.cat([m.scales for m in morceaux]),
            torch.cat([m.zeros for m in morceaux]),
            morceaux[0].group_size, tuple(w.shape))
    # NVFP4 : l'échelle globale est propre à chaque paquet, on garde la plus
    # grande et on ne peut pas recoller sans requantifier — on refuse plutôt
    # que de rendre un tenseur faux.
    raise RuntimeError("nvfp4 par tranches : échelles globales incompatibles")


def _tete_liee(embed: torch.Tensor) -> Any:
    """Le poids de la projection de sortie tirée d'une table partagée."""
    plein = PlainTensor(embed, tuple(embed.shape), "bf16")
    if _TETE_LIEE == "bf16" or not embed.is_cuda:
        return plein
    # La copie quantifiée s'ajoute à la table, elle ne la remplace pas : le
    # gather d'entrée a toujours besoin des poids en 16 bits. Sur un modèle qui
    # remplit déjà la carte, mieux vaut le débit qu'on a qu'un OOM au
    # chargement — on exige le double de la copie en mémoire libre.
    libre = torch.cuda.mem_get_info(embed.device)[0]
    besoin = embed.numel() * (1 if _TETE_LIEE == "int8" else 0.6)
    if libre < 2 * besoin:
        print(f"[acvram] tête liée laissée en bf16 : {libre / 2**20:.0f} Mio "
              f"libres, il en faudrait {2 * besoin / 2**20:.0f}",
              file=sys.stderr, flush=True)
        return plein
    try:
        if _TETE_LIEE == "nvfp4":
            from ..quant.nvfp4 import quantize_nvfp4
            return _par_tranches(embed, lambda t: quantize_nvfp4(t), "nvfp4")
        if _TETE_LIEE == "int8":
            from ..quant.formats import _quantize_int8
            return _par_tranches(embed, lambda t: _quantize_int8(t, 128), "int8")
    except Exception as exc:                      # noqa: BLE001
        # Quantifier la tête est un gain de débit, jamais une condition de
        # chargement : un échec se dit et se replie sur la table bf16.
        print(f"[acvram] tête liée laissée en bf16 ({type(exc).__name__}: {exc})",
              file=sys.stderr, flush=True)
    return plein


# Ce que la carte doit garder hors poids et hors KV : contexte CUDA, activations
# du prefill, graphes capturés, tampons de spéculation. Mesuré sur la 5090 : la
# capture des graphes échoue dès que moins de ~1 Gio reste libre.
_KV_MARGE_MIN = 1536 * 2**20
_KV_MARGE_PART = 0.05


def _borner_kv_par_la_vram(plan: Plan, manifest: dict, dev, reserve: int = 0) -> None:
    """Borne le budget KV de chaque GPU par ce qu'il a réellement de libre.

    Le budget du manifeste vient du planificateur, qui raisonne sur des tailles
    nominales et une capacité de plaque signalétique. Sur la carte, ce sont les
    poids réels qui comptent, plus tout ce que le plan ne voit pas. Résultat
    observé avant ce garde-fou : 50 Mio libres après le chargement d'un
    35B-A3B, capture des graphes CUDA impossible, décodage dégradé. Ici, le
    budget est ramené à ``libre − poids réels − marge`` quand il le dépasse.
    """
    if not torch.cuda.is_available() or not plan.kv_budget:
        return
    attn, mlp, embed, head = _octets_reels(manifest)
    for t in plan.tiers:
        if t.kind != "gpu" or t.name not in plan.kv_budget:
            continue
        try:
            d = dev(t.name)
            libre, capacite = torch.cuda.mem_get_info(d)
        except Exception:                       # noqa: BLE001
            continue
        poids = (embed if plan.embed_device == t.name else 0) \
            + (head if plan.lm_head_device == t.name else 0)
        for l in plan.layers:
            poids += attn.get(l.index, l.attn_bytes) if l.attn_storage == t.name else 0
            poids += mlp.get(l.index, l.mlp_bytes) if l.mlp_storage == t.name else 0
        marge = max(_KV_MARGE_MIN, int(_KV_MARGE_PART * capacite)) + int(reserve)
        borne = libre - poids - marge
        budget = int(plan.kv_budget[t.name])
        if borne < budget:
            plan.kv_budget[t.name] = max(0, borne)
            print(f"[acvram] budget KV de {t.name} borné par la VRAM libre : "
                  f"{budget / 2**30:.2f} → {max(0, borne) / 2**30:.2f} Gio "
                  f"(libre {libre / 2**30:.1f}, poids {poids / 2**30:.1f}, "
                  f"marge {marge / 2**30:.1f} dont préfill {reserve / 2**30:.2f})", file=sys.stderr)


def _kv_plancher(plan: Plan, spec: ModelSpec, max_model_len: Optional[int], dev: str) -> int:
    """Octets KV qu'il faut AU MOINS sur ``dev`` pour qu'UNE séquence de
    ``max_model_len`` jetons y passe : sous ce plancher, l'ordonnanceur
    n'admet jamais la requête et le moteur tourne à vide."""
    bpt = int(getattr(plan, "kv_bytes_per_token", 0) or spec.kv_bytes_per_token())
    total = sum(1 for lp in plan.layers if spec.couche_a_kv(lp.index))
    ici = sum(1 for lp in plan.layers if spec.couche_a_kv(lp.index) and lp.exec_device == dev)
    if not total or not ici:
        return 0
    # Exactement ce que le planificateur accorde à une séquence (`auto_plan` :
    # kv_bytes_per_token × max_model_len) — un plancher plus haut d'un bloc
    # (+16 jetons, 3 Mio) restait inatteignable : le budget du plan ne monte
    # pas au-dessus de sa cible, quatre tours d'exil rendaient « 3 Mio
    # manquants » et le refus (Laure, essai a803254).
    jetons = int(max_model_len or 2048)
    return bpt * jetons * ici // total


def _borner_kv_avec_exil(plan: Plan, manifest: dict, dev, spec: ModelSpec,
                         max_model_len: Optional[int], reserve: int = 0, tours: int = 4) -> None:
    """`_borner_kv_par_la_vram`, puis exile encore si le budget KV est passé
    SOUS le plancher d'une séquence — et refuse explicitement s'il y reste.

    Llama-3.3-70B-nvfp4 (Laure, 17/09, trois essais) : la boucle d'exil
    (`_reajuster_plan`) et la borne KV lisent la VRAM libre à deux instants
    et avec deux marges ; l'exil s'arrêtait à 37/80 satisfait, puis la borne
    ramenait le budget KV « 0,32 → 0,00 Gio ». Zéro bloc : la séquence de
    256 jetons n'était jamais admise, le moteur attendait un travail qui ne
    venait pas (33 min, GPU 0 %, fil principal en poll) — pris pour un
    blocage CUDA. Ici la borne et l'exil se répondent jusqu'à ce que le
    plancher tienne ; sinon le chargement s'arrête avec les chiffres."""
    if not torch.cuda.is_available() or not plan.kv_budget:
        return
    # La cible d'un appareil ne descend pas sous son plancher : si le plan
    # lui accordait moins, la borne (qui ne fait que réduire) ne pourrait
    # jamais l'y amener, quel que soit l'exil.
    cible = {k: max(int(v), _kv_plancher(plan, spec, max_model_len, k))
             for k, v in plan.kv_budget.items()}
    plan.kv_budget = dict(cible)
    top_k = spec.num_experts_per_tok or 8
    supplement = 0
    for tour in range(tours + 1):
        _borner_kv_par_la_vram(plan, manifest, dev, reserve=reserve)
        manque = 0
        for t in plan.tiers:
            if t.kind != "gpu" or t.name not in plan.kv_budget:
                continue
            plancher = _kv_plancher(plan, spec, max_model_len, t.name)
            manque = max(manque, plancher - int(plan.kv_budget[t.name]))
        if manque <= 0:
            return
        if tour == tours:
            break
        supplement += manque
        plan.kv_budget = dict(cible)
        print(f"[acvram] budget KV sous le plancher d'une séquence de {max_model_len} "
              f"jetons ({manque / 2**20:.0f} Mio manquants) : exil supplémentaire (tour {tour + 1})",
              file=sys.stderr)
        _reajuster_plan(plan, manifest, top_k=top_k, reserve=reserve + supplement)
    raise RuntimeError(
        f"refus : budget KV insuffisant après {tours} tours d'exil — "
        f"{ {k: round(v / 2**30, 2) for k, v in plan.kv_budget.items()} } Gio pour un plancher "
        f"d'une séquence de {max_model_len} jetons ({manque / 2**20:.0f} Mio manquants) ; "
        f"réduire max_model_len ou forcer l'exil (ACVRAM_EXIL_COUCHES)")


def _kv_blocks_per_device(plan: Plan, spec: ModelSpec,
                          max_model_len: Optional[int]) -> dict[str, int]:
    """Répartit le budget KV de chaque appareil en blocs, partagés entre ses couches.

    Seules les couches QUI ALLOUENT un cache entrent au dénominateur : plus
    bas, `a_allouer.append` n'est appelé que sur le chemin de l'attention
    pleine, si bien qu'une couche linéaire ou SSM comptée ici réduisait le
    nombre de blocs sans jamais en consommer un seul. Sur les 53 hybrides du
    parc, le diviseur était 4 à 14 fois trop grand — 16 656 jetons de contexte
    sur `Nemotron-Nano-9B` là où le même budget en permet 233 296.

    ``max_model_len`` n'est pas utilisé, et ce n'est pas un oubli : le nombre
    de blocs sort du budget du plan, pas de la longueur demandée. Mesuré le
    9/09/2026, borner par la longueur ne libérerait rien — le budget alloué
    est déjà INFÉRIEUR à ce que seize séquences de 4096 jetons réclament.
    """
    out: dict[str, int] = {}
    layers_on = {}
    for lp in plan.layers:
        if not spec.couche_a_kv(lp.index):
            continue
        layers_on[lp.exec_device] = layers_on.get(lp.exec_device, 0) + 1
    for dev, budget in plan.kv_budget.items():
        n_layers = max(1, layers_on.get(dev, 1))
        per_layer = budget // n_layers
        # Les octets d'un bloc dépendent du FORMAT du palier (int8 8,125
        # bits, lm4 4,125) : compter en int8 un cache lm4 lui volait la moitié
        # de sa capacité, le gain de mémoire du format n'existait pas.
        kv_fmt = _kv_format(plan, dev)
        bytes_per_block = KVCacheConfig(
            num_layers=1, num_kv_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim, num_blocks=1, dtype=kv_fmt).bytes_per_block()
        out[dev] = max(1, per_layer // max(1, bytes_per_block))
    return out


def _octets_reels(manifest: dict) -> tuple[dict, dict, int, int]:
    """Octets réels par couche (attention, MLP) d'après les formats du
    manifeste : le plan a été calculé avec le format nominal (nvfp4), mais
    la conversion promeut en int8 les tenseurs sous le plancher de SNR — un
    70B y gagne une dizaine de Gio que le plan ignore, d'où des OOM au
    chargement."""
    attn: dict = {}; mlp: dict = {}; embed = 0; head = 0
    for nom, e in manifest["tensors"].items():
        if not isinstance(e, dict):
            continue
        shape = e.get("shape") or []
        n = 1
        for x in shape:
            n *= int(x)
        bpw = float(e.get("bpw") or (16.0 if e.get("format") in ("bf16", "fp16", None) else 4.5))
        octets = int(n * bpw / 8)
        if nom.startswith("model.layers."):
            i = int(nom.split(".")[2])
            if ".mlp." in nom:
                mlp[i] = mlp.get(i, 0) + octets
            else:
                attn[i] = attn.get(i, 0) + octets
        elif nom.startswith("model.embed_tokens"):
            embed += octets
        elif nom.startswith("lm_head"):
            head += octets
    # Tête LIÉE : aucun tenseur `lm_head` au manifeste, et pourtant
    # `_tete_liee` en fabrique une copie quantifiée au chargement. La compter
    # ici n'est pas une precaution : `_borner_kv_par_la_vram` calcule
    # `libre − poids − marge` a partir de ce total, si bien que l'omettre lui
    # faisait autoriser un budget KV trop grand — donc MANGER LA MARGE qu'il
    # existe pour proteger, celle qui garde la place d'une capture de graphes.
    mo = manifest.get("model") or {}
    if mo.get("tie_word_embeddings") and mo.get("vocab_size") and mo.get("hidden_size"):
        from .config import ModelSpec
        champs = {k: v for k, v in mo.items() if k in ModelSpec.__dataclass_fields__}
        try:
            head += ModelSpec(**champs).tete_liee_bytes()
        except Exception:                                   # noqa: BLE001
            pass
    return attn, mlp, embed, head


def _compter_experts_manifest(manifest: dict) -> dict:
    """Nombre d'experts par couche, d'après les clés du manifeste — un
    sondage de NOMS, aucun tenseur chargé. Absente du dict : couche dense
    (pas de `mlp.experts.`), comme `_octets_reels` pour `attn`/`mlp`."""
    import re
    motif = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$")
    n: dict = {}
    for nom in manifest["tensors"]:
        m = motif.match(nom)
        if m:
            i, e = int(m.group(1)), int(m.group(2))
            n[i] = max(n.get(i, 0), e + 1)
    return n


def _reajuster_plan(plan: Plan, manifest: dict, top_k: int = 8, reserve: int = 0) -> None:
    """Fait descendre en RAM hôte les MLP des dernières couches d'un GPU
    dont les poids réels dépassent la capacité de l'étage.

    ``reserve`` : octets transitoires du plus grand préfill
    (`ModelSpec.activations_prefill_bytes`), soustraits de la capacité au
    même titre que la marge — avant, le budget ne comptait que résidents + KV
    + marge, et Llama-3.3-70B-nvfp4 chargeait DÉGRADÉ (32/80 exilées) puis
    tombait en OOM de 448 Mio au premier préfill (verdict-palier1-bloc6-17-09).

    `top_k` (bead pds, point 1, Sage §4) : sous ce nombre d'experts résidents,
    un placement PAR EXPERT n'a plus de sens — un jeton qui en route `top_k`
    trouverait presque toujours un froid, et le gain PCIe théorique (moins
    d'octets transférés) disparaîtrait dans les allers-retours qu'un manque
    d'emplacements chauds provoquerait. En dessous, l'exil de la couche
    ENTIÈRE reste le seul geste — c'est le comportement d'avant ce bead.
    """
    attn, mlp, embed, head = _octets_reels(manifest)
    n_experts = _compter_experts_manifest(manifest)
    for t in plan.tiers:
        if t.kind != "gpu":
            continue
        dev = t.name
        for l in plan.layers:
            # une couche décrite par le manifeste mais sans tenseur « .mlp. »
            # n'a réellement pas de MLP (bloc Mamba2/GDN pur) : garder sa
            # taille nominale gonflait le plan de dizaines de Gio et exilait
            # en RAM hôte des experts qui tenaient sur la carte
            decrite = l.index in attn
            l.attn_bytes = attn.get(l.index, l.attn_bytes)
            l.mlp_bytes = mlp.get(l.index, 0 if decrite else l.mlp_bytes)
        def utilise() -> int:
            u = int((plan.kv_budget or {}).get(dev, 0))
            u += embed if plan.embed_device == dev else 0
            u += head if plan.lm_head_device == dev else 0
            for l in plan.layers:
                u += l.attn_bytes if l.attn_storage == dev else 0
                if l.mlp_storage != dev:
                    continue
                if l.experts_residents is not None:
                    # Placement PAR EXPERT : seule la fraction résidente
                    # compte — `mlp_storage == dev` reste vrai (la couche
                    # s'EXÉCUTE ici), mais tous ses octets n'y vivent plus.
                    e = n_experts.get(l.index) or 1
                    u += l.mlp_bytes * l.experts_residents // e
                else:
                    u += l.mlp_bytes
            return u
        # La capacité de l'étage est celle qu'avait la machine le jour de la
        # conversion. Elle ne dit rien de ce que la carte a de libre à cet
        # instant : un serveur qui vient de rendre son port n'a pas encore
        # rendu sa mémoire, et le plan suivant se calcule alors sur des restes
        # qu'il croit disponibles. Le même modèle chargé deux fois de suite a
        # ainsi rendu 15,8 puis 152,2 jetons par seconde. On borne donc par ce
        # qui est réellement libre, mesuré ici.
        capacite = t.capacity
        try:
            # `dev` est ici la chaîne du device, pas la fonction du module :
            # elle est masquée par la variable locale au-dessus.
            libre = torch.cuda.mem_get_info(torch.device(t.name))[0]
            # ATTENTION : ce `min` compare deux nombres de nature différente,
            # pas deux mesures interchangeables. `t.capacity` sort de
            # `build_tiers()` (tiering.py) DÉJÀ NET d'une réserve (800 Mio de
            # contexte CUDA + 3 % de fragmentation) et mesuré plus tôt dans le
            # chargement, avant la compilation JIT des noyaux. `libre` ici est
            # une lecture BRUTE de `mem_get_info`, prise plus tard, sans aucune
            # réserve soustraite. Le plus petit des deux n'est donc pas
            # forcément "le plus à jour" : selon le moment et l'état de la
            # carte, l'un ou l'autre peut gagner sans que ce soit un signe de
            # fraîcheur. Ne pas remplacer par `capacite = libre` : ça
            # supprimerait la réserve de `build_tiers()` (capture de graphes,
            # fragmentation) sans la réintroduire, sur une carte où l'OOM
            # survient à quelques dizaines de Mio libres. Un correctif propre
            # comparerait des bases homogènes : `min(t.capacity, libre - la
            # même réserve)`, pas l'un brut contre l'autre net.
            capacite = min(capacite, libre)
        except Exception:                           # noqa: BLE001
            pass
        # marge pour le contexte CUDA, les activations et les piles d'experts :
        # la capacité de l'étage est déjà nette des réserves du plan, mais un
        # 70B chargé à 99 % tombait encore en OOM à l'allocation du KV
        marge = max(2 * 2**30, int(0.07 * capacite)) + int(reserve)
        deplacees = 0
        while utilise() > capacite - marge:
            # Candidats déjà résidents (couche entière) OU déjà à moitié
            # (placement par expert antérieur, dont on peut encore réduire
            # `experts_residents`) : les deux peuvent encore libérer de la
            # place, contrairement à une couche déjà totalement exilée.
            cand = [l for l in plan.layers
                   if l.mlp_storage == dev
                   and (l.experts_residents is None or l.experts_residents > 0)]
            if not cand:
                break
            l = cand[-1]
            e = n_experts.get(l.index, 0)
            depasse = utilise() - (capacite - marge)
            if e >= top_k and depasse > 0:
                # Placement PAR EXPERT (bead pds) : ne descendre que ce qu'il
                # faut, pas la couche entière — `estimer_cout_exil`
                # (memory/tiering.py) chiffrera le coût du reste exilé contre
                # la bande PCIe mesurée (~21 Go/s, M0).
                actuels = l.experts_residents if l.experts_residents is not None else e
                par_expert = l.mlp_bytes / e
                a_exiler = min(actuels, max(1, math.ceil(depasse / par_expert)))
                c = actuels - a_exiler
                if c >= top_k:
                    l.experts_residents = c
                    deplacees += 1
                    continue
                # c < top_k : un placement par expert n'a plus de sens
                # (docstring) — tombe dans l'exil de couche entière ci-dessous.
                l.experts_residents = None
            l.mlp_storage = "cpu"
            # Descendre un MLP en RAM dit où il est STOCKÉ, pas où il est
            # CALCULÉ. Forcer ici le calcul sur processeur défaisait la
            # décision du planificateur à chaque chargement : c'est ce chemin
            # qui a mis seize couches de Qwen3-Coder-Next sur le processeur,
            # à 10,3 jetons par seconde. Tant que le débit de calcul hôte
            # n'est pas mesuré (host_gemm_gb_s), les poids traversent le bus.
            if hasattr(l, "mlp_exec"):
                l.mlp_exec = ("cpu" if os.environ.get("ACVRAM_MLP_HOTE_CPU")
                              else "gpu")
            deplacees += 1
        if deplacees:
            print(f"[acvram] plan réajusté : {deplacees} MLP de plus en RAM hôte sur {dev} "
                  f"(poids réels {utilise() / 2**30:.1f} Gio pour {capacite / 2**30:.1f} Gio libres, "
                  f"activations de préfill réservées {reserve / 2**30:.2f} Gio)",
                  flush=True)
            continue
        # Symétrique de la descente. Le plan est figé au moment de la
        # conversion, avec le format nominal et la machine d'alors ; les
        # poids réels sont souvent plus compacts (nvfp4 à 4,5 bpw là où le
        # plan comptait 6). Une couche laissée en RAM hôte y coûte un aller
        # PCIe par jeton : dès qu'elle tient sur la carte, elle y remonte.
        if os.environ.get("ACVRAM_PLAN_FIGE"):
            continue
        remontees = 0
        for l in plan.layers:
            if l.exec_device != dev:
                continue
            if l.attn_storage == "cpu" and utilise() + l.attn_bytes <= capacite - marge:
                l.attn_storage = dev
            if l.mlp_storage != "cpu":
                continue
            if utilise() + l.mlp_bytes > capacite - marge:
                continue
            l.mlp_storage = dev
            if hasattr(l, "mlp_exec"):
                l.mlp_exec = "gpu"
            remontees += 1
        if remontees:
            print(f"[acvram] plan réajusté : {remontees} MLP remontés en VRAM sur {dev} "
                  f"(poids réels {utilise() / 2**30:.1f} Gio pour {capacite / 2**30:.1f} Gio libres)",
                  flush=True)
    _rapatrier_sur_une_carte(plan, attn, mlp, embed, head)
    _signaler_cout_exil(plan)


def _signaler_cout_exil(plan: Plan) -> None:
    """Chiffre le surcoût en temps de l'exil décidé sur l'axe des octets.

    `_reajuster_plan` exile pour tenir en capacité (octets) ; l'exil coûte en
    temps (facteur 3-4, docs/TEST-EXIL.md). Ce signal donne le chiffre qui
    manquait — voir `estimer_cout_exil` (memory/tiering.py) et le bead
    anticitoyen-vram-jt5. Il SIGNALE, il ne refuse pas : le plan reste
    chargeable, sans quoi un modèle qui ne tient qu'au prix de l'exil tomberait
    en OOM au lieu de tourner plus lentement. Seuil ajustable par
    `ACVRAM_SEUIL_EXIL` (part du pas, 0,20 par défaut)."""
    from ..memory.tiering import estimer_cout_exil
    try:
        seuil = float(os.environ.get("ACVRAM_SEUIL_EXIL", "0.20"))
    except ValueError:
        seuil = 0.20
    c = estimer_cout_exil(plan, seuil=seuil)
    if c is None:
        return
    msg = (f"exil : {c['n_couches_exilees']} MLP en RAM hôte = "
           f"{c['t_transfert_s'] * 1e3:.1f} ms/jeton de PCIe, soit "
           f"{c['ratio'] * 100:.0f} % du pas de décodage résident "
           f"({c['t_pas_resident_s'] * 1e3:.1f} ms)")
    plan.warnings.append(msg)
    if c["franchit_seuil"]:
        print(f"[acvram] ATTENTION — {msg} ; au-delà du seuil "
              f"{seuil * 100:.0f} % : falaise de l'exil (docs/TEST-EXIL.md)",
              flush=True)


def _rapatrier_sur_une_carte(plan: Plan, attn: dict, mlp: dict,
                             embed: int, head: int) -> None:
    """Ramène tout le modèle sur la première carte quand il y tient.

    Le plan est figé à la conversion, avec le compte de paramètres d'alors.
    Pour les architectures hybrides, ce compte a longtemps triplé la taille
    réelle (Nemotron-H : 103 milliards annoncés pour 31,6), et le planificateur
    étalait sur deux cartes un modèle qui tenait sur une. Chaque frontière
    franchie coûte un aller-retour d'état caché par jeton, et la carte
    d'appoint est trois fois plus lente : mesuré le 5 septembre 2026,
    26 jetons/s au lieu de 170.

    Les octets réels sont connus ici : si la première carte les porte, avec son
    budget KV et la marge, les couches de la seconde y reviennent. Échappement
    par ``ACVRAM_PLAN_FIGE``.
    """
    if os.environ.get("ACVRAM_PLAN_FIGE"):
        return
    gpus = [t for t in plan.tiers if t.kind == "gpu"]
    if len(gpus) < 2:
        return
    occupes = {l.exec_device for l in plan.layers} & {t.name for t in gpus}
    if len(occupes) < 2:
        return
    principal = gpus[0]
    if any(l.attn_storage == "cpu" or l.mlp_storage == "cpu" for l in plan.layers):
        return                          # déjà à l'étroit : ne pas empirer
    besoin = embed if plan.embed_device == principal.name else 0
    besoin += head
    for l in plan.layers:
        besoin += attn.get(l.index, l.attn_bytes) + mlp.get(l.index, l.mlp_bytes)
    besoin += sum(int(v) for v in (plan.kv_budget or {}).values())
    marge = max(2 * 2**30, int(0.07 * principal.capacity))
    if besoin > principal.capacity - marge:
        return
    for l in plan.layers:
        l.exec_device = principal.name
        l.attn_storage = principal.name
        l.mlp_storage = principal.name
        if hasattr(l, "mlp_exec"):
            l.mlp_exec = "gpu"
    plan.lm_head_device = principal.name
    if plan.embed_device in {t.name for t in gpus}:
        plan.embed_device = principal.name
    plan.kv_budget = {principal.name: sum(int(v) for v in (plan.kv_budget or {}).values())}
    plan.stage_ranges = {principal.name: (0, len(plan.layers) - 1)}
    print(f"[acvram] plan réajusté : tout le modèle rapatrié sur {principal.name} "
          f"({besoin / 2**30:.1f} Gio pour {principal.capacity / 2**30:.1f} Gio) — "
          f"une frontière de moins par jeton", flush=True)


def _replanifier(manifest: dict, spec: "ModelSpec",
                 max_model_len: Optional[int] = None,
                 max_concurrent_seqs: Optional[int] = None) -> "Plan | None":
    """Rejoue TOUJOURS le planificateur avec l'état actuel de la machine.

    Le plan est écrit une fois pour toutes à la conversion, et `_reajuster_plan`
    ne sait que faire *descendre* des MLP en RAM hôte : aucune amélioration du
    planificateur n'atteint jamais un modèle déjà converti si on se contente du
    plan figé. Le 7 septembre 2026, Qwen3-Coder-Next tournait à 10,3 jetons par
    seconde sur un plan qui ignorait la seconde carte et calculait seize
    couches sur le processeur, alors que le planificateur du jour recrutait les
    deux cartes et n'en exilait que quatre. **On rejoue donc sans condition**
    (sauf `ACVRAM_PLAN_FIGE`/`ACVRAM_SANS_REPLAN`) — le test `figees !=
    presentes` ci-dessous ne décide PAS s'il faut rejouer, seulement s'il faut
    le signaler : un rejeu silencieux sur la même liste de cartes est la
    normale, pas une exception. (Corrigé le 10/09/2026 : ce docstring disait
    l'inverse pendant que le code faisait déjà ceci — deux heures perdues à
    chercher pourquoi un plan « figé » changeait entre deux chargements.)

    ``max_model_len`` : quand l'appelant l'annonce explicitement, on le CROIT
    et on dimensionne le cache KV pour ce contexte précis, pas pour un chiffre
    par défaut. Une mesure sur 1 024 jetons n'a aucune raison de réserver de la
    VRAM pour 4 096 ou 8 192 — c'est de la place prise aux poids pour un besoin
    qui n'existe pas, et c'est ce qui pousse au sacrifice le plus cher connu
    ici (une couche exilée coûte 70 % du débit). Quand l'appelant ne dit rien
    (``None``), le comportement précédent est conservé à l'identique : un
    serveur qui ne connaît pas encore la taille de ses requêtes doit continuer
    à dimensionner pour le pire cas plausible, pas pour un contexte court par
    défaut.

    ``max_concurrent_seqs`` : même logique, pour le nombre de séquences —
    absent avant le 17/09/2026, ce qui dimensionnait TOUJOURS le budget KV
    pour `PlannerOptions.max_concurrent_seqs` par défaut (8), quel que soit
    le `--max-batch`/`max_batch_size` réellement demandé. À b=12, 4
    séquences sur 12 tronquaient silencieusement avant `max_tokens` (budget
    épuisé, `_finish_budget_epuise`) sans qu'aucun compteur ne le signale —
    trouvé par Laure sur `certifie-b12`, `sage-poste-d-verdict-17-09.md`.
    """
    if os.environ.get("ACVRAM_PLAN_FIGE") or os.environ.get("ACVRAM_SANS_REPLAN"):
        return None
    d = manifest["plan"]
    figees = {t["name"] for t in d.get("tiers", []) if t.get("kind") == "gpu"}
    try:
        from ..hardware.detect import detect_rig
        from ..memory.tiering import PlannerOptions, auto_plan
        rig = detect_rig()
        presentes = {f"cuda:{g.index}" for g in rig.gpus}
        if not presentes:
            return None
        ctx = (max_model_len if max_model_len
               else max(2048, int(d.get("kv_max_tokens") or 0) or 8192))
        slots = (max_concurrent_seqs if max_concurrent_seqs
                else int(d.get("kv_planned_seqs") or 0) or PlannerOptions().max_concurrent_seqs)
        neuf, _ = auto_plan(spec, rig, PlannerOptions(max_model_len=ctx,
                                                       max_concurrent_seqs=slots))
    except Exception as e:                                   # pragma: no cover
        print(f"[acvram] replanification impossible ({e}) ; plan du manifeste "
              f"conservé", flush=True)
        return None
    if figees != presentes:
        print(f"[acvram] plan recalculé : les cartes ont changé "
              f"(manifeste {sorted(figees) or 'aucun GPU'}, "
              f"machine {sorted(presentes)})", flush=True)
        # Un avertissement n'arrête rien. Le banc doit pouvoir REFUSER de
        # publier un débit dans ce cas : le chiffre ne porte plus sur la
        # configuration demandée mais sur celle que le planificateur a
        # choisie. On laisse donc une trace lisible sur le plan lui-même.
        try:
            neuf.replanifie_cartes = (sorted(figees), sorted(presentes))
        except Exception:                    # noqa: BLE001 — une trace ne plante pas
            pass
    return neuf


def _forcer_exil(plan: Plan, n_voulu: int) -> None:
    """Porte le nombre de couches à perceptron exilé à ``n_voulu``.

    Instrument de mesure, pas de production : il sert à tracer le coût réel
    d'une couche exilée *en service*, en faisant varier leur nombre à modèle,
    plan et prompt identiques. On ne peut qu'exiler davantage — remonter des
    poids sur une carte déjà pleine ferait déborder la VRAM — donc la courbe
    se lit à partir du minimum imposé par la capacité.
    """
    exilees = [l for l in plan.layers if l.mlp_storage == "cpu"]
    if n_voulu <= len(exilees):
        print(f"[acvram] ACVRAM_EXIL_COUCHES={n_voulu} ignoré : {len(exilees)} "
              f"couches sont déjà exilées et on ne peut pas les remonter",
              flush=True)
        return
    candidates = [l for l in plan.layers if l.mlp_storage != "cpu"]
    for l in candidates[len(candidates) - (n_voulu - len(exilees)):]:
        l.mlp_storage = "cpu"
        if hasattr(l, "mlp_exec"):
            l.mlp_exec = "gpu"
    print(f"[acvram] mesure : {n_voulu} couches à perceptron exilé "
          f"(minimum imposé par la capacité : {len(exilees)})", flush=True)
    # L'estimation a été calculée AVANT ce déplacement et ne le reflète plus.
    # Mesuré sur Agents-A1-4B : le plan continuait d'annoncer 687,6 jetons/s
    # pour un débit réel de 24,0 à seize couches exilées — un facteur 28,7.
    # Le planificateur, lui, VOIT le placement (624,8 -> 30,1 quand la VRAM
    # simulée tombe à 4,8 Gio) : le défaut est ici, pas dans l'estimateur.
    # « Rendre None quand on ne sait pas vaut mieux qu'un chiffre crédible :
    # celui-ci servirait à choisir un plan » (tiering.py) — on applique la
    # règle du fichier voisin plutôt que de garder un nombre faux.
    plan.est_decode_tok_s = 0.0
    plan.est_bytes_per_token = 0
    try:
        plan.estimation_perimee = (
            f"exil force a {n_voulu} couches apres l'estimation ; "
            f"le debit prevu ne vaut plus rien pour ce plan")
    except Exception:                        # noqa: BLE001 — une trace ne plante pas
        pass


def _exil_demande(plan: Plan) -> None:
    n = os.environ.get("ACVRAM_EXIL_COUCHES")
    if n:
        try:
            _forcer_exil(plan, int(n))
        except ValueError:
            pass


def _forcer_exil_experts(plan: Plan, manifest: dict, fraction: float) -> None:
    """Pendant de `_forcer_exil`, au grain de l'expert plutôt que de la
    couche entière : « mesure qui tue » (bead pds, point 4). Même prompt,
    même modèle, même VRAM libérée dans les deux cas — seul le GRAIN de
    l'exil change, ce que ce couple de fonctions rend comparable.
    """
    n_experts = _compter_experts_manifest(manifest)
    candidates = [l for l in plan.layers
                  if l.mlp_storage != "cpu" and n_experts.get(l.index)]
    for l in candidates:
        e = n_experts[l.index]
        l.experts_residents = max(0, e - round(e * fraction))
    print(f"[acvram] mesure : exil par expert a {fraction:.0%} sur "
          f"{len(candidates)} couches", flush=True)
    plan.est_decode_tok_s = 0.0
    plan.est_bytes_per_token = 0
    try:
        plan.estimation_perimee = (
            f"exil par expert force a {fraction:.0%} apres l'estimation ; "
            f"le debit prevu ne vaut plus rien pour ce plan")
    except Exception:                        # noqa: BLE001 — une trace ne plante pas
        pass


def _exil_experts_demande(plan: Plan, manifest: dict) -> None:
    frac = os.environ.get("ACVRAM_EXIL_EXPERTS_FRACTION")
    if frac:
        try:
            _forcer_exil_experts(plan, manifest, float(frac))
        except ValueError:
            pass


# Jeux de tampons GPU par forme de tenseur pour les poids denses exilés : une
# couche en calcul (gate, up de même forme = 2) + la suivante préchargée (2),
# le pool refuse au-delà (« ExpertPool saturé ») plutôt que d'allouer.
_DENSE_SLOTS = int(os.environ.get("ACVRAM_DENSE_SLOTS", "4"))


def _reserve_prefill(spec, max_model_len: Optional[int], manifest: dict,
                     plan: Optional[Plan] = None) -> int:
    """Octets transitoires à retirer des budgets (KV, exil) :
    `ModelSpec.activations_prefill_bytes` sur ``max_model_len`` (sinon 8 192,
    le défaut du moteur — runner.Engine) PLUS les tampons GPU du streaming
    dense (`_DENSE_SLOTS` × la plus grosse couche du plan, borne haute de
    « _DENSE_SLOTS × chaque forme distincte ») ; 0 sans spec.

    JAMAIS `kv_max_tokens` du manifeste : c'est la CAPACITÉ KV planifiée,
    toutes séquences confondues (56 401 sur Qwen3.8-27B), pas la longueur
    d'une invite — l'avoir pris (1aa767b) réservait 13,04 Gio d'activations
    à un chargement sans max_model_len, exilait 19/64 couches d'un modèle de
    13,1 Gio et faisait refuser l'instrument de préfill (Laure, fla-17-09)."""
    if spec is None:
        return 0
    ctx = int(max_model_len or 8192)
    reserve = int(spec.activations_prefill_bytes(ctx))
    if plan is not None and plan.layers:
        reserve += _DENSE_SLOTS * max(int(l.attn_bytes) + int(l.mlp_bytes) for l in plan.layers)
        # P1 disposition UNIQUE (sage-p1-disposition-unique-18-09) : la pile
        # Marlin REMPLACE la pile NVFP4 (MoEBlock._try_build_stacks la libère
        # après le repack, même masse : codes K·N/2 + échelles K·N/16) — la
        # réserve « seconde disposition » de 5a0f6c4 (Σ mlp_bytes, 17,04 Gio
        # sur Coder : 21/48 couches exilées, Laure 18/09) n'existe plus.
        # tests/test_marlin_prefill_p1.py : Σ × 1, jamais × 2.
    return reserve


def _plan_from_manifest(manifest: dict, spec: "ModelSpec | None" = None,
                        max_model_len: Optional[int] = None,
                        max_concurrent_seqs: Optional[int] = None) -> Plan:
    from ..memory.tiering import LayerPlacement, Plan as _Plan, Tier
    d = manifest["plan"]
    if spec is not None:
        neuf = _replanifier(manifest, spec, max_model_len=max_model_len,
                            max_concurrent_seqs=max_concurrent_seqs)
        if neuf is not None:
            _reajuster_plan(neuf, manifest, top_k=spec.num_experts_per_tok or 8,
                            reserve=_reserve_prefill(spec, max_model_len, manifest, neuf))
            _exil_demande(neuf)
            _exil_experts_demande(neuf, manifest)
            return neuf
    plan = _Plan(model=d["model"])
    plan.tiers = [Tier(**t) for t in d["tiers"]]
    plan.layers = [LayerPlacement(**{k: v for k, v in l.items()
                                     if k not in ("streamed", "total_bytes",
                                                  "resident_bytes")})
                   for l in d["layers"]]
    plan.embed_device = d["embed_device"]
    plan.lm_head_device = d["lm_head_device"]
    _reajuster_plan(plan, manifest, reserve=_reserve_prefill(spec, max_model_len, manifest, plan))
    _exil_demande(plan)
    _exil_experts_demande(plan, manifest)
    plan.kv_budget = d.get("kv_budget", {})
    plan.kv_bytes_per_token = d.get("kv_bytes_per_token", 0)
    plan.kv_max_tokens = d.get("kv_max_tokens", 0)
    plan.kv_planned_seqs = d.get("kv_planned_seqs", 0)
    return plan
