"""Ordonnancement et génération : lot continu sur un cache KV paginé.

Les requêtes arrivent n'importe quand et se terminent à des longueurs
différentes : le moteur fait donc tourner un lot *continu*. À chaque étape, il
admet toutes les nouvelles requêtes que les blocs KV libres peuvent payer,
décode un jeton pour tout ce qui tourne déjà, et évince les séquences dès
qu'elles s'arrêtent. Rien n'attend une frontière de lot.

Le moteur est monothread à dessein. Les couches du modèle sont réparties sur
deux GPU et la mémoire hôte, et une étape les touche toutes en série ; ajouter
des fils par-dessus ferait se disputer les mêmes appareils sans ajouter de
parallélisme. La concurrence vient du lot, pas des fils, et le serveur asyncio
fait entrer et sortir le travail par une file.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch

from ..memory.kvcache import BLOCK_SIZE, BlockAllocator
from .loader import LoadedModel
from .model import _DUMP_MOE, ForwardBatch
from .sampler import SamplingParams, besoin_historique, sample
from .speculative import GardeSpeculation, Proposal, verify_proposal
from .vision import ImageRequete, SansTourVision, TourVision, verifier_plages

__all__ = ["Sequence", "GenerationOutput", "Engine", "EngineStats"]

_ids = itertools.count(1)


@dataclass
class Sequence:
    prompt_ids: list[int]
    params: SamplingParams
    request_id: str = ""
    id: int = field(default_factory=lambda: next(_ids))
    output_ids: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""
    arrival: float = field(default_factory=time.time)
    first_token_at: float = 0.0
    cumulative_logprob: float = 0.0
    prefill_len: int = 0                # jetons d'invite DEJA passes en avant
    cached_len: int = 0                 # jetons d'invite servis par le cache de préfixe
    hashes: list[int] = field(default_factory=list)
    n_accepted: int = 0                 # jetons spéculatifs acceptés
    n_proposed: int = 0
    # Multimodal P1 : images de l'invite (ImageRequete) et, après la tour,
    # leurs traits (debut, fin, embeds bf16 [fin − debut, hidden]).
    images: list = field(default_factory=list)
    image_embeds: Optional[list] = None
    # M-RoPE (Qwen3-VL, engine/mrope) : positions [3, len(prompt_ids)] de
    # l'invite et le décalage entier du décodage (rope_deltas de transformers,
    # ≤ 0). None / 0 : RoPE 1-D — texte seul, ou modèle sans mrope_section.
    mrope_positions: Optional[torch.Tensor] = None
    rope_delta: int = 0
    # Deepstack (Qwen3-VL) : à côté des traits, (debut, fin, niveaux bf16
    # [n_niveaux, fin − debut, hidden]) par image ; None quand la tour n'en rend pas.
    image_niveaux: Optional[list] = None

    @property
    def prefilled(self) -> bool:
        """Vrai quand toute l'invite est passée en avant.

        La progression a son champ à elle. On serait tenté de la lire dans
        `cached_len`, qui est juste à côté et qui avance bien lors d'une
        reprise partielle — mais `cached_len` répond à « combien le cache m'a
        évité », pas à « où j'en suis », et il est faux dans les deux sens :
        il vaut encore son origine après un prefill entier, et il porte déjà
        sa valeur finale AVANT tout passage en avant quand le préfixe est
        servi par le cache.
        """
        return self.prefill_len >= len(self.prompt_ids)

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def longueur_ecrite(self) -> int:
        """Jetons dont les clés et valeurs sont réellement dans le cache.

        `length` compte l'invite ENTIERE des l'admission, avant tout passage
        en avant. Publier des blocs sur cette longueur pendant un prefill
        decoupe livrerait a une requete ulterieure des cles jamais ecrites —
        exactement ce que la garde « uniquement des blocs complets » de
        `_register_complete_blocks` cherche a empecher.
        """
        return self.length if self.prefilled else self.prefill_len

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    def blocks_needed(self, extra: int = 0) -> int:
        return (self.length + extra + BLOCK_SIZE - 1) // BLOCK_SIZE


@dataclass
class GenerationOutput:
    sequence_id: int
    request_id: str
    token_ids: list[int]
    text_delta: str = ""
    finished: bool = False
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class EngineStats:
    steps: int = 0
    # Combien de pas ont contenu un prefill. `steps` les compte tous ; ce
    # champ isole ceux qui ont porte un forward de prefill EN PLUS du
    # decodage. Le prefill n'est pas decoupe (pas de chunked prefill), donc
    # sa duree entiere s'ajoute a la latence de toutes les sequences deja en
    # cours a ce pas-la : un pas mixte est structurellement plus long qu'un
    # pas de decodage pur, et leur proportion explique une part de la
    # dispersion du temps par jeton. Expose par /metrics pour qu'une mesure
    # faite DEHORS du process serveur puisse l'apparier avec ses essais lents
    # — sans lui, seule une mesure in-process voyait le compteur, et les deux
    # bras d'une campagne n'etaient pas instrumentes pareil.
    pas_avec_prefill: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    running: int = 0
    waiting: int = 0
    kv_blocks_free: int = 0
    kv_refills: int = 0
    kv_blocks_total: int = 0
    cached_prompt_tokens: int = 0
    accepted_tokens: int = 0
    proposed_tokens: int = 0
    spec_steps: int = 0
    # Séquences terminées par `_finish_budget_epuise` (budget KV épuisé avant
    # `max_tokens`), jamais par un `EOS`/`max_tokens` normal. Compté pour que
    # `certifie-b12` puisse refuser une cellule où le lot réel a été rogné en
    # cours de mesure au lieu de la lire dans les logs — trouvé le 17/09 par
    # Laure sur une cellule b=12 planifiée pour 8 séquences (`loader.py`).
    sequences_tronquees_budget: int = 0
    # Combien de séquences ADMISES portaient `ignore_eos=True` — porté dans
    # l'en-tête d'une mesure (regime_ligne()) pour qu'un harnais comparatif
    # (sage-harnais-egal-ignore-eos-18-09) sache, en lisant le régime, que
    # cette cellule a bien tourné avec le même comportement que llama.cpp
    # `--ignore-eos` plutôt que de le supposer depuis sa propre requête.
    sequences_ignore_eos: int = 0

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def prefill_tok_s(self) -> float:
        return self.prefill_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "pas_avec_prefill": self.pas_avec_prefill,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "decode_tok_s": round(self.decode_tok_s, 2),
            "prefill_tok_s": round(self.prefill_tok_s, 1),
            # Les temps cumulés, pas seulement les taux. Un taux est une
            # moyenne depuis le démarrage : il ne dit rien d'UNE requête, et
            # deux relevés successifs ne s'en soustraient pas. Les cumuls, si —
            # c'est la seule façon de mesurer le forward d'une requête sans
            # instrumenter le moteur. Leur absence a coûté deux mesures : l'une
            # a divisé un delta de jetons par le taux global et obtenu un
            # forward supérieur au TTFT, l'autre a lu la clé manquante comme un
            # zéro et conclu « 0,0 ms pour 39 410 jetons ».
            "prefill_seconds": round(self.prefill_seconds, 6),
            "decode_seconds": round(self.decode_seconds, 6),
            "running": self.running, "waiting": self.waiting,
            "kv_blocks_free": self.kv_blocks_free,
            "kv_blocks_total": self.kv_blocks_total,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "prefill_tokens_saved": self.cached_prompt_tokens,
            "hit_rate": round(self.hit_rate, 3),
            "host_kv_tokens": self.host_kv_tokens,
            "kv_refills": self.kv_refills,
            "accepted_tokens": self.accepted_tokens,
            "proposed_tokens": self.proposed_tokens,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "tokens_per_step": round(self.tokens_per_step, 3),
            "sequences_tronquees_budget": self.sequences_tronquees_budget,
            "sequences_ignore_eos": self.sequences_ignore_eos,
        }

    @property
    def hit_rate(self) -> float:
        """Fraction des jetons d'invite servis par le cache de préfixe
        (VRAM + étage hôte) plutôt que recalculés — P3, veille TRT-LLM/
        FlashInfer (Sage, 14/09) : un tour d'agent au préfixe froid recalcule
        tout, un tour chaud ne recalcule que ce qui a changé. Publié pour
        qu'une mesure DEHORS du process (banc « tour d'agent ») distingue un
        cache qui rate (blocs qui ne s'apparient pas) d'un vrai coût de
        calcul."""
        total = self.cached_prompt_tokens + self.prefill_tokens
        return self.cached_prompt_tokens / total if total else 0.0

    @property
    def host_kv_tokens(self) -> int:
        """Jetons remontés depuis l'étage hôte du cache KV (P3) — sous-
        ensemble de `cached_prompt_tokens` : ceux qui n'étaient PAS déjà en
        VRAM et ont dû être réimportés (`kv_refills`, un bloc à la fois)."""
        return self.kv_refills * BLOCK_SIZE

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_tokens / self.proposed_tokens
                if self.proposed_tokens else 0.0)

    @property
    def tokens_per_step(self) -> float:
        """Jetons décodés par étape de modèle. Au-dessus de 1, la spéculation a payé."""
        return self.decode_tokens / self.spec_steps if self.spec_steps else 1.0



def _trim_at_stop(joined: str, delta: str,
                  stops: list[str]) -> tuple[str, bool]:
    """Coupe le delta juste avant la première séquence d'arrêt rencontrée.

    L'API s'y engage : la chaîne d'arrêt est *exclue* de la sortie. La
    détection se fait sur le texte assemblé (une séquence peut chevaucher deux
    jetons) ; la coupe, elle, ne peut retrancher que le delta courant — un
    chevauchement sur un delta déjà livré en flux est perdu pour le client,
    comme chez les autres serveurs.
    """
    cut = -1
    for st in stops:
        if st:
            at = joined.find(st)
            if at >= 0 and (cut < 0 or at < cut):
                cut = at
    if cut < 0:
        return delta, False
    prev = len(joined) - len(delta)
    return (joined[prev:cut] if cut > prev else ""), True



def _tranche(seq, a: int, b: int) -> tuple:
    """Les jetons [a, b) d une sequence, sans concatener prompt et sortie.

    Rend exactement `tuple(seq.all_ids[a:b])`, en ne touchant que les elements
    demandes. Une tranche a cheval sur la frontiere prend des deux cotes.
    """
    p = seq.prompt_ids
    n = len(p)
    if b <= n:
        return tuple(p[a:b])
    if a >= n:
        return tuple(seq.output_ids[a - n:b - n])
    return tuple(p[a:]) + tuple(seq.output_ids[:b - n])


# --------------------------------------------------------------------------
# REPIN (bead anticitoyen-vram-pds, point 3) : rétrograder/promouvoir UN
# `QuantLinear` d'expert. Fonctions libres (pas des méthodes de `Engine`) :
# elles ne touchent qu'un objet passé en argument, testables sans modèle.
# --------------------------------------------------------------------------

def _demote_expert(lin, mlp_dev) -> None:
    """Résident -> froid : copie VRAM -> tampon hôte épinglé neuf
    (`to_device(streamed=True)`), puis RÉDUIT `lin.qweight` à un gabarit sans
    octets — `to_device(streamed=True)` ne le touche jamais (voir la
    docstring de `memory.table_adresses.adresse_expert`), et le laisser tel
    quel garderait référencé l'ancien tenseur VRAM : exactement l'octet que
    REPIN devait rendre.

    `lin.qweight` n'est pas mis à `None` : `_resolved_weight()`
    (`_rehydrate(self.qweight, tensors)`) s'en sert comme GABARIT à CHAQUE
    lecture d'un expert streamé — `.shape`, `.padded_in`, et `._gs_f` (le
    scalaire d'échelle mémoïsé, dont l'absence coûterait un `.item()` par
    GEMV, 62 % du décodage au profil NVFP4 cité dans `layers.py`). Le
    gabarit garde donc ces trois informations, mais ses GROS tenseurs
    (`qweight`, `block_scale`) sont vidés — `_rehydrate_brut` ne les lit
    jamais, seule leur forme logique compte."""
    lin.to_device(mlp_dev, streamed=True)
    t = lin.qweight
    # `[:0]` seul resterait une VUE sur le stockage complet de `t.qweight` —
    # `.cpu().clone()` en fait un tenseur INDÉPENDANT, 0 octet utile, sur un
    # stockage neuf : la VRAM de `t` n'est plus référencée que par lui-même,
    # libérable dès que le ramasse-miettes le reprend.
    gabarit = type(t)(t.qweight[:0].cpu().clone(), t.block_scale[:0].cpu().clone(),
                      t.global_scale.detach().cpu().clone(),
                      t.shape, t.padded_in)
    gs = t.__dict__.get("_gs_f")
    if gs is not None:
        gabarit.__dict__["_gs_f"] = gs
    lin.qweight = gabarit


def _promote_expert(lin, mlp_dev) -> None:
    """Froid -> résident : reconstruit `qweight` en VRAM depuis
    `streamed.host` (le tampon épinglé, les VRAIS octets) — réutilise
    `_rehydrate` (`layers.py`, la même fonction que le préchargement par pas
    utilise), avec `lin.qweight` COMME GABARIT (voir `_demote_expert` :
    valide même réduit, `_rehydrate_brut` n'en lit que la forme et `_gs_f`)."""
    from .layers import _rehydrate
    tensors = {k: v.to(mlp_dev) for k, v in lin.streamed.host.items()}
    lin.qweight = _rehydrate(lin.qweight, tensors)
    lin.streamed = None


def _repin_echanger_reel(m, sortant: int, entrant: int) -> None:
    """L'échange RÉEL pour UNE couche déjà placée par expert (`m._table_qw`
    non `None`) : trois projections, copie hôte↔VRAM puis tables mises à
    jour — jamais avant, sinon un lecteur de la table verrait une adresse
    pas encore garnie. `verifier_table` après chaque écriture (jamais 0).

    Fonction libre (comme `_demote_expert`/`_promote_expert`) : ne touche
    que `m` et les deux ids passés, testable sans `Engine`."""
    from ..memory.table_adresses import adresse_expert, verifier_table
    s_mlp, e_mlp = m.experts[sortant], m.experts[entrant]
    mlp_dev = s_mlp.gate_proj.qweight.qweight.device
    for nom in ("gate_proj", "up_proj", "down_proj"):
        s_lin, e_lin = getattr(s_mlp, nom), getattr(e_mlp, nom)
        _demote_expert(s_lin, mlp_dev)
        _promote_expert(e_lin, mlp_dev)
        for tbl, cle in ((m._table_qw[nom], "qweight"),
                        (m._table_bscale[nom], "block_scale")):
            tbl[sortant] = adresse_expert(s_lin, cle)
            tbl[entrant] = adresse_expert(e_lin, cle)
            verifier_table(tbl)


def _couverture_experts(model) -> str:
    """Disposition des experts par couche MoE : « marlin » si toutes, « naturel »
    si aucune, sinon « marlin(N/M) » — N couches sur la disposition unique, M
    couches MoE (les autres refusées, pile naturelle gardée : gate/up distincts)."""
    from .model import MoEBlock
    blocs = [m for m in model.modules() if isinstance(m, MoEBlock)]
    if not blocs:
        return "aucun"
    dispositions = [getattr(m, "experts_layout", None) or "naturel" for m in blocs]
    n_marlin = sum(1 for d in dispositions if d == "marlin")
    if n_marlin == len(blocs):
        return "marlin"
    if n_marlin == 0:
        return dispositions[0] if len(set(dispositions)) == 1 else "naturel"
    return f"marlin({n_marlin}/{len(blocs)})"



def _etat_eco() -> dict:
    """Relecture SOUS CHARGE à chaque `regime()` : le moteur est chargé, la carte
    répond ; au chargement seul (carte verrouillée oisive) l'effectif lirait 225."""
    from .. import eco
    return eco.etat_eco(relire=True)


def _hote_texte() -> str:
    """`hote=thp,omp8[,cpus…]` effectif (acvram/hote.py)."""
    from ..hote import hote_texte
    return hote_texte()


def _glue_texte() -> str:
    """`glue=compact(8)` | `glue=temoin` (C15-3d, regime.glue_texte)."""
    from ..regime import glue_texte
    return glue_texte()


def _mrope_texte(spec) -> str:
    """`mrope=[24,20,20](interleaved)` lu sur le MODÈLE chargé (spec.mrope_section, rope_scaling du config.json),
    rien sans M-RoPE — la ligne de l'Engine ne le portait pas (seule celle de regime.py, 20/09 17:50)."""
    section = getattr(spec, "mrope_section", None)
    if not section:
        return ""
    rs = getattr(spec, "rope_scaling", None) or {}
    return f" mrope=[{','.join(str(int(x)) for x in section)}]({'interleaved' if rs.get('mrope_interleaved') else 'non-entrelace'})"


def _deepstack_texte(tour) -> str:
    """`deepstack=k` lu sur la TOUR chargée (config vision : deepstack_visual_indexes), rien sans niveaux."""
    k = int(getattr(tour, "niveaux_deepstack", 0) or 0) if tour is not None else 0
    return f" deepstack={k}" if k else ""


class ContexteNonTenu(RuntimeError):
    """`max_model_len` demandé mais non tenu par la chauffe au chargement : refus nommé, pas un 500 à la requête.
    Levé seulement si le contexte tenu est < 4096 ou sous ``--ctx-strict`` ; sinon le moteur se CLAMPE (Sage
    sage-s2-k48-feu-vert-21-09 § 2 (c)) et la ligne de régime dit ``ctx_tenu=N(demandé M)``."""

    def __init__(self, demande: int, tenu: int) -> None:
        self.demande, self.tenu = int(demande), int(tenu)
        super().__init__(f"contexte non tenu : max_model_len={self.demande} demandé, {self.tenu} jetons tenus par la "
                         f"chauffe (prefill d'une séquence pleine dans le régime servi) — relancer avec "
                         f"--max-model-len {self.tenu} ou moins ; ACVRAM_CHAUFFE_CTX=0 pour un banc qui pose son plan")


CTX_TENU_MIN = 4096                    # en dessous, un clamp n'est plus un service : refus nommé (Sage § 2 (c))


def _ctx_texte(engine) -> str:
    """`ctx_tenu=N` (chauffe tenue) | `ctx_tenu=non-verifie` (opt-out nommé) | `ctx_tenu=non-chauffe` (pas encore)."""
    v = getattr(engine, "ctx_tenu", "absent")
    if v is None:
        return " ctx_tenu=non-verifie"
    if v == "absent":
        return " ctx_tenu=non-chauffe"
    demande = getattr(engine, "ctx_demande", v)
    return f" ctx_tenu={v}" if demande == v else f" ctx_tenu={v}(demandé {demande})"


def _masque_images_texte(masque) -> str:
    """`masque_images=bidir|causal` sur la ligne dès qu'une tour est servie (famille lue au chargement)."""
    return f" masque_images={masque}" if masque else ""


def _vision_texte(tour) -> str:
    """Le mot ``vision=`` de la ligne du moteur (Manon 15 h 00 : la ligne de l'Engine ne le portait pas,
    seule celle de regime.py l'avait) : `vision=bf16(eager,transformers=5.17.0)` avec la tour, `vision=off`
    sans — la ligne dit ce que le moteur SERT."""
    from .vision import regime_texte
    return " " + (regime_texte() if tour is not None else "vision=off")


def _mla_core_texte() -> str:
    """`mla_core=tf32(≤2048 clés)` hors fp32 (sage-c14-defaut-tf32-8k addendum) ; `flash(fp32)` (C13-c) ;
    puis `mla_prep=grille|temoin` (C14-b geste 3)."""
    from . import mla
    txt = mla.regime_coeur_texte()
    return (f" {txt}" if txt else "") + " " + mla.regime_prep_texte() + " " + mla.regime_glue_texte()


def _eco_texte(e: dict) -> str:
    """``eco=2700(2692)`` conforme, ``eco=2700(libre: refus sudo)`` sinon —
    demandé ≠ effectif est un état nommé, jamais silencieux (sage-eco-2700-defaut § 2)."""
    eff = e.get("effectif") if e.get("effectif") is not None else "?"
    if e.get("etat") == "sans carte":
        return f"eco={e['demande']}(sans carte)"
    return f"eco={e['demande']}({eff})" if e.get("conforme") else f"eco={e['demande']}({eff}: {e.get('etat')})"


class Engine:
    """Détient le modèle, l'allocateur de blocs et les files de requêtes."""

    def __init__(self, loaded: LoadedModel, tokenizer: Any = None,
                 max_batch_size: int = 16, max_model_len: int = 8192,
                 enable_prefix_cache: bool = True,
                 speculator: Any = None, spec_k: int = 4,
                 enable_cuda_graphs: bool = True,
                 host_kv_gib: float = 0.0) -> None:
        # 0.6.31 : réglages hôte rejoués ici (idempotent) — serveur ET instruments (acvram/hote.py)
        from ..hote import regler_hote
        regler_hote()
        self.loaded = loaded
        self.model = loaded.model
        self.spec = loaded.spec
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len

        # Faille REGLES §6 (`load_model` sans `Plan` explicite reste dimensionné
        # pour `PlannerOptions.max_concurrent_seqs` par défaut, 8) refermée ici
        # plutôt que chez chaque appelant (sage-reprise-ordre-18-09 §Suite,
        # verdict-kv-budget-8-a-sec-18-09) : un `max_batch_size` au-delà du
        # nombre de séquences pour lequel le budget KV a été planifié tronque
        # en silence dès qu'une invite dépasse `kv_max_tokens / max_batch_size`
        # jetons — sans erreur, sans compteur. Ne pas confondre avec le budget
        # qui ne troque PAS quand les séquences sont courtes (le cas du 18/09) :
        # ici on refuse le lancement, pas seulement le symptôme.
        kv_planned_seqs = int(getattr(loaded.plan, "kv_planned_seqs", 0) or 0)
        self._kv_plan_override = bool(os.environ.get("ACVRAM_KV_PLAN_OVERRIDE"))
        if kv_planned_seqs and max_batch_size > kv_planned_seqs and not self._kv_plan_override:
            raise ValueError(
                f"max_batch_size={max_batch_size} > plan.kv_planned_seqs="
                f"{kv_planned_seqs} : le budget KV n'a jamais été dimensionné "
                f"pour ce lot (auto_plan/PlannerOptions.max_concurrent_seqs= "
                f"{kv_planned_seqs}, jamais relevé). Reconstruire le Plan avec "
                f"max_concurrent_seqs={max_batch_size}, ou poser "
                f"ACVRAM_KV_PLAN_OVERRIDE=1 pour forcer en connaissance de cause.")

        # sage-devstral-llama4-scaling-18-09 : `llama_4_scaling_beta` (yarn
        # ministral3/Devstral) est porté sur q par model.Attention
        # (`_echelle_llama4`, après le RoPE, préfill et décodage) quand le
        # spec le porte À LA CONSTRUCTION du modèle. Un spec modifié après
        # coup (beta > 0 que nulle Attention ne sert) au-delà de
        # `original_max_position_embeddings` reste un REFUS NOMMÉ plutôt qu'un
        # no-op silencieux (67aa280) ; sous le plafond la formule vaut 1.
        rs = getattr(self.spec, "rope_scaling", None) or {}
        # M-RoPE : section (t, h, w) et fusion spatiale de la tour, lues une fois ;
        # None = les positions restent 1-D pour toute séquence
        self._mrope_section = getattr(self.spec, "mrope_section", None)
        self._mrope_merge = int(getattr(self.spec, "spatial_merge_size", 0) or 0)
        self._llama4_scaling_beta = float(rs.get("llama_4_scaling_beta") or 0.0)
        self._llama4_scaling_plafond = int(rs.get("original_max_position_embeddings") or 0)
        from .model import Attention as _Attn
        self._llama4_servi = any(getattr(m, "llama4", None) is not None
                                 for m in self.model.modules() if isinstance(m, _Attn))
        if (str(rs.get("rope_type") or rs.get("type") or "") == "yarn"
                and self._llama4_scaling_beta > 0 and self._llama4_scaling_plafond
                and not self._llama4_servi and max_model_len > self._llama4_scaling_plafond):
            raise ValueError(
                f"llama_4_scaling_beta non servi : max_model_len={max_model_len} "
                f"> original_max_position_embeddings={self._llama4_scaling_plafond} "
                f"(rope yarn, beta={self._llama4_scaling_beta}) -- aucune Attention du "
                f"modèle ne porte le scaling (spec sans rope_scaling à la construction) ; "
                f"au-delà de ce plafond la formule 1+beta*log(1+floor(position/"
                f"{self._llama4_scaling_plafond})) n'est plus 1,0.")

        # Hybrides à récurrence linéaire : l'état GDN vit par séquence, hors
        # du cache paginé ; le cache de préfixe n'aurait pas de sens (les
        # blocs KV ne suffisent pas à restaurer l'état), on le coupe.
        self.est_hybride = bool(getattr(self.spec, "layer_types", None))
        # Tour de vision (multimodal P1) : None sans tour (manifeste
        # `vision: non`) ; alors toute image est refusée, nommément.
        from .. import regime as _regime
        _regime.declarer_modele_charge(loaded.manifest)          # vision=off sur la ligne d'un alias texte (pièce a)
        self.vision: Optional[TourVision] = TourVision.depuis_dossier(
            loaded.path, loaded.manifest, self.model.embed_tokens.device)
        # Une tour servie exige un masque de plage image connu pour sa famille : refus NOMMÉ au chargement
        # (MasqueImageInconnu), jamais un bloc bidirectionnel appliqué par défaut (Qwen3-VL, 20/09 18:57)
        self.masque_images: Optional[str] = None
        if self.vision is not None:
            from .vision import masque_images_famille
            self.masque_images = masque_images_famille(self.spec)
        self.gdn_states: dict = {}
        # Hybrides à récurrence linéaire : les blocs KV ne suffisent pas à
        # reprendre une invite, l'état récurrent vit hors du cache paginé. On
        # le photographie donc aux frontières régulières du prefill, en RAM
        # hôte épinglée (150 Mio pour les quarante-huit couches d'un 27B), et
        # un préfixe n'est repris que jusqu'à la frontière dont on tient
        # l'instantané.
        self._pas_insta = int(os.environ.get("ACVRAM_INSTA_PAS", "256"))
        self._max_insta = int(os.environ.get("ACVRAM_INSTA_MAX", "3"))
        self._insta: "OrderedDict[int, list]" = OrderedDict()

        if self.model.caches:
            n_blocks = min(c.cfg.num_blocks for c in self.model.caches.values())
        else:
            # Modèle sans cache paginé (MLA latent contigu, GLM ; ou
            # récurrence linéaire pure) : `self.model.caches` est vide, le
            # générateur ci-dessus ne rend jamais rien. Le budget de jetons
            # vivants doit venir du plan réel (`auto_plan`, `kv_max_tokens`)
            # — pas d'un défaut arbitraire qui écraserait silencieusement un
            # budget calculé pour ce rig. Trouvé le 16/09 : le défaut de
            # 1024 blocs (16 384 jetons) s'appliquait à tout modèle MLA,
            # quel que soit le rig ou le plan réel.
            kv_max = getattr(self.loaded.plan, "kv_max_tokens", 0) or 0
            n_blocks = max(1, kv_max // BLOCK_SIZE) if kv_max else 1024
        self.allocator = BlockAllocator(n_blocks, enable_prefix_cache)
        # Les hybrides ne vérifient pas encore q_len > 1 à formes fixes : une
        # proposition n-gram y coûte une passe eager (≈3× le pas) pour un gain
        # incertain. Pas de spéculation sur eux tant que ce chemin manque.
        self.speculator = speculator
        self.spec_k = spec_k
        # Garde de lot : mesure 14/09 (verdict-cout-verification-ngram-b12),
        # la speculation coute a b_reel=12 (carte deja pleine) et gagne a
        # b_reel=1 (verdict-taux-ngram-code-13-09) -- cf. GardeSpeculation.
        self._garde_spec = GardeSpeculation(
            int(os.environ.get("ACVRAM_SPECULATION_LOT_MAX", "2")))
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.stats = EngineStats(kv_blocks_total=n_blocks)
        self._lock = threading.Lock()
        self._refusees: list[GenerationOutput] = []     # admissions impossibles, rendues au pas suivant
        self._eos = self._eos_ids()
        # Étage hôte du cache KV : les blocs de préfixe évincés descendent en
        # RAM et remontent au réemploi, au lieu d'être recalculés.
        self.host_kv = None
        if host_kv_gib > 0 and self.model.caches:
            from ..memory.kvcache import HostKVPool
            self.host_kv = HostKVPool(int(host_kv_gib * 1024 ** 3))

            def _deverser(blk: int, h: int) -> None:
                self.host_kv.store(h, [c.export_block(blk)
                                       for c in self.model.caches.values()])
            self.allocator.spill_cb = _deverser

        self.graphs = None
        self._graphes_demandes = enable_cuda_graphs
        self._graphes_raison: Optional[str] = ("désactivés (enable_cuda_graphs=False)"
                                               if not enable_cuda_graphs else None)
        if enable_cuda_graphs:
            from .graphs import GraphRunner
            gr = GraphRunner(self.model, max_model_len, max_batch_size=self.max_batch_size)
            self.graphs = gr if gr.enabled else None
            if not gr.enabled:
                self._graphes_raison = gr.raison or "raison non nommée"

        # REPIN (bead anticitoyen-vram-pds, point 3 — Sage §4). `_pin` :
        # {index_couche: set(experts résidents)} — peuplé depuis les couches
        # que loader.py a placées par expert (`MoEBlock._pin_experts`, point
        # 1). VIDE si aucune ne l'est encore (comportement d'aujourd'hui,
        # tout `mlp_storage`) : `_repin_pass` ne trouve alors rien à échanger
        # et ne journalise rien — un no-op de fait, pas déguisé.
        from .model import MoEBlock
        self._pin: dict = {m.index_couche: set(m._pin_experts)
                          for m in self.model.modules()
                          if isinstance(m, MoEBlock) and m._pin_experts is not None}
        self._dernier_repin = 0

        # Recouvrement pas n+1 / rejeu n (bead runner, 14/09 — Jérôme, accord
        # d'interface avec Laurine). `_pipeline_pendiente` porte le résultat
        # DÉJÀ REJOUÉ mais pas encore rapatrié d'un pas antérieur :
        # {"seqs", "tokens_dev", "logprobs_dev", "event"}. None = rien en vol
        # (au repos, ou juste après une recomposition du lot). Défaut ON depuis
        # 0.6.34 (Maîtresse 21/09, verdict A/B rendu : ids identiques au bit) ;
        # `ACVRAM_PIPELINE=0` = témoin sans recouvrement. La ligne de régime
        # porte `pipeline=1|0`.
        self._pipeline_pendiente: Optional[dict] = None
        self.pipeline_actif = os.environ.get("ACVRAM_PIPELINE", "1") not in ("0", "")
        # `evenement_jetons` de GraphRunner ne borne QUE le rejeu — enregistré
        # par `rejouer_suivant()` avant que `_sample_only` (l'argmax) soit
        # même lancé. Le synchroniser au pas suivant garantirait le rejeu,
        # pas l'échantillonnage enfilé APRÈS lui sur le même flux : un jeton
        # parfois encore en vol au moment du `.tolist()` — trouvé par le
        # test bit-identique (bead runner, 14/09), pas par relecture. Un
        # `torch.cuda.Event()` NEUF à chaque pas (pas un seul réutilisé) :
        # `_plain_decode_pipeline` enfile le pas n+1 avant de synchroniser
        # sur l'événement du pas n — un objet partagé ré-enregistré par le
        # pas n+1 pointerait alors vers SA PROPRE fin, pas celle du pas n
        # (bogue trouvé le 14/09 soir, même famille que le premier).

        # `piles_ok` n'est pas encore décidé à ce point (paresseux, au
        # premier passage GPU) : cette ligne ne peut donc pas être un
        # verdict complet, seulement ce qui est déjà connu au chargement.
        if not os.environ.get("ACVRAM_REGIME_MUET"):
            print(f"[acvram] {self.regime_ligne()}", flush=True)

    # -- régime ------------------------------------------------------------
    def regime(self) -> dict:
        """État réel du moteur après chargement — pas ce qu'on espérait,
        ce qui tourne. Demandé par Jérôme (14/09 soir) après qu'un
        deuxième chargement de modèle dans le même processus ait
        silencieusement dégradé le plan (exil supplémentaire, graphes CUDA
        coupés, modèle réparti sur 2 cartes) et faussé une mesure sans
        qu'aucune ligne ne le dise. « Plus jamais un chiffre mesuré sur un
        moteur dégradé sans le savoir. »

        `piles_ok` : `MoEBlock._stack_state` est décidé PARESSEUSEMENT, au
        premier passage GPU de chaque couche — encore "?" (non vérifié)
        juste après le chargement, avant tout `step()`/`warm_graphs()`.
        `experts_exiles` compte les EXPERTS (au moins une de leurs
        projections `QuantLinear.streamed`), pas les couches — l'exil du
        plan (`couches_exilees`) est un exil de couche ENTIÈRE ; les deux
        coexistent et ne se déduisent pas l'un de l'autre.
        """
        from .. import kernels
        from ..regime import regime_noyaux
        from .gdn import gdn_regime as _gdn_regime
        from .model import MoEBlock

        plan = self.loaded.plan
        couches_exilees = sum(1 for lp in plan.layers if lp.streamed)
        cartes = sorted({lp.exec_device for lp in plan.layers}
                        | {plan.embed_device, plan.lm_head_device})

        etats_piles: set[str] = set()
        raisons_piles: set[str] = set()
        experts_total = experts_exiles = 0
        for m in self.model.modules():
            if not isinstance(m, MoEBlock):
                continue
            etats_piles.add(m._stack_state)
            if m._stack_state == "non" and getattr(m, "_raison_repli", ""):
                raisons_piles.add(m._raison_repli)
            for expert in m.experts:
                experts_total += 1
                if any(getattr(getattr(expert, nom, None), "streamed", None) is not None
                      for nom in ("gate_proj", "up_proj", "down_proj")):
                    experts_exiles += 1

        if not etats_piles:
            piles_ok: Optional[bool] = None          # pas de couche MoE
        elif "non" in etats_piles:
            piles_ok = False
        elif etats_piles == {"oui"}:
            piles_ok = True
        else:
            piles_ok = None                           # au moins une "?" : non vérifié

        # « mma-a4 », pas « mma » : le chemin MMA du préfill quantifie les activations en E2M1
        # (W4A4) — le nom porte le régime de précision (sage-glm-cellules-w4a4-hote-20-09)
        chemin_moe = ("mma-a4" if os.environ.get("ACVRAM_MOE_MMA", "1") not in ("0", "")
                     else "gemv")
        if os.environ.get("ACVRAM_GRAPHES_TABLE") == "0":
            chemin_moe += "+pile" if piles_ok else "+pile(désactivé)"

        # `self.graphs` reste le MÊME OBJET après une capture ratée en cours
        # de service (`GraphRunner._capture` bascule `enabled=False` mais ne
        # se retire pas de `self.graphs`, graphs.py:431) : lire seulement
        # « l'objet existe » disait `graphes=on` alors que le moteur avait
        # déjà replié en eager — signalé par plusieurs verdicts (poste E, KV
        # lm4) où `regime_ligne()` mentait sur le régime réellement mesuré.
        return {
            # état VIVANT : `GraphRunner.enabled` retombe à False quand une
            # capture échoue au premier pas (graphs.py ~431) ; lu sur l'objet,
            # la ligne disait `graphes=on` sur des bras entièrement en eager
            # (sage-kv-lm4-clos-17-09 § 1)
            "graphes": self.graphs is not None and bool(self.graphs.enabled),
            "repli_eager": int(getattr(self.graphs, "replis_eager", 0)) if self.graphs is not None else 0,
            "replis_eager_raisons": sorted(getattr(self.graphs, "_raisons_eager_vues", set())) if self.graphs is not None else [],
            "slots_hybrides": getattr(self.graphs, "max_slots", None) if self.graphs is not None else None,
            # photos VRAM (octets) prises par GraphRunner avant sa première capture
            # et après un échec (chantier-gemma-capture-godet1-20-09) ; None à sec
            "graphes_memoire_avant_capture": getattr(self.graphs, "memoire_avant_capture", None) if self.graphs is not None else None,
            "graphes_memoire_apres_echec": getattr(self.graphs, "memoire_apres_echec", None) if self.graphs is not None else None,
            "graphes_demandes": self._graphes_demandes,
            "graphes_raison": (self._graphes_raison if self.graphs is None
                               else (None if self.graphs.enabled
                                     else (self.graphs.raison or "capture impossible"))),
            "couches_exilees": couches_exilees,
            "couches_total": len(plan.layers),
            "experts_exiles": experts_exiles,
            "experts_total": experts_total,
            "piles_ok": piles_ok,
            "piles_raison": sorted(raisons_piles),
            "cartes": cartes,
            "chemin_moe": chemin_moe,
            # régime du prefill NVFP4 non groupé : bf16 (W4A16) | w8a8 | w4a4 —
            # jamais plus tacite (sage-prefill-a8-verdict-17-09)
            "prefill": kernels.prefill_regime() + self._prefill_coupe_texte(),
            # linéaires INT8 du préfill (P0) : bf16 | a8 — toujours écrit
            "prefill_int8": kernels.prefill_int8_regime(),
            # P1 disposition unique : « marlin » (pile Marlin seule, préfill et
            # décodage, la pile NVFP4 rendue) | « naturel » (pile NVFP4 seule)
            # couverture PAR COUCHE (Sage 19/09, budget GLM : 33 couches Marlin + 13 refusées
            # « distinctes » sur la pile naturelle — « experts_layout=marlin » seul mentait)
            "experts_layout": _couverture_experts(self.model),
            # linéaires INT8 du décodage : triton≥b|cuda (poste C, bascule mesurée)
            "dense": kernels.narrow_regime(),
            "gdn": _gdn_regime(),
            "noyaux": regime_noyaux()["hors_defaut"],
            "eco": _etat_eco(),
            "kv_plan_override": self._kv_plan_override,
            # sage-devstral-llama4-scaling-18-09 : visible meme sous le
            # plafond (non refuse ici), pour ne jamais laisser croire que le
            # scaling est applique alors qu'il ne l'est nulle part.
            "llama4_scaling_beta": self._llama4_scaling_beta or None,
        }

    def kv_format_servi(self) -> str:
        """Le format du cache KV SERVI, lu sur les caches construits (pas sur une variable) : `int8`, `bf16`, `fp16`,
        `fp8_e4m3`, … ; `int8-canal16` sous C5-b ; `latent-bf16` / `latent-fp8` pour un MLA pur (cache latent contigu,
        aucun cache paginé). 20/09 (Jérôme, prise (b) 12B vision) : la ligne ne disait pas que le palier cuda:0 du
        manifeste servait le KV en int8 — un chiffre de log-prob comparé à transformers (KV bf16) sans le savoir."""
        try:
            if self.model.caches:
                fmts = sorted({str(c.cfg.dtype) for c in self.model.caches.values()})
                fmt = fmts[0] if len(fmts) == 1 else "|".join(fmts)
                try:
                    from ..memory import kv_canal as _kvc
                    if _kvc.ACTIF and fmt == "int8":
                        fmt = "int8-canal16"
                except Exception:                                    # noqa: BLE001
                    pass
                return fmt
            from . import mla as _mla
            return "latent-fp8" if getattr(_mla, "_MLA_LATENT_FP8", False) else "latent-bf16"
        except Exception as exc:                                    # noqa: BLE001
            return f"?({type(exc).__name__})"

    def regime_ligne(self) -> str:
        """Une ligne, pour le log au chargement et `acvram serve --regime`."""
        r = self.regime()
        nominal = ((r["graphes"] or not r["graphes_demandes"])
                  and r["couches_exilees"] == 0
                  and r["experts_exiles"] == 0 and r["piles_ok"] is not False
                  and len(r["cartes"]) <= 1)
        etat = "NOMINAL" if nominal else "DÉGRADÉ"
        piles_txt = f"piles_ok={r['piles_ok']}"
        if r["piles_ok"] is False and r["piles_raison"]:
            # DÉGRADÉ nommé, pas seulement constaté : sans la cause, un
            # "piles_ok=False" oblige à relire le code pour savoir si c'est
            # une vraie anomalie (formats réellement mélangés) ou un cas
            # attendu (bf16 non quantifié, sans noyau groupé — trouvé le
            # 15/09 sur un GLM converti --format bf16, pris pour un bogue).
            piles_txt += " (" + " ; ".join(r["piles_raison"]) + ")"
        kv_seqs = getattr(self.loaded.plan, "kv_planned_seqs", 0) or "?"
        slots = r.get("slots_hybrides")
        # graphes demandés mais retombés (capture impossible) : la CAUSE est sur la
        # ligne — `graphes=off(repli eager: AcceleratorError: CUDA error: out of memory)`
        # (gemma b=1, sage-cloture-nuit-0540-20-09 rang 3) ; off demandé : `graphes=off`
        raison_off = (f"(repli eager: {r['graphes_raison']})"
                      if not r["graphes"] and r.get("graphes_demandes") and r.get("graphes_raison") else "")
        return (f"régime {etat} — graphes={'on' if r['graphes'] else 'off' + raison_off}"
                f"{'' if slots is None else f'(hybrides≤{slots})'} "
                f"repli_eager={r.get('repli_eager', 0)} "
               f"couches_exilées={r['couches_exilees']}/{r['couches_total']} "
               f"experts_exilés={r['experts_exiles']}/{r['experts_total']} "
               f"{piles_txt} cartes={r['cartes']} "
               f"chemin_moe={r['chemin_moe']} prefill={r['prefill']} prefill_int8={r['prefill_int8']} dense={r['dense']} "
               f"ACVRAM_GDN={r['gdn']} experts_layout={r['experts_layout']} "
               + (f"noyaux={r['noyaux']} " if r["noyaux"] else "")
               + f"kv_budget={self.allocator.num_blocks * BLOCK_SIZE}/{kv_seqs} "
               + f"kv={self.kv_format_servi()} "
               + f"pipeline={int(bool(self.pipeline_actif and self.graphs is not None))} "   # effectif : demandé ET graphes
               + (f"kv_plan_override=1 " if r["kv_plan_override"] else "")
               + (f"llama4_scaling_beta={r['llama4_scaling_beta']}"
                  f"({'servi' if self._llama4_servi else 'non_servi'}) "
                  if r["llama4_scaling_beta"] else "")
               + (f"ignore_eos={self.stats.sequences_ignore_eos} "
                  if self.stats.sequences_ignore_eos else "")
               + f"cache_prefixe={self.stats.hit_rate:.3f} "
               f"({self.stats.cached_prompt_tokens} vram+hôte, "
               f"{self.stats.host_kv_tokens} hôte) "
               + _eco_texte(r["eco"])
               + _mla_core_texte()
               + " " + _glue_texte()
               + " " + _hote_texte()
               + _vision_texte(self.vision)
               + _ctx_texte(self)
               + _masque_images_texte(self.masque_images)
               + _mrope_texte(self.spec)
               + _deepstack_texte(self.vision))

    def fermer(self) -> None:
        """Arrêt du moteur : rend l'horloge éco posée par ce processus
        (sage-eco-2700-defaut-19-09 § 1) — le `-rgc` suit la vie du serveur.
        Idempotent ; l'atexit et les signaux font le même geste."""
        from .. import eco
        eco.rendre_horloge()

    # -- admission -------------------------------------------------------
    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        cfg = self.loaded.manifest.get("model", {})
        raw = self.loaded.manifest.get("generation_config", {})
        # Le generation_config.json est recopie a cote du modele converti :
        # c'est lui qui porte <|im_end|> chez Qwen.
        disque: dict = {}
        d = getattr(self.loaded, "path", "") or ""
        gen_path = os.path.join(d, "generation_config.json") if d else ""
        if gen_path and os.path.isfile(gen_path):
            try:
                with open(gen_path, "r", encoding="utf-8") as fh:
                    disque = json.load(fh)
            except (OSError, json.JSONDecodeError):
                disque = {}
        for key in ("eos_token_id", "eos_token_ids"):
            for src in (cfg, raw, disque):
                v = src.get(key)
                if isinstance(v, int):
                    ids.add(v)
                elif isinstance(v, list):
                    ids.update(int(x) for x in v if isinstance(x, int))
        return ids

    def add_request(self, prompt_ids: list[int], params: SamplingParams,
                    request_id: str = "", images: Any = None) -> Sequence:
        """``images`` (multimodal P1) : itérable de (debut, fin, pixel_values,
        sha256) ou d'objets à ces attributs — voir engine/vision.ImageRequete."""
        if len(prompt_ids) >= self.max_model_len:
            raise ValueError(
                f"invite de {len(prompt_ids)} jetons au-delà de max_model_len "
                f"{self.max_model_len}")
        ims = [ImageRequete.depuis(i) for i in (images or [])]
        if ims and self.vision is None:
            raise SansTourVision(
                f"{len(ims)} image(s) pour un modèle sans tour de vision "
                f"(manifeste vision: non) — request_id={request_id!r}")
        verifier_plages(ims, len(prompt_ids))
        seq = Sequence(list(prompt_ids), params, request_id)
        seq.images = sorted(ims, key=lambda i: i.debut)
        if params.ignore_eos:
            self.stats.sequences_ignore_eos += 1
        with self._lock:
            self.waiting.append(seq)
        return seq

    def abort(self, request_id: str) -> None:
        with self._lock:
            for seq in list(self.running) + list(self.waiting):
                if seq.request_id == request_id:
                    self._finish(seq, "abort")

    def _admit(self) -> list[Sequence]:
        """Fait entrer dans le lot autant de séquences en attente que les blocs le permettent."""
        admitted = []
        with self._lock:
            while self.waiting and len(self.running) < self.max_batch_size:
                seq = self.waiting[0]
                # On réserve l'invite plus un peu de marge, pour que les
                # premières étapes de décodage n'aient pas besoin aussitôt d'un
                # bloc supplémentaire.
                need = min(seq.blocks_needed(extra=BLOCK_SIZE), self._blocs_plafond())
                if need > self.allocator.num_blocks:
                    # Jamais admissible, même carte vide : la garder en file
                    # laissait le moteur tourner à vide sans une ligne (70B
                    # en exil, budget KV 0 → 1 bloc, invite de 17 blocs).
                    self.waiting.pop(0)
                    print(f"[acvram] requête refusée : {need} blocs KV nécessaires, "
                          f"{self.allocator.num_blocks} en tout (request_id={seq.request_id})",
                          flush=True)
                    self._finish(seq, "refus")
                    self._refusees.append(GenerationOutput(
                        sequence_id=seq.id, request_id=seq.request_id, token_ids=[],
                        finished=True, finish_reason="refus",
                        prompt_tokens=len(seq.prompt_ids), completion_tokens=0))
                    continue
                if need > self.allocator.num_free:
                    break
                self.waiting.pop(0)

                # Tour de vision : une passe eager par image, ici, avant le
                # prefill et hors de tout graphe. Une tour qui échoue refuse
                # la requête, nommément.
                if seq.images and seq.image_embeds is None:
                    try:
                        # M-RoPE (engine/mrope) : positions [3, T] et delta de
                        # l'invite depuis (debut, fin, supplement["image_grid_thw"])
                        # de chaque image — avant la tour, elles n'en dépendent pas
                        if self._mrope_section is not None:
                            from .mrope import grille_de, positions_mrope
                            seq.mrope_positions, seq.rope_delta = positions_mrope(
                                len(seq.prompt_ids),
                                [(im.debut, im.fin, grille_de(im)) for im in seq.images],
                                self._mrope_merge)
                        rendus = [
                            (im.debut, im.fin,
                             *self.vision.traits_niveaux(im.pixel_values, im.fin - im.debut,
                                                         supplement=getattr(im, "supplement", None)))
                            for im in seq.images]
                        seq.image_embeds = [(d, f, e) for d, f, e, _ in rendus]
                        # Deepstack : les niveaux suivent les traits, tous ou aucun
                        # (une tour rend le même nombre de niveaux pour chaque image)
                        seq.image_niveaux = ([(d, f, n) for d, f, _, n in rendus]
                                             if all(n is not None for *_, n in rendus) else None)
                        if seq.image_niveaux is None and any(n is not None for *_, n in rendus):
                            raise ValueError("tour de vision : niveaux deepstack rendus pour une partie "
                                             "des images seulement")
                        # Le journal dit que la tour a tourné (Manon 15 h 00 : « aucune ligne tour ») —
                        # une somme des traits par image, pour qu'une image différente se voie
                        print("[engine] tour : " + " ; ".join(
                            f"[{d},{f}) {tuple(e.shape)} sha={im.sha256[:8]} Σ={float(e.float().abs().sum()):.4g}"
                            for (d, f, e), im in zip(seq.image_embeds, seq.images))
                              + f" request_id={seq.request_id}", flush=True)
                    except Exception as exc:                 # noqa: BLE001
                        print(f"[engine] refus : tour de vision en échec "
                              f"({type(exc).__name__}: {exc}) request_id={seq.request_id}",
                              flush=True)
                        self._finish(seq, "refus")
                        self._refusees.append(GenerationOutput(
                            sequence_id=seq.id, request_id=seq.request_id, token_ids=[],
                            finished=True, finish_reason="refus",
                            prompt_tokens=len(seq.prompt_ids), completion_tokens=0))
                        continue

                # On sert les blocs de tête que le cache détient déjà. Un bloc
                # est toujours retenu : une requête dont l'invite est
                # entièrement en cache a tout de même besoin d'un jeton à faire
                # traverser le modèle.
                hashes = BlockAllocator.block_hashes(seq.prompt_ids, BLOCK_SIZE,
                                                     images=seq.images)
                limit = max(0, (len(seq.prompt_ids) - 1) // BLOCK_SIZE)
                # Sur un hybride, les blocs KV ne valent que si l'état récurrent
                # de la même frontière est disponible : on plafonne l'appariement
                # à la plus grande frontière photographiée.
                insta_f = 0
                if self.est_hybride:
                    insta_f = self._frontiere_disponible(seq, limit * BLOCK_SIZE)
                    limit = insta_f // BLOCK_SIZE
                matched = self.allocator.match_prefix(hashes, limit=limit)
                if self.est_hybride:
                    if len(matched) * BLOCK_SIZE != insta_f or insta_f == 0:
                        # Appariement partiel : l'état et les clés ne
                        # coincideraient pas, on repart de l'invite entiere.
                        if matched:
                            self.allocator.free(matched)
                        matched = []
                    else:
                        self._reprendre_insta_a(seq, insta_f)
                # L'étage hôte prolonge la suite : chaque bloc suivant présent
                # en RAM remonte dans un bloc VRAM fraîchement alloué.
                if self.host_kv is not None:
                    while len(matched) < (limit or 0):
                        data = self.host_kv.fetch(hashes[len(matched)])
                        if data is None or self.allocator.num_free < 1:
                            break
                        blk = self.allocator.allocate(1)[0]
                        for c, d in zip(self.model.caches.values(), data):
                            c.import_block(blk, d)
                        self.allocator.register(blk, hashes[len(matched)])
                        matched.append(blk)
                        self.stats.kv_refills += 1
                seq.blocks = list(matched)
                seq.cached_len = len(matched) * BLOCK_SIZE
                # Le cache dispense de recalculer ces jetons : la progression
                # du prefill part de là, elle ne part pas de zéro.
                seq.prefill_len = seq.cached_len
                seq.hashes = list(hashes[:len(matched)])
                seq.blocks.extend(self.allocator.allocate(need - len(matched)))
                self.stats.cached_prompt_tokens += seq.cached_len

                self.running.append(seq)
                admitted.append(seq)
        return admitted

    # -- instantanés d'état récurrent -------------------------------------
    def _prefill_coupe_texte(self) -> str:
        """« (coupé@256) » derrière `prefill=` quand le régime servi coupe le prefill à la
        frontière d'instantané (`_frontiere_insta` : hybride ET cache de préfixe ON) — un
        prefill en deux morceaux n'a pas la numérique d'un morceau (MECANISMES, 20/09 :
        contrôle 3.2, 14,9613 contre 14,7888 sur GLM 8 192 + 512). Dense, ou cache OFF :
        rien. Le pas est celui lu (`ACVRAM_INSTA_PAS`), pas une constante."""
        if self.est_hybride and self.allocator.enable_prefix_cache:
            return f"(coupé@{self._pas_insta})"
        return ""

    def _frontiere_insta(self, seq: Sequence) -> Optional[int]:
        """Position où couper le prefill pour photographier l'état, ou None.

        Un multiple du pas d'instantané, strictement à l'intérieur de l'invite
        et au-delà de ce que le cache a déjà servi. Le choix ne dépend que de la
        longueur de l'invite : deux requêtes partageant une amorce tombent sur
        la même frontière tant qu'elles restent dans la même tranche.
        """
        if not self.est_hybride or not self.allocator.enable_prefix_cache:
            return None
        pas = self._pas_insta
        n = len(seq.prompt_ids)
        f = ((n - 1) // pas) * pas
        if f < pas or f <= seq.cached_len or f % BLOCK_SIZE:
            return None
        return f

    @staticmethod
    def _vers_hote(etat):
        if torch.is_tensor(etat):
            # Mémoire épinglée : la restitution est un transfert asynchrone de
            # 150 Mio, deux fois plus rapide qu'en mémoire paginable.
            h = torch.empty_like(etat, device="cpu", pin_memory=True)
            h.copy_(etat.detach(), non_blocking=False)
            return h
        if isinstance(etat, (tuple, list)):
            return type(etat)(Engine._vers_hote(x) for x in etat)
        if isinstance(etat, dict):
            return {k: Engine._vers_hote(v) for k, v in etat.items()}
        return etat

    @staticmethod
    def _vers_gpu(etat, ref):
        if torch.is_tensor(etat):
            dev = ref.device if torch.is_tensor(ref) else torch.device("cuda")
            return etat.to(dev, copy=True, non_blocking=True)
        if isinstance(etat, (tuple, list)):
            return type(etat)(Engine._vers_gpu(x, ref) for x in etat)
        if isinstance(etat, dict):
            return {k: Engine._vers_gpu(v, ref) for k, v in etat.items()}
        return etat

    def _cle_insta(self, ids: list[int], jusqu_a: int) -> int:
        return hash(tuple(ids[:jusqu_a]))

    def _photographier(self, seq: Sequence, coupe: int) -> None:
        """Range l'état récurrent de toutes les couches, pris à ``coupe``."""
        instant = []
        for idx, par_seq in self.gdn_states.items():
            etat = par_seq.get(seq.id)
            if etat is None:
                continue
            instant.append((idx, self._vers_hote(etat)))
        if not instant:
            return
        cle = self._cle_insta(seq.prompt_ids, coupe)
        self._insta[cle] = instant
        self._insta.move_to_end(cle)
        while len(self._insta) > self._max_insta:
            self._insta.popitem(last=False)

    def _frontiere_disponible(self, seq: Sequence, plafond: int) -> int:
        """Plus grande frontière photographiée pour cette invite, sous
        ``plafond`` jetons. Zéro si aucune."""
        if not self._insta:
            return 0
        pas = self._pas_insta
        f = (min(plafond, len(seq.prompt_ids) - 1) // pas) * pas
        while f >= pas:
            if self._cle_insta(seq.prompt_ids, f) in self._insta:
                return f
            f -= pas
        return 0

    def _reprendre_insta_a(self, seq: Sequence, f: int) -> None:
        """Restaure l'état récurrent photographié à la frontière ``f``."""
        cle = self._cle_insta(seq.prompt_ids, f)
        instant = self._insta.get(cle)
        if instant is None:
            return
        for idx, etat in instant:
            par_seq = self.gdn_states.setdefault(idx, {})
            ref = next(iter(par_seq.values()), None)
            par_seq[seq.id] = self._vers_gpu(etat, ref)
        self._insta.move_to_end(cle)
        self.stats.kv_refills += 1

    def _register_complete_blocks(self, seq: Sequence) -> None:
        """Publie les blocs désormais pleins, pour que des requêtes ultérieures les
        réutilisent.

        Uniquement des blocs complets : un bloc à moitié rempli, retrouvé par un
        hachage nommant un contenu qu'il ne porte pas encore, livrerait à une
        requête ultérieure des clés et des valeurs jamais écrites.
        """
        # Pas de `seq.all_ids` ici : il concatene tout le contexte pour n en
        # lire SEIZE elements, a chaque pas et pour chaque sequence. `length`
        # donne la meme longueur sans rien recopier, et `_tranche` ne touche
        # que les seize jetons du bloc.
        n_full = min(seq.longueur_ecrite // BLOCK_SIZE, len(seq.blocks))
        while len(seq.hashes) < n_full:
            i = len(seq.hashes)
            prev = seq.hashes[-1] if seq.hashes else 0
            span = _tranche(seq, i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)
            h = BlockAllocator.hash_bloc(
                prev, span, BlockAllocator.sel_images(seq.images, i * BLOCK_SIZE,
                                                      (i + 1) * BLOCK_SIZE))
            seq.hashes.append(h)
            self.allocator.register(seq.blocks[i], h)

    def _finish_budget_epuise(self, seq: Sequence) -> GenerationOutput:
        """`_grow` a échoué : le budget KV est épuisé, pas la séquence qui
        a fini naturellement. `finish_reason` reste "length" (contrat API
        OpenAI/Anthropic, ne pas y toucher) mais jamais silencieux par
        ailleurs (REGLES : un échec est un résultat) — trouvé le 16/09,
        les quatre sites qui appellent `_grow` puis `_finish(seq, "length")`
        ne distinguaient pas ce cas d'un `max_tokens` atteint normalement,
        et aucun n'émettait de `GenerationOutput` pour cette séquence : elle
        disparaissait du lot sans jamais signaler sa fin à l'appelant. Rendu
        ici pour que l'appelant l'ajoute à ce que le pas rend ; le rendre est
        sûr partout (aucun état à recomposer), câblé dans le retour
        seulement au chemin sans graphes pour l'instant (indépendant du
        chantier de recouvrement de Laurine)."""
        self.stats.sequences_tronquees_budget += 1
        print(f"[acvram] budget KV épuisé, séquence tronquée avant "
             f"max_tokens : request_id={seq.request_id} "
             f"sortis={len(seq.output_ids)}/{seq.params.max_tokens} "
             f"blocs_libres={self.allocator.num_free}", flush=True)
        sortie = GenerationOutput(
            sequence_id=seq.id, request_id=seq.request_id, token_ids=[],
            finished=True, finish_reason="length",
            prompt_tokens=len(seq.prompt_ids), completion_tokens=len(seq.output_ids))
        self._finish(seq, "length")
        return sortie

    def _blocs_plafond(self) -> int:
        """Jamais plus de blocs qu il n en faut pour ``max_model_len`` (une séquence n écrit aucune position au-delà :
        `length >= max_model_len` finit la séquence). La marge d admission ``+BLOCK_SIZE`` au-delà donnait une table de
        257 blocs à un graphe dimensionné au contexte (256) : P3 (4) 30B, confirmation avec graphes à 4096 →
        ``size of tensor a (256) must match b (257)`` — et le même 500 pour toute invite > max_model_len − 16 sous graphes."""
        return (int(self.max_model_len) + BLOCK_SIZE - 1) // BLOCK_SIZE

    def _grow(self, seq: Sequence, extra: int = 0) -> bool:
        need = min(seq.blocks_needed(extra=extra), self._blocs_plafond())
        if need <= len(seq.blocks):
            return True
        if self.allocator.num_free < need - len(seq.blocks):
            return False
        seq.blocks.extend(self.allocator.allocate(need - len(seq.blocks)))
        return True

    def _finish(self, seq: Sequence, reason: str) -> None:
        seq.finished = True
        if self.speculator is not None:
            self.speculator.release(seq)
        seq.finish_reason = reason
        if seq.blocks:
            # Publier avant de rendre les blocs : une requête qui s'arrête au
            # premier jeton voyait son invite entierement perdue, puisque
            # `_register_complete_blocks` s'execute apres `_emit` et ne trouvait
            # plus aucun bloc. Les invites courtes — un systeme partage, une
            # question breve — ne peuplaient donc jamais le cache de prefixe.
            self._register_complete_blocks(seq)
            self.allocator.free(seq.blocks)
            seq.blocks = []
        # HORS du bloc ci-dessus : l'état récurrent n'a aucun rapport avec le
        # fait que la séquence détienne encore des blocs KV. Les deux étaient
        # liés, si bien qu'un `_finish` appelé sur une séquence déjà libérée —
        # annulation tardive, second appel — laissait son état sur la carte.
        # Cela ne fuyait probablement pas aujourd'hui, un état ne naissant
        # qu'au forward et un forward exigeant des blocs ; mais c'était vrai
        # par l'état du moteur, pas par construction.
        for etats in self.gdn_states.values():
            etats.pop(seq.id, None)
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)

    # -- batch construction ----------------------------------------------
    def _budget_jetons(self) -> int:
        """Plafond de jetons d'invite par séquence et par pas — 0 = illimité.

        Coupé par défaut : a zero, le moteur se comporte au jeton pres comme
        avant. Actif, il découpe le prefill d'une longue invite en tranches
        et REND LA MAIN entre chaque, au lieu de faire attendre toutes les
        séquences en cours derriere elle. Le découpage existait deja
        (`_build_batch(..., limite=)`) mais chainait ses passes dans le meme
        pas : il découpait le calcul sans découper la latence.
        """
        try:
            return max(0, int(os.environ.get("ACVRAM_BUDGET_JETONS", "0")))
        except ValueError:
            return 0

    def _decodables(self) -> list[Sequence]:
        """Les séquences prêtes à décoder — un seul endroit qui le décide.

        Les essais recopiaient ce prédicat au lieu de l'appeler : ils
        auraient continué à passer en éprouvant l'ancienne notion.
        """
        return [s for s in self.running if s.prefilled and not s.finished]

    def _build_batch(self, seqs: list[Sequence], prefill: bool,
                     limite: Optional[int] = None) -> ForwardBatch:
        tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        block_tables: list[torch.Tensor] = []

        for seq in seqs:
            if prefill:
                # On saute ce que le cache de préfixe détient déjà.
                # `prefill_len` et non `cached_len` : identiques tant que le
                # prefill n'est pas decoupe, distincts des qu'il l'est. La
                # borne `limite` est une borne de FIN, absolue.
                ids = seq.prompt_ids[seq.prefill_len:limite]
                start = seq.prefill_len
            else:
                ids = [seq.output_ids[-1]] if seq.output_ids else [seq.prompt_ids[-1]]
                start = seq.length - 1
            for j, tok in enumerate(ids):
                pos = start + j
                tokens.append(tok)
                positions.append(pos if prefill else self._pos_decodage(seq, pos))
                slots.append(seq.blocks[pos // BLOCK_SIZE] * BLOCK_SIZE
                             + pos % BLOCK_SIZE)
            query_lens.append(len(ids))
            seq_lens.append(start + len(ids))
            block_tables.append(torch.tensor(seq.blocks, dtype=torch.long))

        # Multimodal P1 : traits d'image par séquence au prefill seulement ;
        # None pour tout le lot quand aucune n'en porte (chemin texte au bit).
        images = None
        deepstack = None
        if prefill and any(s.image_embeds for s in seqs):
            images = [list(s.image_embeds) if s.image_embeds else None for s in seqs]
            if any(s.image_niveaux for s in seqs):
                deepstack = [list(s.image_niveaux) if s.image_niveaux else None for s in seqs]

        return ForwardBatch(
            tokens=torch.tensor(tokens, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
            seq_lens=seq_lens, query_lens=query_lens,
            block_tables=block_tables,
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=prefill,
            seq_ids=[s.id for s in seqs], gdn_store=self.gdn_states,
            images=images, deepstack=deepstack,
            **self._mrope_du_lot(seqs, prefill, query_lens, seq_lens))

    @staticmethod
    def _mrope_du_lot(seqs: list[Sequence], prefill: bool, query_lens: list[int],
                      seq_lens: list[int]) -> dict:
        """Champs M-RoPE du lot : {} tant qu'aucune séquence n'a de positions
        3-D (texte, modèle sans mrope : le lot est celui d'avant, au bit).
        Prefill : ``positions_3d`` [3, t] — la tranche [début, fin) de chaque
        séquence à positions 3-D (morceau, reprise après le cache de préfixe),
        ses positions 1-D ×3 pour une séquence texte du même lot. Décodage :
        rien — ``positions`` porte déjà le delta (voir ``_pos_decodage``)."""
        if not prefill or not any(s.mrope_positions is not None for s in seqs):
            return {}
        morceaux = []
        for s, ql, sl in zip(seqs, query_lens, seq_lens):
            debut = sl - ql
            if s.mrope_positions is not None:
                morceaux.append(s.mrope_positions[:, debut:sl])
            else:
                morceaux.append(torch.arange(debut, sl, dtype=torch.long).view(1, -1).expand(3, -1))
        return {"positions_3d": torch.cat(morceaux, dim=1).contiguous(),
                "rope_delta": [s.rope_delta for s in seqs]}

    @staticmethod
    def _pos_decodage(seq: Sequence, pos: int) -> int:
        """La position que le RoPE lit au décodage : 1-D + ``rope_delta`` de la
        séquence (les trois axes M-RoPE valent max + 1 après l'image, un seul
        entier par créneau) ; 0 pour le texte, l'ancien chemin au bit."""
        return pos + seq.rope_delta

    @staticmethod
    def _eviter_coupe_image(seq: Sequence, fin: Optional[int]) -> Optional[int]:
        """Une borne de fin de morceau qui tombe DANS une plage image est
        déplacée : au début de l'image si celui-ci est encore à faire, sinon
        à sa fin. Les jetons d'une image se voient tous (masque bidirectionnel)
        et ne peuvent pas être calculés en deux morceaux : un morceau coupé
        dans l'image serait refusé par `layers.masque_images`."""
        if fin is None:
            return None
        for im in seq.images:
            if im.debut < fin < im.fin:
                return im.debut if im.debut > seq.prefill_len else im.fin
        return fin

    # -- the step --------------------------------------------------------
    def step(self) -> list[GenerationOutput]:
        """Exécute une passe avant et rend ce qu'elle a produit."""
        outputs = self._step()
        if _DUMP_MOE:
            self._sauver_dump_moe()
        return outputs

    def _sauver_dump_moe(self) -> None:
        """ACVRAM_DUMP_MOE (model._DUMP_MOE) : tous les tampons statiques de
        sortie MoE, par couche et par forme, après chaque pas (eager ou rejeu)."""
        from .model import MoEBlock
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        n = self.__dict__.get("_dump_pas", 0)
        self.__dict__["_dump_pas"] = n + 1
        blocs = [m for m in self.model.modules() if isinstance(m, MoEBlock)]
        d = {i: {f"{cle[0]}": b.detach().cpu().clone() for cle, b in m.__dict__.get("_dump_bufs", {}).items()}
             for i, m in enumerate(blocs)}
        torch.save(d, os.path.join(_DUMP_MOE, f"pas-{n:05d}.pt"))

    def _step(self) -> list[GenerationOutput]:
        new = self._admit()
        if new:
            self.stats.pas_avec_prefill += 1
        outputs: list[GenerationOutput] = self._refusees
        self._refusees = []

        # Lot groupé : un seul prefill pour toutes les séquences de `new` au
        # lieu d'un par séquence. `input_layernorm` et le MoE
        # (DecoderLayerGDN.forward) s'appliquent déjà sur le lot entier, hors
        # de toute boucle par séquence ; seule MLAttention y reste bouclée
        # (mla.py) — le gain est donc complet sur un modèle dense, partiel sur
        # un hybride MLA. Garde conservatrice : seulement si AUCUNE séquence
        # n'a de frontière instantanée à geler (`_frontiere_insta`), sinon la
        # photographie par séquence (longueur d'invite différente d'une
        # requête à l'autre) rendrait le lot incohérent — repli inchangé.
        budget = self._budget_jetons()
        a_prefiller = list(new)
        if budget:
            # Les inachevees d'un pas precedent reprennent AVANT les nouvelles :
            # sinon une arrivee continue les affamerait indefiniment.
            a_prefiller = [s for s in self.running
                           if not s.prefilled and not s.finished
                           and s not in new] + a_prefiller

        if new and not budget and os.environ.get("ACVRAM_PREFILL_BATCH") != "0" \
                and all(self._frontiere_insta(s) is None for s in new):
            t0 = time.perf_counter()
            batch = self._build_batch(new, prefill=True)
            logits = self.model(batch)
            if os.environ.get("ACVRAM_CHRONO_SYNC"):
                # Instrumentation temporaire (Laure, mandat Jerome) : sans
                # synchronisation, `self.model(batch)` est un lancement
                # asynchrone -- ce chrono mesurait potentiellement le
                # lancement, pas l'execution. Sous garde d'env, cf graphs.py.
                torch.cuda.synchronize()
            self.stats.prefill_seconds += time.perf_counter() - t0
            self.stats.prefill_tokens += sum(len(s.prompt_ids) - s.cached_len for s in new)
            for seq in new:
                seq.prefill_len = len(seq.prompt_ids)
            outputs += self._emit(logits, new)
            for seq in new:
                self._register_complete_blocks(seq)
        else:
            # On précalcule les séquences nouvellement admises une par une.
            # Mêler une longue invite à un lot de décodage bloquerait derrière
            # elle toutes les séquences en cours.
            for seq in a_prefiller:
                t0 = time.perf_counter()
                debut = seq.prefill_len
                # La frontiere ne se gele qu'au tout premier passage de la
                # sequence : une tranche suivante est deja au-dela.
                coupe = (self._frontiere_insta(seq)
                         if seq.prefill_len == seq.cached_len else None)
                if coupe is not None and self._eviter_coupe_image(seq, coupe) != coupe:
                    # La frontière d'instantané doit rester un multiple du pas :
                    # dans une image, on renonce à l'instantané pour cette invite.
                    coupe = None
                if coupe is not None:
                    # Première passe jusqu'à la frontière, instantané, puis le
                    # reste : le point de reprise est ainsi le même d'une requête à
                    # l'autre tant que l'invite partage ses premiers jetons.
                    self.model(self._build_batch([seq], prefill=True, limite=coupe))
                    self._photographier(seq, coupe)
                    seq.cached_len = coupe
                    seq.prefill_len = coupe
                fin = len(seq.prompt_ids)
                if budget:
                    fin = min(fin, seq.prefill_len + budget)
                    fin = self._eviter_coupe_image(seq, fin)
                batch = self._build_batch(
                    [seq], prefill=True,
                    limite=fin if fin < len(seq.prompt_ids) else None)
                logits = self.model(batch)
                if os.environ.get("ACVRAM_CHRONO_SYNC"):
                    torch.cuda.synchronize()
                self.stats.prefill_seconds += time.perf_counter() - t0
                # Compte depuis `debut`, releve AVANT la passe de frontiere :
                # l'ancienne formule partait de `cached_len` deja avance a la
                # coupe, et perdait donc la premiere moitie sur un hybride.
                self.stats.prefill_tokens += fin - debut
                seq.prefill_len = fin
                # Une tranche intermediaire ne produit pas de jeton : ses
                # logits ne sont pas ceux du dernier jeton de l'invite.
                if seq.prefilled:
                    outputs += self._emit(logits, [seq])
                self._register_complete_blocks(seq)

        decodable = self._decodables()
        if decodable:
            t0 = time.perf_counter()
            b_reel = len(decodable)
            if self.speculator is not None and self._garde_spec.eligible(b_reel):
                n0 = self.stats.decode_tokens
                outputs += self._speculative_decode(decodable)
                self._garde_spec.enregistrer(self.stats.decode_tokens - n0, b_reel)
            else:
                outputs += self._plain_decode(decodable)
            t1 = time.perf_counter()
            self.stats.decode_seconds += t1 - t0
            self.stats.spec_steps += 1
            for seq in decodable:
                if not seq.finished:
                    self._register_complete_blocks(seq)
            if os.environ.get("ACVRAM_TRACE_STEPS"):
                t2 = time.perf_counter()
                if (t2 - t0) * 1000 > 15:
                    print(f"[pas-lent] decode {(t1-t0)*1000:.1f} registre "
                          f"{(t2-t1)*1000:.1f} ms len={decodable[0].length}",
                          flush=True)

        self.stats.steps += 1
        self.stats.running = len(self.running)
        self.stats.waiting = len(self.waiting)
        self.stats.kv_blocks_free = self.allocator.num_free
        self._repin_pass()
        return outputs

    def _repin_pass(self) -> None:
        """REPIN à chaud, hors pas (bead anticitoyen-vram-pds, point 3).

        Appelé à la FIN de `step()` — jamais pendant un pas, jamais pendant
        une capture (même règle que `MoEBlock._compter_routage`) : la
        décision peut être lente (elle parcourt toutes les couches MoE), la
        capacité peut changer (`_pin`), et un graphe capturé ne doit jamais
        voir cette mutation se produire pendant qu'il rejoue.

        Cadence : `ACVRAM_REPIN` jetons décodés (défaut 64, `0` désactive).
        Ne fait rien tant qu'aucune couche n'a de `_pin` non vide (point (2)
        pas encore câblé) — voir `Engine.__init__`.
        """
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        try:
            n = int(os.environ.get("ACVRAM_REPIN", "64") or "64")
        except ValueError:
            n = 64
        from ..memory.repin import cadence_atteinte, choisir_echanges
        if not cadence_atteinte(self.stats.decode_tokens - self._dernier_repin, n):
            return
        self._dernier_repin = self.stats.decode_tokens
        if not self._pin:
            return
        from .model import MoEBlock
        etat = {}
        couches: dict = {}
        for m in self.model.modules():
            if not isinstance(m, MoEBlock):
                continue
            pin = self._pin.get(m.index_couche)
            if not pin or m._usage_routage is None:
                continue
            heat = {e: int(c) for e, c in
                    enumerate(m._usage_routage.detach().to("cpu").tolist())}
            etat[m.index_couche] = (pin, heat)
            couches[m.index_couche] = m
        for echange in choisir_echanges(etat, max_echanges=4):
            pin = self._pin[echange.couche]
            m = couches[echange.couche]
            reel = m._table_qw is not None
            if reel:
                self._repin_echanger(m, echange.sortant, echange.entrant)
            pin.discard(echange.sortant)
            pin.add(echange.entrant)
            suite = "" if reel else (" — PAS de table (NVFP4 seulement) : "
                                     "placement Python seul, aucun octet déplacé")
            print(f"[REPIN] couche {echange.couche} : "
                  f"expert {echange.sortant} sort, {echange.entrant} entre "
                  f"(gain {echange.gain}){suite}", flush=True)

    def _repin_echanger(self, m, sortant: int, entrant: int) -> None:
        _repin_echanger_reel(m, sortant, entrant)

    def _plain_decode(self, decodable: list[Sequence]) -> list[GenerationOutput]:
        if self.pipeline_actif and self.graphs is not None:
            return self._plain_decode_pipeline(decodable)
        return self._plain_decode_sync(decodable)

    def _plain_decode_sync(self, decodable: list[Sequence]) -> list[GenerationOutput]:
        trace = os.environ.get("ACVRAM_TRACE_STEPS")
        tg = time.perf_counter()
        epuisees = [self._finish_budget_epuise(seq) for seq in decodable
                   if not self._grow(seq)]
        decodable = [s for s in decodable if not s.finished]
        if not decodable:
            return epuisees
        t0 = time.perf_counter()
        batch = self._build_batch(decodable, prefill=False)
        t1 = time.perf_counter()
        logits = self.graphs.run(batch) if self.graphs is not None else None
        voie = "graphe"
        if logits is None:
            self._sonde_eager(batch)
            logits = self.model(batch)
            voie = "eager"
        self.model._mtp_hidden_n = len(decodable)     # un rejeu ne pose rien en Python
        t2 = time.perf_counter()
        self.stats.decode_tokens += len(decodable)
        outs = self._emit(logits, decodable)
        t3 = time.perf_counter()
        getattr(self, "temps_pas_total", None) is None and setattr(self, "temps_pas_total", [])
        getattr(self, "temps_pas_voie", None) is None and setattr(self, "temps_pas_voie", [])
        self.temps_pas_total.append((t3 - tg) * 1000)
        self.temps_pas_voie.append(voie)
        if trace and (t3 - tg) * 1000 > 12:
            print(f"[pas-lent] {voie} grow {(t0-tg)*1000:.1f} batch "
                  f"{(t1-t0)*1000:.1f} avant {(t2-t1)*1000:.1f} emit "
                  f"{(t3-t2)*1000:.1f} ms len={decodable[0].length}", flush=True)
        return epuisees + outs

    def _build_batch_device(self, seqs: list[Sequence],
                            tokens_dev: torch.Tensor) -> ForwardBatch:
        """Comme `_build_batch(seqs, prefill=False)`, mais le jeton à
        plonger vient d'un tenseur DEVICE (l'argmax du pas précédent, jamais
        rapatrié) plutôt que de `seq.output_ids[-1]` — c'est justement ce qui
        permet de lancer ce pas SANS attendre que le pas précédent soit
        rapatrié sur l'hôte (bead runner, 14/09, accord d'interface avec
        Laurine côté `graphs.py`).

        Appelée AVANT `_consommer` (recouvrement : c'est justement pour ça
        qu'on n'attend pas) — `output_ids` ne porte PAS encore ce jeton,
        contrairement à `_build_batch`. Convention DÉCALÉE d'un cran :
        `pos = seq.length` (pas `- 1`), et l'appelant doit réserver le bloc
        avec `_grow(seq, extra=1)`."""
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        block_tables: list[torch.Tensor] = []
        for seq in seqs:
            pos = seq.length
            positions.append(self._pos_decodage(seq, pos))
            slots.append(seq.blocks[pos // BLOCK_SIZE] * BLOCK_SIZE
                         + pos % BLOCK_SIZE)
            query_lens.append(1)
            seq_lens.append(pos + 1)
            block_tables.append(torch.tensor(seq.blocks, dtype=torch.long))
        return ForwardBatch(
            tokens=tokens_dev,
            positions=torch.tensor(positions, dtype=torch.long),
            seq_lens=seq_lens, query_lens=query_lens,
            block_tables=block_tables,
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=False,
            seq_ids=[s.id for s in seqs], gdn_store=self.gdn_states)

    def _pipeline_amorcer(self, decodable: list[Sequence]) -> list[GenerationOutput]:
        """Un pas NORMAL (synchrone, comme `_plain_decode_sync`), qui pose ou
        REPOSE l'état du pipeline plutôt que de le poursuivre en
        recouvrement — au tout premier pas, et chaque fois que la
        composition du lot change (une séquence finit, une autre est
        admise) : `_plain_decode_pipeline` retombe ici plutôt que de
        deviner. Un seul pas de recouvrement perdu à chaque recomposition,
        jamais de famine pour une arrivée."""
        for seq in decodable:
            if not self._grow(seq):
                self._finish_budget_epuise(seq)
        decodable = [s for s in decodable if not s.finished]
        if not decodable:
            return []
        batch = self._build_batch(decodable, prefill=False)
        ok = self.graphs.preparer(batch)
        self.stats.decode_tokens += len(decodable)
        if not ok:
            self._sonde_eager(batch)
            logits = self.model(batch)
            return self._emit(logits, decodable)
        logits = self.graphs.rejouer_suivant()
        tokens_dev, logprobs_dev = self._sample_only(logits, decodable)
        evenement = torch.cuda.Event()
        evenement.record()
        self._pipeline_pendiente = {
            "seqs": decodable, "tokens_dev": tokens_dev,
            "logprobs_dev": logprobs_dev, "event": evenement}
        return []

    def _pipeline_suite(self, roster: list[Sequence],
                        tokens_dev: torch.Tensor) -> list[GenerationOutput]:
        """Prépare et lance le pas SUIVANT en recouvrement — enfilée AVANT
        que `_consommer` n'ait rapatrié le pas courant (c'est le
        recouvrement : `_plain_decode_pipeline` appelle celle-ci D'ABORD).
        `roster` est le sous-ensemble encore actif tel que connu au pas
        PRÉCÉDENT (`output_ids` ne porte pas encore son jeton de ce pas-ci),
        `tokens_dev` ses jetons DEVICE alignés dans le même ordre. `_grow`
        avec `extra=1` : réserve le bloc du jeton pas encore ajouté à
        `output_ids` (cf. `_build_batch_device`)."""
        for seq in roster:
            if not self._grow(seq, extra=1):
                self._finish_budget_epuise(seq)
        vivants = [s for s in roster if not s.finished]
        if not vivants:
            return []
        if len(vivants) != len(roster):
            mask = torch.tensor([not s.finished for s in roster],
                                device=tokens_dev.device)
            tokens_dev = tokens_dev[mask]
        batch = self._build_batch_device(vivants, tokens_dev)
        ok = self.graphs.preparer(batch)
        self.stats.decode_tokens += len(vivants)
        if not ok:
            self._sonde_eager(batch)
            logits = self.model(batch)
            return self._emit(logits, vivants)
        logits = self.graphs.rejouer_suivant()
        tokens_dev2, logprobs_dev2 = self._sample_only(logits, vivants)
        evenement = torch.cuda.Event()
        evenement.record()
        self._pipeline_pendiente = {
            "seqs": vivants, "tokens_dev": tokens_dev2,
            "logprobs_dev": logprobs_dev2, "event": evenement}
        return []

    def _plain_decode_pipeline(self, decodable: list[Sequence]) -> list[GenerationOutput]:
        """Décodage à un pas de retard (`ACVRAM_PIPELINE=1`, bead runner
        14/09) : le pas n+1 est préparé et lancé AVANT de rapatrier les
        jetons du pas n — c'est le recouvrement lui-même, `.tolist()`
        continue d'attendre le GPU (Jérôme : « le gain vient du
        recouvrement, pas de la suppression de l'attente »).

        `decodable` est calculé par `step()` AVANT l'appel — donc avant que
        le rapatriement ci-dessous ait pu faire finir une séquence. Il ne
        sert qu'à détecter un changement de composition (une arrivée) ; la
        composition RÉELLE du pas en vol vient de `_pipeline_pendiente`."""
        pend = self._pipeline_pendiente
        if pend is None:
            return self._pipeline_amorcer(decodable)

        self._pipeline_pendiente = None
        roster_avant = [s for s in pend["seqs"] if not s.finished]
        # `.id`, pas `in`/`==` : `Sequence` est un dataclass à égalité par
        # champs (output_ids inclus) — comparer les objets eux-mêmes serait
        # à la fois faux (deux séquences ne sont jamais "égales" en ce sens)
        # et lent (compare tous les champs, listes comprises).
        ids_pend = {s.id for s in pend["seqs"]}
        nouveaux = [s for s in decodable if s.id not in ids_pend]

        if nouveaux or not roster_avant:
            # Recomposition du lot : pas de rejeu à enfiler par avance (sa
            # forme dépend de la nouvelle composition), donc rien à
            # recouvrir ici — on synchronise puis on retombe sur le pas
            # normal, comme documenté.
            pend["event"].synchronize()
            outputs = self._consommer(pend["tokens_dev"], pend["logprobs_dev"], pend["seqs"])
            outputs += self._pipeline_amorcer(roster_avant + nouveaux)
            return outputs

        # Lot stable : enfiler le rejeu n+1 D'ABORD (il ne lit que
        # `tokens_dev`, un tenseur DEVICE écrit par le rejeu n — l'ordre du
        # flux CUDA garantit la dépendance, aucun événement requis ici) puis
        # SEULEMENT ENSUITE synchroniser pour rapatrier les jetons du pas n
        # — sinon le rejeu n+1 ne part qu'après tout le travail hôte de
        # `_consommer`, et il n'y a plus rien à recouvrir (ce qui était le
        # bogue : le +0,6 % mesuré le 14/09 venait de cet ordre inversé).
        mask = [not s.finished for s in pend["seqs"]]
        tokens_dev = (pend["tokens_dev"] if all(mask) else
                     pend["tokens_dev"][torch.tensor(mask, device=pend["tokens_dev"].device)])
        outputs = self._pipeline_suite(roster_avant, tokens_dev)

        pend["event"].synchronize()
        outputs = self._consommer(pend["tokens_dev"], pend["logprobs_dev"], pend["seqs"]) + outputs
        return outputs

    def _speculative_decode(self, decodable: list[Sequence]
                            ) -> list[GenerationOutput]:
        """Proposer, vérifier en une passe avant, garder le préfixe accepté.

        Tout le lot est vérifié ensemble bien que les propositions diffèrent en
        longueur : le chemin d'attention gère déjà un bloc de requêtes par
        séquence avec son propre décalage absolu, la machinerie même dont le
        cache de préfixe avait besoin.
        """
        hyb = self.est_hybride
        if hyb:
            # hybrides : une séquence, sous graphe seulement (les états
            # récurrents reviennent en arrière par l'historique des tampons
            # fixes) ; forme fixe k+1 pour ne capturer qu'un graphe de plus
            if len(decodable) != 1 or self.graphs is None or not self.graphs.enabled:
                return self._plain_decode(decodable)
            self.graphs.max_ql = self.spec_k + 1
        proposals: dict[int, Proposal] = {}
        for seq in decodable:
            budget = max(0, seq.params.max_tokens - len(seq.output_ids) - 1)
            k = min(self.spec_k, budget)
            prop = self.speculator.propose(seq, k) if k > 0 else Proposal([])
            if hyb:
                if not len(prop) or k < self.spec_k:
                    return self._plain_decode(decodable)
                if len(prop) < k:
                    prop = Proposal(list(prop.tokens) + [prop.tokens[-1]] * (k - len(prop)))
            # Une proposition qui dépasserait la limite de contexte est rognée
            # plutôt qu'abandonnée : une spéculation plus courte paie encore.
            room = self.max_model_len - seq.length - 1
            if len(prop) > room:
                prop = Proposal(prop.tokens[:max(0, room)],
                                None if prop.probs is None
                                else prop.probs[:max(0, room)])
            proposals[seq.id] = prop
            if not self._grow(seq, extra=len(prop)):
                self._finish_budget_epuise(seq)

        decodable = [s for s in decodable if not s.finished]
        if not decodable:
            return []
        # Proposeur en veille (rendement trop faible) : le pas spéculatif à
        # largeur 1 ne ferait qu'ajouter du travail Python.
        if all(not len(proposals[s.id]) for s in decodable):
            return self._plain_decode(decodable)

        batch = self._build_spec_batch(decodable, proposals)
        flat = self.graphs.run(batch) if self.graphs is not None else None
        if flat is None:
            if hyb:
                return self._plain_decode(decodable)
            flat = self.model(batch, logits_positions=batch.all_token_indices())
        self.model._mtp_hidden_n = int(flat.shape[0])  # un rejeu ne pose rien en Python

        outputs: list[GenerationOutput] = []
        cursor = 0
        for seq in decodable:
            prop = proposals[seq.id]
            width = len(prop) + 1
            rows = flat[cursor:cursor + width]
            cursor += width
            tokens, n_acc = verify_proposal(rows, prop, seq.params)
            if hyb and n_acc + 1 < width:
                self.graphs.rollback_hybrid(n_acc + 1)
            self.stats.proposed_tokens += len(prop)
            self.stats.accepted_tokens += n_acc
            self.stats.decode_tokens += len(tokens)
            seq.n_proposed += len(prop)
            seq.n_accepted += n_acc
            self.speculator.commit(seq, tokens)
            outputs.append(self._append(seq, tokens))
        return outputs

    def _build_spec_batch(self, seqs: list[Sequence],
                          proposals: dict[int, Proposal]) -> ForwardBatch:
        """Le lot de vérification : le dernier vrai jeton, puis les propositions.

        Présenter le dernier jeton produit à côté des propositions ne coûte
        rien de plus : ses clés et valeurs n'ont jamais été écrites, puisqu'un
        jeton n'entre dans le cache qu'au moment où on le présente. Les K+1
        positions sont donc exactement les K+1 prédictions nécessaires.
        """
        tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        block_tables: list[torch.Tensor] = []

        for seq in seqs:
            prop = proposals[seq.id]
            start = seq.length - 1
            block = [seq.output_ids[-1] if seq.output_ids else seq.prompt_ids[-1]]
            block += list(prop.tokens)
            for j, tok in enumerate(block):
                pos = start + j
                tokens.append(tok)
                positions.append(self._pos_decodage(seq, pos))
                slots.append(seq.blocks[pos // BLOCK_SIZE] * BLOCK_SIZE
                             + pos % BLOCK_SIZE)
            query_lens.append(len(block))
            seq_lens.append(start + len(block))
            block_tables.append(torch.tensor(seq.blocks, dtype=torch.long))

        return ForwardBatch(
            tokens=torch.tensor(tokens, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
            seq_lens=seq_lens, query_lens=query_lens,
            block_tables=block_tables,
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=False,
            seq_ids=[s.id for s in seqs], gdn_store=self.gdn_states)

    def _append(self, seq: Sequence, tokens: list[int]) -> GenerationOutput:
        """Ajoute plusieurs jetons acceptés, en s'arrêtant au premier qui termine."""
        reason = ""
        kept: list[int] = []
        for tok in tokens:
            seq.output_ids.append(int(tok))
            kept.append(int(tok))
            if not seq.first_token_at:
                seq.first_token_at = time.time()
            if (tok in self._eos and not seq.params.ignore_eos) or tok in seq.params.stop_token_ids:
                reason = "stop"
                break
            if len(seq.output_ids) >= seq.params.max_tokens:
                reason = "length"
                break
            if seq.length >= self.max_model_len:
                reason = "length"
                break

        text = self._decode_delta(seq, len(kept)) if self.tokenizer else ""
        if not reason and seq.params.stop and text:
            text, coupe = _trim_at_stop(self._decode_all(seq), text,
                                        seq.params.stop)
            if coupe:
                reason = "stop"
        if reason:
            self._finish(seq, reason)
        return GenerationOutput(
            sequence_id=seq.id, request_id=seq.request_id, token_ids=kept,
            text_delta=text, finished=bool(reason), finish_reason=reason,
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=len(seq.output_ids))

    def _sonde_eager(self, batch: ForwardBatch) -> None:
        """ACVRAM_TRACE_ENTREES=1 : le pendant eager de graphs._sonde_entrees —
        jetons, positions, seq_lens et le plongement que le modèle va calculer."""
        if not os.environ.get("ACVRAM_TRACE_ENTREES"):
            return
        torch.cuda.synchronize()
        n = min(4, batch.batch_size)
        m = self.model
        idx = batch.tokens[:n]
        if torch.is_tensor(idx):
            idx = idx.to(m.embed_tokens.device)
        else:
            idx = torch.tensor(list(idx), device=m.embed_tokens.device)
        x = torch.nn.functional.embedding(idx, m.embed_tokens).to(m.dtype)
        if m.spec.embedding_multiplier != 1.0:
            x = x * m.spec.embedding_multiplier
        print(f"[ENTREES-EAGER] jetons={idx.tolist()} positions={batch.positions[:n].tolist()} "
              f"seq_lens={list(batch.seq_lens[:n])} | x[:4]={[[round(v, 5) for v in r] for r in x[:, :4].float().tolist()]}",
              flush=True)

    def _sample_only(self, logits: torch.Tensor,
                     seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """La moitié de `_emit` qui reste SUR DEVICE — aucun `.tolist()`/
        `.item()`. Partagée par le pas normal (`_emit` l'appelle puis lit
        tout de suite) et le pas recouvert (bead runner, 14/09), qui différe
        la lecture d'un pas pour la faire pendant que le rejeu suivant tourne
        déjà, au lieu de l'ajouter en série après chaque rejeu."""
        params = [s.params for s in seqs]
        # `all_ids` CONCATENE prompt et sortie : une liste neuve de la taille
        # du contexte entier, a chaque pas et pour chaque sequence. En
        # decodage glouton sans penalites — le cas courant — `sample` rend son
        # argmax AVANT de la lire, et ce travail proportionnel au contexte
        # etait entierement perdu. La condition vit dans `sampler` pour que
        # celle qui construit et celle qui lit ne puissent pas diverger.
        history = ([s.all_ids for s in seqs] if besoin_historique(params)
                   else [() for _ in seqs])
        return sample(logits, params, history)

    def _consommer(self, tokens: torch.Tensor, logprobs: torch.Tensor,
                   seqs: list[Sequence]) -> list[GenerationOutput]:
        """Le corps de `_emit` après l'échantillonnage — LE seul `.tolist()`
        du pas, qu'il soit immédiat (`_emit`) ou différé d'un pas (pipeline).

        `seq.finished` déjà vrai (chemin pipeline : la séquence s'est arrêtée
        — longueur — APRÈS que ce rejeu a été lancé, avant qu'il soit
        rapatrié) : son jeton est un jeton FANTÔME, ignoré — même principe
        que le masquage des créneaux fantômes du remplissage godet
        (`MoEBlock.forward`, bead pds), au niveau du planificateur cette
        fois plutôt que du routage MoE."""
        out = []
        for seq, tok, lp in zip(seqs, tokens.tolist(), logprobs.tolist()):
            if seq.finished:
                continue
            seq.output_ids.append(int(tok))
            seq.cumulative_logprob += float(lp)
            if not seq.first_token_at:
                seq.first_token_at = time.time()

            reason = ""
            if (tok in self._eos and not seq.params.ignore_eos) or tok in seq.params.stop_token_ids:
                reason = "stop"
            elif len(seq.output_ids) >= seq.params.max_tokens:
                reason = "length"
            elif seq.length >= self.max_model_len:
                reason = "length"

            text = ""
            if self.tokenizer is not None:
                text = self._decode_delta(seq)
            if not reason and seq.params.stop and text:
                text, coupe = _trim_at_stop(self._decode_all(seq), text,
                                            seq.params.stop)
                if coupe:
                    reason = "stop"

            if reason:
                self._finish(seq, reason)
            out.append(GenerationOutput(
                sequence_id=seq.id, request_id=seq.request_id,
                token_ids=[int(tok)], text_delta=text,
                finished=bool(reason), finish_reason=reason,
                prompt_tokens=len(seq.prompt_ids),
                completion_tokens=len(seq.output_ids)))
        return out

    def _emit(self, logits: torch.Tensor,
              seqs: list[Sequence]) -> list[GenerationOutput]:
        tokens, logprobs = self._sample_only(logits, seqs)
        return self._consommer(tokens, logprobs, seqs)

    def _decode_delta(self, seq: Sequence, n_new: int = 1) -> str:
        """Décode au fil de l'eau, en respectant les séquences UTF-8 multi-jetons.

        Décoder le seul dernier jeton couperait les caractères multi-octets et
        émettrait des caractères de remplacement en plein mot : on décode donc
        une petite fenêtre et on ne rend que ce qui est nouveau.
        """
        tok = self.tokenizer
        span = max(8, n_new + 4)
        window = seq.output_ids[-span:]
        prev = seq.output_ids[-span:-n_new] if n_new else seq.output_ids[-span:]
        self._refuser_si_attention_amputee()
        try:
            full = tok.decode(window)
            head = tok.decode(prev) if prev else ""
        except Exception:                            # noqa: BLE001
            return ""
        return full[len(head):] if full.startswith(head) else full

    @staticmethod
    def _refuser_si_attention_amputee() -> None:
        """Les bras B et C d'ACVRAM_PA_ARM amputent l'attention paginee : la
        sortie du modele est fausse. Elle ne doit pas pouvoir etre lue, sinon
        elle deviendra le prochain chiffre plausible qui se transporte."""
        bras = os.environ.get("ACVRAM_PA_ARM", "A")
        if bras and bras[0] != "A":
            raise RuntimeError(
                f"ACVRAM_PA_ARM={bras} : l'attention paginee est amputee, la "
                "sortie est FAUSSE et le decodage du texte est refuse. Ce bras "
                "ne sert qu'a chronometrer le pas."
            )

    def _decode_all(self, seq: Sequence) -> str:
        self._refuser_si_attention_amputee()
        try:
            return self.tokenizer.decode(seq.output_ids)
        except Exception:                            # noqa: BLE001
            return ""

    # -- convenience -----------------------------------------------------
    def generate(self, prompt_ids: list[int], params: SamplingParams,
                 images: Any = None) -> Iterator[GenerationOutput]:
        """Générateur bloquant pour une requête unique. Utilisé par le CLI et les tests."""
        seq = self.add_request(prompt_ids, params, images=images)
        while not seq.finished:
            for out in self.step():
                if out.sequence_id == seq.id:
                    yield out
            if not self.running and not self.waiting:
                break

    def sequence_de_chauffe(self, L: int) -> list[int]:
        """Séquence pseudo-aléatoire sur le vocabulaire, graine fixe 0, longueur ``L`` (Sage § 2 (a)) : une séquence
        homogène ``[1]×N`` route tous les jetons vers les mêmes experts et sous-estime la crête d'un vrai prefill
        (GLM k48 : chauffe TENUE 32768/32768 puis 500 CUDA OOM réel à ctx − 64, 318/298 Mio)."""
        g = torch.Generator().manual_seed(0)
        vocab = int(getattr(self.spec, "vocab_size", 0) or 0) or 2
        return torch.randint(0, vocab, (L,), generator=g).tolist()

    def _avant_essai_de_chauffe(self) -> None:
        """Entre deux pas de la dichotomie, les segments réservés par le pas précédent (plus grand, ou tenu sans
        réserve) restent dans l allocateur CUDA : le pas suivant lit moins de libre et la recherche descend trop
        (Coder i8c : 15 360 hier avec [1]×N, 4 096 ce matin — Maîtresse 21/09 (3)). Synchroniser, vider."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _essai_de_chauffe(self, L: int, max_tokens: int = 1) -> bool:
        """Une passe de chauffe de ``L`` jetons dans le régime courant : tenue ssi pas d OOM ET ≥ max(5 %, 64 Mio)
        libres après elle. ``max_tokens=2`` force un pas de décodage (les graphes, s ils sont actifs, rejouent)."""
        self._avant_essai_de_chauffe()                                   # (3) chaque pas part d un allocateur vide
        try:
            for _ in self.generate(self.sequence_de_chauffe(L - 2), SamplingParams(max_tokens=max_tokens, temperature=0.0)):
                pass
        except Exception as exc:                                         # noqa: BLE001
            oom = isinstance(exc, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(exc).lower() \
                or "blocs KV" in str(exc)
            if not oom:
                raise
            self._apres_oom_de_chauffe()
            return False
        libre, total = self._libre_apres_chauffe()
        seuil = max(total * 5 // 100, 64 << 20)
        self._oublier_la_chauffe()
        if libre < seuil:                                                # (b) : tenu sans réserve = non tenu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        self.reserve_chauffe = (libre, seuil)
        return True

    def _libre_apres_chauffe(self) -> tuple[int, int]:
        """(libre, total) octets du pilote après la passe, avant tout `empty_cache` : le réservé du prefill y est
        encore compté. Hors carte : (total, total) — la réserve ne se juge que sur carte."""
        dev = self.model.embed_tokens.device
        if torch.cuda.is_available() and dev.type == "cuda":
            torch.cuda.synchronize(dev)                                  # la crête est passée avant la lecture
            libre, total = torch.cuda.mem_get_info(dev)
            return int(libre), int(total)
        return (1 << 40, 1 << 40)

    def _experts_sollicites(self, seq: list[int], params) -> int:
        """Prefill de ``seq`` en comptant les experts DISTINCTS touchés par couche MoE (somme sur les couches) :
        c'est la grandeur qui sépare ``[1]×N`` (top_k experts par couche) d'une séquence réelle (jusqu'à tous) —
        chaque expert sollicité a ses poids mis en jeu (Marlin nvfp4 : atelier par expert actif). 0 = modèle dense."""
        from .model import MoEBlock
        blocs = [m for m in self.model.modules() if isinstance(m, MoEBlock)]
        vus: dict[int, set] = {}
        originaux = []
        for i, m in enumerate(blocs):
            orig = m._route

            def route(x, _o=orig, _i=i):
                topw, topi = _o(x)
                vus.setdefault(_i, set()).update(int(e) for e in torch.unique(topi.detach().cpu()).tolist())
                return topw, topi
            originaux.append((m, orig)); m._route = route
        try:
            for _ in self.generate(seq, params):
                pass
        finally:
            for m, orig in originaux:
                m._route = orig
        return sum(len(v) for v in vus.values())

    def chauffer_contexte(self, pas: int = 1024, strict: bool = False) -> Optional[int]:
        """Prouve ``max_model_len`` AU CHARGEMENT (Sage, sage-3b-lanceur-contexte-20-09 (ii), resserré par
        sage-s2-k48-feu-vert-21-09 § 2) : une séquence PSEUDO-ALÉATOIRE (graine 0, ``sequence_de_chauffe``) de
        ``max_model_len − 2`` jetons passe le prefill dans le RÉGIME SERVI, puis ses blocs sont rendus et le cache
        de préfixe ne la garde pas. Tenu(L) = la passe ne lève pas d'OOM ET laisse ≥ max(5 %, 64 Mio) du pilote
        libres après elle (réserve pour la crête d'une vraie requête). ``ctx_tenu`` = plus grand multiple de
        ``pas`` tenu (dichotomie depuis ``max_model_len``). Non tenu → CLAMP : ``max_model_len`` prend
        ``ctx_tenu``, la ligne dit ``ctx_tenu=N(demandé M)``, une invite au-delà reçoit un 400 nommé ; refus
        ``ContexteNonTenu`` seulement si ``ctx_tenu < CTX_TENU_MIN`` (4096) ou ``strict`` (--ctx-strict).
        Opt-out nommé ``ACVRAM_CHAUFFE_CTX=0`` → ``ctx_tenu=non-verifie`` ; jamais sous ``ACVRAM_TYPE=service``."""
        n = int(self.max_model_len)
        if os.environ.get("ACVRAM_CHAUFFE_CTX") == "0":
            if os.environ.get("ACVRAM_TYPE") == "service":
                print("[acvram] ACVRAM_CHAUFFE_CTX=0 ignoré : un service prouve son contexte", flush=True)
            else:
                self.ctx_tenu = None
                return None
        self.reserve_chauffe: Optional[tuple[int, int]] = None           # (libre, seuil) de la dernière passe tenue
        # Ordre imposé (Maîtresse 21/09, GLM k48 : OOM à la CAPTURE, avant la chauffe — la capture alloue au ctx
        # demandé) : plan → clamp → capture → chauffe. Les graphes sont masqués pendant les essais : aucune
        # capture ne peut se faire à un contexte que la chauffe n a pas encore tenu ; la taille de capture prend
        # le contexte tenu (voir la fin).
        graphes, self.graphs = self.graphs, None

        essai = self._essai_de_chauffe

        t0 = time.time()
        tenu: Optional[int]
        if essai(n):
            tenu = n
        else:
            bas, haut = 0, n                                              # bas tenu (0 : rien), haut non tenu
            for _ in range(8):
                milieu = max(pas, ((bas + haut) // 2) // pas * pas)
                if milieu <= bas or milieu >= haut:
                    break
                if essai(milieu):
                    bas = milieu
                else:
                    haut = milieu
            tenu = bas
        self._oublier_la_chauffe()
        self.graphs = graphes
        self.ctx_demande = n
        self.ctx_tenu = tenu
        r = self.reserve_chauffe
        print(f"[acvram] chauffe du contexte : {tenu}/{n} jetons tenus en {time.time() - t0:.1f} s"
              + (f", {r[0] >> 20} Mio libres après la passe (réserve ≥ {r[1] >> 20})" if r else ""), flush=True)
        if tenu < n:
            if strict or tenu < CTX_TENU_MIN:
                raise ContexteNonTenu(n, tenu)
            self.max_model_len = tenu                                    # (c) clamp : 400 nommé au-delà, pas un 500
            if self.graphs is not None:
                self.graphs.max_model_len = tenu                         # la capture se dimensionne au tenu
            print(f"[acvram] contexte clampé à {tenu} (demandé {n}) : une invite au-delà reçoit un 400 nommé",
                  flush=True)
        return tenu

    def _recapturer(self, warm_max_len: int) -> int:
        """Graphes neufs au ``max_model_len`` courant (les captures précédentes sont rendues), puis capture d avance."""
        from .graphs import GraphRunner
        self.graphs = None
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        gr = GraphRunner(self.model, self.max_model_len, max_batch_size=self.max_batch_size)
        self.graphs = gr if gr.enabled else None
        return self.warm_graphs(warm_max_len) if self.graphs is not None else 0

    def demarrer_service(self, strict: bool = False, warm_max_len: int = 2048,
                         pas: int = 1024, pas_confirmation: int = 1024) -> tuple[Optional[int], int]:
        """La séquence de chargement d un service, dans l ordre qui ne peut pas mentir (Maîtresse 21/09) :
        plan → clamp → dichotomie SANS graphes → capture au tenu → passe de CONFIRMATION au tenu, graphes actifs
        (la requête réelle tourne avec leurs réserves : c est le contrôle qui peut rendre faux). Confirmation
        échouée → tenu − ``pas_confirmation``, recapture, deux fois au plus, puis refus nommé. Rend (ctx_tenu,
        captures). Test cassant : faux graphes qui coûtent 2 pas ⇒ tenu final = tenu − 2·pas, chargement réussi."""
        tenu = self.chauffer_contexte(pas=pas, strict=strict)
        captures = self.warm_graphs(warm_max_len) if self.graphs is not None else 0
        if tenu is None or self.graphs is None:
            return tenu, captures
        for baisse in range(3):
            if self._essai_de_chauffe(int(self.max_model_len), max_tokens=2):
                if baisse:
                    print(f"[acvram] confirmation avec graphes : {self.ctx_tenu} tenus après {baisse} baisse(s)", flush=True)
                self._oublier_la_chauffe()
                return self.ctx_tenu, captures
            nouveau = int(self.max_model_len) - pas_confirmation
            if baisse == 2 or strict or nouveau < CTX_TENU_MIN:
                raise ContexteNonTenu(self.ctx_demande, nouveau if baisse < 2 else int(self.max_model_len))
            print(f"[acvram] confirmation avec graphes échouée à {self.max_model_len} : contexte {nouveau}, recapture",
                  flush=True)
            self.max_model_len = self.ctx_tenu = nouveau
            captures = self._recapturer(warm_max_len)
        return self.ctx_tenu, captures                                   # jamais atteint

    def _apres_oom_de_chauffe(self) -> None:
        """Après un OOM de chauffe : séquences en cours abandonnées, blocs rendus, allocateur neuf, cache CUDA vidé."""
        for seq in list(self.running) + list(self.waiting):
            try:
                self._finish(seq, "chauffe")
            except Exception:                                            # noqa: BLE001
                pass
        self.running.clear(); self.waiting.clear()
        self._oublier_la_chauffe()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _oublier_la_chauffe(self) -> None:
        """Le cache de préfixe ne garde pas la séquence de chauffe (des « 1 » : un vrai préfixe de 1 y trouverait un
        KV) : l'allocateur est recréé à l'identique (mêmes blocs, même mode)."""
        self.allocator = BlockAllocator(self.allocator.num_blocks, self.allocator.enable_prefix_cache)

    def warm_graphs(self, max_len: int = 2048) -> int:
        """Capture d'avance les graphes de décodage des godets jusqu'à
        ``max_len`` jetons : une capture coûte 40 à 130 ms, mieux vaut la
        payer au démarrage qu'au milieu d'une réponse. Rend le nombre de
        captures faites."""
        if self.graphs is None:
            return 0
        avant = self.graphs.captures
        L = 128
        while L <= min(max_len, self.max_model_len - 4):
            # longueur choisie pour que prefill + 2 jetons restent dans le
            # godet de L/16 blocs (puissance de deux)
            for _ in self.generate([1] * (L - 2), SamplingParams(max_tokens=2,
                                                                 temperature=0.0)):
                pass
            if (self.est_hybride and self.speculator is not None
                    and os.environ.get("ACVRAM_WARM_SPEC", "1") != "0"):
                # motif répété : le proposeur n-gramme spécule dès le
                # premier pas, d'où la capture du graphe de forme k+1
                ids = ([5, 6, 7, 8] * (L // 4))[:L - 2]
                for _ in self.generate(ids, SamplingParams(max_tokens=self.spec_k + 2,
                                                           temperature=0.0)):
                    pass
            L *= 2
        return self.graphs.captures - avant

    @property
    def idle(self) -> bool:
        return not self.running and not self.waiting
