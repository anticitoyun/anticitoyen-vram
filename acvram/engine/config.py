"""Description d'un modèle, dérivée d'un ``config.json`` Hugging Face.

Uniquement ce dont le planificateur et l'exécution ont besoin : les formes, le
nombre de paramètres par groupe de couches, et le fait qu'une couche soit ou
non à mélange d'experts creux — ce qui change d'un ordre de grandeur le coût de
la placer en mémoire vive.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

__all__ = ["ModelSpec", "LayerSpec", "load_model_spec"]


@dataclass
class LayerSpec:
    """Un bloc de transformeur, découpé selon les tenseurs que déplace le planificateur."""

    index: int
    attn_params: int
    mlp_params: int
    norm_params: int
    is_moe: bool = False
    n_experts: int = 0
    n_experts_active: int = 0
    shared_expert_params: int = 0

    @property
    def total_params(self) -> int:
        return self.attn_params + self.mlp_params + self.norm_params

    @property
    def active_params(self) -> int:
        """Paramètres réellement lus pour un seul jeton.

        Pour un bloc dense, c'est la totalité. Pour un bloc à mélange d'experts,
        seuls le routeur, l'expert partagé et ``n_experts_active`` experts sont
        touchés — et c'est pourquoi un MoE de 235 milliards de paramètres se
        transfère depuis la mémoire vive à une vitesse exploitable, là où un
        modèle dense de 70 milliards ne le peut pas.
        """
        if not self.is_moe or self.n_experts == 0:
            return self.total_params
        per_expert = (self.mlp_params - self.shared_expert_params) / self.n_experts
        return int(self.attn_params + self.norm_params + self.shared_expert_params
                   + per_expert * self.n_experts_active)

    @property
    def activation_ratio(self) -> float:
        return self.active_params / max(1, self.total_params)



# Types de couche qui allouent un cache KV, et ceux qui portent un état
# récurrent. Les deux listes vivent ici, à côté du budget qui les utilise :
# une règle écrite deux fois finit par diverger, comme `stack_int8_linears` et
# `stack_nvfp4_linears` ont divergé sur le biais jusqu'au 9/09/2026.
_TYPES_AVEC_KV = frozenset({
    "full_attention", "sliding_attention", "attention", "parallel"})
_TYPES_RECURRENTS = frozenset({"linear_attention", "mamba", "conv"})
# Ni cache KV ni état récurrent : ce sont des types de MLP, pas d'attention.
_TYPES_SANS_ETAT = frozenset({"moe", "mlp"})


@dataclass
class ModelSpec:
    name: str
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    head_dim: int = 0
    # MoE
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    first_k_dense_replace: int = 0
    # PROVENANCE (11/09) : None = la source ne declare pas le champ.
    # torch_dtype n est CONSOMME nulle part au runtime (le dtype du KV
    # cache est calcule de son propre format), donc None y est sans
    # risque ; il n a d effet que sur ce que le manifeste enregistre.
    torch_dtype: Optional[str] = None
    # Jetons d'arret. Sans eux le moteur ne s'arrete jamais de lui-meme et
    # rend toujours max_tokens jetons, en repartant en roue libre apres la
    # reponse.
    eos_token_id: list[int] = field(default_factory=list)
    bos_token_id: Optional[int] = None
    # Hybrides à récurrence linéaire (qwen3-next) : type de chaque couche et
    # géométrie de la partie Gated DeltaNet. Vide = transformeur pur.
    layer_types: list[str] = field(default_factory=list)
    linear_num_value_heads: int = 0
    linear_num_key_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4
    rotary_dim: Optional[int] = None      # RoPE partiel (None = tête entière)
    attn_output_gate: bool = False
    # gemma4 : couches locales (fenêtre) / globales (têtes plus larges, RoPE
    # proportionnel), v normalisé, k = v global, softcap final
    sliding_window: int = 0
    global_head_dim: int = 0
    num_global_key_value_heads: int = 0
    rope_theta_swa: float = 0.0
    partial_rotary_factor_full: float = 1.0
    final_logit_softcapping: float = 0.0
    # None = source muette. Le repli "silu" vit au point d usage
    # (propriete mlp_activation ci-dessous), pas ici : ecrire "silu"
    # par defaut le rendrait indiscernable d un silu mesure.
    hidden_activation: Optional[str] = None
    attention_k_eq_v: bool = False
    # nemotron_h : Mamba2 (SSD) + attention sans RoPE + MLP ReLU²
    mamba_num_heads: int = 0
    mamba_head_dim: int = 0
    mamba_n_groups: int = 1
    mamba_state_size: int = 128
    mamba_conv_kernel: int = 4
    attention_rope: bool = True
    # muse_glimmer : epsilon des normes post (1e-8)
    post_norm_eps: float = 0.0
    # starcoder2 : normes LayerNorm (biais), MLP non gaté GELU avec biais
    norm_type: str = "rms_norm"
    mlp_gated: bool = True
    # lfm2 : longueur du noyau de la conv courte
    conv_L_cache: int = 3
    # granite : multiplicateurs scalaires (attention, plongement, résidu, logits)
    attention_multiplier: Optional[float] = None
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    # kimi-linear : KDA + MLA + routeur DeepSeek
    model_type: str = ""
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    mla_rope: bool = False                # RoPE sur la partie pe (DeepSeek), pas Kimi
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    router_scoring: str = "softmax"
    routed_scaling_factor: float = 1.0
    # Qwen3-VL : fusion spatiale de la tour de vision (vision_config.spatial_merge_size),
    # le pas des grilles (t, h//merge, w//merge) des positions M-RoPE ; 0 = sans tour.
    spatial_merge_size: int = 0
    layers: list[LayerSpec] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.head_dim:
            self.head_dim = self.hidden_size // max(1, self.num_attention_heads)
        if not self.layers:
            self.layers = [self._build_layer(i) for i in range(self.num_layers)]

    def _build_layer(self, i: int) -> LayerSpec:
        h, hd = self.hidden_size, self.head_dim
        q = h * self.num_attention_heads * hd
        kv = 2 * h * self.num_key_value_heads * hd
        o = self.num_attention_heads * hd * h
        attn = q + kv + o
        norm = 2 * h

        is_moe = self.num_experts > 0 and i >= self.first_k_dense_replace
        if is_moe:
            inter = self.moe_intermediate_size or self.intermediate_size
            per_expert = 3 * h * inter
            router = h * self.num_experts
            shared = 3 * h * self.shared_expert_intermediate_size \
                if self.shared_expert_intermediate_size else 0
            mlp = per_expert * self.num_experts + router + shared
            return LayerSpec(i, attn, mlp, norm, True, self.num_experts,
                             self.num_experts_per_tok, shared + router)
        mlp = 3 * h * self.intermediate_size
        return LayerSpec(i, attn, mlp, norm)

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_params(self) -> int:
        return 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size

    def tete_liee_bytes(self, group_size: int = 128) -> int:
        """Octets de la copie quantifiée d'une tête LIÉE, que rien ne comptait.

        Quand ``tie_word_embeddings`` est vrai, il n'existe aucun tenseur
        `lm_head` : `lm_head_params` vaut zéro et `_octets_reels` ne trouve
        rien à compter. Mais `_tete_liee` (`loader.py`) fabrique au chargement
        une copie INT8 de la table d'embedding — la projection lit une matrice
        entière à chaque jeton, et la lire en 16 bits coûterait le double —
        et cette copie **s'ajoute** à la table sans la remplacer, le gather
        d'entrée ayant toujours besoin des 16 bits.

        Le plan sous-estimait donc les poids d'autant : 0,369 Gio sur un
        Qwen3-4B, mesuré le 9/09/2026. Et la conséquence dépassait le
        comptage — `_borner_kv_par_la_vram` calcule ``libre − poids − marge``
        avec ce même total : sous-estimer les poids lui faisait autoriser un
        budget KV trop grand, donc **manger la marge qu'il est chargé de
        protéger**, celle-là même qui doit garder la place d'une capture de
        graphes CUDA.

        Le format retenu est l'INT8 par défaut de `_TETE_LIEE`. Un réglage
        contraire — `bf16`, ou le repli automatique quand la carte est trop
        pleine — rend ce compte majorant, ce qui est le bon sens de l'erreur
        pour une provision.
        """
        if not self.tie_word_embeddings:
            return 0
        n = self.vocab_size * self.hidden_size
        g = max(1, group_size)
        echelles = self.vocab_size * (self.hidden_size // g) * 2      # fp16
        zeros = self.vocab_size * (self.hidden_size // g + 1) // 2    # uint4 packés
        return n + echelles + zeros

    @property
    def total_params(self) -> int:
        return (self.embed_params + self.lm_head_params + self.hidden_size
                + sum(l.total_params for l in self.layers))

    @property
    def active_params(self) -> int:
        return (self.embed_params + self.lm_head_params
                + sum(l.active_params for l in self.layers))

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def couches_avec_kv(self) -> int:
        """Couches qui allouent un cache KV — pas toutes, sur un hybride.

        Une couche à attention linéaire ou à SSM porte un état de taille FIXE,
        pas un cache qui croît avec le contexte : `loader.py` ne lui alloue
        aucun `PagedKVCache`. La compter dans le budget KV le surestime d'un
        facteur 4 à 14 sur les 53 hybrides du parc.

        Énumération POSITIVE et non soustraction, parce que deux pièges
        guettent : `sliding_attention` a bien un cache, et `moe`/`mlp` sont des
        types de couche sans attention du tout — compter « tout sauf les
        linéaires » donnait 29 couches à cache sur Nemotron là où il y en a 6.
        Vérifié sur le parc : 65 modèles d'accord, aucun en désaccord.

        Sans `layer_types`, le repli sur `num_layers` n'est pas seulement
        prudent, il est EXACT : les 49 manifestes qui n'en portent pas sont
        tous purement quadratiques.
        """
        if not self.layer_types:
            return self.num_layers
        return sum(1 for t in self.layer_types if t in _TYPES_AVEC_KV)

    @property
    def est_mla(self) -> bool:
        """Attention à latent compressé (DeepSeek, GLM-4.x, Kimi).

        Le critère est la compression elle-même, pas le `model_type` : une
        architecture nouvelle qui compresse son KV se reconnaîtra sans qu'on
        ait à l'ajouter à une liste.
        """
        return self.kv_lora_rank > 0

    def couche_a_kv(self, index: int) -> bool:
        """La couche ``index`` alloue-t-elle un cache PAGINÉ ?

        Distinct de `couches_avec_kv`, et la nuance décide : une couche MLA
        garde bien un cache par jeton, mais un latent contigu que
        `loader.py` n'enregistre jamais dans `a_allouer` — sa branche fait
        `continue` avant. Elle STOCKE sans PAGINER. Confondre les deux, c'est
        soit diviser les blocs par des couches qui n'en prennent aucun, soit
        ne rien budgéter pour ce qu'elles gardent vraiment.
        """
        if self.est_mla:
            return False
        if not self.layer_types or index >= len(self.layer_types):
            return True
        return self.layer_types[index] in _TYPES_AVEC_KV

    @property
    def couches_recurrentes(self) -> int:
        """Couches qui portent un état récurrent — linéaire, SSM ou convolution."""
        if not self.layer_types:
            return 0
        return sum(1 for t in self.layer_types if t in _TYPES_RECURRENTS)

    @property
    def types_de_couche_inconnus(self) -> list[str]:
        """Types présents que ni le budget KV ni la provision ne savent traiter.

        Un type inconnu ne doit pas provisionner zéro EN SILENCE : c'est
        exactement ainsi que l'état des couches `mamba` et `conv` est resté
        hors budget, et le silence a valu un commit à reprendre. Une
        architecture neuve doit se signaler d'elle-même, pas attendre qu'un
        déficit de VRAM la révèle.
        """
        connus = _TYPES_AVEC_KV | _TYPES_RECURRENTS | _TYPES_SANS_ETAT
        return sorted({t for t in (self.layer_types or []) if t not in connus})

    def kv_bytes_per_token(self, kv_bits: int = 8) -> int:
        """Octets de cache pour un jeton, sur les couches QUI EN GARDENT UN.

        L'attention à requêtes groupées est déjà prise en compte : seules
        ``num_key_value_heads`` têtes sont stockées.

        Une attention à latent compressé ne garde NI K NI V par tête : elle
        garde le latent ``[rang + rope]``, en 16 bits, hors du système paginé.
        Lui appliquer la formule à requêtes groupées se trompait dans les deux
        sens selon le modèle, et pas d'un peu — mesuré sur le parc, le rapport
        entre le vrai coût et celui qu'on budgétait va de **0,28 à 7,78** :

            DeepSeek-Coder-V2-Lite   0,28   on reservait 3,6 fois trop
            GLM-4.7-Flash            5,54   on reservait 5,5 fois trop peu
            GLM-4.7-Grande-42B       5,54
            Kimi-Linear-35B          7,78   on reservait 7,8 fois trop peu

        Un biais aurait été un réglage ; deux sens opposés sont une formule
        qui ne décrit pas l'objet.
        """
        if self.est_mla:
            latent = (self.kv_lora_rank + self.qk_rope_head_dim) * 2
            return int(latent * self.couches_avec_kv)
        per_layer = 2 * self.num_key_value_heads * self.head_dim * kv_bits / 8
        # échelles groupées du KV quantifié : un fp16 par tête, par jeton, par kv
        overhead = 0.0 if kv_bits >= 16 else 2 * self.num_key_value_heads * 2
        # C5-b (REGLES § 6 : le régime porte la taille de bloc) : sous
        # ACVRAM_KV_INT8_CANAL=1 chaque bloc int8 porte en plus sc E4M3 [HKV, D]
        # et tampon_de int32 (`KVCacheConfig.bytes_per_block`) ; budgété en
        # octets « par jeton » sans ce terme, le planificateur accordait 3,1 %
        # de blocs en moins (Coder : 23 824 jetons pour 12 × 2 048 demandés,
        # une séquence tronquée, `certifie` refusait — Manon, 20/09).
        if kv_bits == 8:
            from ..memory import kv_canal
            from ..memory.kvcache import BLOCK_SIZE
            if kv_canal.ACTIF:
                overhead += (self.num_key_value_heads * self.head_dim + 4) / BLOCK_SIZE
        return int((per_layer + overhead) * self.couches_avec_kv)

    def activations_prefill_bytes(self, n_jetons: int) -> int:
        """Octets TRANSITOIRES de VRAM qu'un préfill de ``n_jetons`` demande
        au-delà des poids résidents et du cache KV — le terme que le budget
        d'exil ne comptait pas (Laure, verdict-palier1-bloc6-17-09 :
        Llama-3.3-70B-nvfp4 chargé DÉGRADÉ à 32/80 couches exilées, puis OOM de
        448 Mio au tout premier préfill).

        Le préfill n'est pas découpé (runner : pas de chunked prefill) : le
        plus grand chunk est une invite de ``max_model_len`` jetons. Par
        jeton, en bf16 sauf mention : le flux résiduel, la ligne normée, la
        sortie d'attention et celle du MLP (4·H), q/k/v ((HQ + 2·HKV)·D), et
        pour le MLP gate, up, act (3·I) plus une copie fp32 de act (chemins
        torch) ; pour un MoE, ces trois-là sur les ``top_k`` lignes routées
        par jeton (moe_intermediate) et l'expert partagé. S'y ajoute, une fois,
        la plus grosse matrice déquantifiée en bf16 que le chemin W4A16 du
        préfill matérialise (`nvfp4_matmul` au-delà du seuil GEMV ; pour les
        experts, la pile `_pile_bf16` = E·I_moe·H) : c'est l'allocation de
        470 Mio (28672 × 8192 × 2) qui manquait au 70B. Les logits n'y sont
        pas : le moteur ne les calcule que pour les positions échantillonnées.
        Un préfill groupé de plusieurs invites (ACVRAM_PREFILL_BATCH) peut
        dépasser cette estimation : elle couvre une invite, la plus longue."""
        T = max(1, int(n_jetons))
        H = self.hidden_size
        D = self.head_dim or (H // max(1, self.num_attention_heads))
        qkv = (self.num_attention_heads + 2 * self.num_key_value_heads) * D
        par_jeton = 4 * H * 2 + qkv * 2
        if self.num_experts and self.moe_intermediate_size:
            k = max(1, self.num_experts_per_tok)
            im = self.moe_intermediate_size
            par_jeton += k * (3 * im * 2 + im * 4) + self.shared_expert_intermediate_size * 3 * 2
            plus_grosse = max(self.num_experts * im * H,
                              self.intermediate_size * H if self.first_k_dense_replace else 0,
                              qkv * H) * 2
        else:
            im = self.intermediate_size
            par_jeton += 3 * im * 2 + im * 4
            plus_grosse = max(im * H, qkv * H) * 2
        return T * par_jeton + plus_grosse

    def etat_recurrent_bytes(self, max_batch: int = 16) -> int:
        """Octets d'état récurrent à provisionner, toutes couches linéaires.

        Cet état vit sur la carte, une copie PAR SÉQUENCE (`gdn_states` du
        runner), et il n'était budgété nulle part : `kda.py` l'alloue à la
        volée dans le forward. La surestimation du cache KV lui servait de
        provision de fait — deux erreurs de sens opposé qui se compensaient par
        accident. Corriger `couches_avec_kv` sans provisionner ici déplacerait
        le défaut au lieu de le lever.

        Les formules suivent les `new_static` de chaque famille, qui sont les
        chemins à formes fixes, et il y en a QUATRE — pas une :

            kda.py      conv x3 + S[têtes, d, d]        `linear_attention`
            gdn.py      conv    + S[1, nv, dk, dv]      `linear_attention`
            mamba2.py   conv    + h[H, N, P]            `mamba`
            lfm2.py     conv seule                      `conv`

        KDA et GDN partagent le type `linear_attention`, `model_type` les
        départage — mais leurs états coïncident sur le poste qui domine
        (``nv·dk·dv`` contre ``têtes·d²``, égaux ici puisque dk = dv = d), et
        la convolution de KDA majore celle de GDN. Une seule formule couvre
        donc les deux, du bon côté.

        Ne provisionner que `linear_attention`, comme le faisait la première
        version, laissait onze modèles à découvert — dont `Nemotron-Nano-9B`
        et ses 2,11 Gio d'état mamba à seize séquences, pour 1,68 Gio de
        budget KV rendus : la correction y creusait un déficit au lieu de le
        combler. Chercher `new_static` dans tout le moteur coûtait une
        commande ; s'en tenir au fichier qu'on m'avait montré a coûté un
        commit.

        Mesuré sur le parc : 2304 Mio à seize séquences sur un Qwen3.8-27B,
        contre 1854 Mio de budget KV entier — ces modèles allouaient déjà hors
        budget dès que la concurrence monte.
        """
        if not self.layer_types:
            return 0
        seq = max(1, max_batch)
        total = 0

        n_lin = sum(1 for t in self.layer_types if t == "linear_attention")
        if n_lin and self.linear_num_value_heads:
            nh, d = self.linear_num_value_heads, self.linear_value_head_dim
            k1 = max(0, self.linear_conv_kernel_dim - 1)
            total += (3 * nh * d * k1 + nh * d * d) * 4 * n_lin

        n_mamba = sum(1 for t in self.layer_types if t == "mamba")
        if n_mamba and self.mamba_num_heads:
            H, P = self.mamba_num_heads, self.mamba_head_dim
            N, G = self.mamba_state_size, self.mamba_n_groups
            k1 = max(0, self.mamba_conv_kernel - 1)
            conv_dim = H * P + 2 * G * N
            total += (conv_dim * k1 + H * N * P) * 4 * n_mamba

        n_conv = sum(1 for t in self.layer_types if t == "conv")
        if n_conv:
            # LFM2 : convolution courte seule, sur `self.dim` canaux, que la
            # configuration n'expose pas séparément — `hidden_size` en est la
            # borne, et majorer une provision est le bon sens de l'erreur.
            k1 = max(0, self.mamba_conv_kernel - 1)
            total += self.hidden_size * k1 * 4 * n_conv

        return total * seq

    def summary(self) -> str:
        b = self.total_params / 1e9
        a = self.active_params / 1e9
        moe = (f", MoE {self.num_experts} experts, top-{self.num_experts_per_tok}"
               if self.is_moe else "")
        return (f"{self.name} : {self.architecture}, {self.num_layers} couches, "
                f"h={self.hidden_size}, {b:.1f} G parametres "
                f"({a:.1f} G actifs par jeton){moe}")

    # Cles de `raw` que le CHARGEUR consulte : elles doivent survivre a la
    # conversion, sinon le manifeste ne transporte pas ce que le chargeur
    # attend et la branche correspondante ne s'execute jamais en service.
    # Le 8/09/2026, `gdn_a_log_negexp` manquait ainsi au manifeste : le facteur
    # de decroissance des couches recurrentes etait transforme deux fois, pour
    # onze pour cent d'ecart au lieu d'un contre llama.cpp.
    CLES_BRUTES_UTILES = ("gdn_a_log_negexp",)

    @property
    def mrope_section(self) -> Optional[list[int]]:
        """[t, h, w] de ``rope_scaling`` (Qwen3-VL, entrelacé), None sans M-RoPE."""
        from .mrope import section_depuis
        return section_depuis(self.rope_scaling)

    @property
    def mlp_activation(self) -> str:
        """Le repli "silu" au POINT D USAGE, pas a l enregistrement.

        `hidden_activation` vaut None quand la source est muette ; le MLP a
        besoin d une fonction concrete, et silu est la convention d un
        transformeur. Le manifeste garde None (honnete), le moteur voit silu.
        """
        return self.hidden_activation or "silu"

    def to_dict(self) -> dict:
        d = asdict(self)
        brut = d.pop("raw", None) or {}
        d.pop("layers", None)
        for cle in self.CLES_BRUTES_UTILES:
            if cle in brut:
                d[cle] = brut[cle]
        d["total_params"] = self.total_params
        d["active_params"] = self.active_params
        return d


_ARCH_ALIASES = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    # Devstral-Small-2 (Mistral3, text_config ministral3) : deplie generi-
    # quement par le bloc `text_config` ci-dessous, meme convention que les
    # modeles vision-langage. Explicite plutot que laisse au repli "llama"
    # par defaut (`_ARCH_ALIASES.get(archs[0], "llama")`) -- un repli
    # silencieux qui marche aujourd'hui casse sans bruit si le defaut change.
    "Mistral3ForConditionalGeneration": "llama",
    "Qwen2ForCausalLM": "llama",
    "Qwen3ForCausalLM": "llama",
    "Qwen2MoeForCausalLM": "moe",
    "Qwen3MoeForCausalLM": "moe",
    "MixtralForCausalLM": "moe",
    "DeepseekV2ForCausalLM": "moe",
    "DeepseekV3ForCausalLM": "moe",
    "GemmaForCausalLM": "llama",
    "Gemma2ForCausalLM": "llama",
    "Phi3ForCausalLM": "llama",
    "GraniteForCausalLM": "llama",
    "Lfm2ForCausalLM": "llama",
    "NemotronHForCausalLM": "llama",
    "FalconH1ForCausalLM": "llama",
    "Starcoder2ForCausalLM": "llama",
    "MuseGlimmerForConditionalGeneration": "llama",
    "Ernie4_5_MoeForCausalLM": "llama",
    "Lfm2MoeForCausalLM": "moe",
    "Gemma4ForCausalLM": "llama",
    "Gemma4ForConditionalGeneration": "llama",
    # vision-langage (partie texte seule)
    "Qwen2VLForConditionalGeneration": "llama",
    "Qwen2_5_VLForConditionalGeneration": "llama",
    "Qwen3VLForConditionalGeneration": "llama",
    "Qwen3VLMoeForConditionalGeneration": "moe",
    "Qwen3_5ForConditionalGeneration": "llama",
    "Qwen3_5MoeForConditionalGeneration": "moe",
}


def _as_id_list(v: Any) -> list[int]:
    """``eos_token_id`` vaut tantot un entier, tantot une liste."""
    if isinstance(v, int):
        return [v]
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v if isinstance(x, int)]
    return []


def load_model_spec(path: str, name: Optional[str] = None) -> ModelSpec:
    """Lit un répertoire de modèle Hugging Face, ou un simple ``config.json``."""
    cfg_path = path if path.endswith(".json") else os.path.join(path, "config.json")
    if not path.endswith(".json") and not os.path.isfile(cfg_path):
        # Un point de contrôle GGUF porte sa configuration dans son en-tête.
        from ..quant.gguf import GGUFFile, is_gguf
        if is_gguf(path):
            g = GGUFFile(path)
            g.check_executable()
            cfg = g.hf_config()
        else:
            raise FileNotFoundError(f"ni config.json ni .gguf sous {path}")
    else:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

    # Certaines configurations imbriquent le modèle de langage (modèles visuels).
    if "text_config" in cfg and "hidden_size" not in cfg:
        cfg = {**cfg, **cfg["text_config"]}
    rp_gen = cfg.get("rope_parameters") or {}
    if "rope_theta" not in cfg and isinstance(rp_gen, dict) and rp_gen.get("rope_theta"):
        cfg = {**cfg, "rope_theta": rp_gen["rope_theta"]}
    if cfg.get("model_type") == "ministral3" and not cfg.get("rope_scaling") and rp_gen:
        # Devstral-Small-2 (jerome, 18/09) : `rope_parameters` porte le yarn
        # (type/factor/original_max_position_embeddings/beta_fast/beta_slow),
        # memes noms de cles que le `rope_scaling` que lit deja layers.py
        # (_build_inv_freq) -- passe tel quel.
        #
        # `llama_4_scaling_beta` (seul champ propre a ministral3) N'EST PAS
        # PORTE ICI -- PAS UN NO-OP CONFIRME. Le vrai config.json publie
        # (huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512,
        # verifie le 18/09, pas la valeur par defaut de la bibliotheque
        # transformers) porte `original_max_position_embeddings=8192`, pas
        # 16384. La formule HF (modeling_ministral3.py, get_llama_4_attn_
        # scale) vaut `1 + beta*log(1+floor(position/original_max_position_
        # embeddings))` : EXACTEMENT 1.0 pour toute position < 8192, mais
        # PAS au-dela (position 8192 : 1 + 0,1*log(2) = 1,069). Un manque a
        # 0 % pres seulement si aucune fenetre servie/mesuree ne depasse
        # 8192 jetons de position -- a confirmer avec Laure avant la
        # conversion, pas suppose ici.
        cfg = {**cfg, "rope_scaling": rp_gen}
    if not cfg.get("rope_scaling") and isinstance(rp_gen, dict) and rp_gen.get("mrope_section"):
        # Qwen3-VL enregistré par transformers ≥ 5 : le M-RoPE vit dans
        # `rope_parameters` (mrope_section, mrope_interleaved, rope_type) — même
        # dictionnaire que `rope_scaling`, lu par layers.RotaryEmbedding
        cfg = {**cfg, "rope_scaling": rp_gen}
    if cfg.get("model_type") == "gemma4_unified_text":
        cfg = {**cfg, "model_type": "gemma4_text"}     # même modèle texte
    if cfg.get("model_type") == "nemotron_h" and (
            cfg.get("hybrid_override_pattern") or cfg.get("layers_block_type")):
        # point de contrôle HF (bf16 ou EXL3) : motif M/E/-/* (hybrid_override_
        # pattern) ou liste de mots (layers_block_type, le seul champ présent
        # sur le bf16 source — hybrid_override_pattern n'y est écrit que sur
        # le dérivé EXL3) → types de couches, mêmes conventions que la
        # synthèse GGUF (attention sans RoPE, MLP ReLU², routage sigmoïde +
        # biais, experts partagés)
        if cfg.get("hybrid_override_pattern"):
            motif = str(cfg["hybrid_override_pattern"])
            kinds = {"M": "mamba", "E": "moe", "-": "mlp", "*": "full_attention"}
            couches = [kinds[c] for c in motif[:int(cfg["num_hidden_layers"])]]
        else:
            mots = {"mamba": "mamba", "moe": "moe", "attention": "full_attention",
                    "mlp": "mlp", "-": "mlp"}
            couches = [mots[m] for m in cfg["layers_block_type"][:int(cfg["num_hidden_layers"])]]
        cfg = {**cfg,
               "layer_types": couches,
               "attention_rope": False, "hidden_act": "relu2",
               "rms_norm_eps": cfg.get("norm_eps") or cfg.get("layer_norm_epsilon") or 1e-5,
               "num_experts": cfg.get("n_routed_experts") or 0,
               "shared_expert_intermediate_size":
                   int(cfg.get("moe_shared_expert_intermediate_size") or 0) * int(cfg.get("n_shared_experts") or 0),
               "router_scoring": "sigmoid", "first_k_dense_replace": 0}
    if cfg.get("model_type") in ("lfm2", "lfm2_moe") and "norm_eps" in cfg:
        cfg = {**cfg, "rms_norm_eps": cfg["norm_eps"],
               "first_k_dense_replace": cfg.get("num_dense_layers") or 0,
               "router_scoring": "sigmoid"}
    if cfg.get("model_type") == "ernie4_5_moe":
        # MoE ERNIE : top-k sur softmax + biais de correction (sélection
        # seule), renormalisation, experts partagés fusionnés en un MLP,
        # première couche dense ; RoPE entrelacé → q/k dé-permutés à la
        # conversion (comme les GGUF llama)
        cfg = {**cfg, "num_experts": cfg.get("moe_num_experts"),
               "num_experts_per_tok": cfg.get("moe_k"),
               "shared_expert_intermediate_size":
                   int(cfg.get("moe_intermediate_size") or 0) * int(cfg.get("moe_num_shared_experts") or 0),
               "first_k_dense_replace": cfg.get("moe_layer_start_index") or 0,
               "scoring_func": "softmax", "norm_topk_prob": True}
    if cfg.get("model_type") in ("muse_glimmer", "muse_glimmer_text"):
        # attention à porte de sortie (gate_proj fusionné dans q_proj à la
        # conversion), q/k normalisés sans poids (facteur 3.87 replié dans
        # q_norm), logits × output_multiplier puis softcap
        cfg = {**cfg, "attn_output_gate": True, "model_type": "muse_glimmer",
               "hidden_activation": cfg.get("hidden_activation") or "silu",
               "logits_scaling": 1.0 / float(cfg.get("output_multiplier") or 1.0)}

    # generation_config.json fait autorite sur les jetons d'arret : Qwen y
    # declare <|im_end|>, absent du eos_token_id de config.json sur certains
    # points de controle.
    gen_path = (os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                             "generation_config.json")
                if os.path.isfile(cfg_path) else "")
    if os.path.isfile(gen_path):
        try:
            with open(gen_path, "r", encoding="utf-8") as fh:
                gen = json.load(fh)
            merged = _as_id_list(cfg.get("eos_token_id")) + \
                _as_id_list(gen.get("eos_token_id"))
            if merged:
                cfg = {**cfg, "eos_token_id": sorted(set(merged))}
        except (OSError, json.JSONDecodeError):
            pass

    archs = cfg.get("architectures") or ["LlamaForCausalLM"]
    mt = str(cfg.get("model_type", ""))
    if mt in ("gemma4", "gemma4_text"):
        rp = cfg.get("rope_parameters") or {}
        full, swa = rp.get("full_attention", {}), rp.get("sliding_attention", {})
        cfg = {**cfg,
               "rope_theta": float(full.get("rope_theta", cfg.get("rope_theta", 1e6))),
               "rope_theta_swa": float(swa.get("rope_theta", cfg.get("rope_theta_swa", 1e4))),
               "partial_rotary_factor_full": float(full.get("partial_rotary_factor",
                                                             cfg.get("partial_rotary_factor_full", 1.0))),
               "embedding_multiplier": float(cfg["hidden_size"]) ** 0.5,
               "attention_multiplier": 1.0}
    if mt == "glm4_moe_lite":
        # HF (modeling_glm4_moe_lite.py:402-423, Sage 14/09) : sigmoid +
        # biais de correction (`e_score_correction_bias`) INCONDITIONNEL
        # pour cette famille — sa config ne declare pas `scoring_func`, et
        # le repli generique plus bas tombait sur "softmax", qui IGNORE le
        # biais (branche softmax de model.py, le biais n'est lu qu'ailleurs)
        # et se trompe sur un sous-ensemble des jetons (8/16 mesures,
        # equivalence CPU 2 couches vs HF, revue/sage-refutation-glm-
        # routage-14-09.md). PAS un heuristique general "sigmoid si biais
        # present" : ernie4_5_moe a biais ET softmax, legitimement (ligne
        # ~534 ci-dessus) — ce cas est nomme par model_type, pas devine.
        cfg = {**cfg, "router_scoring": "sigmoid"}
    # qwen3_next est désormais exécutable (couches Gated DeltaNet) quand la
    # configuration porte nos champs layer_types/linear_* ; les autres
    # hybrides restent refusés.
    # Le critere est la compression du KV elle-meme (cf. ModelSpec.est_mla),
    # pas une liste de model_type : glm4_moe_lite (GLM-4.7-Flash) en manquait,
    # la conversion serait passee en silence par le chemin non-MLA (bead
    # anticitoyen-vram-992, 14/09).
    if cfg.get("kv_lora_rank"):
        cfg = {**cfg, "mla_rope": True,
               "layer_types": cfg.get("layer_types") or
               ["full_attention"] * int(cfg.get("num_hidden_layers", 0))}
        if not cfg.get("shared_expert_intermediate_size") and cfg.get("n_shared_experts"):
            # point de contrôle HF : experts partagés fusionnés en un MLP
            cfg = {**cfg, "shared_expert_intermediate_size":
                   int(cfg.get("moe_intermediate_size") or 0) * int(cfg["n_shared_experts"]),
                   "router_scoring": cfg.get("router_scoring") or cfg.get("scoring_func") or "softmax"}
    if mt in ("qwen3_next", "kimi_linear", "qwen3_5", "qwen3_5_text",
              "qwen3_5_moe", "qwen3_5_moe_text") \
            and cfg.get("linear_num_value_heads"):
        if cfg.get("rotary_dim") is None and cfg.get("partial_rotary_factor"):
            hd = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
            cfg = {**cfg, "rotary_dim": int(hd * float(cfg["partial_rotary_factor"]))}
    elif (mt in ("kimi_linear", "mamba", "mamba2", "jamba")
            or "linear_attn" in json.dumps(cfg.get("layer_types", ""))):
        raise ValueError(
            f"architecture « {archs[0]} » (model_type={mt}) : recurrence "
            f"lineaire ou hybride SSM — le moteur acvram est un transformeur "
            f"pur et ne peut pas l'executer.")
    arch = _ARCH_ALIASES.get(archs[0], "llama")

    n_heads = cfg.get("num_attention_heads", 32)
    spec = ModelSpec(
        name=name or cfg.get("_name_or_path") or os.path.basename(os.path.abspath(path)),
        architecture=arch,
        hidden_size=cfg.get("hidden_size", 4096),
        intermediate_size=cfg.get("intermediate_size", 11008),
        num_layers=cfg.get("num_hidden_layers", 32),
        num_attention_heads=n_heads,
        num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
        vocab_size=cfg.get("vocab_size", 32000),
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        rms_norm_eps=cfg.get("rms_norm_eps", cfg.get("norm_epsilon", 1e-5)),
        rope_theta=cfg.get("rope_theta", 10000.0),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        head_dim=cfg.get("head_dim", 0),
        num_experts=cfg.get("num_experts") or cfg.get("num_local_experts")
        or cfg.get("n_routed_experts") or 0,
        num_experts_per_tok=cfg.get("num_experts_per_tok") or cfg.get("top_k") or 0,
        moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
        shared_expert_intermediate_size=cfg.get("shared_expert_intermediate_size", 0),
        first_k_dense_replace=cfg.get("first_k_dense_replace", 0),
        torch_dtype=(str(cfg["torch_dtype"])
                     if cfg.get("torch_dtype") is not None else None),
        layer_types=list(cfg.get("layer_types") or []),
        linear_num_value_heads=int(cfg.get("linear_num_value_heads") or 0),
        linear_num_key_heads=int(cfg.get("linear_num_key_heads") or 0),
        linear_key_head_dim=int(cfg.get("linear_key_head_dim") or 0),
        linear_value_head_dim=int(cfg.get("linear_value_head_dim") or 0),
        linear_conv_kernel_dim=int(cfg.get("linear_conv_kernel_dim") or 4),
        rotary_dim=cfg.get("rotary_dim"),
        attn_output_gate=bool(cfg.get("attn_output_gate")),
        model_type=mt,
        mamba_num_heads=int(cfg.get("mamba_num_heads") or 0),
        mamba_head_dim=int(cfg.get("mamba_head_dim") or 0),
        mamba_n_groups=int(cfg.get("n_groups") or cfg.get("mamba_n_groups") or 1),
        mamba_state_size=int(cfg.get("ssm_state_size") or cfg.get("mamba_state_size") or 128),
        mamba_conv_kernel=int(cfg.get("conv_kernel") or cfg.get("mamba_conv_kernel") or 4),
        attention_rope=bool(cfg.get("attention_rope", True)),
        post_norm_eps=float(cfg.get("post_norm_eps") or 0.0),
        norm_type=str(cfg.get("norm_type") or "rms_norm"),
        mlp_gated=bool(cfg.get("mlp_gated", cfg.get("model_type") != "starcoder2")),
        conv_L_cache=int(cfg.get("conv_L_cache") or 3),
        sliding_window=int(cfg.get("sliding_window") or 0),
        global_head_dim=int(cfg.get("global_head_dim") or 0),
        num_global_key_value_heads=int(cfg.get("num_global_key_value_heads") or 0),
        rope_theta_swa=float(cfg.get("rope_theta_swa") or 0.0),
        partial_rotary_factor_full=float(cfg.get("partial_rotary_factor_full") or 1.0),
        final_logit_softcapping=float(cfg.get("final_logit_softcapping") or 0.0),
        hidden_activation=(str(cfg.get("hidden_activation")
                               or cfg.get("hidden_act"))
                           if (cfg.get("hidden_activation")
                               or cfg.get("hidden_act")) else None),
        attention_k_eq_v=bool(cfg.get("attention_k_eq_v")),
        attention_multiplier=(float(cfg["attention_multiplier"])
                              if cfg.get("attention_multiplier") else None),
        embedding_multiplier=float(cfg.get("embedding_multiplier") or 1.0),
        residual_multiplier=float(cfg.get("residual_multiplier") or 1.0),
        logits_scaling=float(cfg.get("logits_scaling") or 1.0),
        kv_lora_rank=int(cfg.get("kv_lora_rank") or 0),
        q_lora_rank=int(cfg.get("q_lora_rank") or 0),
        mla_rope=bool(cfg.get("mla_rope")),
        qk_rope_head_dim=int(cfg.get("qk_rope_head_dim") or 0),
        qk_nope_head_dim=int(cfg.get("qk_nope_head_dim") or 0),
        v_head_dim=int(cfg.get("v_head_dim") or 0),
        router_scoring=str(cfg.get("router_scoring") or "softmax"),
        routed_scaling_factor=float(cfg.get("routed_scaling_factor") or 1.0),
        eos_token_id=_as_id_list(cfg.get("eos_token_id")),
        bos_token_id=(cfg.get("bos_token_id")
                      if isinstance(cfg.get("bos_token_id"), int) else None),
        spatial_merge_size=int((cfg.get("vision_config") or {}).get("spatial_merge_size")
                               or cfg.get("spatial_merge_size") or 0),
        raw=cfg,
    )
    _affiner_couches(spec, path)
    return spec


def _formes_du_point_de_controle(path: str) -> dict:
    """Nom et forme de chaque tenseur, sans charger un seul poids.

    Trois sources, par ordre de préférence : le manifeste d'un modèle déjà
    converti, l'en-tête d'un fichier GGUF, les en-têtes des fragments
    safetensors d'un dépôt Hugging Face. Tous trois portent les formes en
    clair, à quelques kilooctets de lecture.
    """
    if path.endswith(".json"):
        return {}
    manifeste = os.path.join(path, "acvram_manifest.json")
    if os.path.isfile(manifeste):
        try:
            with open(manifeste, "r", encoding="utf-8") as fh:
                t = json.load(fh)["tensors"]
            return {n: e["shape"] for n, e in t.items()
                    if isinstance(e, dict) and e.get("shape")}
        except Exception:                              # noqa: BLE001
            return {}
    if os.path.isfile(path) or not os.path.isdir(path):
        try:
            from ..quant.gguf import GGUFFile, is_gguf
            if not is_gguf(path):
                return {}
            g = GGUFFile(path)
        except Exception:                              # noqa: BLE001
            return {}
        # Les noms de llama.cpp (« blk.3.ffn_up_exps.weight ») ne ressemblent pas
        # à ceux de Hugging Face, et les experts y sont empilés en un tenseur
        # [E, ...] par projection. On traduit ce qu'il faut pour peser une
        # couche : sa famille et le nombre de paramètres qu'elle porte.
        formes: dict = {}
        for nom, info in g.tensors.items():
            dims = list(info[0])
            if not nom.startswith("blk."):
                continue
            morceaux = nom.split(".", 2)
            if len(morceaux) < 3:
                continue
            idx, reste = morceaux[1], morceaux[2]
            if "_exps" in reste:
                # [E, sortie, entrée] : une entrée par expert, pour que le
                # compte des experts et leur taille soient tous deux justes
                e = dims[0] if len(dims) == 3 else 1
                par_expert = dims[1:] if len(dims) == 3 else dims
                proj = reste.split(".")[0].replace("ffn_", "").replace("_exps", "")
                for k in range(int(e)):
                    formes[f"model.layers.{idx}.mlp.experts.{k}.{proj}_proj.weight"] = par_expert
            elif reste.startswith("ffn_"):
                formes[f"model.layers.{idx}.mlp.{reste}"] = dims
            elif "norm" in reste:
                formes[f"model.layers.{idx}.{reste}"] = dims
            else:
                formes[f"model.layers.{idx}.self_attn.{reste}"] = dims
        return formes
    brut: dict = {}
    try:
        import struct
        fragments = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        for f in fragments:
            with open(os.path.join(path, f), "rb") as fh:
                taille = struct.unpack("<Q", fh.read(8))[0]
                if taille > 200 * 1024 * 1024:         # en-tête aberrant
                    return {}
                entete = json.loads(fh.read(taille))
            for n, e in entete.items():
                if n != "__metadata__" and isinstance(e, dict) and e.get("shape"):
                    brut[n] = e["shape"]
    except Exception:                                  # noqa: BLE001
        return {}
    return _traduire_exl3(brut) if any(n.endswith(".suh") for n in brut) else brut


def _traduire_exl3(brut: dict) -> dict:
    """Rend les formes logiques d'un dépôt exllamav3.

    EXL3 ne range pas des matrices mais des treillis : ``…suh`` porte la
    dimension d'entrée, ``…svh`` la sortie, ``…trellis`` les poids compressés.
    Le produit des deux premières donne le nombre de paramètres de la
    projection. Les noms suivent aussi une autre convention — ``backbone``
    pour ``model``, ``mixer`` pour l'attention ou le bloc à experts.
    """
    formes: dict = {}
    for nom, forme in brut.items():
        if nom.endswith(".suh"):
            base = nom[:-4]
            sortie = brut.get(base + ".svh")
            if not sortie:
                continue
            logique = base.replace("backbone.", "model.")
            if ".mixer.experts." in logique or ".mixer.shared_experts" in logique:
                logique = logique.replace(".mixer.", ".mlp.")
            elif ".mixer." in logique:
                logique = logique.replace(".mixer.", ".self_attn.")
            formes[logique + ".weight"] = [int(sortie[0]), int(forme[0])]
        elif len(forme) == 2 and nom.endswith(".weight"):
            formes[nom.replace("backbone.", "model.")] = forme
    return formes


def _affiner_couches(spec: ModelSpec, path: str) -> None:
    """Recompte les paramètres par couche depuis les tenseurs réels du modèle.

    ``_build_layer`` déduit la taille d'une couche de la configuration, en
    supposant que toutes se ressemblent : une attention plus un MLP, MoE au-delà
    de ``first_k_dense_replace``. Les architectures hybrides démentent cette
    supposition — Nemotron-H alterne 23 couches Mamba sans MLP, 23 couches MoE
    sans attention et 6 couches d'attention pure, et le compte analytique
    donnait 103 milliards de paramètres pour un modèle qui en a 31,6. Le
    planificateur croyait alors devoir exiler 27 Gio en mémoire vive, étalait
    le modèle sur les deux cartes, et le décodage tombait à 26 jetons/s contre
    170 pour llama.cpp (mesuré le 5 septembre 2026).

    Quand le répertoire est un modèle déjà converti, son manifeste donne la
    forme exacte de chaque tenseur : on s'en sert. C'est juste pour toute
    architecture, présente ou à venir, sans rien deviner.
    """
    tenseurs = _formes_du_point_de_controle(path)
    if not tenseurs:
        return
    attn: dict[int, int] = {}
    mlp: dict[int, int] = {}
    norm: dict[int, int] = {}
    partage: dict[int, int] = {}
    experts: dict[int, set] = {}
    for nom, forme in tenseurs.items():
        if not nom.startswith("model.layers."):
            continue
        n = 1
        for x in forme:
            n *= int(x)
        try:
            i = int(nom.split(".")[2])
        except ValueError:
            continue
        if "norm" in nom.rsplit(".", 2)[-2:][0] or nom.endswith("norm.weight"):
            norm[i] = norm.get(i, 0) + n
        elif ".mlp." in nom or ".feed_forward." in nom:
            mlp[i] = mlp.get(i, 0) + n
            if ".experts." in nom:
                experts.setdefault(i, set()).add(nom.split(".experts.")[1].split(".")[0])
            else:
                partage[i] = partage.get(i, 0) + n      # routeur et expert partagé
        else:
            attn[i] = attn.get(i, 0) + n
    if not attn and not mlp:
        return
    couches = []
    for l in spec.layers:
        i = l.index
        if i not in attn and i not in mlp:
            couches.append(l)                           # couche absente : on garde l'estimation
            continue
        n_ex = len(experts.get(i, ()))
        couches.append(LayerSpec(i, attn.get(i, 0), mlp.get(i, 0), norm.get(i, l.norm_params),
                                 n_ex > 0, n_ex,
                                 min(l.n_experts_active, n_ex) if n_ex else 0,
                                 partage.get(i, 0)))
    spec.layers = couches
