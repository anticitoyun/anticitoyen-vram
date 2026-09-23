"""Attention (dense, MLA, fenêtre, RoPE, cache KV paginé) et MLP dense : `Attention`, `MLP`, `MLP2`, avec le seuil de
fusion des projections (`SEUIL_FUSION`) et le témoin `_ROPE_KV`. Déplacement PUR depuis `engine/model.py` (scission
21/09, module 3/5), corps octet pour octet ; `model` réexporte tout (regime.py lit `acvram.engine.model.SEUIL_FUSION`
et `_ROPE_KV` : les valeurs sont les mêmes objets)."""

from __future__ import annotations

from typing import Optional

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels
from ..memory import kv_lm4
from ..memory.kvcache import PagedKVCache
from .config import ModelSpec
from .layers import (QuantLinear, RMSNorm, RotaryEmbedding, add_norm, apply_rope,
                     attention, decode_attention_fixed, repeat_kv, rope_fusee)
from .lot import ForwardBatch

__all__ = ["Attention", "MLP", "MLP2", "SEUIL_FUSION", "_ROPE_KV", "_multi_projection", "_multi_utilisable"]

# `ACVRAM_SEUIL_FUSION` permet de l'explorer sans toucher au code.
# Au-dela de ce nombre de jetons, on cesse d'empiler les projections.
#
# La fusion remplace deux ou trois GEMM par un seul. Elle gagne quand ils sont
# trop PETITS pour occuper la carte -- au decodage, k et v font 1024 lignes
# pour UNE ligne d'entree. Le vide qu'elle comble se resorbe quand le lot
# grandit.
#
# L'ancien seuil valait 8, ce qui excluait tout preremplissage reel : le prompt
# du banc fait 88 jetons. Gain mesure sur le forward complet, meme chargement,
# graphes actifs, aucun poids en flux (Qwen2.5-Coder-14B, 9 septembre 2026).
# DOUZE tailles, parce qu'un point unique se serait cite comme un gain general
# alors que la courbe est dentelee :
#
#      16 jetons  +6,18 %      128  +4,44 %      200  +3,25 %
#      36         +2,58        160  +5,05        224  +3,37
#      64         +2,61        171  +4,29        240  +3,42
#      88         +4,23        ---               256  +3,51
#
# Plage : +2,58 % a +6,18 %, positif partout, mediane ~+3,8 %.
#
# AU-DELA, LE GAIN N'EST PLUS MONOTONE, et ce n'est pas du bruit : trois series
# independantes donnent les memes valeurs a 0,04 ms pres. Sur les GEMM nus,
# 48 couches : +1,1 ms a 256, -3,3 a 320, +1,1 a 384, +2,3 a 448, -1,3 a 512.
# cuBLAS choisit un autre noyau selon la taille exacte, et la forme empilee
# tombe parfois du mauvais cote. Le point de bascule depend aussi du modele --
# mesure sur trois geometries, il varie et disparait quand `intermediate`
# grandit.
#
# 256 est donc une borne PRUDENTE, pas un optimum : en dessous le gain est
# stable et mesure, au-dela il faudrait le mesurer par modele et par taille.
# `ACVRAM_SEUIL_FUSION` permet de l'explorer sans toucher au code.
SEUIL_FUSION = int(os.environ.get("ACVRAM_SEUIL_FUSION", "256"))

# Poste F, fusion (3a) : normes + RoPE + kv_write en un noyau Triton — RÉFUTÉ
# (verdict-f3a-finale-17-09 : corrompt le cache sous graphe, ou codes ≠
# kv_write_int8 selon la version ; kernels/rope_kv.py) ; témoin, jamais défaut
_ROPE_KV = os.environ.get("ACVRAM_ROPE_KV", "0") == "1"


def _multi_projection(lins):
    """`MultiProjection` (kernels/gemm_dense_etroit) des projections NVFP4
    de même entrée, ou None si une n'est pas NVFP4, porte un biais, une
    rotation, ou n'est pas sur la carte."""
    try:
        from ..kernels.gemm_dense_etroit import MultiProjection, disponible
        from ..quant.nvfp4 import NVFP4Tensor
        interprete = os.environ.get("TRITON_INTERPRET") == "1"
        if not disponible() or not all(isinstance(l.qweight, NVFP4Tensor)
                                       and (l.qweight.qweight.is_cuda or interprete) for l in lins):
            return None
        return MultiProjection(lins)
    except (AssertionError, ImportError):
        return None


# Palier 2 (multi-projection q/k/v, qkv/gate/α/β) : TÉMOIN NOMMÉ, jamais
# défaut — 0,88 To/s pondéré au banc, qkv_multi 0,52 pas mieux que kv seule
# 0,47 avec N triplé : le mécanisme « sous-occupation » est réfuté, cause
# inconnue (sage-gemm-dense-palier2-non-ouvert-17-09 § 1). Réouverture par
# un micro-banc qui EXPLIQUE le 0,52. ACVRAM_MULTI_PROJ=1 pour le rejouer.
_MULTI_PROJ = os.environ.get("ACVRAM_MULTI_PROJ", "0") == "1"


def _multi_utilisable(mp, x: torch.Tensor, t: int) -> bool:
    return (mp is not None and _MULTI_PROJ and kernels._DENSE_NVFP4 == "triton"
            and kernels._DENSE_NVFP4_MIN_M <= t <= 32
            and (x.dtype == torch.bfloat16 and x.is_cuda
                 or x.dtype == torch.float16 and os.environ.get("TRITON_INTERPRET") == "1"))


class Attention(nn.Module):
    def __init__(self, spec: ModelSpec, q: QuantLinear, k: QuantLinear,
                 v: QuantLinear, o: QuantLinear, rope: RotaryEmbedding,
                 q_norm: Optional[nn.Module] = None,
                 k_norm: Optional[nn.Module] = None,
                 output_gate: bool = False,
                 n_kv_heads: Optional[int] = None,
                 head_dim: Optional[int] = None,
                 scale: Optional[float] = None,
                 v_norm_eps: Optional[float] = None,
                 k_eq_v: bool = False,
                 window: int = 0) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q, k, v, o
        # qwen3-next : q_proj sort, par tête, [q | porte] ; la sortie de
        # l'attention est multipliée par sigmoïde(porte) avant o_proj.
        self.output_gate = output_gate
        # Qwen3, Gemma 3 et Olmo 2 normalisent Q et K par tete, avant la RoPE.
        # L'ordre compte : normaliser apres ferait tourner un vecteur puis
        # ecraserait sa norme, ce que le modele n'a pas appris.
        self.q_norm, self.k_norm = q_norm, k_norm
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = n_kv_heads or spec.num_key_value_heads
        self.head_dim = head_dim or spec.head_dim
        self.n_rep = self.n_heads // max(1, self.n_kv_heads)
        self.scale = scale if scale is not None \
            else (spec.attention_multiplier or self.head_dim ** -0.5)
        # Gemma 4 : v normalisé (RMS sans poids), v = k brut sur les couches
        # globales (k_eq_v), fenêtre glissante sur les couches locales
        self.v_norm_eps = v_norm_eps
        self.k_eq_v = k_eq_v
        self.window = window
        self.rope = rope
        self.spec = spec
        self._masque_images: Optional[str] = None     # 'bidir' | 'causal' (vision.masque_images_famille), posé au premier lot à images
        # Scaling « llama 4 » de ministral3/Devstral (transformers
        # modeling_ministral3.get_llama_4_attn_scale) : q ← q · (1 + β·ln(1 +
        # ⌊pos/plafond⌋)) après le RoPE, par jeton, sous rope yarn avec
        # llama_4_scaling_beta > 0. Vaut 1 exactement sous le plafond
        # (original_max_position_embeddings, 8 192 sur Devstral) ; au-delà,
        # sans lui les logits sont faux sans le dire (refus de 67aa280, levé
        # ici). tests/test_devstral_llama4_scaling.py contre transformers.
        rs = getattr(spec, "rope_scaling", None) or {}
        beta = float(rs.get("llama_4_scaling_beta") or 0.0)
        plafond = int(rs.get("original_max_position_embeddings") or 0)
        yarn = str(rs.get("rope_type") or rs.get("type") or "") == "yarn"
        self.llama4 = (beta, plafond) if (yarn and beta > 0 and plafond > 0 and rope is not None) else None
        # Déclaré ici et pas au niveau de la classe : un attribut de classe
        # masque le module enregistré par nn.Module.__setattr__, et la
        # projection empilée resterait invisible (self.qkv_proj toujours None).
        self.qkv_proj = None
        self.qkv_tailles = ()

    # q, k et v lisent la même entrée : au décodage, trois GEMV dont deux
    # minuscules (têtes KV groupées) coûtent plus que la seule grande qui
    # les contient toutes.
    def fuse(self) -> bool:
        from .layers import (stack_int8_linears, stack_nvfp4_linears,
                             stack_int4_awq_linears, stack_plain_linears)
        lins = [self.q_proj, self.k_proj] + ([] if self.k_eq_v else [self.v_proj])
        if any(l is None for l in lins):
            return False

        def _empiler(sous):
            return (stack_int8_linears(sous) or stack_nvfp4_linears(sous)
                    or stack_int4_awq_linears(sous) or stack_plain_linears(sous))

        self.qkv_proj = _empiler(lins)
        if self.qkv_proj is not None:
            self.qkv_tailles = tuple(l.qweight.shape[0] for l in lins)
            self.qkv_partiel = None
            return True
        # Empilement refusé (scalers différents, conversion calibrée par
        # projection) : à 2 ≤ b ≤ 32, une multi-projection NVFP4 sert les
        # trois en un lancement, chacune avec son échelle (17/09, palier 2 de
        # sage-gemm-dense-porte-fermee-palier-17-09) — voir _proj
        self.qkv_multi = _multi_projection(lins)

        # FUSION PARTIELLE. Le groupe entier ne s'empile pas — un format
        # different, une echelle differente — mais un SOUS-ENSEMBLE le peut, et
        # le gain n'est pas marginal : mesure le 9/09/2026 sur les formes de
        # Qwen2.5-14B, trois appels valent 27,69 us, un seul 22,06, et une
        # paire plus un appel isole 24,48 — soit 57 % du gain pour zero octet
        # et zero changement de qualite.
        #
        # L'ordre compte, a contre-sens de l'intuition : le gain vient du
        # SAUVETAGE DES PETITS noyaux, pas de l'agrandissement du gros.
        # Empiler les deux plus petites projections en sauve deux (57 %) ;
        # empiler la grosse avec une petite n'en sauve qu'une (49 %), et cela
        # quelle que soit la petite — mesure a 0,01 us pres.
        if len(lins) < 3:
            return False
        # MESUREE A -12,16 % le 9/09/2026 sur qwen25-coder-14b :
        # 82,15 pas/s sans fusion partielle contre 72,16 avec, seuil de
        # detection 0,51 %. Le premier passage vaut encore 81,72 puis tout
        # bascule a 72 et y reste — une bascule, pas une dispersion. Tant que
        # le decrochage n'est pas explique ET remesure, cette voie ne sert
        # personne par defaut. L'equivalence numerique, elle, tenait : 48
        # jetons identiques. Le chemin est juste, il est couteux.
        if os.environ.get("ACVRAM_FUSION_PARTIELLE", "0") != "1":
            return False
        from .layers import explorer_sans_compter
        # UNE SEULE PILE CONSTRUITE. La version d'avant en batissait jusqu'a
        # TROIS par groupe et en jetait deux — mais `_empiler` ne se contente
        # pas de rendre une pile : il REECRIT le `qweight` de chaque
        # projection en une vue de cette pile. Apres trois essais, lins[0]
        # etait une vue de la pile du dernier essai pendant que la pile
        # retenue etait celle d'un essai precedent : deux tampons vivants la
        # ou un suffit, jamais liberes puisque toujours references. Les
        # valeurs restaient exactes — d'ou les 48 jetons identiques — et
        # seule la vitesse payait. Premier suspect du -12,16 %.
        #
        # Le cout est connu SANS construire : c'est la somme des lignes.
        # On trie donc les paires par cout croissant et on s'arrete a la
        # PREMIERE qui s'empile.
        paires = sorted(((0, 1), (0, 2), (1, 2)),
                        key=lambda ij: (lins[ij[0]].qweight.shape[0]
                                        + lins[ij[1]].qweight.shape[0]))
        pile = None
        # Les tentatives ne sont pas des refus : sans ce silence, le bilan
        # compterait 240 refus la ou il y a 80 groupes.
        for i, j in paires:
            with explorer_sans_compter():
                pile = _empiler([lins[i], lins[j]])
            if pile is not None:
                break
        if pile is None:
            return False
        reste = [k for k in range(3) if k not in (i, j)][0]
        self.qkv_partiel = (pile, (i, j), reste,
                            (lins[i].qweight.shape[0], lins[j].qweight.shape[0]))
        self.qkv_tailles = tuple(l.qweight.shape[0] for l in lins)
        return True

    def _proj_i8c_partage(self, x: torch.Tensor):
        """C15-prefill : q, k, v INT8 par canal (convertis -qkvo-i8c) lus sur la
        MÊME ligne normée — l'A8 par jeton quantifiée une fois pour les trois
        (`kernels.int8_matmul_partage`), au lieu d'une fois par projection
        (nsys P2 19/09 : `_quant_a8_kernel` ×4 par couche, trois fois les mêmes
        octets). None si une projection n'est pas un INT8 plein sans échelle ni
        biais ni exil, ou si le dispatcher ne prendrait pas le chemin cublas :
        le chemin d'avant (trois forwards) reste, au bit."""
        from ..quant.formats import INT8Tensor
        lins = [self.q_proj, self.k_proj] + ([] if self.k_eq_v else [self.v_proj])
        for lin in lins:
            if (not isinstance(lin, QuantLinear) or not isinstance(lin.qweight, INT8Tensor)
                    or lin.streamed is not None or lin.bias is not None
                    or (lin.scaler is not None and not lin.scaler.is_identity)):
                return None
        sorties = kernels.int8_matmul_partage(x, [lin.qweight for lin in lins])
        if sorties is None:
            return None
        qr, kr = sorties[0], sorties[1]
        return qr, kr, (kr if self.k_eq_v else sorties[2])

    def _proj(self, x: torch.Tensor, t: int, qkv=None):
        """q, k, v (et la porte de sortie) : une GEMV empilée si possible ;
        ``qkv`` déjà calculé (3b : norme absorbée par le GEMV) est découpé tel quel."""
        if qkv is not None:
            p = torch.split(qkv, self.qkv_tailles, dim=-1)
            qr, kr = p[0], p[1]
            vr = kr if self.k_eq_v else p[2]
        elif self.qkv_proj is not None and t <= SEUIL_FUSION:
            p = torch.split(self.qkv_proj(x), self.qkv_tailles, dim=-1)
            qr, kr = p[0], p[1]
            vr = kr if self.k_eq_v else p[2]
        elif _multi_utilisable(getattr(self, "qkv_multi", None), x, t):
            p = torch.split(self.qkv_multi(x), self.qkv_multi.tailles, dim=-1)
            qr, kr = p[0], p[1]
            vr = kr if self.k_eq_v else p[2]
        elif getattr(self, "qkv_partiel", None) is not None and t <= SEUIL_FUSION:
            pile, (i, j), reste, tailles = self.qkv_partiel
            deux = torch.split(pile(x), tailles, dim=-1)
            seul = (self.q_proj, self.k_proj, self.v_proj)[reste](x)
            sorties = [None, None, None]
            sorties[i], sorties[j], sorties[reste] = deux[0], deux[1], seul
            qr, kr, vr = sorties
        else:
            partage = self._proj_i8c_partage(x) if kernels.prefill_compact("a8") else None
            if partage is not None:
                qr, kr, vr = partage
            else:
                qr, kr = self.q_proj(x), self.k_proj(x)
                vr = kr if self.k_eq_v else self.v_proj(x)
        gate = None
        if self.output_gate:
            # par tête : [q_h | porte_h] — l'ordre du point de contrôle HF,
            # conservé par le convertisseur GGUF (vérifié : l'ordre plat
            # dégénère immédiatement, celui-ci non)
            qg = qr.view(t, self.n_heads, 2 * self.head_dim)
            q, gate = qg[..., :self.head_dim].contiguous(), qg[..., self.head_dim:]
        else:
            q = qr.view(t, self.n_heads, self.head_dim)
        k = kr.view(t, self.n_kv_heads, self.head_dim)
        v = k if self.k_eq_v else vr.view(t, self.n_kv_heads, self.head_dim)
        if self.v_norm_eps is not None:
            v32 = v.to(torch.float32)
            v = (v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True)
                                   + self.v_norm_eps)).to(x.dtype)
        return q, k, v, gate

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        t = x.shape[0]
        q, k, v, gate = self._proj(x, t)

        r = None
        if self.rope is not None:
            pos = batch.positions_on(x.device)
            mx = max(batch.seq_lens)
            # M-RoPE : positions [3, t] → tables par axe (layers.forward_mrope),
            # hors du noyau fusionné qui n'indexe qu'une position par jeton.
            pos_rope = batch.positions_rope_on(x.device)
            if pos_rope is pos:
                r = rope_fusee(q, k, self.rope, pos, mx, self.q_norm, self.k_norm)
        if r is not None:
            q, k = r                       # normes par tête comprises
        else:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.rope is not None:
                cos, sin = self.rope(pos_rope, x.device, x.dtype, max_pos=mx)
                q, k = apply_rope(q, k, cos, sin)
        if self.llama4 is not None:
            q = self._echelle_llama4(q, pos)

        if cache is not None:
            cache.write(batch.slots_on(x.device), k, v, positions=batch.positions_on(x.device))

        if batch.is_decode:
            return self._decode(q, k, v, batch, cache, t, gate)
        return self._prefill(q, k, v, batch, cache, t, gate)

    def _echelle_llama4(self, q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """q · (1 + β·ln(1 + ⌊pos/plafond⌋)), calculé en fp32 puis arrondi au
        dtype de q — la même arithmétique que transformers (positions/plafond
        en flottant, torch.log, `.to(query_states.dtype)`)."""
        beta, plafond = self.llama4
        s = 1.0 + beta * torch.log1p(torch.floor(positions.to(torch.float32) / plafond))
        return q * s.to(q.dtype).view(-1, *([1] * (q.dim() - 1)))

    def norme_fusee_possible(self, t: int) -> bool:
        """(3b) : la projection q/k/v empilée est un INT8 sans échelle AWQ ni
        biais, lot ≤ 8 — le GEMV peut absorber la norme d'entrée."""
        lin = self.qkv_proj
        return (lin is not None and t <= SEUIL_FUSION and t <= 8
                and getattr(getattr(lin, "qweight", None), "format", "") == "int8"
                and getattr(lin, "streamed", None) is None
                and (lin.scaler is None or lin.scaler.is_identity) and lin.bias is None)

    def decode_fixed_norme(self, res, delta, norme, mult, positions, slots, block_tables,
                           seq_lens, max_pos, cache, q_len: int = 1):
        """(3b) : `add_norm(res, delta)` absorbé par le GEMV q/k/v — rend
        (x = res + mult·delta, sortie d'attention) ; None si le noyau décline."""
        r = kernels.int8_matmul_norme(delta, self.qkv_proj.qweight, res, norme.weight, norme.eps, mult)
        if r is None:
            return None
        qkv, x = r
        return x, self.decode_fixed(x, positions, slots, block_tables, seq_lens, max_pos,
                                    cache, q_len, qkv=qkv)

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache: PagedKVCache, q_len: int = 1, qkv=None) -> torch.Tensor:
        """Le pas de décodage à formes fixes — le chemin que capture le graphe.

        Même mathématique que ``forward`` en décodage, mais aucun scalaire
        Python tiré des données : positions, emplacements, tables et longueurs
        sont des tenseurs dont seul le *contenu* change entre deux rejeux.
        ``max_pos`` majore les positions (la longueur maximale du godet).
        ``q_len`` > 1 est le pas de vérification spéculative : chaque séquence
        pose q_len positions, chacune voyant son propre préfixe causal — le
        noyau paginé le gère nativement, et c'est lui qui est exigé ici (le
        repli déquantifier-puis-SDPA ne connaît que q_len = 1).
        """
        b = x.shape[0]
        q, k, v, gate = self._proj(x, b, qkv=qkv)
        # Poste F, fusion (3a) : normes par tête + RoPE + écriture int8 du
        # cache en UN noyau Triton (kernels/rope_kv) — à la place de
        # rope_inplace puis kv_write_int8 ; sans diagnostic lm4 (positions).
        if (_ROPE_KV and self.rope is not None and q.is_cuda and q.dtype == torch.bfloat16
                and cache.cfg.dtype == "int8" and cache.k_scale is not None
                and not getattr(cache, "canal", False)      # C5-b : rope_kv écrit par jeton
                and not kv_lm4.diagnostic_actif()):
            r = self._rope_kv_fusee(q, k, v, positions, max_pos, slots, cache)
            if r is not None:
                q, k = r
                if self.llama4 is not None:
                    q = self._echelle_llama4(q, positions)
                out = kernels.paged_attention(q, cache, block_tables, seq_lens,
                                              self.n_rep, self.scale, q_len=q_len,
                                              window=self.window)
                if out is None:
                    if q_len != 1:
                        raise RuntimeError("verification speculative a formes fixes "
                                           "sans noyau pagine : chemin inéligible")
                    kk, vv = cache.gather_fixed(block_tables, q.dtype)
                    out = decode_attention_fixed(q, kk, vv, seq_lens, self.n_rep,
                                                 self.scale, window=self.window)
                out = self._gated(out.to(x.dtype), gate, b)
                return self.o_proj(out.reshape(b, self.n_heads * self.head_dim))
        r = None
        if self.rope is not None:
            r = rope_fusee(q, k, self.rope, positions, max_pos,
                           self.q_norm, self.k_norm)
        if r is not None:
            q, k = r                       # normes par tête comprises
        else:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.rope is not None:
                cos, sin = self.rope(positions, x.device, x.dtype, max_pos=max_pos)
                q, k = apply_rope(q, k, cos, sin)
        if self.llama4 is not None:
            q = self._echelle_llama4(q, positions)
        cache.write(slots, k, v, positions=positions)
        out = kernels.paged_attention(q, cache, block_tables, seq_lens,
                                      self.n_rep, self.scale, q_len=q_len,
                                      window=self.window)
        if out is None:                    # cache non int8, ou pas de noyau
            if q_len != 1:
                raise RuntimeError("verification speculative a formes fixes "
                                   "sans noyau pagine : chemin inéligible")
            kk, vv = cache.gather_fixed(block_tables, q.dtype)
            out = decode_attention_fixed(q, kk, vv, seq_lens, self.n_rep,
                                         self.scale, window=self.window)
        out = self._gated(out.to(x.dtype), gate, b)
        return self.o_proj(out.reshape(b, self.n_heads * self.head_dim))

    def _rope_kv_fusee(self, q, k, v, positions, max_pos, slots, cache):
        """Conditions de `rope_fusee` (normes RMSNorm bf16 de la taille d'une
        tête, eps partagé, tables fp32) puis le noyau fusionné ; None si inéligible."""
        from ..kernels import rope_kv as _rk
        if not _rk.disponible():
            return None
        normes = [n for n in (self.q_norm, self.k_norm) if n is not None]
        if any(type(n).__name__ != "RMSNorm" or n.weight.dtype != torch.bfloat16
               or n.weight.shape[-1] != q.shape[-1] for n in normes):
            return None
        if len(normes) == 2 and abs(self.q_norm.eps - self.k_norm.eps) > 1e-12:
            return None
        cos32, sin32 = self.rope.tables32(max_pos, q.device)
        d = cos32.shape[-1]
        if d % 2 or d > q.shape[-1] or q.shape[-1] & (q.shape[-1] - 1):
            return None
        if q.stride(2) != 1 or q.stride(1) != q.shape[2]:
            q = q.contiguous()
        if k.stride(2) != 1 or k.stride(1) != k.shape[2]:
            k = k.contiguous()
        if v.stride(2) != 1:
            v = v.contiguous()
        _rk.rope_kv(q, k, v, cos32, sin32, positions, slots,
                    None if self.q_norm is None else self.q_norm.weight,
                    None if self.k_norm is None else self.k_norm.weight,
                    normes[0].eps if normes else 1e-6, cache)
        return q, k

    def _gated(self, out: torch.Tensor, gate, t: int) -> torch.Tensor:
        if gate is not None:
            out = out * torch.sigmoid(gate.reshape(out.shape))
        return out

    def _prefill(self, q, k, v, batch: ForwardBatch,
                 cache: Optional[PagedKVCache], t: int,
                 gate=None) -> torch.Tensor:
        # C15-prefill (fusion « attn ») : les têtes KV diffusées par SDPA
        # (enable_gqa) au lieu de `repeat_kv` qui matérialisait K et V ×n_rep
        # ([t, têtes, d] bf16 deux fois par couche : 0,98 ms au budget nsys P2
        # du 19/09), et, à une séquence, la sortie rendue telle quelle au lieu
        # d'être recopiée dans `out` (16,8 Mo de plus par couche à L = 2 047).
        compact = kernels.prefill_compact("attn")
        une_seq = compact and len(batch.query_lens) == 1
        out = None if une_seq else torch.empty_like(q)
        start = 0
        for i, qlen in enumerate(batch.query_lens):
            end = start + qlen
            offset = batch.seq_lens[i] - qlen
            if cache is not None and offset > 0:
                # Une partie de cette séquence est déjà en cache : un préfixe
                # servi, ou un morceau antérieur. On la relit et on masque selon
                # le décalage absolu de la requête.
                kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                      batch.seq_lens[i], q.dtype)
            else:
                # Rien avant : on utilise les clés qu'on vient de calculer
                # plutôt que de les relire par le cache. Cela évite un
                # aller-retour de quantification sur chaque jeton de prefill, ce
                # qui est à la fois plus rapide et un peu plus précis.
                kk, vv = k[start:end], v[start:end]
            plages = batch.images_de(i)          # [] sans image : masque d'avant
            if plages:
                # porte de famille (vision.masque_images_famille) : Gemma → bloc bidirectionnel, Qwen → causal ;
                # inconnu → MasqueImageInconnu (déjà levé au chargement quand une tour est servie)
                if self._masque_images is None:
                    from .vision import masque_images_famille
                    self._masque_images = masque_images_famille(self.spec)
                if self._masque_images != "bidir":
                    plages = []
            if compact:
                a = attention(q[start:end], kk, vv, True, self.scale,
                              q_offset=offset, window=self.window, n_rep=self.n_rep,
                              images=plages)
            else:
                kk = repeat_kv(kk, self.n_rep)
                vv = repeat_kv(vv, self.n_rep)
                a = attention(q[start:end], kk, vv, True, self.scale,
                              q_offset=offset, window=self.window, images=plages)
            if une_seq:
                out = a
            else:
                out[start:end] = a
            start = end
        out = self._gated(out, gate, t)
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

    def _decode(self, q, k, v, batch: ForwardBatch,
                cache: Optional[PagedKVCache], t: int,
                gate=None) -> torch.Tensor:
        """Décodage, et vérification spéculative, pour tout le lot d'un coup."""
        if cache is None:
            return self._prefill(q, k, v, batch, cache, t, gate)

        if all(ql == 1 for ql in batch.query_lens):
            # Décodage pur : le chemin à formes fixes, celui-là même que le
            # graphe CUDA capture — un seul gather vectorisé, pas de boucle
            # Python, et une sortie identique au bit près entre eager et rejeu.
            tables, lens = batch.fixed_decode_views(q.device)
            out = kernels.paged_attention(q, cache, tables, lens,
                                          self.n_rep, self.scale,
                                          window=self.window)
            if out is None:                # cache non int8, ou pas de noyau
                kk, vv = cache.gather_fixed(tables, q.dtype)
                out = decode_attention_fixed(q, kk, vv, lens, self.n_rep,
                                             self.scale, window=self.window)
            out = self._gated(out.to(q.dtype), gate, t)
            return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

        # Vérification spéculative : plusieurs positions de requête par
        # séquence, chacune ne voyant que son propre préfixe. Quand toutes les
        # séquences vérifient le même nombre de positions — le cas normal —
        # le noyau paginé les traite en un lancement, chaque ligne de requête
        # avec sa longueur causale propre.
        ql = batch.query_lens[0]
        if all(q_ == ql for q_ in batch.query_lens):
            tables, lens = batch.fixed_decode_views(q.device)
            out = kernels.paged_attention(q, cache, tables, lens,
                                          self.n_rep, self.scale, q_len=ql,
                                          window=self.window)
            if out is not None:
                out = self._gated(out.to(q.dtype), gate, t)
                return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

        keys, values = [], []
        for i in range(batch.batch_size):
            kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                  batch.seq_lens[i], q.dtype)
            keys.append(kk)
            values.append(vv)

        out = torch.empty_like(q)
        start = 0
        for i, qlen in enumerate(batch.query_lens):
            end = start + qlen
            offset = batch.seq_lens[i] - qlen
            out[start:end] = attention(
                q[start:end], repeat_kv(keys[i], self.n_rep),
                repeat_kv(values[i], self.n_rep), True, self.scale,
                q_offset=offset, window=self.window)
            start = end
        out = self._gated(out, gate, t)
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, gate: QuantLinear, up: QuantLinear, down: QuantLinear,
                 act: str = "silu") -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down
        self.act = act
        self.gate_up = None       # attribut d'instance : voir Attention.fuse

    def _act(self, g: torch.Tensor) -> torch.Tensor:
        if self.act in ("gelu_pytorch_tanh", "gelu_tanh"):
            return F.gelu(g, approximate="tanh")
        if self.act == "gelu":
            return F.gelu(g)
        return F.silu(g)

    def fuse(self) -> bool:
        """gate et up lisent la même entrée : une GEMV empilée au lieu de deux.

        Les NVFP4 ont chacun leur échelle globale ; le noyau en accepte une par
        ligne de sortie, ce qui les empile sans réarrondi (v0.4.62)."""
        from .layers import (stack_int8_linears, stack_nvfp4_linears,
                             stack_int4_awq_linears, stack_plain_linears)
        paire = [self.gate_proj, self.up_proj]
        self.gate_up = (stack_int8_linears(paire) or stack_nvfp4_linears(paire)
                        or stack_int4_awq_linears(paire)
                        or stack_plain_linears(paire))
        return self.gate_up is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up is not None and x.shape[0] <= SEUIL_FUSION:
            gu = self.gate_up(x)
            # SiLU et le produit sont deux lancements elementaires pour un
            # travail derisoire : sur un pas de decodage, la latence de
            # lancement pese plus que le calcul. Le noyau fusionne les fait en
            # un seul, avec l'arrondi intermediaire de torch pour que la sortie
            # reste identique. Le repli couvre gelu et l'absence d'extension.
            if self.act == "silu" and gu.is_cuda and gu.dtype == torch.bfloat16:
                from .. import kernels
                ext = kernels.get_extension()
                if ext is not None and hasattr(ext, "swiglu_bf16"):
                    return self.down_proj(ext.swiglu_bf16(gu))
            g, u = gu.split(gu.shape[-1] // 2, dim=-1)
            return self.down_proj(self._fusionner(g, u))
        return self.down_proj(self._fusionner(self.gate_proj(x),
                                              self.up_proj(x)))

    def _fusionner(self, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """SiLU(g) * u en un lancement quand le noyau est la.

        Le chemin non fusionne payait silu PUIS produit -- deux noyaux par
        couche -- alors que le chemin fusionne n'en payait qu'un. Or le nvfp4
        n'empile pas ses projections (echelles d'activation differentes) : il
        prenait donc systematiquement le chemin a deux noyaux. Mesure sous ncu :
        48 `silu_kernel` par pas cote nvfp4, zero cote bf16 fusionne.
        """
        if self.act == "silu" and g.is_cuda and g.dtype == torch.bfloat16:
            from .. import kernels
            ext = kernels.get_extension()
            if ext is not None and hasattr(ext, "swiglu2_bf16"):
                return ext.swiglu2_bf16(g, u)
        return self._act(g) * u


class MLP2(nn.Module):
    """MLP sans porte (Nemotron-H) : down(act(up(x))), act = ReLU² ou GELU."""

    def __init__(self, up: QuantLinear, down: QuantLinear, act: str = "relu2") -> None:
        super().__init__()
        self.up_proj, self.down_proj = up, down
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(x)
        if self.act == "relu2":
            h = F.relu(h); h = h * h
        elif self.act.startswith("gelu"):
            h = F.gelu(h, approximate="tanh")
        else:
            h = F.silu(h)
        return self.down_proj(h)
