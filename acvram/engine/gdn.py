"""Gated DeltaNet — les couches à récurrence linéaire des hybrides Qwen3-Next
(« qwen35 » côté GGUF), dont les modèles « kimi » du parc dérivent.

La règle delta elle-même — le cœur mathématique, facile à se tromper et
invérifiable à l'œil — vient de l'implémentation de référence de
``transformers`` (`torch_chunk_gated_delta_rule` au prefill,
`torch_recurrent_gated_delta_rule` au décodage), comme la reconstruction EXL3
vient d'exllamav3 : dépendance optionnelle, mathématique garantie. Ce module
fournit ce qui l'entoure : les projections (nos ``QuantLinear``), la
convolution causale à état, la normalisation gated et l'état par séquence.

L'état d'une séquence pour une couche : ``(conv_state [conv_dim, k-1],
S [1, num_v_heads, d_k, d_v])`` — quelques mégaoctets, porté par le moteur,
hors du cache paginé (ces couches n'ont pas de KV).
"""

from __future__ import annotations

from typing import Optional

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lot_etats import nouveau_static, redistribuer, tranches

__all__ = ["GatedDeltaNet", "gdn_available", "gdn_regime"]

# ACVRAM_GDN = fla (défaut) | torch : la récurrence par les noyaux Triton de
# flash-linear-attention (`chunk_gated_delta_rule` au prefill,
# `fused_recurrent_gated_delta_rule` au décodage, un lancement par couche
# pour tout le lot) ou par la référence torch de transformers (une séquence
# à la fois, des dizaines de lancements par séquence et par couche — Qwen3.8
# b=12 : 97 j/s pour 621 chez vLLM, sage-priorite-apres-campagne-17-09).
# « 1 » vaut fla, « 0 » reste le refus des hybrides (quant/gguf.py).
_GDN_VOIE = os.environ.get("ACVRAM_GDN", "fla")


def _fla():
    """(chunk, fused_recurrent) de fla, ou None si absent ou non demandé."""
    if _GDN_VOIE in ("torch", "0"):
        return None
    try:
        from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                              fused_recurrent_gated_delta_rule)
        return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except ImportError:
        return None


def gdn_regime() -> str:
    """Voie effective, pour `regime_ligne()` : fla | torch(raison)."""
    if _GDN_VOIE in ("torch", "0"):
        return "torch"
    if _fla() is None:
        return "torch(fla absent)"
    if not (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"):
        return "torch(sans carte)"
    return "fla"


def _voie_fla(x: torch.Tensor) -> bool:
    return _fla() is not None and (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1")


def _refs():
    """La référence torch de transformers — la fonction NUE : depuis que fla
    est installé (17/09), transformers l'enveloppe (integrations/hub_kernels)
    et la renvoie vers fla, silencieusement ; la voie « torch » serait fla.
    `__wrapped__` (functools.wraps) rend l'originale."""
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
    return (getattr(torch_chunk_gated_delta_rule, "__wrapped__", torch_chunk_gated_delta_rule),
            getattr(torch_recurrent_gated_delta_rule, "__wrapped__", torch_recurrent_gated_delta_rule))


def gdn_available() -> bool:
    try:
        _refs()
        return True
    except ImportError:
        return False


class GatedDeltaNet(nn.Module):
    def __init__(self, qkv: nn.Module, gate: nn.Module, alpha: nn.Module,
                 beta: nn.Module, out: nn.Module,
                 conv_weight: torch.Tensor,        # [conv_dim, kernel]
                 dt_bias: torch.Tensor,            # [num_v_heads]
                 a_log: torch.Tensor,              # [num_v_heads]
                 norm_weight: torch.Tensor,        # [head_v_dim]
                 num_k_heads: int, num_v_heads: int,
                 head_k_dim: int, head_v_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.qkv, self.gate = qkv, gate
        self.alpha, self.beta_proj, self.out_proj = alpha, beta, out
        self.conv_weight = nn.Parameter(conv_weight, requires_grad=False)
        self.dt_bias = nn.Parameter(dt_bias.float(), requires_grad=False)
        self.a_log = nn.Parameter(a_log.float(), requires_grad=False)
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.nk, self.nv = num_k_heads, num_v_heads
        self.dk, self.dv = head_k_dim, head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = conv_weight.shape[-1]
        self.eps = eps

    # -- quatre projections de même entrée en un lancement (palier 2) -----
    def fuse(self) -> bool:
        """qkv, gate, α, β lisent x : une multi-projection NVFP4 (chacune
        avec son scaler) à 2 ≤ b ≤ 32 sous ACVRAM_DENSE_NVFP4=triton."""
        from .model import _multi_projection
        self.multi = _multi_projection([self.qkv, self.gate, self.alpha, self.beta_proj])
        return self.multi is not None

    def _projections(self, x: torch.Tensor):
        """(qkv, z, b, a) en fp32 — un lancement si la multi-projection sert."""
        from .model import _multi_utilisable
        mp = getattr(self, "multi", None)
        if _multi_utilisable(mp, x, x.shape[0]):
            qkv, z, a, b = torch.split(mp(x), mp.tailles, dim=-1)
            return qkv.to(torch.float32), z.to(torch.float32), b.to(torch.float32), a.to(torch.float32)
        return (self.qkv(x).to(torch.float32), self.gate(x).to(torch.float32),
                self.beta_proj(x).to(torch.float32), self.alpha(x).to(torch.float32))

    # -- normalisation gated (RMSNorm de la sortie, porte SiLU(z)) --------
    def _norm_gated(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        x32 = x32 * self.norm_weight.to(torch.float32)
        return (x32 * F.silu(z.to(torch.float32))).to(x.dtype)

    def forward(self, x: torch.Tensor,
                state: Optional[tuple] = None
                ) -> tuple[torch.Tensor, tuple]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, nouvel état)."""
        chunk_rule, recurrent_rule = _refs()
        t = x.shape[0]
        decode = (t == 1 and state is not None)

        # toute la récurrence se calcule en float32 : la règle delta cumule
        # des produits d'état où le bfloat16 dérive vite
        qkv, z, b, a = self._projections(x)                 # [t, conv_dim], [t, value_dim], [t, nv] × 2

        # convolution causale depthwise, avec état (kernel-1 colonnes)
        seq = qkv.t().unsqueeze(0)                          # [1, conv_dim, t]
        if state is not None:
            conv_state = state[0]
            seq = torch.cat([conv_state.unsqueeze(0), seq], dim=-1)
        else:
            seq = F.pad(seq, (self.kernel - 1, 0))
        new_conv_state = seq[0, :, -(self.kernel - 1):].detach().clone()
        conv = F.conv1d(seq, self.conv_weight.unsqueeze(1),
                        groups=self.conv_dim)               # [1, conv_dim, t]
        conv = F.silu(conv)

        mixed = conv.transpose(1, 2)                        # [1, t, conv_dim]
        q, k, v = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, t, self.nk, self.dk)
        k = k.reshape(1, t, self.nk, self.dk)
        v = v.reshape(1, t, self.nv, self.dv)

        beta = b.sigmoid().unsqueeze(0)                     # [1, t, nv]
        g = (-self.a_log.exp() * F.softplus(a + self.dt_bias)).unsqueeze(0)
        if self.nv // self.nk > 1:
            q = q.repeat_interleave(self.nv // self.nk, dim=2)
            k = k.repeat_interleave(self.nv // self.nk, dim=2)

        s_prev = state[1] if state is not None else None
        # sous l'interpréteur Triton (tests sans carte) seul le noyau
        # récurrent tourne : le noyau par blocs y bute sur `i_t.to(...)`
        fla = _fla() if _voie_fla(x) and (decode or x.is_cuda) else None
        if fla is not None:
            # même mathématique, même disposition d'état [B, H, K, V] que la
            # référence (transformers l'a portée de fla) ; entrées en fp32,
            # TRITON_F32_DEFAULT=ieee posé par fla — exact, pas TF32
            rule = fla[1] if decode else fla[0]
            core, s_new = rule(q.contiguous(), k.contiguous(), v.contiguous(), g=g.contiguous(),
                               beta=beta.contiguous(), initial_state=s_prev,
                               output_final_state=True, use_qk_l2norm_in_kernel=True)
        else:
            rule = recurrent_rule if decode else chunk_rule
            core, s_new = rule(q, k, v, g=g, beta=beta,
                               initial_state=s_prev, output_final_state=True,
                               use_qk_l2norm_in_kernel=True)

        core = core.reshape(-1, self.dv)
        y = self._norm_gated(core, z.reshape(-1, self.dv))
        y = y.reshape(t, self.value_dim)
        return self.out_proj(y.to(x.dtype)), (new_conv_state, s_new)

    # -- chemin à formes fixes (graphes CUDA) --------------------------------
    # La règle delta de référence est déjà à formes fixes pour t = 1 : on la
    # rejoue telle quelle et l'on recopie ses sorties dans les tampons fixes
    # — même mathématique, mêmes noyaux, donc mêmes arrondis que ``forward``.
    def new_static(self, device: torch.device) -> dict:
        """Un créneau = des VUES dans un tampon groupé (`lot_etats`) : les
        créneaux 0..b-1 forment des tranches contiguës, `decode_static_batch`
        sert le lot en un lancement sans rassembler ni redistribuer."""
        st = nouveau_static(self, device, {"conv": (self.conv_dim, self.kernel - 1),
                                           "S_": (self.nv, self.dk, self.dv)})
        st["S"] = st.pop("S_").unsqueeze(0)             # [1, nv, dk, dv], la forme de l'état fonctionnel
        return st

    @staticmethod
    def static_load(st: dict, etat) -> None:
        if etat is None:
            st["conv"].zero_(); st["S"].zero_()
            return
        st["conv"].copy_(etat[0]); st["S"].copy_(etat[1])

    @staticmethod
    def static_export(st: dict) -> tuple:
        return (st["conv"].clone(), st["S"].clone())

    def decode_static(self, x: torch.Tensor, st: dict) -> torch.Tensor:
        y, (conv, S) = self.forward(x, (st["conv"], st["S"]))
        st["conv"].copy_(conv)
        st["S"].copy_(S.to(torch.float32))
        return y

    # -- décodage du lot en un lancement (fla) ------------------------------
    def peut_batcher_decode(self, h: torch.Tensor) -> bool:
        return _voie_fla(h)

    def _lot_projete(self, x: torch.Tensor, conv_state: torch.Tensor):
        """Projections, convolution causale à état et portes pour ``b``
        jetons (un par séquence) ; ``conv_state`` [b, conv_dim, k-1] est mis
        à jour EN PLACE. Rend (q, k, v, g, beta, z) aux formes de fla."""
        b = x.shape[0]
        qkv, z, bt, a = self._projections(x)
        seq = torch.cat([conv_state, qkv.unsqueeze(-1)], dim=-1)   # [b, conv_dim, k]
        conv_state.copy_(seq[:, :, 1:])
        conv = F.silu(F.conv1d(seq, self.conv_weight.unsqueeze(1), groups=self.conv_dim))
        mixed = conv.transpose(1, 2)                        # [b, 1, conv_dim]
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, 1, self.nk, self.dk)
        k = k.reshape(b, 1, self.nk, self.dk)
        v = v.reshape(b, 1, self.nv, self.dv)
        if self.nv // self.nk > 1:
            q = q.repeat_interleave(self.nv // self.nk, dim=2)
            k = k.repeat_interleave(self.nv // self.nk, dim=2)
        beta = bt.sigmoid().unsqueeze(1)                    # [b, 1, nv]
        g = (-self.a_log.exp() * F.softplus(a + self.dt_bias)).unsqueeze(1)
        return q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), z

    def forward_batch(self, h: torch.Tensor, etats: list) -> tuple[torch.Tensor, list]:
        """Décodage eager de ``b`` séquences (un jeton chacune) en un
        lancement de la récurrence ; ``etats`` = tuples (conv, S) ou None."""
        b = h.shape[0]
        conv_state = torch.stack([e[0] if e is not None else
                                  torch.zeros(self.conv_dim, self.kernel - 1, dtype=torch.float32, device=h.device)
                                  for e in etats])
        S = torch.cat([e[1] if e is not None else
                       torch.zeros(1, self.nv, self.dk, self.dv, dtype=torch.float32, device=h.device)
                       for e in etats])
        q, k, v, g, beta, z = self._lot_projete(h, conv_state)
        core, S_new = _fla()[1](q, k, v, g=g, beta=beta, initial_state=S.contiguous(),
                                output_final_state=True, use_qk_l2norm_in_kernel=True)
        y = self._norm_gated(core.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(b, self.value_dim)
        y = self.out_proj(y.to(h.dtype))
        return y, [(conv_state[i].clone(), S_new[i:i + 1].to(torch.float32)) for i in range(b)]

    def decode_static_batch(self, h: torch.Tensor, statics: list) -> torch.Tensor:
        """Chemin à formes fixes pour ``b`` créneaux : états lus et écrits
        dans les tranches contiguës du tampon groupé (aucune copie si les
        créneaux 0..b-1 vivent dans le même lot), un lancement de fla."""
        b = h.shape[0]
        g_, contigu = tranches(self, statics, b, ("conv", "S_"))
        conv_state, S = g_["conv"], g_["S_"]
        q, k, v, g, beta, z = self._lot_projete(h, conv_state)
        core, S_new = _fla()[1](q, k, v, g=g, beta=beta, initial_state=S.contiguous(),
                                output_final_state=True, use_qk_l2norm_in_kernel=True)
        S.copy_(S_new)
        if not contigu:
            for i, st in enumerate(statics[:b]):
                st["conv"].copy_(conv_state[i]); st["S"].copy_(S[i:i + 1])
        y = self._norm_gated(core.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(b, self.value_dim)
        return self.out_proj(y.to(h.dtype))
