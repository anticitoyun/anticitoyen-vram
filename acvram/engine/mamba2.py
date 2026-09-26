"""Mamba2 (SSD) — les couches récurrentes des hybrides Nemotron-H.

Sémantique de la référence transformers (NemotronHMamba2Mixer, chemin
torch) et du graphe llama.cpp (mamba-base.cpp) :
``in_proj`` -> [z | xBC | dt] ; conv causale depthwise (+biais, SiLU) sur
xBC ; x, B, C par groupes ; ``dt = softplus(dt + dt_bias)`` borné en bas ;
récurrence ``h ← exp(dt·A) h + dt·x ⊗ B`` par tête, ``y = C·h + D·x`` ;
norme RMS groupée après ``y · silu(z)`` ; ``out_proj``. Le GGUF porte déjà
``A = −exp(A_log)``. Prefill par le noyau ``chunk_simple_gla`` de fla (SSD =
attention linéaire à décroissance scalaire par tête), décodage récurrent.
État : ``(conv [conv_dim, K−1], h [H, N, P])`` en float32.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lot_etats import nouveau_static, redistribuer, tranches

__all__ = ["Mamba2Mixer"]


def _chunk_gla():
    import os
    if os.environ.get("ACVRAM_MAMBA_CHUNK", "1") == "0":
        return None
    try:
        from fla.ops.simple_gla import chunk_simple_gla
        return chunk_simple_gla
    except ImportError:
        return None


def _recurrent_gla():
    """Le noyau récurrent de fla pour le décodage du LOT (17/09) — même
    interrupteur que le prefill (`ACVRAM_MAMBA_CHUNK=0` : tout torch) ; sous
    l'interpréteur Triton (tests sans carte) il tourne aussi."""
    import os
    if os.environ.get("ACVRAM_MAMBA_CHUNK", "1") == "0":
        return None
    try:
        from fla.ops.simple_gla import fused_recurrent_simple_gla
        return fused_recurrent_simple_gla
    except ImportError:
        return None


def _interprete() -> bool:
    import os
    return os.environ.get("TRITON_INTERPRET") == "1"


class Mamba2Mixer(nn.Module):
    def __init__(self, in_proj: nn.Module, out_proj: nn.Module,
                 conv_weight: torch.Tensor, conv_bias: Optional[torch.Tensor],
                 dt_bias: torch.Tensor, A: torch.Tensor, D: torch.Tensor,
                 norm_weight: torch.Tensor,
                 num_heads: int, head_dim: int, n_groups: int, state_size: int,
                 eps: float = 1e-5, dt_min: float = 0.001) -> None:
        super().__init__()
        self.in_proj, self.out_proj = in_proj, out_proj
        f = lambda t: nn.Parameter(t.to(torch.float32).reshape(-1) if t.dim() > 1 and t.shape[-1] == 1 else t.to(torch.float32), requires_grad=False)
        self.conv_w = nn.Parameter(conv_weight.to(torch.float32), requires_grad=False)   # [conv_dim, K]
        self.conv_b = nn.Parameter(conv_bias.to(torch.float32), requires_grad=False) if conv_bias is not None else None
        self.dt_bias = f(dt_bias)          # [H]
        self.A = f(A)                      # [H], négatif
        self.D = f(D)                      # [H]
        self.norm_w = nn.Parameter(norm_weight.to(torch.float32).reshape(-1), requires_grad=False)  # [inner]
        self.H, self.P, self.G, self.N = num_heads, head_dim, n_groups, state_size
        self.inner = num_heads * head_dim
        self.conv_dim = self.inner + 2 * n_groups * state_size
        self.kernel = conv_weight.shape[-1]
        self.eps, self.dt_min = eps, dt_min

    def _conv(self, xBC: torch.Tensor, state: Optional[torch.Tensor]):
        seq = xBC.t().unsqueeze(0)                              # [1, conv_dim, t]
        seq = torch.cat([state.unsqueeze(0), seq], -1) if state is not None \
            else F.pad(seq, (self.kernel - 1, 0))
        new_state = seq[0, :, -(self.kernel - 1):].detach().clone()
        y = F.conv1d(seq, self.conv_w.unsqueeze(1), self.conv_b, groups=self.conv_dim)[0].t()
        return F.silu(y), new_state

    def _norm(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = y * F.silu(z)
        g = y.reshape(*y.shape[:-1], self.G, self.inner // self.G)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps)
        return g.reshape(y.shape) * self.norm_w

    def forward(self, x: torch.Tensor, state: Optional[tuple] = None
                ) -> tuple[torch.Tensor, tuple]:
        t = x.shape[0]
        conv_prev, h_prev = state if state is not None else (None, None)
        proj = self.in_proj(x).to(torch.float32)
        z, xBC, dt = proj.split([self.inner, self.conv_dim, self.H], dim=-1)
        xBC, conv_new = self._conv(xBC, conv_prev)
        xs, B, C = xBC.split([self.inner, self.G * self.N, self.G * self.N], dim=-1)
        dt = F.softplus(dt + self.dt_bias).clamp(min=self.dt_min)        # [t, H]
        xs = xs.reshape(t, self.H, self.P)
        B = B.reshape(t, self.G, self.N).repeat_interleave(self.H // self.G, dim=1)
        C = C.reshape(t, self.G, self.N).repeat_interleave(self.H // self.G, dim=1)
        h = h_prev if h_prev is not None else torch.zeros(
            self.H, self.N, self.P, dtype=torch.float32, device=x.device)
        chunk = _chunk_gla() if (t > 1 and x.is_cuda) else None
        if chunk is not None:
            o, h_fin = chunk(C.unsqueeze(0), B.unsqueeze(0),
                             (xs * dt.unsqueeze(-1)).unsqueeze(0),
                             g=(self.A * dt).unsqueeze(0), scale=1.0,
                             initial_state=h.unsqueeze(0), output_final_state=True)
            y = o[0].to(torch.float32); h = h_fin[0].to(torch.float32)
        else:
            ys = torch.empty(t, self.H, self.P, dtype=torch.float32, device=x.device)
            for i in range(t):
                h = h * torch.exp(self.A * dt[i]).view(self.H, 1, 1) \
                    + B[i].unsqueeze(-1) * (dt[i].view(self.H, 1) * xs[i]).unsqueeze(-2)
                ys[i] = torch.einsum('hnp,hn->hp', h, C[i])
            y = ys
        y = (y + xs * self.D.view(1, self.H, 1)).reshape(t, self.inner)
        y = self._norm(y, z)
        return self.out_proj(y.to(x.dtype)), (conv_new, h)

    # -- chemin à formes fixes (graphes CUDA) --------------------------------
    def new_static(self, device: torch.device) -> dict:
        return nouveau_static(self, device, {"conv": (self.conv_dim, self.kernel - 1),
                                             "h": (self.H, self.N, self.P)})

    @staticmethod
    def static_load(st: dict, etat) -> None:
        if etat is None:
            st["conv"].zero_(); st["h"].zero_()
        else:
            st["conv"].copy_(etat[0]); st["h"].copy_(etat[1])

    @staticmethod
    def static_export(st: dict) -> tuple:
        return (st["conv"].clone(), st["h"].clone())

    def decode_static(self, x: torch.Tensor, st: dict) -> torch.Tensor:
        y, (conv, h) = self.forward(x, (st["conv"], st["h"]))
        st["conv"].copy_(conv); st["h"].copy_(h)
        return y

    # -- décodage du lot en un lancement (fla, 17/09) ------------------------
    def peut_batcher_decode(self, h: torch.Tensor) -> bool:
        return _recurrent_gla() is not None and (h.is_cuda or _interprete())

    def _lot_projete(self, x: torch.Tensor, conv_state: torch.Tensor):
        """``b`` jetons, un par séquence ; ``conv_state`` [b, conv_dim, k-1]
        mis à jour EN PLACE. Rend (C, B, v = xs·dt, g = A·dt, xs, z) aux formes de fla."""
        b = x.shape[0]
        proj = self.in_proj(x).to(torch.float32)
        z, xBC, dt = proj.split([self.inner, self.conv_dim, self.H], dim=-1)
        seq = torch.cat([conv_state, xBC.unsqueeze(-1)], dim=-1)          # [b, conv_dim, k]
        conv_state.copy_(seq[:, :, 1:])
        xBC = F.silu(F.conv1d(seq, self.conv_w.unsqueeze(1), self.conv_b, groups=self.conv_dim))[:, :, 0]
        xs, B, C = xBC.split([self.inner, self.G * self.N, self.G * self.N], dim=-1)
        dt = F.softplus(dt + self.dt_bias).clamp(min=self.dt_min)         # [b, H]
        xs = xs.reshape(b, 1, self.H, self.P)
        B = B.reshape(b, 1, self.G, self.N).repeat_interleave(self.H // self.G, dim=2)
        C = C.reshape(b, 1, self.G, self.N).repeat_interleave(self.H // self.G, dim=2)
        v = xs * dt.reshape(b, 1, self.H, 1)
        g = (self.A * dt).reshape(b, 1, self.H)
        return C.contiguous(), B.contiguous(), v.contiguous(), g.contiguous(), xs, z

    def _lot_sortie(self, o, xs, z, x_dtype):
        b = o.shape[0]
        y = (o[:, 0].to(torch.float32) + xs[:, 0] * self.D.view(1, self.H, 1)).reshape(b, self.inner)
        return self.out_proj(self._norm(y, z).to(x_dtype))

    def forward_batch(self, h: torch.Tensor, etats: list) -> tuple[torch.Tensor, list]:
        b = h.shape[0]
        z_ = lambda *f: torch.zeros(*f, dtype=torch.float32, device=h.device)
        conv_state = torch.stack([e[0] if e is not None else z_(self.conv_dim, self.kernel - 1) for e in etats])
        hs = torch.stack([e[1] if e is not None else z_(self.H, self.N, self.P) for e in etats])
        C, B, v, g, xs, z = self._lot_projete(h, conv_state)
        o, h_new = _recurrent_gla()(C, B, v, g=g, scale=1.0, initial_state=hs.contiguous(),
                                    output_final_state=True)
        y = self._lot_sortie(o, xs, z, h.dtype)
        return y, [(conv_state[i].clone(), h_new[i].to(torch.float32)) for i in range(b)]

    def decode_static_batch(self, h: torch.Tensor, statics: list) -> torch.Tensor:
        b = h.shape[0]
        g_, contigu = tranches(self, statics, b, ("conv", "h"))
        conv_state, hs = g_["conv"], g_["h"]
        C, B, v, g, xs, z = self._lot_projete(h, conv_state)
        o, h_new = _recurrent_gla()(C, B, v, g=g, scale=1.0, initial_state=hs.contiguous(),
                                    output_final_state=True)
        hs.copy_(h_new)
        if not contigu:
            redistribuer(statics, b, {"conv": conv_state, "h": hs})
        return self._lot_sortie(o, xs, z, h.dtype)
