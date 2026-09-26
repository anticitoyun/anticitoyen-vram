"""KDA — Kimi Delta Attention (architecture GGUF « kimi-linear »).

Traduction op par op du graphe de llama.cpp (src/models/kimi-linear.cpp +
delta-net-base.cpp), notre seul exécuteur de référence local pour ce format.
Conventions du convertisseur, déjà appliquées dans le GGUF :
- ``ssm_a`` stocke **−exp(A_log)** par tête (comme qwen35) — utilisé tel quel
  ici, car la formule llama.cpp est ``g1 = ssm_a · softplus(f_b(f_a(x)) + dt)``.
- Convolutions causales séparées pour q, k, v (noyau 4, SiLU), chacune avec
  son état de ``kernel−1`` colonnes.
- q et k L2-normalisés par tête ; q mis à l'échelle 1/√d.
- Récurrence delta par canal : ``S ← S ⊙ exp(g1)`` sur l'**axe clé** (la
  convention fla/vLLM — tranché par bissection contre llama.cpp : l'autre axe
  dérive en boucles après ~7 jetons), puis ``S += β(v−S·k)⊗k ; o = S·q``.
- Sortie : RMSNorm(ssm_norm) par tête × sigmoid(g_b(g_a(x))), puis out_proj.

L'état d'une séquence : ``(conv_q, conv_k, conv_v [d_inner, kernel−1],
S [heads, d, d] float32)`` — porté par le moteur, hors cache paginé.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lot_etats import nouveau_static, redistribuer, tranches

__all__ = ["KimiDeltaAttention"]


def _chunk_kda():
    """Le noyau par blocs de fla pour le prefill, si disponible et permis."""
    import os
    if os.environ.get("ACVRAM_KDA_CHUNK", "1") == "0":
        return None
    try:
        from fla.ops.kda import chunk_kda
        return chunk_kda
    except ImportError:
        return None


def _recurrent_kda():
    """Le noyau récurrent de fla pour le décodage du LOT (17/09) — même
    interrupteur que le prefill (`ACVRAM_KDA_CHUNK=0` : tout torch)."""
    import os
    if os.environ.get("ACVRAM_KDA_CHUNK", "1") == "0":
        return None
    try:
        from fla.ops.kda import fused_recurrent_kda
        return fused_recurrent_kda
    except ImportError:
        return None


def _interprete() -> bool:
    import os
    return os.environ.get("TRITON_INTERPRET") == "1"


def _extension():
    """L'extension CUDA si elle porte le noyau KDA (et si on ne l'a pas coupée)."""
    import os
    if os.environ.get("ACVRAM_HYBRID_KERNELS", "1") == "0":
        return None
    from .. import kernels
    ext = kernels.get_extension()
    return ext if ext is not None and hasattr(ext, "kda_decode") else None


class KimiDeltaAttention(nn.Module):
    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module,
                 out_proj: nn.Module,
                 f_a: nn.Module, f_b: nn.Module,
                 g_a: nn.Module, g_b: nn.Module,
                 beta: nn.Module,
                 conv_q: torch.Tensor, conv_k: torch.Tensor,  # [d_inner, kernel]
                 conv_v: torch.Tensor,
                 dt_bias: torch.Tensor,                       # [d_inner]
                 a: torch.Tensor,                             # [heads] = -exp(A_log)
                 norm_weight: torch.Tensor,                   # [head_dim]
                 num_heads: int, head_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = q_proj, k_proj, v_proj
        self.out_proj = out_proj
        self.f_a, self.f_b, self.g_a, self.g_b = f_a, f_b, g_a, g_b
        self.beta_proj = beta
        self.conv_q = nn.Parameter(conv_q, requires_grad=False)
        self.conv_k = nn.Parameter(conv_k, requires_grad=False)
        self.conv_v = nn.Parameter(conv_v, requires_grad=False)
        self.dt_bias = nn.Parameter(dt_bias.float(), requires_grad=False)
        self.a = nn.Parameter(a.float().reshape(-1), requires_grad=False)
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.nh, self.d = num_heads, head_dim
        self.d_inner = num_heads * head_dim
        self.kernel = conv_q.shape[-1]
        self.eps = eps

    # -- empilement des projections (moins de lancements au décodage) -------
    def fuse_projections(self) -> bool:
        """Empile q/k/v et f_a/g_a en deux QuantLinear quand leurs poids sont
        des INT8Tensor de même géométrie : 9 GEMV -> 6 par pas."""
        from .layers import stack_int8_linears as pile

        self.qkv_proj = pile([self.q_proj, self.k_proj, self.v_proj])
        self.fa_ga = pile([self.f_a, self.g_a])      # même entrée x : empilables
        return self.qkv_proj is not None

    def _conv(self, x: torch.Tensor, w: torch.Tensor,
              state: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Conv1d causale depthwise + SiLU ; rend (sortie [t, d_inner], état)."""
        seq = x.t().unsqueeze(0)                              # [1, d_inner, t]
        if state is not None:
            seq = torch.cat([state.unsqueeze(0), seq], dim=-1)
        else:
            seq = F.pad(seq, (self.kernel - 1, 0))
        new_state = seq[0, :, -(self.kernel - 1):].detach().clone()
        y = F.conv1d(seq, w.unsqueeze(1), groups=self.d_inner)
        return F.silu(y[0].t()), new_state                    # [t, d_inner]

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor,
                state: Optional[tuple] = None
                ) -> tuple[torch.Tensor, tuple]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, nouvel état)."""
        t = x.shape[0]
        cq = ck = cv = None
        s_prev = None
        if state is not None:
            cq, ck, cv, s_prev = state

        q, new_cq = self._conv(self.q_proj(x).to(torch.float32), self.conv_q.float(), cq)
        k, new_ck = self._conv(self.k_proj(x).to(torch.float32), self.conv_k.float(), ck)
        v, new_cv = self._conv(self.v_proj(x).to(torch.float32), self.conv_v.float(), cv)

        q_raw = q.reshape(t, self.nh, self.d)
        k_raw = k.reshape(t, self.nh, self.d)
        q = self._l2norm(q_raw)
        k = self._l2norm(k_raw)
        v = v.reshape(t, self.nh, self.d)
        q = q * (self.d ** -0.5)

        g1 = F.softplus(self.f_b(self.f_a(x)).to(torch.float32) + self.dt_bias)
        g1 = g1.reshape(t, self.nh, self.d) * self.a.reshape(1, self.nh, 1)
        beta = torch.sigmoid(self.beta_proj(x).to(torch.float32))   # [t, nh]

        S = s_prev if s_prev is not None else torch.zeros(
            self.nh, self.d, self.d, dtype=torch.float32, device=x.device)
        chunk = _chunk_kda() if (t > 1 and x.is_cuda) else None
        if chunk is not None:
            # prefill : noyau Triton par blocs de fla (même mathématique que
            # la boucle ; l'état fla est [K, V], le nôtre [V, K])
            core, S_fin = chunk(
                q_raw.unsqueeze(0), k_raw.unsqueeze(0), v.unsqueeze(0),
                g=g1.unsqueeze(0), beta=beta.unsqueeze(0),
                initial_state=S.transpose(-1, -2).contiguous().unsqueeze(0),
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            sorties = core[0].to(torch.float32)
            S = S_fin[0].transpose(-1, -2).contiguous().to(torch.float32)
        else:
          sorties = torch.empty(t, self.nh, self.d,
                                dtype=torch.float32, device=x.device)
          for i in range(t):
              S = S * torch.exp(g1[i]).unsqueeze(-2)          # décroissance (axe clé)
              pred = torch.einsum('hij,hj->hi', S, k[i])
              d = beta[i].unsqueeze(-1) * (v[i] - pred)
              S = S + d.unsqueeze(-1) * k[i].unsqueeze(-2)
              sorties[i] = torch.einsum('hij,hj->hi', S, q[i])

        # RMSNorm par tête × porte sigmoïde g2
        var = sorties.pow(2).mean(-1, keepdim=True)
        normed = sorties * torch.rsqrt(var + self.eps) \
            * self.norm_weight.to(torch.float32)
        g2 = self.g_b(self.g_a(x)).to(torch.float32).reshape(t, self.nh, self.d)
        y = (normed * torch.sigmoid(g2)).reshape(t, self.d_inner)
        return (self.out_proj(y.to(x.dtype)),
                (new_cq, new_ck, new_cv, S))

    # -- chemin à formes fixes (graphes CUDA) --------------------------------
    def new_static(self, device: torch.device) -> dict:
        k1 = self.kernel - 1
        return nouveau_static(self, device, {"cq": (self.d_inner, k1), "ck": (self.d_inner, k1),
                                             "cv": (self.d_inner, k1), "S": (self.nh, self.d, self.d)})

    @staticmethod
    def static_load(st: dict, etat) -> None:
        if etat is None:
            for v in st.values():
                v.zero_()
            return
        cq, ck, cv, S = etat
        st["cq"].copy_(cq); st["ck"].copy_(ck); st["cv"].copy_(cv)
        st["S"].copy_(S)

    @staticmethod
    def static_export(st: dict) -> tuple:
        return (st["cq"].clone(), st["ck"].clone(), st["cv"].clone(),
                st["S"].clone())

    # -- décodage du lot en un lancement (fla, 17/09) ------------------------
    def peut_batcher_decode(self, h: torch.Tensor) -> bool:
        return _recurrent_kda() is not None and (h.is_cuda or _interprete())

    def _lot_projete(self, x: torch.Tensor, cq: torch.Tensor, ck: torch.Tensor, cv: torch.Tensor):
        """``b`` jetons ; les trois états de convolution [b, d_inner, k-1] mis
        à jour EN PLACE. Rend (q_raw, k_raw, v, g1, beta, g2) aux formes de fla
        (q/k bruts : la L2-norme et l'échelle 1/√d sont dans le noyau)."""
        b = x.shape[0]

        def conv(xp, w, buf):
            seq = torch.cat([buf, xp.unsqueeze(-1)], dim=-1)               # [b, d_inner, k]
            buf.copy_(seq[:, :, 1:])
            return F.silu(F.conv1d(seq, w.unsqueeze(1), groups=self.d_inner)[:, :, 0])
        q = conv(self.q_proj(x).to(torch.float32), self.conv_q.float(), cq)
        k = conv(self.k_proj(x).to(torch.float32), self.conv_k.float(), ck)
        v = conv(self.v_proj(x).to(torch.float32), self.conv_v.float(), cv)
        g1 = F.softplus(self.f_b(self.f_a(x)).to(torch.float32) + self.dt_bias)
        g1 = g1.reshape(b, 1, self.nh, self.d) * self.a.reshape(1, 1, self.nh, 1)
        beta = torch.sigmoid(self.beta_proj(x).to(torch.float32)).reshape(b, 1, self.nh)
        g2 = self.g_b(self.g_a(x)).to(torch.float32).reshape(b, self.nh, self.d)
        forme = (b, 1, self.nh, self.d)
        return (q.reshape(forme).contiguous(), k.reshape(forme).contiguous(), v.reshape(forme).contiguous(),
                g1.contiguous(), beta.contiguous(), g2)

    def _lot_sortie(self, o, g2, x_dtype):
        b = o.shape[0]
        o = o[:, 0].to(torch.float32)                                        # [b, nh, d]
        var = o.pow(2).mean(-1, keepdim=True)
        normed = o * torch.rsqrt(var + self.eps) * self.norm_weight.to(torch.float32)
        y = (normed * torch.sigmoid(g2)).reshape(b, self.d_inner)
        return self.out_proj(y.to(x_dtype))

    def forward_batch(self, h: torch.Tensor, etats: list) -> tuple[torch.Tensor, list]:
        b = h.shape[0]
        z_ = lambda *f: torch.zeros(*f, dtype=torch.float32, device=h.device)
        k1 = self.kernel - 1
        cq, ck, cv, S = (torch.stack([e[j] if e is not None else z_(*forme) for e in etats])
                         for j, forme in ((0, (self.d_inner, k1)), (1, (self.d_inner, k1)),
                                          (2, (self.d_inner, k1)), (3, (self.nh, self.d, self.d))))
        q, k, v, g1, beta, g2 = self._lot_projete(h, cq, ck, cv)
        # l'état fla est [K, V], le nôtre [V, K] (docstring du module)
        o, S_fin = _recurrent_kda()(q, k, v, g=g1, beta=beta, initial_state=S.transpose(-1, -2).contiguous(),
                                    output_final_state=True, use_qk_l2norm_in_kernel=True)
        S_new = S_fin.transpose(-1, -2).to(torch.float32)
        y = self._lot_sortie(o, g2, h.dtype)
        return y, [(cq[i].clone(), ck[i].clone(), cv[i].clone(), S_new[i].contiguous()) for i in range(b)]

    def decode_static_batch(self, h: torch.Tensor, statics: list) -> torch.Tensor:
        b = h.shape[0]
        g_, contigu = tranches(self, statics, b, ("cq", "ck", "cv", "S"))
        q, k, v, g1, beta, g2 = self._lot_projete(h, g_["cq"], g_["ck"], g_["cv"])
        o, S_fin = _recurrent_kda()(q, k, v, g=g1, beta=beta,
                                    initial_state=g_["S"].transpose(-1, -2).contiguous(),
                                    output_final_state=True, use_qk_l2norm_in_kernel=True)
        g_["S"].copy_(S_fin.transpose(-1, -2))
        if not contigu:
            redistribuer(statics, b, g_)
        return self._lot_sortie(o, g2, h.dtype)

    def decode_static(self, x: torch.Tensor, st: dict) -> torch.Tensor:
        """Un jeton, une séquence, états mis à jour EN PLACE dans ``st``."""
        ext = _extension() if x.is_cuda else None
        if ext is not None:
            # noyau fusionné : convs, normalisations, portes, récurrence,
            # norme de sortie — un seul lancement après les projections
            bf = torch.bfloat16
            if getattr(self, "qkv_proj", None) is not None:
                xqkv = self.qkv_proj(x).to(bf).reshape(-1)
                xq, xk, xv = xqkv.split(self.d_inner)
                xq, xk, xv = xq.contiguous(), xk.contiguous(), xv.contiguous()
            else:
                xq = self.q_proj(x).to(bf).reshape(-1)
                xk = self.k_proj(x).to(bf).reshape(-1)
                xv = self.v_proj(x).to(bf).reshape(-1)
            if getattr(self, "fa_ga", None) is not None:
                low = self.fa_ga(x)                       # [1, 2*r]
                r = low.shape[-1] // 2
                # f_b ne voit que f_a(x), g_b que g_a(x) : deux GEMV sur les
                # moitiés (l'empilement [f_b;g_b] exigerait la même entrée)
                g1_pre = self.f_b(low[:, :r]).to(bf).reshape(-1)
                g2 = self.g_b(low[:, r:]).to(bf).reshape(-1)
            else:
                g1_pre = self.f_b(self.f_a(x)).to(bf).reshape(-1)
                g2 = self.g_b(self.g_a(x)).to(bf).reshape(-1)
            beta_pre = self.beta_proj(x).to(bf).reshape(-1)
            y = ext.kda_decode(xq, xk, xv, g1_pre, g2, beta_pre,
                               self.conv_q, self.conv_k, self.conv_v,
                               st["cq"], st["ck"], st["cv"],
                               self.dt_bias, self.a, self.norm_weight,
                               st["S"], self.eps)
            return self.out_proj(y.to(x.dtype))
        def conv(xp, w, buf):
            # mêmes noyaux que le chemin fonctionnel (cuDNN, même forme) :
            # les deux chemins doivent arrondir pareil
            seq = torch.cat([buf, xp.t()], dim=-1)          # [d_inner, k]
            y = F.conv1d(seq.unsqueeze(0), w.unsqueeze(1), groups=self.d_inner)
            buf.copy_(seq[:, 1:])
            return F.silu(y[0, :, 0])
        q = conv(self.q_proj(x).float(), self.conv_q, st["cq"])
        k = conv(self.k_proj(x).float(), self.conv_k, st["ck"])
        v = conv(self.v_proj(x).float(), self.conv_v, st["cv"])
        q = self._l2norm(q.view(self.nh, self.d)) * (self.d ** -0.5)
        k = self._l2norm(k.view(self.nh, self.d))
        v = v.view(self.nh, self.d)
        g1 = F.softplus(self.f_b(self.f_a(x)).float()[0] + self.dt_bias)
        g1 = g1.view(self.nh, self.d) * self.a.view(self.nh, 1)
        beta = torch.sigmoid(self.beta_proj(x).float()[0])   # [nh]
        S = st["S"]
        S.mul_(torch.exp(g1).unsqueeze(-2))                   # axe clé
        pred = torch.einsum('hij,hj->hi', S, k)
        d = beta.unsqueeze(-1) * (v - pred)
        S.add_(d.unsqueeze(-1) * k.unsqueeze(-2))          # mul puis add, comme forward
        o = torch.einsum('hij,hj->hi', S, q)
        var = o.pow(2).mean(-1, keepdim=True)
        normed = o * torch.rsqrt(var + self.eps) * self.norm_weight.float()
        g2 = self.g_b(self.g_a(x)).float()[0].view(self.nh, self.d)
        y = (normed * torch.sigmoid(g2)).reshape(1, self.d_inner)
        return self.out_proj(y.to(x.dtype))
