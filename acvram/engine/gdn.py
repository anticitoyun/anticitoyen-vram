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
# b=12 : 97 j/s pour 621 chez vLLM, poste7-priorite-apres-campagne-17-09).
# « 1 » vaut fla, « 0 » reste le refus des hybrides (quant/gguf.py).
_GDN_VOIE = os.environ.get("ACVRAM_GDN", "fla")
# Pièce 156 F4 (défaut depuis le verdict 156 c, au bit ; 0 = témoin) : au décodage du lot, la récurrence fla écrit son état
# final DANS le tampon statique au lieu d'en allouer un puis de le recopier
# (25 Mo par couche à b=8 sur Qwen3.8). Au bit : voir `_recurrence_en_place`.
_GDN_ETAT_EN_PLACE = os.environ.get("ACVRAM_GDN_ETAT_EN_PLACE", "1") == "1"
# Pièce 156 F2 (défaut depuis le verdict 156 c, au bit ; 0 = témoin) : conv du décodage du lot en un noyau Triton, visé au
# bit (`gdn_conv.py`) ; q/k sans répétition des têtes (fla les indexe).
_GDN_CONV_FUSEE = os.environ.get("ACVRAM_GDN_CONV_FUSEE", "1") == "1"
# Pièce 175 : les portes α et β (poids bf16 [nv, K] de l'alias mixte, `PlainTensor`) en UN appel par couche —
# auto (DÉFAUT depuis 175 b, chef 25/09 : M = 1 concat, 2 ≤ M ≤ 8 triton, au-delà les deux appels — AU BIT des deux F.linear à
# chaque M, test_gdn_ab_175 ; b=8 mixte −2,3 ms/pas, −12 %) | separe (témoin : deux F.linear, 31 µs chacun à b=8, 173) | concat
# (un F.linear sur β‖α [2nv, K] : au bit à M = 1 seulement) | triton (GEMM étroite fp32 déterministe, kernels/gemv_bf16_etroit.py).
AB_DEFAUT = "auto"
_GDN_AB = os.environ.get("ACVRAM_GDN_AB", AB_DEFAUT)
AB_BILAN = {"fusionnees": 0, "raisons": {}}          # 175 : ce que le chargement a fait, pour la ligne de régime (preuve)
# Pièce 194 (b2) : β‖α (3 programmes, 9-14 µs, jusqu'ici sur le chemin critique) lancé sur un SECOND flux pendant la pile
# qkv‖gate (1 024 programmes, 60 µs) qui lit la même entrée ; jointure avant de rendre, la récurrence les consomme. Au bit
# par construction (mêmes noyaux, mêmes entrées). Banc : −12,15 µs/couche, 0,58 ms/pas à b=8 (poste1-194-b2-banc-25-09).
# DÉFAUT depuis le verdict 194 b2 (chef 25/09 : servi mixte b=8 +2,20 %, J/jeton −1,9 %, b=1 +0,46 %, témoins nvfp4 nuls,
# capture 4/4) ; 0 = témoin (série).
_GDN_AB_FLUX = os.environ.get("ACVRAM_GDN_AB_FLUX", "1") == "1"
_FLUX_AB: dict = {}


def _flux_ab(device) -> "torch.cuda.Stream":
    """Un second flux par carte, créé au premier appel (l'échauffement eager précède toute capture)."""
    f = _FLUX_AB.get(device.index)
    if f is None:
        f = _FLUX_AB[device.index] = torch.cuda.Stream(device)
    return f


def _joindre(courant, flux) -> None:
    """Le flux principal attend β‖α avant tout consommateur. Isolé pour que test_gdn_ab_flux_194 prouve qu'il casse sans.
    Pas de record_stream : chaque fourche attend le flux principal, donc un bloc de β‖α libéré n'est réutilisé sur le
    second flux qu'après ses consommateurs, et un bloc de x réutilisé sur le flux principal l'est après la jointure."""
    courant.wait_stream(flux)


def ab_bilan_reinit() -> None:
    """175 b : le bilan est celui du DERNIER chargement — le chargeur l'appelle avant sa passe de fusion, sinon il s'accumule
    entre les modèles d'un même processus (vu dans une suite pytest : « ab=auto(28:…) » sur un modèle chargé à sec)."""
    AB_BILAN["fusionnees"] = 0
    AB_BILAN["raisons"].clear()
# Pièce 156 F1 (DÉFAUT depuis 156 d ; 0 = témoin) : portes (softplus, exp, sigmoid) calculées dans le noyau fla de
# la voie F4 — ± ulp fp32 (softplus de fla en ex2/lg2 approchés) : KL contre témoins tenue sur Qwen3.8 et Qwen3.5-35B.
_GDN_PORTES_NOYAU = os.environ.get("ACVRAM_GDN_PORTES_NOYAU", "1") == "1"
# Pièce 156 F3 (DÉFAUT depuis 156 d ; 0 = témoin) : norme gated en un noyau Triton (gdn_norme.py) — ± ulp (ordre de
# la somme des carrés) ; même KL.
_GDN_NORME_FUSEE = os.environ.get("ACVRAM_GDN_NORME_FUSEE", "1") == "1"
# Pièce 182 (1) : z (porte de la norme) rendu par `_projections` dans le dtype de la projection, sans cast fp32 — ses
# consommateurs castent au chargement (exact) ; au décodage, 48 copies de moins par pas sur Qwen3.8. 0 = témoin (cast).
_GDN_Z_BF16 = os.environ.get("ACVRAM_GDN_Z_BF16", "1") == "1"


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
    return "fla" + _ab_texte()


def _ab_texte() -> str:
    """175 : ` ab=concat(48)` (couches fusionnées) ; ` ab=triton(0:scaler=…)` = demandé mais pas pris, raison nommée."""
    if _GDN_AB == "separe":
        return ""
    raisons = ",".join(f"{k}×{v}" for k, v in sorted(AB_BILAN["raisons"].items()))
    return (f" ab={_GDN_AB}({AB_BILAN['fusionnees']}" + (f":{raisons}" if raisons else "") + ")"
            + (" abflux" if _GDN_AB_FLUX and AB_BILAN["fusionnees"] else ""))   # 194 : seulement si β‖α existe (inerte sinon)


def _recurrence_en_place(q, k, v, g, beta, S: torch.Tensor, A_log=None, dt_bias=None) -> torch.Tensor:
    """`fused_recurrent_gated_delta_rule` (fla 0.5.2, `use_qk_l2norm_in_kernel`, échelle K^-1/2) avec ``h0 = ht =
    S`` : même noyau, mêmes arguments que `fused_recurrent_gated_delta_rule_fwd`, seule l'adresse de l'état final
    change. Au bit parce que `p_h0` et `p_ht` ont la même indexation et que chaque programme de la grille
    (NV, N·HV) lit sa tuile [K, BV] une fois AVANT la boucle et l'écrit une fois APRÈS, sans lire celle d'un autre
    (fused_recurrent.py, chargement de h0 puis `tl.store(p_ht, …)`). Rend la sortie ``o`` ; ``S`` est mis à jour.
    F1 : avec ``A_log`` (et ``dt_bias``), ``g`` et ``beta`` sont les projections BRUTES ; le noyau fait
    g = −exp(A_log)·softplus(g + dt_bias) et beta = sigmoid(beta) (fused_recurrent.py:124-135)."""
    import triton
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as noyau
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BK, BV = triton.next_power_of_2(K), min(8, triton.next_power_of_2(V))
    o = torch.empty_like(v)
    with torch.cuda.device(q.device.index):
        noyau[(triton.cdiv(V, BV), B * HV)](
            q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=A_log, dt_bias=dt_bias, o=o, h0=S, ht=S,
            cu_seqlens=None, scale=K ** -0.5, T=T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
            IS_BETA_HEADWISE=beta.ndim != v.ndim, USE_QK_L2NORM_IN_KERNEL=True, APPLY_BETA_SIGMOID=A_log is not None,
            ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False, num_warps=1, num_stages=3)
    return o


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
        # 175 : β‖α fusionnés (QuantLinear) ou None — attribut d'INSTANCE : un `ab = None` de classe masquait le sous-module
        # (nn.Module range les modules dans `_modules`, lus par __getattr__ seulement si la recherche normale échoue) et le
        # chemin fusionné n'était jamais pris (prises 1-2 de la 175 : aucun effet, compteur à 48 pourtant).
        self.ab = None

    # -- quatre projections de même entrée en un lancement (palier 2) -----
    def fuse(self) -> bool:
        """qkv, gate, α, β lisent x : une multi-projection NVFP4 (chacune
        avec son scaler) à 2 ≤ b ≤ 32 sous ACVRAM_DENSE_NVFP4=triton."""
        from .model import _multi_projection
        self.multi = _multi_projection([self.qkv, self.gate, self.alpha, self.beta_proj])
        self.ab = self._fusionner_ab()                                  # pièce 175 : β‖α en un appel (alias mixte)
        # Pièce 176 : qkv et gate INT8 lisent la même entrée — UNE pile (vues, aucune copie), servie au décodage
        # (M ≤ 16) en un appel au lieu de deux ; `_segments` garde à chaque segment sa partition K (au bit des deux
        # appels, gemm_etroit._etroit_segments_kernel). Préfill : les vues, appels séparés (cuBLAS choisit selon N).
        if os.environ.get("ACVRAM_GDN_QKV_GATE", "1") != "0":
            from .layers import stack_int8_linears
            n_qkv, n_gate = self.qkv.qweight.shape[0], self.gate.qweight.shape[0]
            pile = stack_int8_linears([self.qkv, self.gate])
            if pile is not None:
                pile.qweight._segments = (n_qkv, n_gate)
                self.qkv_gate = pile
        return self.multi is not None or getattr(self, "qkv_gate", None) is not None

    def _fusionner_ab(self):
        """Pièce 175 : β‖α en un `QuantLinear` bf16 [2nv, K] quand les deux sont des `PlainTensor` sur la carte
        (alias mixte) et que ACVRAM_GDN_AB ≠ separe. Les poids nvfp4 (défaut) ne sont pas concernés : None."""
        from ..quant.formats import PlainTensor
        from .layers import QuantLinear
        if _GDN_AB not in ("concat", "triton", "auto"):
            if _GDN_AB != "separe":
                raise ValueError(f"ACVRAM_GDN_AB={_GDN_AB!r} : attendu auto | separe | concat | triton")
            return None
        wb, wa = getattr(self.beta_proj, "qweight", None), getattr(self.alpha, "qweight", None)
        identite = lambda lin: getattr(lin, "scaler", None) is None or getattr(lin.scaler, "is_identity", False)  # noqa: E731
        raison = ("format" if not (isinstance(wb, PlainTensor) and isinstance(wa, PlainTensor)) else
                  "hote" if not wb.weight.is_cuda else
                  "dtype" if wb.weight.dtype != torch.bfloat16 or wa.weight.dtype != wb.weight.dtype else
                  "biais" if self.beta_proj.bias is not None or self.alpha.bias is not None else
                  "scaler" if not (identite(self.beta_proj) and identite(self.alpha)) else None)
        if raison:
            AB_BILAN["raisons"][raison] = AB_BILAN["raisons"].get(raison, 0) + 1
            return None
        AB_BILAN["fusionnees"] += 1
        w = torch.cat([wb.weight, wa.weight]).contiguous()
        return QuantLinear(PlainTensor(w, tuple(w.shape), wb.format), None, None, w.shape[0], w.shape[1])

    def _ab(self, x: torch.Tensor, fp32: bool = False):
        """(b, a) par le poids fusionné. `concat` : F.linear sur β‖α (cuBLAS choisit à N = 96 un noyau 128x2 : 3,9 µs à M = 8
        mais HORS bit des deux appels, KL 3,4 × les témoins ; au bit à M = 1). `triton` : GEMM étroite fp32 (8,9 µs à M = 8,
        AU BIT des deux appels cuBLAS wmma 128x1 à M = 8 sur 32 pas × 8 séquences ; plus lente que cuBLAS à M = 1).
        `auto` (candidat au défaut, prises 3-4 de la 175) : M = 1 → concat, 2 ≤ M ≤ 8 → triton, au-delà → les deux appels."""
        m = x.shape[0]
        mode = _GDN_AB
        if mode == "auto":
            # prise 4 : triton == les deux appels au bit à M = 2, 4, 8 ; PLUS à M = 12 et 16 (cuBLAS change de noyau) → 8
            mode = "concat" if m == 1 else "triton" if m <= 8 else "separe"
        if mode == "separe":
            b, a = self.beta_proj(x), self.alpha(x)
            return (b.to(torch.float32), a.to(torch.float32)) if fp32 else (b, a)
        if mode == "triton" and m <= 16:
            from ..kernels.gemv_bf16_etroit import gemv_bf16_etroit
            ba = gemv_bf16_etroit(x, self.ab.qweight.weight, fp32=fp32)   # 175 (poste1, 182) : le cast dans le noyau
        else:
            ba = self.ab(x)
            if fp32:
                ba = ba.to(torch.float32)                                   # UN cast sur β‖α au lieu de deux
        return ba[:, : self.nv], ba[:, self.nv:]

    def _projections(self, x: torch.Tensor, qkv_brut: bool = False):
        """(qkv, z, b, a) en fp32 — un lancement si la multi-projection sert. Pièce 182 : z reste dans le dtype de la
        projection hors multi-projection (bf16), ses consommateurs le castent au chargement (exact).
        ``qkv_brut`` : qkv rendu tel que la projection le sort (F2 le lit)."""
        from .model import _multi_utilisable
        mp = getattr(self, "multi", None)
        if _multi_utilisable(mp, x, x.shape[0]):
            qkv, z, a, b = torch.split(mp(x), mp.tailles, dim=-1)
            return (qkv if qkv_brut else qkv.to(torch.float32)), z.to(torch.float32), \
                b.to(torch.float32), a.to(torch.float32)
        flux = None
        if _GDN_AB_FLUX and self.ab is not None and x.is_cuda and x.shape[0] <= 16:    # pièce 194 (b2)
            flux, courant = _flux_ab(x.device), torch.cuda.current_stream(x.device)
            flux.wait_stream(courant)
            with torch.cuda.stream(flux):
                b, a = self._ab(x, fp32=True)
        pile = getattr(self, "qkv_gate", None)
        if pile is not None and x.shape[0] <= 16:                            # pièce 176 : un appel au décodage
            qkv, z = torch.split(pile(x), pile.qweight._segments, dim=-1)
        else:
            qkv, z = self.qkv(x), self.gate(x)
        if not _GDN_Z_BF16:
            z = z.to(torch.float32)
        if flux is not None:
            _joindre(courant, flux)
            return ((qkv if qkv_brut else qkv.to(torch.float32)), z, b, a)
        if self.ab is not None:                                        # pièce 175 : β‖α en un appel, casts absorbés
            b, a = self._ab(x, fp32=True)
            return ((qkv if qkv_brut else qkv.to(torch.float32)), z, b, a)
        return ((qkv if qkv_brut else qkv.to(torch.float32)), z,
                self.beta_proj(x).to(torch.float32), self.alpha(x).to(torch.float32))

    # -- normalisation gated (RMSNorm de la sortie, porte SiLU(z)) --------
    def _norm_gated(self, x: torch.Tensor, z: torch.Tensor, sortie: Optional[torch.dtype] = None) -> torch.Tensor:
        """``sortie`` = dtype que l'appelant donnera à out_proj : F3 écrit directement du bf16 (le cast fp32 → bf16
        de l'appelant devient sans effet, même arrondi au plus près)."""
        if _GDN_NORME_FUSEE and x.is_cuda and sortie == torch.bfloat16:
            from .gdn_norme import norme_gated
            return norme_gated(x, z, self.norm_weight, self.eps)
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        x32 = x32 * self.norm_weight.to(torch.float32)
        return (x32 * F.silu(z.reshape(x.shape).to(torch.float32))).to(x.dtype)

    def forward(self, x: torch.Tensor,
                state: Optional[tuple] = None
                ) -> tuple[torch.Tensor, tuple]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, nouvel état)."""
        # toute la récurrence se calcule en float32 : la règle delta cumule
        # des produits d'état où le bfloat16 dérive vite
        qkv, z, b, a = self._projections(x)                 # [t, conv_dim], [t, value_dim], [t, nv] × 2
        y, etat = self._coeur(x, qkv, z, b, a, state)
        return self.out_proj(y.to(x.dtype)), etat

    def forward_lot(self, x: torch.Tensor, etats: list,
                    query_lens: list[int]) -> tuple[torch.Tensor, list]:
        """Préfill de plusieurs séquences (pièce 150 bis) : les cinq projections
        en UN appel sur les Σ t lignes, convolution et règle delta par séquence
        (ni la causalité ni l'état ne passent la frontière entre séquences).
        Séquence par séquence, chaque projection tournait à M = t (78 au banc
        chat) : un int8 par canal y prend le GEMV par tranches (≤ 80 lignes),
        un nvfp4 relit son poids b fois. Mêmes lignes, autre M : le noyau
        choisi peut changer, donc pas « au bit » par construction."""
        qkv, z, b, a = self._projections(x)
        ys, etats_new = [], []
        d = 0
        for ql, etat in zip(query_lens, etats):
            f = d + ql
            y, e = self._coeur(x[d:f], qkv[d:f], z[d:f], b[d:f], a[d:f], etat)
            ys.append(y)
            etats_new.append(e)
            d = f
        return self.out_proj(torch.cat(ys).to(x.dtype)), etats_new

    def _coeur(self, x: torch.Tensor, qkv: torch.Tensor, z: torch.Tensor,
               b: torch.Tensor, a: torch.Tensor, state: Optional[tuple]):
        """Convolution, règle delta et norme gated d'UNE séquence, projections
        faites ; rend (y avant out_proj, nouvel état)."""
        chunk_rule, recurrent_rule = _refs()
        t = x.shape[0]
        decode = (t == 1 and state is not None)

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
        y = self._norm_gated(core, z.reshape(-1, self.dv), x.dtype)
        return y.reshape(t, self.value_dim), (new_conv_state, s_new)

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

    def _lot_projete(self, x: torch.Tensor, conv_state: torch.Tensor, portes_brutes: bool = False):
        """Projections, convolution causale à état et portes pour ``b``
        jetons (un par séquence) ; ``conv_state`` [b, conv_dim, k-1] est mis
        à jour EN PLACE. Rend (q, k, v, g, beta, z) aux formes de fla.
        ``portes_brutes`` (F1, voie F2 seulement) : g et beta rendus AVANT
        softplus et sigmoid, pour le noyau ; rend alors un 7e élément vrai."""
        b = x.shape[0]
        if _GDN_CONV_FUSEE and x.is_cuda and conv_state.is_contiguous():
            from .gdn_conv import conv_decode
            qkv, z, bt, a = self._projections(x, qkv_brut=True)
            q, k, v = conv_decode(qkv, conv_state, self.conv_weight.contiguous(), self.key_dim, self.value_dim)
            if portes_brutes:
                return (q.view(b, 1, self.nk, self.dk), k.view(b, 1, self.nk, self.dk), v.view(b, 1, self.nv, self.dv),
                        a.unsqueeze(1).contiguous(), bt.unsqueeze(1).contiguous(), z, True)
            beta = bt.sigmoid().unsqueeze(1)
            g = (-self.a_log.exp() * F.softplus(a + self.dt_bias)).unsqueeze(1)
            return (q.view(b, 1, self.nk, self.dk), k.view(b, 1, self.nk, self.dk), v.view(b, 1, self.nv, self.dv),
                    g.contiguous(), beta.contiguous(), z)
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
        out = (q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), z)
        return out + (False,) if portes_brutes else out

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
        y = self._norm_gated(core.reshape(-1, self.dv), z.view(b, self.nv, self.dv), h.dtype).reshape(b, self.value_dim)
        y = self.out_proj(y.to(h.dtype))
        return y, [(conv_state[i].clone(), S_new[i:i + 1].to(torch.float32)) for i in range(b)]

    def decode_static_batch(self, h: torch.Tensor, statics: list) -> torch.Tensor:
        """Chemin à formes fixes pour ``b`` créneaux : états lus et écrits
        dans les tranches contiguës du tampon groupé (aucune copie si les
        créneaux 0..b-1 vivent dans le même lot), un lancement de fla."""
        b = h.shape[0]
        g_, contigu = tranches(self, statics, b, ("conv", "S_"))
        conv_state, S = g_["conv"], g_["S_"]
        en_place = _GDN_ETAT_EN_PLACE and S.is_cuda and S.is_contiguous()
        brutes = False
        if en_place and _GDN_PORTES_NOYAU:
            q, k, v, g, beta, z, brutes = self._lot_projete(h, conv_state, portes_brutes=True)
        else:
            q, k, v, g, beta, z = self._lot_projete(h, conv_state)
        # F4 (156 c) : la récurrence en place est un noyau de carte ; sur processeur (CI publique,
        # machine sans GPU) le chemin fla de référence reste seul valable.
        if en_place:
            core = (_recurrence_en_place(q, k, v, g, beta, S, self.a_log, self.dt_bias) if brutes
                    else _recurrence_en_place(q, k, v, g, beta, S))
        else:
            core, S_new = _fla()[1](q, k, v, g=g, beta=beta, initial_state=S.contiguous(),
                                    output_final_state=True, use_qk_l2norm_in_kernel=True)
            S.copy_(S_new)
        if not contigu:
            for i, st in enumerate(statics[:b]):
                st["conv"].copy_(conv_state[i]); st["S"].copy_(S[i:i + 1])
        y = self._norm_gated(core.reshape(-1, self.dv), z.view(b, self.nv, self.dv), h.dtype).reshape(b, self.value_dim)
        return self.out_proj(y.to(h.dtype))
