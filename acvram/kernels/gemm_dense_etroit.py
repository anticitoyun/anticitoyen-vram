"""GEMM W4A16 dense à petit M (2 ≤ M ≤ 32), bornée par la bande — le poste
révélé par la fenêtre fla (sage-hybrides-etape1-close-gemm-dense-17-09 § 2) :
à b > 1, `nvfp4_gemv` relit les poids UNE FOIS PAR SÉQUENCE (Qwen3.8 b = 12 :
20 Go relus ligne par ligne à 0,23 To/s, 86,6 ms sur un pas de 93,9 ; plancher
de bande 11 ms).

Ici les M lignes d'activation forment une tuile [BM, BK] (M rembourré à 16 ou
32) qui reste en registres ; les poids NVFP4 sont balayés UNE fois par pas,
décodés en registres par les deux tables constantes de B1' (E2M1 → bf16,
E4M3 → bf16, produit code × échelle exact en bf16), et le produit passe par
``tl.dot`` sur la tuile — 0,65 TFLOP par pas sur Qwen3.8, la crête utile est
la bande, pas les tensor cores. Grille (tuiles N, tranches K) pour couvrir la
carte quand N est petit (q/k/v/o : N = 5 120 → 80 tuiles de 64) ; les
tranches K écrivent des partiels fp32 réduits par une somme torch
(déterministe, même ordre d'accumulation par tuile) ; échelle globale (par
ligne si le tenseur en porte) dans l'épilogue.

Porte : micro-banc `outils/banc-gemm-dense-etroit-17-09.py` ≥ 1,3 To/s sur les
formes de Qwen3.8 (faux < 0,9 : noyau CUDA, décision séparée). Régime
``ACVRAM_DENSE_NVFP4 = gemv (défaut, témoin) | triton``. Sans carte :
``TRITON_INTERPRET=1`` en fp16.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:                                        # noqa: BLE001
    triton = None
    tl = None

from .gemm_groupe import tables

M_MAX = 32
_BN = int(os.environ.get("ACVRAM_DENSE_ETROIT_BN", "64"))
_BK = int(os.environ.get("ACVRAM_DENSE_ETROIT_BK", "128"))
_WARPS = int(os.environ.get("ACVRAM_DENSE_ETROIT_WARPS", "4"))
_STAGES = int(os.environ.get("ACVRAM_DENSE_ETROIT_STAGES", "3"))
_PROGRAMMES_PAR_SM = 2          # tranches K visées : ≥ 2 programmes par SM


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _dense_etroit_kernel(x_ptr, qw_ptr, bs_ptr, gs_ptr, y_ptr, out_ptr, cpt_ptr, lut4_ptr, lut8_ptr,
                             M, N, K, k_par_tranche, T,
                             stride_xm, stride_qn, stride_bn, stride_gs, stride_yt, stride_ym, stride_om,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Épilogue « dernier bloc » (sage-lm-head-392-verdict-17-09) : les T
        tranches K d'une tuile N écrivent leur partiel fp32 puis incrémentent
        un compteur ; la dernière arrivée somme les T partiels DANS L'ORDRE
        t = 0..T-1 (déterministe, même somme que `_reduire_kernel`), écrit la
        sortie et remet le compteur à zéro (rejouable sous graphe sans
        memset). Un lancement au lieu de deux : la réduction séparée valait
        12 % du pas Qwen3.8 b=12."""
        pn = tl.program_id(0)
        pt = tl.program_id(1)
        rows = tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        masque_m = rows < M
        masque_n = cols < N
        k_debut = pt * k_par_tranche
        k_fin = tl.minimum(k_debut + k_par_tranche, K)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(k_debut, k_fin, BK):
            ks = k0 + tl.arange(0, BK)
            masque_k = ks < k_fin
            a = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            kb = k0 // 2 + tl.arange(0, BK // 2)
            masque_kb = kb < k_fin // 2
            oct = tl.load(qw_ptr + cols[:, None] * stride_qn + kb[None, :],
                          mask=masque_n[:, None] & masque_kb[None, :], other=0)
            lo = tl.load(lut4_ptr + (oct & 15))
            hi = tl.load(lut4_ptr + (oct >> 4))
            w = tl.reshape(tl.join(lo, hi), (BN, BK))
            ksc = k0 // 16 + tl.arange(0, BK // 16)
            masque_sc = ksc < k_fin // 16
            sc = tl.load(bs_ptr + cols[:, None] * stride_bn + ksc[None, :],
                         mask=masque_n[:, None] & masque_sc[None, :], other=0)
            scb = tl.load(lut8_ptr + sc)
            scf = tl.reshape(tl.broadcast_to(tl.expand_dims(scb, 2), (BN, BK // 16, 16)), (BN, BK))
            b = (w * scf).to(a.dtype)
            acc = tl.dot(a, tl.trans(b), acc)
        gs = tl.load(gs_ptr + cols * stride_gs, mask=masque_n, other=0.0)
        acc = acc * gs[None, :]
        masque = masque_m[:, None] & masque_n[None, :]
        if T == 1:
            tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :],
                     acc.to(out_ptr.dtype.element_ty), mask=masque)
        else:
            tl.store(y_ptr + pt * stride_yt + rows[:, None] * stride_ym + cols[None, :], acc, mask=masque)
            tl.debug_barrier()
            arrivee = tl.atomic_add(cpt_ptr + pn, 1, sem="acq_rel")
            if arrivee == T - 1:
                somme = tl.zeros((BM, BN), dtype=tl.float32)
                for t in range(0, T):
                    somme += tl.load(y_ptr + t * stride_yt + rows[:, None] * stride_ym + cols[None, :],
                                     mask=masque, other=0.0, volatile=True)
                tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :],
                         somme.to(out_ptr.dtype.element_ty), mask=masque)
                tl.atomic_xchg(cpt_ptr + pn, 0)


    @triton.jit
    def _dense_etroit_multi_kernel(x_ptr, qw_tab, bs_tab, sc_tab, gs_ptr, y_ptr, lut4_ptr, lut8_ptr,
                                   tuile_p, tuile_c0, n_tab, off_tab,
                                   M, K, k_par_tranche, stride_xm, stride_yt, stride_ym,
                                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Plusieurs projections de MÊME entrée (q, k, v ; qkv, gate, α, β du
        GDN) en UN lancement, chacune avec SON échelle d'activation
        (`ChannelScaler` : x / s, calculée ici en fp32 puis arrondie en bf16
        — le même arrondi que torch) : la fusion par empilement (`stack_
        nvfp4_linears`) refuse dès que les scalers diffèrent, ce qui est le
        cas de toute conversion calibrée par projection (Qwen3.8 calibA :
        act_scale q ≠ k ≠ v), et trois petites GEMM sous-occupent la carte
        (banc Laure 21033b7 : kv 0,47 To/s, q 0,72 ; qkv empilée 1,01)."""
        pn = tl.program_id(0)
        pt = tl.program_id(1)
        p = tl.load(tuile_p + pn)
        c0 = tl.load(tuile_c0 + pn)
        n_p = tl.load(n_tab + p)
        off = tl.load(off_tab + p)
        qw_ptr = tl.load(qw_tab + p).to(tl.pointer_type(tl.uint8))
        bs_ptr = tl.load(bs_tab + p).to(tl.pointer_type(tl.uint8))
        sc_ptr = tl.load(sc_tab + p).to(tl.pointer_type(x_ptr.dtype.element_ty))
        rows = tl.arange(0, BM)
        cols = c0 + tl.arange(0, BN)
        masque_m = rows < M
        masque_n = cols < n_p
        k_debut = pt * k_par_tranche
        k_fin = tl.minimum(k_debut + k_par_tranche, K)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(k_debut, k_fin, BK):
            ks = k0 + tl.arange(0, BK)
            masque_k = ks < k_fin
            a = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            s = tl.load(sc_ptr + ks, mask=masque_k, other=1.0)
            a = (a.to(tl.float32) / s.to(tl.float32)[None, :]).to(a.dtype)
            kb = k0 // 2 + tl.arange(0, BK // 2)
            masque_kb = kb < k_fin // 2
            oct = tl.load(qw_ptr + cols[:, None] * (K // 2) + kb[None, :],
                          mask=masque_n[:, None] & masque_kb[None, :], other=0)
            lo = tl.load(lut4_ptr + (oct & 15))
            hi = tl.load(lut4_ptr + (oct >> 4))
            w = tl.reshape(tl.join(lo, hi), (BN, BK))
            ksc = k0 // 16 + tl.arange(0, BK // 16)
            masque_sc = ksc < k_fin // 16
            sc = tl.load(bs_ptr + cols[:, None] * (K // 16) + ksc[None, :],
                         mask=masque_n[:, None] & masque_sc[None, :], other=0)
            scb = tl.load(lut8_ptr + sc)
            scf = tl.reshape(tl.broadcast_to(tl.expand_dims(scb, 2), (BN, BK // 16, 16)), (BN, BK))
            b = (w * scf).to(a.dtype)
            acc = tl.dot(a, tl.trans(b), acc)
        gs = tl.load(gs_ptr + off + cols, mask=masque_n, other=0.0)
        acc = acc * gs[None, :]
        tl.store(y_ptr + pt * stride_yt + rows[:, None] * stride_ym + off + cols[None, :],
                 acc, mask=masque_m[:, None] & masque_n[None, :])

    @triton.jit
    def _reduire_kernel(y_ptr, out_ptr, T, M, N, stride_yt, stride_ym, stride_om,
                        BM: tl.constexpr, BN: tl.constexpr):
        """Somme des partiels fp32 des tranches K et sortie bf16, en un
        lancement (torch : `sum` puis `to`, deux)."""
        pn = tl.program_id(0)
        rows = tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        masque = (rows < M)[:, None] & (cols < N)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for t in range(0, T):
            acc += tl.load(y_ptr + t * stride_yt + rows[:, None] * stride_ym + cols[None, :], mask=masque, other=0.0)
        tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :], acc.to(out_ptr.dtype.element_ty), mask=masque)


def _reduire(y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[T, M, N] fp32 → [M, N] ``dtype`` : un noyau si T > 1 et carte, sinon torch."""
    T, M, N = y.shape
    if T == 1:
        return y[0].to(dtype)
    if y.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        return y.sum(0).to(dtype)
    out = torch.empty(M, N, dtype=dtype, device=y.device)
    BM = 16 if M <= 16 else 32
    _reduire_kernel[(-(-N // 256),)](y, out, T, M, N, y.stride(0), y.stride(1), out.stride(0),
                                     BM=BM, BN=256, num_warps=4)
    return out


def _sms(device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 4


def _tranches(n: int, k: int, device, bn: int, bk: int) -> tuple[int, int]:
    """(tranches K, jetons K par tranche) : assez de programmes pour couvrir
    la carte, tranche multiple de BK (les octets de codes et d'échelles d'une
    tranche restent alignés sur 2 et 16)."""
    tuiles_n = -(-n // bn)
    voulu = max(1, -(-_PROGRAMMES_PAR_SM * _sms(device) // tuiles_n))
    pas_max = -(-k // bk)
    t = max(1, min(voulu, pas_max))
    par_tranche = -(-pas_max // t) * bk
    return -(-k // par_tranche), par_tranche


def gemm_dense_etroit(x: torch.Tensor, t, bn: int = 0, bk: int = 0, warps: int = 0,
                      stages: int = 0, sortie_fp32: bool = False) -> torch.Tensor:
    """``x`` [M ≤ 32, K] bf16 (fp16 sous l'interpréteur), ``t`` NVFP4Tensor
    (échelle globale scalaire ou par ligne) → [M, N] dans le dtype de x, ou
    fp32 (``sortie_fp32`` : la tête — les logits sont LE tenseur à comparer,
    accumulés en fp32 depuis des produits bf16 × bf16 exacts)."""
    bn, bk = bn or _BN, bk or _BK
    warps, stages = warps or _WARPS, stages or _STAGES
    M, K = x.shape
    N, k_oct = t.qweight.shape
    k_pad = t.padded_in
    assert 1 <= M <= M_MAX and K <= k_pad and k_pad % 16 == 0 and bk % 16 == 0, (M, K, k_pad, bk)
    if K != k_pad:
        x = torch.nn.functional.pad(x, (0, k_pad - K))
    x = x.contiguous()
    BM = 16 if M <= 16 else 32
    lut4, lut8 = tables(x.device, x.dtype)
    gsr = getattr(t, "global_scale_rows", None)
    gs = (gsr.to(torch.float32).reshape(-1).contiguous() if gsr is not None
          else t.global_scale.to(torch.float32).reshape(1))
    tuiles_n = -(-N // bn)
    tranches, par_tranche = _tranches(N, k_pad, x.device, bn, bk)
    out = torch.empty(M, N, dtype=torch.float32 if sortie_fp32 else x.dtype, device=x.device)
    y = torch.empty(tranches if tranches > 1 else 0, M, N, dtype=torch.float32, device=x.device)
    _dense_etroit_kernel[(tuiles_n, tranches)](
        x, t.qweight, t.block_scale.view(torch.uint8), gs, y, out, _compteurs(x.device, tuiles_n), lut4, lut8,
        M, N, k_pad, par_tranche, tranches,
        x.stride(0), t.qweight.stride(0), t.block_scale.stride(0), 1 if gsr is not None else 0,
        y.stride(0), y.stride(1), out.stride(0),
        BM=BM, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


_COMPTEURS: dict = {}


def _compteurs(device, n: int) -> torch.Tensor:
    """Compteurs d'arrivée par tuile N, à zéro entre deux appels (le dernier
    bloc les remet) : un tampon par appareil, grandi au besoin, jamais
    réalloué en dessous (une adresse stable pour les graphes)."""
    cle = str(device)
    vivants = _COMPTEURS.setdefault(cle, [])          # les anciens restent vivants : un graphe capturé les tient
    if not vivants or vivants[-1].numel() < n:
        vivants.append(torch.zeros(max(n, 8192), dtype=torch.int32, device=device))
    return vivants[-1]


class MultiProjection:
    """Plusieurs projections NVFP4 de MÊME entrée servies en un lancement à
    2 ≤ M ≤ 32 (`_dense_etroit_multi_kernel`), chacune avec son
    `ChannelScaler` (x / s, sans rotation) — là où l'empilement des poids est
    refusé parce que les scalers diffèrent. Rend [M, Σ N] ; l'appelant
    découpe (`tailles`). Hors de sa fenêtre (M = 1, M > 32, autre dtype),
    l'appelant repasse par les projections séparées."""

    def __init__(self, lins: list) -> None:
        from ..quant.nvfp4 import NVFP4Tensor
        ts = [l.qweight for l in lins]
        assert ts and all(isinstance(t, NVFP4Tensor) for t in ts), "NVFP4 seulement"
        assert all(getattr(l, "bias", None) is None for l in lins), "pas de biais"
        K = ts[0].padded_in
        assert all(t.padded_in == K for t in ts) and K % 16 == 0, "même entrée"
        for l in lins:
            sc = getattr(l, "scaler", None)
            assert sc is None or sc.is_identity or not sc.hadamard_block, "rotation Hadamard non portée"
        self.lins, self.K = lins, K
        self.tailles = tuple(int(t.shape[0]) for t in ts)
        self.N = sum(self.tailles)
        self.device = ts[0].qweight.device
        dev = self.device
        offs = [0]
        for n in self.tailles:
            offs.append(offs[-1] + n)
        self.off_tab = torch.tensor(offs[:-1], dtype=torch.int32, device=dev)
        self.n_tab = torch.tensor(self.tailles, dtype=torch.int32, device=dev)
        gs = []
        for t in ts:
            gsr = getattr(t, "global_scale_rows", None)
            gs.append(gsr.to(torch.float32).reshape(-1) if gsr is not None
                      else t.global_scale.to(torch.float32).reshape(1).expand(int(t.shape[0])))
        self.gs = torch.cat(gs).contiguous().to(dev)
        self.qw_tab = torch.tensor([t.qweight.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        self.bs_tab = torch.tensor([t.block_scale.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        self._qw, self._bs = [t.qweight for t in ts], [t.block_scale for t in ts]   # gardent les adresses vivantes
        self._sc: dict = {}
        self._tuiles: dict = {}

    def _scalers(self, dtype: torch.dtype):
        if dtype not in self._sc:
            uns = torch.ones(self.K, dtype=dtype, device=self.device)
            vecs = []
            for l in self.lins:
                sc = getattr(l, "scaler", None)
                if sc is None or sc.is_identity:
                    vecs.append(uns)
                else:
                    v = sc._au_dtype(dtype).reshape(-1)
                    vecs.append(torch.nn.functional.pad(v, (0, self.K - v.numel()), value=1.0).contiguous()
                                if v.numel() < self.K else v.contiguous())
            tab = torch.tensor([v.data_ptr() for v in vecs], dtype=torch.int64, device=self.device)
            self._sc[dtype] = (tab, vecs)
        return self._sc[dtype][0]

    def _grille(self, bn: int):
        if bn not in self._tuiles:
            tp, tc = [], []
            for p, n in enumerate(self.tailles):
                for c0 in range(0, n, bn):
                    tp.append(p); tc.append(c0)
            self._tuiles[bn] = (torch.tensor(tp, dtype=torch.int32, device=self.device),
                                torch.tensor(tc, dtype=torch.int32, device=self.device), len(tp))
        return self._tuiles[bn]

    def __call__(self, x: torch.Tensor, bn: int = 0, bk: int = 0, warps: int = 0, stages: int = 0) -> torch.Tensor:
        bn, bk = bn or _BN, bk or _BK
        warps, stages = warps or _WARPS, stages or _STAGES
        M, K = x.shape
        assert 1 <= M <= M_MAX and K <= self.K, (M, K, self.K)
        if K != self.K:
            x = torch.nn.functional.pad(x, (0, self.K - K))
        x = x.contiguous()
        BM = 16 if M <= 16 else 32
        lut4, lut8 = tables(x.device, x.dtype)
        tp, tc, nt = self._grille(bn)
        tranches, par_tranche = _tranches(self.N, self.K, x.device, bn, bk)
        y = torch.empty(tranches, M, self.N, dtype=torch.float32, device=x.device)
        _dense_etroit_multi_kernel[(nt, tranches)](
            x, self.qw_tab, self.bs_tab, self._scalers(x.dtype), self.gs, y, lut4, lut8,
            tp, tc, self.n_tab, self.off_tab, M, self.K, par_tranche,
            x.stride(0), y.stride(0), y.stride(1),
            BM=BM, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
        return _reduire(y, x.dtype)
