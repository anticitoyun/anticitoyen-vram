"""GEMM étroit W8A16 en Triton — poste C (poste7-profil-verdict-17-09 § 2) :
les linéaires INT8 denses du décodage à b ≤ 16 (`narrow_gemm_kernel<32>`
2,08 ms pour 0,9 Go à 24 % de la bande passante, `int8_gemv` de la tête
0,88 ms pour 0,31 Go — poste3 6a4fd57).

Poids uint8 affine par groupes (`INT8Tensor` : q [N, K], échelle fp16 et
zéro uint8 [N, K/G]) : par groupe g, ``y += (x_g · q_gᵀ − Σx_g · z_g) · s_g``
— le produit se fait sur les entiers (uint8 ≤ 255 exact en bf16) par
``tl.dot`` en fp32, le zéro ne coûte qu'une somme de ligne, l'échelle un
produit par colonne. Grille (tuile N, tranche K) pour couvrir la carte à
b = 1 comme à b = 12 ; tranches réduites par une somme torch (déterministe).
Sortie bf16, ou fp32 pour la tête (logits).

Scellé (poste7) : dense b = 12 ≤ 1,0 ms par pas (−1,8 ms), PPL inchangée ;
sortie = chemin actuel ± 2⁻⁸ (juge : tests/test_gemm_etroit.py). Sans
carte : ``TRITON_INTERPRET=1`` en fp16.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:                                        # noqa: BLE001
    triton = None
    tl = None

BM = 16                    # lignes de la tuile : b ≤ 16 rembourré
BN = 64
_WARPS, _STAGES = 4, 3


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _acc_tranche(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, g0, groupes_par_tranche,
                     rows, cols, masque_m, masque_n, stride_xm, stride_qn, stride_sn,
                     BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        """Accumulateur fp32 [BM, BN] d'UNE tranche K (groupes g0 .. g0 + gpt) :
        le corps commun de `_etroit_kernel` (témoin, somme torch des tranches)
        et de `_etroit_reduit_kernel` (C15 niveau 3)."""
        kk = tl.arange(0, G)
        acc = tl.zeros((BM_, BN_), dtype=tl.float32)
        for g in range(g0, g0 + groupes_par_tranche):
            # la dernière tranche déborde quand NG n'est pas multiple de la
            # taille de tranche : g ≥ NG lisait l'échelle et le zéro AU-DELÀ
            # du tenseur (cols·NG + g), accès illégal sous graphe dans le
            # moteur (poste3 b34a1bb) — invisible au banc, dont les tenseurs
            # voisins absorbaient la lecture
            masque_g = g < NG
            ks = g * G + kk
            masque_k = (ks < K) & masque_g
            x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            q = tl.load(q_ptr + cols[:, None] * stride_qn + ks[None, :],
                        mask=masque_n[:, None] & masque_k[None, :], other=0)
            s = tl.load(s_ptr + cols * stride_sn + g, mask=masque_n & masque_g, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + cols * stride_sn + g, mask=masque_n & masque_g, other=0).to(tl.float32)
            prod = tl.dot(x, tl.trans(q.to(x.dtype)))                     # [BM, BN] fp32
            sx = tl.sum(x.to(tl.float32), 1)                              # Σ_k x[m, k] du groupe
            acc += (prod - sx[:, None] * z[None, :]) * s[None, :]
        return acc


    @triton.jit
    def _etroit_kernel(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, M, N, K, NG, groupes_par_tranche,
                       stride_xm, stride_qn, stride_sn, stride_ys, stride_ym,
                       BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        pn = tl.program_id(0)
        ps = tl.program_id(1)
        rows = tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque_m = rows < M
        masque_n = cols < N
        acc = _acc_tranche(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, ps * groupes_par_tranche,
                           groupes_par_tranche, rows, cols, masque_m, masque_n,
                           stride_xm, stride_qn, stride_sn, BM_, BN_, G)
        tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :],
                 acc, mask=masque_m[:, None] & masque_n[None, :])


    @triton.jit
    def _etroit_reduit_kernel(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, cnt_ptr, out_ptr, M, N, K, NG,
                              groupes_par_tranche, tranches,
                              stride_xm, stride_qn, stride_sn, stride_ys, stride_ym, stride_om,
                              BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        """C15 niveau 3 : le GEMM étroit en UN nœud — chaque programme (tuile N,
        tranche K) écrit son accumulateur fp32 dans y[ps], incrémente le
        compteur de sa tuile ; le DERNIER arrivé somme les tranches dans
        l'ordre 0..tranches−1 (déterministe quel que soit l'arrivant), écrit la
        sortie dans le dtype de out (bf16 : un seul arrondi, comme
        `y.sum(0).to(bf16)` ; fp32 pour la tête) et remet le compteur à zéro.
        Remplace torch.zeros + noyau + y.sum(0) + cast (4 nœuds). Ordre de la
        somme : sériel, celui du témoin quand torch le fait sériel (≤ 4
        tranches) ; sinon ± 1 ulp bf16. Visibilité : barrière, atomique
        acq_rel portée gpu, barrière, lectures .cg."""
        pn = tl.program_id(0)
        ps = tl.program_id(1)
        rows = tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque_m = rows < M
        masque_n = cols < N
        masque = masque_m[:, None] & masque_n[None, :]
        acc = _acc_tranche(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, ps * groupes_par_tranche,
                           groupes_par_tranche, rows, cols, masque_m, masque_n,
                           stride_xm, stride_qn, stride_sn, BM_, BN_, G)
        tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :], acc, mask=masque)
        tl.debug_barrier()
        n = tl.atomic_add(cnt_ptr + pn, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if n == tranches - 1:
            somme = tl.zeros((BM_, BN_), dtype=tl.float32)
            for t in range(0, tranches):
                somme += tl.load(y_ptr + t * stride_ys + rows[:, None] * stride_ym + cols[None, :],
                                 mask=masque, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :],
                     somme.to(out_ptr.dtype.element_ty), mask=masque)
            tl.store(cnt_ptr + pn, 0)


    @triton.jit
    def _etroit_segments_kernel(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, cnt_ptr, out_ptr, gpt_ptr, ntr_ptr, M, N, K, NG,
                                stride_xm, stride_qn, stride_sn, stride_ys, stride_ym, stride_om,
                                BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        """Pièce 176 : `_etroit_reduit_kernel` pour une PILE de segments de même entrée (GDN qkv‖gate) où chaque
        tuile garde la partition K de SON segment (gpt, tranches lus dans des tables par tuile) : chaque colonne est
        sommée exactement comme par l'appel séparé de son segment — au bit. Grille Y = max des tranches ; un programme
        au-delà des tranches de sa tuile sort aussitôt (la 4e tranche, vide, des tuiles qkv)."""
        pn = tl.program_id(0)
        ps = tl.program_id(1)
        groupes_par_tranche = tl.load(gpt_ptr + pn)
        tranches = tl.load(ntr_ptr + pn)
        if ps < tranches:
            rows = tl.arange(0, BM_)
            cols = pn * BN_ + tl.arange(0, BN_)
            masque_m = rows < M
            masque_n = cols < N
            masque = masque_m[:, None] & masque_n[None, :]
            acc = _acc_tranche(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, ps * groupes_par_tranche,
                               groupes_par_tranche, rows, cols, masque_m, masque_n,
                               stride_xm, stride_qn, stride_sn, BM_, BN_, G)
            tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :], acc, mask=masque)
            tl.debug_barrier()
            n = tl.atomic_add(cnt_ptr + pn, 1, sem="acq_rel", scope="gpu")
            tl.debug_barrier()
            if n == tranches - 1:
                somme = tl.zeros((BM_, BN_), dtype=tl.float32)
                for t in range(0, tranches):
                    somme += tl.load(y_ptr + t * stride_ys + rows[:, None] * stride_ym + cols[None, :],
                                     mask=masque, other=0.0, cache_modifier=".cg")
                tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :],
                         somme.to(out_ptr.dtype.element_ty), mask=masque)
                tl.store(cnt_ptr + pn, 0)


def _tables_segments(t, device):
    """Pièce 176 : (gpt [tuiles], tranches [tuiles], max des tranches) d'une pile à ``t._segments`` — la partition K
    que `decouper_k` donne à CHAQUE segment appelé seul. Construites une fois (jamais pendant une capture)."""
    tab = t.__dict__.get("_tables_segments")
    if tab is not None and tab[0].device == device:
        return tab
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("gemm_etroit : tables de segments construites pendant une capture — l'échauffement eager "
                           "doit précéder")
    ng = t.qweight.shape[1] // t.group_size
    gpts, ntrs = [], []
    for i, n in enumerate(t._segments):
        assert i == len(t._segments) - 1 or n % BN == 0, "segment non aligné sur BN : une tuile chevaucherait deux segments"
        tu = -(-n // BN)
        tr, gpt = decouper_k(ng, tu, device)
        gpts += [gpt] * tu
        ntrs += [tr] * tu
    tab = (torch.tensor(gpts, dtype=torch.int32, device=device), torch.tensor(ntrs, dtype=torch.int32, device=device),
           max(ntrs))
    t.__dict__["_tables_segments"] = tab
    return tab


def _programmes(device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 4


G_MAX = 128


def eligible(t) -> bool:
    """Le noyau étroit prend le GROUPE comme tuile K : un poids symétrique par
    canal (group_size = K, convertis -qkvo-i8c, P2) demanderait 393 Kio de
    mémoire partagée pour 101 Kio disponibles (OutOfResources en service,
    19/09, godet 2 et préfill chunké) — dimension « taille de groupe » jamais
    posée (MECANISMES). Refusé au-delà de G_MAX : l'appelant prend narrow_gemm
    CUDA ou le GEMV, sortie inchangée pour tout poids à groupes de 128."""
    return t.group_size <= G_MAX


_COMPTEURS: dict = {}


def _compteur(n: int, device) -> torch.Tensor:
    """Compteurs int32 [n] des tuiles N, nuls entre deux lancements (le dernier
    programme remet le sien à zéro) : réservés une fois par forme et appareil,
    jamais pendant une capture de graphe (l'échauffement eager les crée)."""
    cle = (n, str(device))
    c = _COMPTEURS.get(cle)
    if c is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("gemm_etroit : compteurs du noyau fusionné réservés pendant une capture "
                               "de graphe — l'échauffement eager doit précéder (REGLES § 7)")
        c = _COMPTEURS[cle] = torch.zeros(n, dtype=torch.int32, device=device)
    return c


# Forme du noyau (BM/BN fixes ; `warps` et `etages` réglables pour
# l occupation — pièce 35, verdict croisé Q(3) : `o` [2048, 4096] tourne à
# 0,71 To/s, hypothèse = occupation par programme, pas partition de K ; le
# split-K à facteur variable a été RÉFUTÉ au banc (poste2 b08a3d34) et son
# crochet est retiré ici). Changer warps ou étages ne change NI l ordre des
# sommes en K (même boucle par groupe), NI l ordre des tranches (même
# `tranches`) : la sortie est AU BIT — test dans le même commit.
_FORME: Optional[tuple] = None


def forme_noyau() -> tuple[int, int]:
    """(warps, étages) du lancement : défaut (4, 3), ou `ACVRAM_ETROITES_FORME=W,S`."""
    if _FORME is not None:
        return _FORME
    v = os.environ.get("ACVRAM_ETROITES_FORME", "")
    if v:
        w, e = v.split(",")
        return int(w), int(e)
    return _WARPS, _STAGES


def regler_forme(forme: Optional[tuple]) -> None:
    """Banc : impose (warps, étages) ; None = relire l environnement / le défaut."""
    global _FORME
    _FORME = forme


def etroites_texte() -> str:
    """Ligne de régime : `serie` (4 warps, 3 étages) ou `w{W}s{S}`, puis `+canal(table|temoin|BNxBKxWxS)` (195)."""
    w, e = forme_noyau()
    base = "serie" if (w, e) == (_WARPS, _STAGES) else f"w{w}s{e}"
    return base + f"+canal({canal_texte()})"


def decouper_k(ng: int, tuiles_n: int, device) -> tuple[int, int]:
    """(tranches, groupes par tranche) — 2 programmes par SM visés, arithmétique du 17/09."""
    voulu = -(-2 * _programmes(device) // tuiles_n)
    tranches = max(1, min(ng, voulu))
    gpt = -(-ng // tranches)
    return -(-ng // gpt), gpt


def tranches_de(x: torch.Tensor, t) -> int:
    N, k_pad = t.qweight.shape
    return decouper_k(k_pad // t.group_size, -(-N // BN), x.device)[0]


def programmes_de(x: torch.Tensor, t) -> int:
    N, k_pad = t.qweight.shape
    tuiles_n = -(-N // BN)
    return tuiles_n * decouper_k(k_pad // t.group_size, tuiles_n, x.device)[0]


def gemm_etroit(x: torch.Tensor, t, sortie_fp32: bool = False, compact: bool = False) -> torch.Tensor:
    """``x`` [M ≤ 16, K] bf16 (fp16 sous l'interpréteur), ``t`` INT8Tensor
    → [M, N] dans le dtype de x, ou fp32 (tête). ``compact`` (C15 niveau 3,
    ACVRAM_GLUE_COMPACT) : un seul lancement, tranches réduites par le
    dernier programme de chaque tuile, sortie écrite dans son dtype."""
    M, K = x.shape
    N, k_pad = t.qweight.shape
    G = t.group_size
    assert M <= BM and K <= k_pad and k_pad % G == 0, (M, K, k_pad, G)
    ng = k_pad // G
    assert t.scales.shape == (N, ng) and t.zeros.shape == (N, ng), (t.scales.shape, t.zeros.shape, N, ng)
    tuiles_n = -(-N // BN)
    seg = getattr(t, "_segments", None)
    if seg is not None:
        if not compact:                   # voie témoin : chaque segment par son propre appel (au bit par construction)
            outs, d = [], 0
            for n in seg:
                vue = type(t)(t.qweight[d:d + n], t.scales[d:d + n], t.zeros[d:d + n], G, (n, t.shape[1]))
                outs.append(gemm_etroit(x, vue, sortie_fp32, compact=False))
                d += n
            return torch.cat(outs, dim=-1)
        gpt_t, ntr_t, tmax = _tables_segments(t, x.device)
        y = torch.empty(tmax, M, N, dtype=torch.float32, device=x.device)
        out = torch.empty(M, N, dtype=torch.float32 if sortie_fp32 else x.dtype, device=x.device)
        _etroit_segments_kernel[(tuiles_n, tmax)](
            x, t.qweight, t.scales, t.zeros, y, _compteur(tuiles_n, x.device), out, gpt_t, ntr_t, M, N, K, ng,
            x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1), out.stride(0),
            BM_=BM, BN_=BN, G=G, num_warps=forme_noyau()[0], num_stages=forme_noyau()[1])
        return out
    tranches, gpt = decouper_k(ng, tuiles_n, x.device)
    if compact:
        y = torch.empty(tranches, M, N, dtype=torch.float32, device=x.device)
        out = torch.empty(M, N, dtype=torch.float32 if sortie_fp32 else x.dtype, device=x.device)
        _etroit_reduit_kernel[(tuiles_n, tranches)](
            x, t.qweight, t.scales, t.zeros, y, _compteur(tuiles_n, x.device), out, M, N, K, ng, gpt, tranches,
            x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1), out.stride(0),
            BM_=BM, BN_=BN, G=G, num_warps=forme_noyau()[0], num_stages=forme_noyau()[1])
        return out
    y = torch.zeros(tranches, M, N, dtype=torch.float32, device=x.device)
    _etroit_kernel[(tuiles_n, tranches)](
        x, t.qweight, t.scales, t.zeros, y, M, N, K, ng, gpt,
        x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1),
        BM_=BM, BN_=BN, G=G, num_warps=forme_noyau()[0], num_stages=forme_noyau()[1])
    out = y.sum(0) if tranches > 1 else y[0]
    return out if sortie_fp32 else out.to(x.dtype)


# ---------------------------------------------------------------------------
# Pièce 195 (AU DÉFAUT depuis le 25/09, décision déléguée par l'utilisateur ; HORS BIT « ± 1 ulp », REGLES § 1) : poids int8 symétrique
# PAR CANAL (échelle [N, 1], zéro 128 — tous les int8 de l'alias mixte-i8c),
# K ENTIER par programme, sans tranche ni partiel ni atomique : la géométrie
# de NInfer (`fp8_gemv_kernel<Fp8Geometry<N, K>>`, 194 § c). La somme fp32 sur
# K entier n'a pas l'ordre des 2-5 tranches du noyau d'avant, qui reste le TÉMOIN NOMMÉ : `ACVRAM_ETROIT_CANAL=0`.
# `=1` (défaut : table par forme) ou `=BN,BK,W,S` (géométrie imposée). Qualifié par `revue/poste6-piece195-verdict-25-09.md`
# (banc 0,86 ms/pas, ABBA mixte b=8 +4,01 % / −3,76 % J, KL ≤ 2 × témoin à échantillon égal sur deux modèles, argmax égal).
# ---------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _etroit_canal_kernel(x_ptr, q_ptr, s_ptr, out_ptr, M, N, K, stride_xm, stride_qn, stride_om,
                             BM_: tl.constexpr, BN_: tl.constexpr, BK: tl.constexpr):
        """y[m, n] = s[n] · Σ_k x[m, k] · (q[n, k] − 128) ; (q − 128) ∈ [−128, 127] est exact en
        bf16, donc aucun terme Σx·z à retrancher (pas de perte par annulation sur K entier)."""
        pn = tl.program_id(0)
        rows = tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque_m = rows < M
        masque_n = cols < N
        acc = tl.zeros((BM_, BN_), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            masque_k = ks < K
            x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            q = tl.load(q_ptr + cols[:, None] * stride_qn + ks[None, :],
                        mask=masque_n[:, None] & masque_k[None, :], other=128)
            acc += tl.dot(x, tl.trans(q.to(x.dtype) - 128.0))
        s = tl.load(s_ptr + cols, mask=masque_n, other=0.0).to(tl.float32)
        tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :],
                 (acc * s[None, :]).to(out_ptr.dtype.element_ty),
                 mask=masque_m[:, None] & masque_n[None, :])


# (N, K) → (BN, BK, warps, étages), mesuré par `scratchpad/poste6-p195-25-09/banc-canal.py` (L2 froid, M = 8) ;
# une forme absente reste sur le noyau servi. Mesuré le 25/09 (`banc-canal.log`, 54 géométries × 5 formes, servi = vue
# g128 compact) : µs canal / servi — o‖out 22,70 / 25,23 · qkv attn 49,80 / 50,30 · GDN qkv‖gate 56,47 / 65,57 ·
# down 58,92 / 66,27 · gate‖up 117,01 / 141,41 ; 0,86 ms/pas sur l'alias mixte-i8c à b = 8. BN 16 (NInfer littéral)
# perd partout en Triton ; BN 64 × BK 512 × 4 étages déborde la mémoire partagée (2 échecs).
GEOMETRIE_CANAL: dict = {
    (5120, 6144): (32, 128, 4, 4),        # o_proj attention, out GDN (64 appels/pas)
    (14336, 5120): (32, 256, 4, 3),       # q‖k‖v attention empilé (gain ≈ 0 : 49,80 contre 50,30)
    (16384, 5120): (64, 256, 8, 2),       # qkv‖gate GDN empilé (176)
    # Segments de cette pile pris SÉPARÉMENT (qkv 10240, gate 6144) : jamais servis au décodage (176 : pile au défaut),
    # mais la géométrie doit être CELLE DE LA PILE pour que la pile reste au bit des deux appels (test 176) — à K entier,
    # chaque colonne ne dépend que de sa propre partition K (même BK, mêmes warps), pas de sa tuile N.
    (10240, 5120): (64, 256, 8, 2),
    (6144, 5120): (64, 256, 8, 2),
    (5120, 17408): (64, 512, 8, 3),       # down int8 (couches MLP int8 du mixte)
    (34816, 5120): (64, 256, 4, 2),       # gate‖up int8
}


CANAL_DEFAUT = "1"          # 25/09 : au défaut ; 0 = témoin nommé (noyau à tranches, vue g128)


def canal_actif() -> bool:
    return os.environ.get("ACVRAM_ETROIT_CANAL", CANAL_DEFAUT) not in ("", "0")


def canal_texte() -> str:
    """`table` (défaut), `temoin` (0 : noyau d'avant), ou la géométrie imposée `BNxBKxWxS`."""
    v = os.environ.get("ACVRAM_ETROIT_CANAL", CANAL_DEFAUT)
    if v in ("", "0"):
        return "temoin"
    return "table" if v == "1" else v.replace(",", "x")


def geometrie_canal(N: int, k_pad: int) -> Optional[tuple[int, int, int, int]]:
    """Géométrie imposée (`BN,BK,W,S`), sinon celle de la table pour cette forme, sinon None : une forme que le banc
    n'a pas mesurée (α/β int8 48 × 5120 de l'alias attn-gdn-i8c : 2 programmes à K entier) reste sur le noyau servi."""
    v = os.environ.get("ACVRAM_ETROIT_CANAL", CANAL_DEFAUT)
    if "," in v:
        bn, bk, w, e = (int(t) for t in v.split(","))
        return bn, bk, w, e
    return GEOMETRIE_CANAL.get((N, k_pad))


def canal_eligible(t) -> bool:
    """INT8 symétrique par canal : échelle [N, 1], zéro 128 partout, groupe = K (vérifié une fois par tenseur)."""
    c = t.__dict__.get("_canal")
    if c is None:
        N, k_pad = t.qweight.shape
        c = bool(t.group_size == k_pad and tuple(t.scales.shape) == (N, 1) and tuple(t.zeros.shape) == (N, 1)
                 and bool((t.zeros == 128).all()))
        t.__dict__["_canal"] = c
    return c


def gemm_canal(x: torch.Tensor, t, geometrie: Optional[tuple] = None) -> torch.Tensor:
    """``x`` [M ≤ 16, K] bf16 × poids int8 par canal → [M, N] bf16, K entier par programme (195, opt-in)."""
    M, K = x.shape
    N, k_pad = t.qweight.shape
    assert M <= BM and K <= k_pad, (M, K, k_pad)
    geo = geometrie or geometrie_canal(N, k_pad)
    assert geo is not None, ("forme sans géométrie mesurée", N, k_pad)
    bn, bk, w, e = geo
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _etroit_canal_kernel[(-(-N // bn),)](
        x, t.qweight, t.scales, out, M, N, K, x.stride(0), t.qweight.stride(0), out.stride(0),
        BM_=BM, BN_=bn, BK=bk, num_warps=w, num_stages=e)
    return out
