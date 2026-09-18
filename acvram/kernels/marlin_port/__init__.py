"""Port des noyaux Marlin de vLLM v0.29.0 (Apache-2.0, `LICENSE-vllm`) pour la
porte P1 (poste7-reprise-ordre-18-09 § 4 (a) ; note-marlin-classe-a-sec-18-09) :
GEMM groupée W4A16 classe Marlin sur nos piles NVFP4, noyau SEUL, sans
intégration moteur — micro-banc `outils/banc-marlin-p1-18-09.py`.

Sources (verbatim, espace de noms des ops renommé `acvram_marlin`) :
`csrc/libtorch_stable/quantization/marlin/{marlin.cuh, marlin_dtypes.cuh,
dequant.h, marlin_mma.h, gptq_marlin_repack.cu}`, `csrc/libtorch_stable/moe/
marlin_moe_wna16/{kernel.h, marlin_template.h, ops.cu}` (+ instanciations
générées pour NVFP4 seul : E2M1 + échelle E4M3 par 16, activation bf16),
`csrc/core/scalar_type.hpp`, `csrc/libtorch_stable/torch_utils.h`. Les
préparations Python (repack, permutation et conversion des échelles en
S0E5M3, alignement des jetons par blocs d'expert) sont transcrites de
`vllm/model_executor/layers/quantization/utils/marlin_utils{,_fp4}.py` et
`fused_moe/moe_align_block_size.py`.

Compilation : `torch.utils.cpp_extension.load` à la première demande
(`charger()`), cache `~/.cache/acvram/marlin_port` ; les ops sont
`torch.ops.acvram_marlin.{gptq_marlin_repack, moe_wna16_marlin_gemm}`.
"""
from __future__ import annotations

import glob
import os
import pathlib

import torch

ICI = pathlib.Path(__file__).resolve().parent
GROUP_SIZE = 16
FE2M1F_ID = None            # rempli au chargement (vllm::kFE2M1f.id())


def _kfe2m1f_id() -> int:
    """`vllm::kFE2M1f.id()` = ScalarType::float_(2, 1, finite_values_only=True,
    NanRepr::NONE) — champs dans l'ordre de scalar_type.hpp / scalar_type.py :
    exponent (8 bits, décalage 0), mantissa (8, 8), signed (1, 16), bias
    (32, 17), finite_values_only (1, 49), nan_repr (8, 50). Contrôlé contre
    `vllm.scalar_type.scalar_types.float4_e2m1f.id` = 562949953487106."""
    exponent, mantissa, signed, bias, finite_only, nan_repr = 2, 1, 1, 0, 1, 0
    return exponent | (mantissa << 8) | (signed << 16) | ((bias & 0xFFFFFFFF) << 17) | (finite_only << 49) | (nan_repr << 50)


_EXT = None
VERSION_SOURCE = "vLLM v0.29.0 (csrc/libtorch_stable, commit de la balise v0.29.0)"
COMPILE_ICI = False          # vrai si la compilation a eu lieu dans CE processus (REGLES § 6, garde b)


def dossier_cache() -> pathlib.Path:
    return pathlib.Path(os.environ.get("ACVRAM_MARLIN_CACHE",
                                       pathlib.Path.home() / ".cache" / "acvram" / "marlin_port"))


def chemin_so() -> pathlib.Path:
    return dossier_cache() / "acvram_marlin.so"


def sha_so() -> str:
    """sha256 du binaire compilé — l'en-tête d'une cellule (INDEX) le porte."""
    import hashlib
    p = chemin_so()
    if not p.exists():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def charger(verbose: bool = False, compiler: bool = True):
    """Compile (une fois) et charge l'extension ; rend l'espace `torch.ops.acvram_marlin`.

    JIT hors capture (REGLES § 6) : à compiler AVANT la prise de carte —
    `CUDA_VISIBLE_DEVICES="" python outils/banc-marlin-p1-18-09.py --compiler-seulement`
    (nvcc seul, aucune carte requise : archs sm_86 + sm_120 par défaut). Si le
    binaire n'est pas en cache au moment du banc, `COMPILE_ICI` passe à vrai
    et le banc se déclare invalide."""
    global _EXT, COMPILE_ICI
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load
    COMPILE_ICI = not chemin_so().exists()
    if COMPILE_ICI and not compiler:
        return None                      # le moteur ne compile jamais sous le verrou (REGLES § 6)
    moe = ICI / "libtorch_stable" / "moe" / "marlin_moe_wna16"
    sources = [str(ICI / "bindings.cpp"), str(moe / "ops.cu"),
               str(ICI / "libtorch_stable" / "quantization" / "marlin" / "gptq_marlin_repack.cu")]
    sources += sorted(glob.glob(str(moe / "sm80_kernel_*.cu")))
    cache = dossier_cache()
    cache.mkdir(parents=True, exist_ok=True)
    from .. import _arch_flags
    load(name="acvram_marlin", sources=sources, is_python_module=False, verbose=verbose,
         build_directory=str(cache),
         extra_include_paths=[str(ICI)],
         # USE_CUDA : les déclarations du shim CUDA de l'ABI stable (flux courant,
         # cublas) sont derrière cette garde ; l'espace de noms Marlin est posé
         # par kernel.h (moe) et par défaut (repack), pas ici
         extra_cflags=["-O3", "-std=c++20", "-DUSE_CUDA"],
         # -static-global-template-stub=false : depuis CUDA 12.8 nvcc donne une
         # liaison INTERNE aux stubs hôte des gabarits __global__ instanciés
         # explicitement ; sans ce drapeau (que vLLM pose, CMakeLists.txt:1377)
         # l'édition de liens rend « undefined hidden symbol Marlin<…> »
         extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-DENABLE_BF16", "-DUSE_CUDA",
                            "-static-global-template-stub=false"] + _arch_flags())
    _EXT = torch.ops.acvram_marlin
    return _EXT


# --- préparation des poids (transcrit de marlin_utils.py / marlin_utils_fp4.py, Apache-2.0) ---

def _scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def permuter_echelles(s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = _scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def facteur_nvfp4(marlin_scales: torch.Tensor) -> float:
    """Puissance de 2 telle que toute échelle non nulle × 2⁷ soit ≥ 2 en bf16
    (le bit de poids fort de la représentation S0E5M3 toujours à 1)."""
    ws = marlin_scales.float() * (2 ** 7)
    nz = ws > 0
    if nz.any():
        mx = ws[nz].max()
        if mx < 448 * (2 ** 7):
            return (448 * (2 ** 7) / mx).log2().floor().exp2().item()
    return 1.0


def traiter_echelles_nvfp4(marlin_scales: torch.Tensor, facteur: float) -> torch.Tensor:
    """E4M3 (S1E4M3) → « S0E5M3 » : demi-précision × facteur × 2⁷, < 2 → 0,
    décalage d'un bit, moitié haute des paires."""
    assert bool((marlin_scales >= 0).all()), "Marlin NVFP4 suppose des échelles ≥ 0"
    s = marlin_scales.to(torch.half)
    s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
    if facteur > 1.0:
        s = (s.float() * facteur).to(torch.half)
    s = s * (2 ** 7)
    s[s < 2] = 0
    s = s.view(torch.int16) << 1
    s = s.view(torch.float8_e4m3fn)
    return s[:, 1::2].contiguous()


def traiter_echelle_globale(g: torch.Tensor, facteur: float) -> torch.Tensor:
    """bf16 : biais d'exposant 2⁷−2¹ = 126, replié ×2^(126−7), puis ÷ facteur."""
    return (g.float() * (2.0 ** (126 - 7))) / facteur


def preparer_pile(qw: torch.Tensor, bs: torch.Tensor, gs: torch.Tensor):
    """Pile NVFP4 acvram → format Marlin. ``qw`` [E, N, K/2] uint8 (paires
    E2M1, bas d'abord = ModelOpt), ``bs`` [E, N, K/16] E4M3, ``gs`` [E] fp32.
    Rend (w_marlin [E, K/16, N·2] int32 repacké, s_marlin [E, K/16, N] E4M3
    S0E5M3, g_marlin [E] fp32). K multiple de 64, N multiple de 64."""
    ops = charger()
    E, N, K2 = qw.shape
    K = K2 * 2
    assert K % 64 == 0 and N % 64 == 0, (K, N)
    perm = torch.empty(0, dtype=torch.int, device=qw.device)
    w_out = None
    for e in range(E):
        q = qw[e].contiguous().view(torch.int32).T.contiguous()      # [K/8, N] int32 (GPTQ : K empaqueté)
        m = ops.gptq_marlin_repack(q, perm, K, N, 4, False)
        if w_out is None:
            w_out = torch.empty((E, *m.shape), dtype=m.dtype, device=m.device)
        w_out[e] = m
    scales = bs.to(torch.bfloat16)                                       # [E, N, K/16]
    facteur = facteur_nvfp4(scales)
    s_list = []
    for e in range(E):
        s = permuter_echelles(scales[e].T, K, N, GROUP_SIZE)              # [K/16, N]
        s_list.append(traiter_echelles_nvfp4(s, facteur))
    s_out = torch.stack(s_list)
    g_out = traiter_echelle_globale(gs.float().reshape(-1), facteur).contiguous()
    return w_out, s_out, g_out


# --- alignement des jetons par blocs d'expert (moe_align_block_size, en torch) ---

def aligner_blocs(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """``topk_ids`` [M, top_k] int → (sorted_token_ids [P] int32 : indices de
    PAIRES triés par expert, chaque expert rembourré à un multiple de
    ``block_size`` par la sentinelle M·top_k ; expert_ids [P/block] int32 ;
    num_tokens_post_padded [1] int32). Déterministe (tri stable)."""
    plat = topk_ids.reshape(-1)
    n = plat.numel()
    ordre = torch.argsort(plat, stable=True)
    e_tri = plat[ordre]
    comptes = torch.bincount(e_tri, minlength=num_experts)
    rembourres = ((comptes + block_size - 1) // block_size) * block_size
    total = int(rembourres.sum())
    sorted_ids = torch.full((max(total, block_size),), n, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.zeros((max(total, block_size)) // block_size, dtype=torch.int32, device=topk_ids.device)
    debut_pad = torch.cumsum(rembourres, 0) - rembourres                 # début de chaque expert (rembourré)
    debut_tri = torch.cumsum(comptes, 0) - comptes                       # début de chaque expert (trié)
    pos = debut_pad[e_tri] + (torch.arange(n, device=topk_ids.device) - debut_tri[e_tri])
    sorted_ids[pos] = ordre.to(torch.int32)
    blocs = rembourres // block_size
    expert_ids[: int(blocs.sum())] = torch.repeat_interleave(
        torch.arange(num_experts, device=topk_ids.device, dtype=torch.int32), blocs)
    return sorted_ids, expert_ids, torch.tensor([total], dtype=torch.int32, device=topk_ids.device)


def choisir_block_size(M: int, top_k: int, E: int) -> int:
    for b in [8, 16, 32, 48, 64]:
        if M * top_k / E / b < 0.9:
            break
    return b


def espace_travail(device, blocs_par_sm: int = 4) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * blocs_par_sm, dtype=torch.int32, device=device)


def gemm_moe(a: torch.Tensor, w_marlin, s_marlin, g_marlin, sorted_ids, expert_ids, num_post,
             topk_weights, block_size: int, top_k: int, size_m: int, size_n: int, size_k: int,
             workspace, mul_topk_weights: bool = False, c=None) -> torch.Tensor:
    """Une GEMM groupée Marlin (bf16 → bf16), fp32 reduce, sans atomique."""
    ops = charger()
    return ops.moe_wna16_marlin_gemm(
        a, c, w_marlin, None, s_marlin, None, g_marlin, None, None, None, workspace,
        sorted_ids, expert_ids, num_post, topk_weights, block_size, top_k, mul_topk_weights,
        _kfe2m1f_id(), size_m, size_n, size_k, True, False, True, False, -1, -1, -1)
