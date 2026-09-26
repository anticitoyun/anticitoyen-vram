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
ECHEC_COMPILATION = None     # pièce 161 : raison du dernier échec de compilation (repli nommé), sinon None
VERSION_SOURCE = "vLLM v0.29.0 (csrc/libtorch_stable, commit de la balise v0.29.0)"
COMPILE_ICI = False          # vrai si la compilation a eu lieu dans CE processus (REGLES § 6, garde b)


_EMPREINTE = None
_SUFFIXES_SOURCES = (".cu", ".cuh", ".h", ".hpp", ".cpp")


def _fichiers_sources() -> list:
    """Tout ce que nvcc lit dans le port (sources et en-têtes), chemins triés."""
    return sorted(q for q in ICI.rglob("*")
                  if q.is_file() and q.suffix in _SUFFIXES_SOURCES and "__pycache__" not in q.parts)


def empreinte_sources() -> str:
    """Pièce 161 : sha256 (12 hex) des sources du port — chemins relatifs et octets.
    C'est la CLÉ du cache : deux arbres aux mêmes sources partagent un binaire, deux versions n'en partagent aucun.
    Le 24/09 le cache était UN dossier pour tous les worktrees : chaque changement d'arbre appelant rebâtissait le
    .so (chemin -I dans build.ninja), même sous `charger(compiler=False)` et sous le verrou d'une autre prise, et
    pendant l'édition de liens un chargement concurrent voyait le .so ABSENT → repli « port non compilé » muet
    (revue/poste6-piece161-verdict-24-09.md)."""
    global _EMPREINTE
    if _EMPREINTE is None:
        import hashlib
        h = hashlib.sha256()
        for q in _fichiers_sources():
            h.update(str(q.relative_to(ICI)).encode()); h.update(b"\0"); h.update(q.read_bytes()); h.update(b"\0")
        # Pas les architectures : `_arch_flags()` dépend du contexte (à sec, CUDA_VISIBLE_DEVICES vide, il ne voit
        # pas la carte) et la compilation se fait à sec AVANT la prise — la même empreinte doit sortir des deux côtés
        # (première prise 161 : 9e4f5791 à sec contre 2eb083f8 sur carte → .so « absent », repli). Les cibles sont
        # constantes sur un poste.
        _EMPREINTE = h.hexdigest()[:12]
    return _EMPREINTE


def dossier_cache() -> pathlib.Path:
    """Un cache PAR EMPREINTE des sources (pièce 161), sous `~/.cache/acvram/` ou sous `ACVRAM_MARLIN_CACHE`
    (racine, observation et tests). Jamais le dossier partagé `marlin_port/` d'avant."""
    racine = pathlib.Path(os.environ.get("ACVRAM_MARLIN_CACHE") or (pathlib.Path.home() / ".cache" / "acvram"))
    return racine / f"marlin_port-{empreinte_sources()}"


def _sources_en_cache(cache: pathlib.Path) -> pathlib.Path:
    """Copie des sources DANS le cache (`src/`), d'où ninja compile : les lignes de commande ne portent plus le
    chemin du worktree, donc un second arbre aux mêmes sources ne rebâtit rien. L'empreinte garantit le contenu ;
    le marqueur `.complet` garde d'une copie interrompue."""
    import shutil
    src = cache / "src"
    if not (src / ".complet").exists():
        for q in _fichiers_sources():
            dest = src / q.relative_to(ICI)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(q, dest)
        (src / ".complet").write_text(empreinte_sources())
    return src


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
    global _EXT, COMPILE_ICI, ECHEC_COMPILATION
    if _EXT is not None:
        return _EXT
    so = chemin_so()
    if not so.exists():
        # Pièce 161 (réserve de chef, Marlin étant le DÉFAUT) : empreinte absente → compilation UNE fois par version
        # des sources, sous un verrou de fichier propre au cache ; le cache par empreinte rend le ping-pong impossible,
        # donc les ~25 s ne reviennent qu'à chaque nouvelle version. Seul un échec (nvcc absent…) laisse le repli, nommé.
        # ``compiler`` est gardé pour les appelants (banc) : il ne change plus rien.
        try:
            _compiler(verbose)
        except Exception as e:                             # noqa: BLE001 — la raison va sur la ligne de régime
            ECHEC_COMPILATION = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
            return None
    # Le .so de CETTE empreinte est chargé tel quel — jamais ninja quand il existe (c'est ce que
    # `load(is_python_module=False)` finit par faire, sans la reconstruction qui le précède).
    torch.ops.load_library(str(so))
    _EXT = torch.ops.acvram_marlin
    return _EXT


def _compiler(verbose: bool = False) -> None:
    """Compile le port dans le cache de son empreinte, sous `flock` (un seul processus compile ; les autres attendent
    puis trouvent le .so). Pose `COMPILE_ICI` (le banc se déclare invalide si la compilation a eu lieu chez lui)."""
    global COMPILE_ICI
    import fcntl
    from torch.utils.cpp_extension import load
    cache = dossier_cache()
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / ".verrou-compilation", "w") as verrou:
        fcntl.flock(verrou, fcntl.LOCK_EX)
        if chemin_so().exists():                           # un autre processus vient de compiler
            return
        COMPILE_ICI = True
        _lancer_ninja(cache, verbose, load)


def _lancer_ninja(cache: pathlib.Path, verbose: bool, load) -> None:
    SRC = _sources_en_cache(cache)
    moe = SRC / "libtorch_stable" / "moe" / "marlin_moe_wna16"
    sources = [str(SRC / "bindings.cpp"), str(moe / "ops.cu"),
               str(SRC / "libtorch_stable" / "quantization" / "marlin" / "gptq_marlin_repack.cu"),
               str(SRC / "libtorch_stable" / "moe" / "moe_align_sum_kernels.cu")]      # pièce 62 : aligneur CUDA
    sources += sorted(glob.glob(str(moe / "sm80_kernel_*.cu")))
    dense = SRC / "libtorch_stable" / "quantization" / "marlin"                       # pièce 101 : Marlin dense
    sources += [str(dense / "marlin.cu")] + sorted(glob.glob(str(dense / "dense_sm80_kernel_*.cu")))
    from .. import _arch_flags
    empreinte = f"-DMARLIN_PORT_EMPREINTE={empreinte_sources()}"   # 161 : ccache ne rend jamais un objet d'une autre version
    load(name="acvram_marlin", sources=sources, is_python_module=False, verbose=verbose,   # ninja, une fois par empreinte
         build_directory=str(cache),
         extra_include_paths=[str(SRC)],
         # USE_CUDA : les déclarations du shim CUDA de l'ABI stable (flux courant,
         # cublas) sont derrière cette garde ; l'espace de noms Marlin est posé
         # par kernel.h (moe) et par défaut (repack), pas ici
         extra_cflags=["-O3", "-std=c++20", "-DUSE_CUDA", empreinte],
         # -static-global-template-stub=false : depuis CUDA 12.8 nvcc donne une
         # liaison INTERNE aux stubs hôte des gabarits __global__ instanciés
         # explicitement ; sans ce drapeau (que vLLM pose, CMakeLists.txt:1377)
         # l'édition de liens rend « undefined hidden symbol Marlin<…> »
         extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-DENABLE_BF16", "-DUSE_CUDA",
                            "-static-global-template-stub=false", empreinte] + _arch_flags())


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


def _facteur_depuis_max(mx: torch.Tensor) -> torch.Tensor:
    """Le facteur de `facteur_nvfp4`, élément par élément (puissance de 2, 1 pour un max nul ou ≥ 448)."""
    f = torch.ones_like(mx, dtype=torch.float32)
    ok = (mx > 0) & (mx < 448)
    f[ok] = (448.0 / mx[ok].float()).log2().floor().exp2()
    return f


def echelles_ecrasees(bs: torch.Tensor, par_ligne: bool = False) -> int:
    """Pièce 157 : nombre d'échelles de bloc > 0 que la conversion Marlin (`traiter_echelles_nvfp4` : ×facteur×2⁷ en
    demi-précision, < 2 → 0) écraserait À ZÉRO — le bloc de 16 poids avec. S0E5M3 n'a qu'environ 2^14,8 de plage contre
    2^17,8 pour e4m3 : un poids (ou une pile d'experts, facteur commun) qui mêle 448 et des sous-normales perd les petites.
    ``par_ligne`` : facteur par ligne de sortie (dernière dimension réduite). Qwen3-14B couche 2 down : 47 368 valeurs
    fausses au dépaquetage (24/09), Qwen3-Coder-30B couche 0 : 0,57 % des blocs de 43 experts à zéro."""
    s = bs.float()
    mx = s.amax(-1, keepdim=True) if par_ligne else s.amax().reshape([1] * s.dim())
    f = _facteur_depuis_max(mx)
    return int(((s > 0) & (s.half().float() * f * 128 < 2)).sum())


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


def preparer_pile(qw: torch.Tensor, bs: torch.Tensor, gs: torch.Tensor, repack=None, par_ligne: bool = True):
    """Pile NVFP4 acvram → format Marlin. ``qw`` [E, N, K/2] uint8 (paires
    E2M1, bas d'abord = ModelOpt), ``bs`` [E, N, K/16] E4M3, ``gs`` [E] fp32.
    Rend (w_marlin [E, K/16, N·2] int32 repacké, s_marlin [E, K/16, N] E4M3
    S0E5M3, g_marlin [E] fp32 — ou **[E, N]** par (expert, colonne) quand un facteur
    commun écraserait des sous-normales : pièce 209, facteur PAR LIGNE d'expert
    (puissance de 2, e4m3 exact), g divisé d'autant ; fl((s·f)·(g/f)) = fl(s·g) au
    bit. Une pile qu'un facteur commun n'écrase pas garde EXACTEMENT la
    préparation d'avant. ``par_ligne=False`` : l'ancien comportement (témoin des
    tests cassants). K multiple de 64, N multiple de 64.
    ``repack`` : (q [K/8, N] int32, K, N) → [K/16, 2N] ; défaut l'op CUDA
    `gptq_marlin_repack` (carte) — `repack_torch` à sec (même disposition,
    tests/test_depaqueter_marlin.py)."""
    E, N, K2 = qw.shape
    K = K2 * 2
    assert K % 64 == 0 and N % 64 == 0, (K, N)
    if repack is None:
        ops = charger()
        perm = torch.empty(0, dtype=torch.int, device=qw.device)
        repack = lambda q, K, N: ops.gptq_marlin_repack(q, perm, K, N, 4, False)   # noqa: E731
    w_out = None
    for e in range(E):
        q = qw[e].contiguous().view(torch.int32).T.contiguous()      # [K/8, N] int32 (GPTQ : K empaqueté)
        m = repack(q, K, N)
        if w_out is None:
            w_out = torch.empty((E, *m.shape), dtype=m.dtype, device=m.device)
        w_out[e] = m
    f_en = None
    if par_ligne and echelles_ecrasees(bs):
        # Pièce 209 (157 pour les denses) : Coder-30B couches 0, 1, 2, 4 — 43 experts portent 448 et 2⁻⁹ ensemble
        f_en = _facteur_depuis_max(bs.float().amax(-1)).to(bs.device)     # [E, N]
        bs = (bs.float() * f_en[..., None]).to(bs.dtype)
        assert not echelles_ecrasees(bs), "facteur par ligne insuffisant : une ligne d'expert dépasse S0E5M3"
    scales = bs.to(torch.bfloat16)                                       # [E, N, K/16]
    facteur = facteur_nvfp4(scales)
    s_list = []
    for e in range(E):
        s = permuter_echelles(scales[e].T, K, N, GROUP_SIZE)              # [K/16, N]
        s_list.append(traiter_echelles_nvfp4(s, facteur))
    s_out = torch.stack(s_list)
    if f_en is None:
        g_out = traiter_echelle_globale(gs.float().reshape(-1), facteur).contiguous()
    else:
        g_out = traiter_echelle_globale(gs.float().reshape(-1, 1).to(f_en.device) / f_en, facteur).reshape(E, N).contiguous()
    return w_out, s_out, g_out


# --- disposition Marlin ↔ poids bf16, en torch (chantier C2, revue/chantier-c2-19-09) ---
#
# Le repack CUDA (gptq_marlin_repack.cu:122-209 : num_bits = 4, sans perm,
# activation 16 bits) range chaque tuile 16 k × 64 n en 128 mots int32 : le
# mot t·4 + w (t = voie 0..31, w = warp 0..3) porte les colonnes n = w·16 + t/4
# et n + 8 aux k = (t % 4)·2 + {0, 1, 8, 9} ; quartets du mot, bas d'abord
# (pack_idx {0, 2, 4, 6, 1, 3, 5, 7}) : (n,k0) (n,k8) | (n+8,k0) (n+8,k8) |
# (n,k1) (n,k9) | (n+8,k1) (n+8,k9) — c'est aussi ce que documente et lit
# `nvfp4_gemv_marlin_kernel` (acvram_kernels.cu, en-tête « GEMV groupée sur la
# DISPOSITION MARLIN »). Les échelles (permuter_echelles puis le [0, 2, 1, 3]
# de traiter_echelles_nvfp4) : la colonne o de la tuile est à l'octet
# 8·(o % 8) + swap4(o / 8), swap4 = (q & ~3) | {0, 2, 1, 3}[q & 3].
# À sec, tout se prouve par repack_torch → depaqueter_marlin = identité et par
# l'égalité AU BIT avec dequantize_nvfp4 (tests/test_depaqueter_marlin.py) ;
# l'égalité de repack_torch avec l'op CUDA elle-même reste à faire sur carte.

_PACK_IDX = (0, 2, 4, 6, 1, 3, 5, 7)
_TC_OFFSETS = (0, 1, 8, 9)
_SWAP4 = (0, 2, 1, 3)
# code E2M1 signé (bit 3) → valeur, -0.0 compris (dequantize_nvfp4 rend -0.0 pour le code 8)
_NIVEAUX16 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_INDICES: dict = {}


def _indices(device):
    """Indices fixes d'une tuile, calculés une fois par appareil :
    ``repack`` [32, 4, 8] = source (kl·64 + nl) de chaque quartet du mot
    (t, w) ; ``depaquet`` [64, 16] = quartet plat (t, w, octet, moitié) de
    chaque (nl, kl) ; ``echelles`` [64] = octet d'échelle de la colonne o ;
    ``niveaux`` [16] fp32."""
    cle = str(device)
    if cle in _INDICES:
        return _INDICES[cle]
    t = torch.arange(32).view(32, 1, 1)
    w = torch.arange(4).view(1, 4, 1)
    v = torch.tensor(_PACK_IDX).view(1, 1, 8)
    nl = w * 16 + t // 4 + 8 * (v // 4)
    kl = (t % 4) * 2 + torch.tensor(_TC_OFFSETS)[v % 4]
    repack = (kl * 64 + nl).to(torch.long)
    nl = torch.arange(64).view(64, 1)
    kl = torch.arange(16).view(1, 16)
    w, r = nl // 16, nl % 16
    h, c = r // 8, r % 8
    j, rem = kl // 8, kl % 8
    tm4, kk = rem // 2, rem % 2
    tt, b = c * 4 + tm4, kk * 2 + h
    depaquet = (((tt * 4 + w) * 4 + b) * 2 + j).to(torch.long)
    o = torch.arange(64)
    q = o // 8
    echelles = (8 * (o % 8) + ((q & ~3) | torch.tensor(_SWAP4)[q & 3])).to(torch.long)
    niveaux = torch.tensor(_NIVEAUX16, dtype=torch.float32)
    res = tuple(x.to(device) for x in (repack, depaquet, echelles, niveaux))
    _INDICES[cle] = res
    return res


def repack_torch(q: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Jumeau torch de `ops.gptq_marlin_repack(q, ∅, K, N, 4, False)` : ``q``
    [K/8, N] int32 GPTQ (le quartet k aux bits 4·(k % 8) du mot k/8) →
    [K/16, 2N] int32, tuiles 16 × 64 de 128 mots. Sert à sec (preparer_pile
    (repack=repack_torch)) ; vérifié contre depaqueter_marlin, pas encore
    contre l'op CUDA."""
    assert tuple(q.shape) == (K // 8, N) and K % 64 == 0 and N % 64 == 0, (tuple(q.shape), K, N)
    idx, _, _, _ = _indices(q.device)
    dec = (torch.arange(8, device=q.device, dtype=torch.int32) * 4).view(1, 8, 1)
    codes = ((q.to(torch.int32).unsqueeze(1) >> dec) & 0xF).to(torch.uint8)           # [K/8, 8, N] = [K, N]
    tuiles = codes.view(K // 16, 16, N // 64, 64).permute(0, 2, 1, 3).reshape(K // 16, N // 64, 1024)
    quartets = tuiles[:, :, idx.view(-1)].view(K // 16, N // 64, 32, 4, 4, 2)          # [.., t, w, octet, moitié]
    octets = quartets[..., 0] | (quartets[..., 1] << 4)
    return octets.contiguous().view(torch.int32).view(K // 16, 2 * N)


def depaqueter_marlin(w_marlin: torch.Tensor, s_marlin: torch.Tensor, g_marlin, K: int, N: int,
                      out: torch.Tensor = None, par: int = 16, noyau: str = "auto") -> torch.Tensor:
    """Disposition Marlin (preparer_pile) → poids bf16 [N, K], ou [E, N, K]
    pour une pile ([E, K/16, 2N], [E, K/16, N], [E]). EXACTEMENT les valeurs
    de `dequantize_nvfp4` sur la pile NVFP4 d'origine : mêmes opérations fp32
    dans le même ordre (échelle de bloc × globale, puis × code, puis arrondi
    bf16) — l'octet S0E5M3 décodé vaut s·facteur et g_marlin vaut g·2¹¹⁹/facteur,
    leur produit fp32 est fl(s·g)·2¹¹⁹ (puissances de deux) ; seule exception,
    les échelles que le repack a annulées (s·facteur·2⁷ < 2) restent nulles.
    ``out`` : tampon bf16 réutilisé ([E·N·K] ou [E, N, K]), rempli par tranches
    de ``par`` experts (pic transitoire borné, jamais la pile entière en fp32).
    ``noyau`` : torch (référence, toute machine) | triton (une passe, un
    programme par tuile : le chemin de la carte, `_depaqueter_kernel`) | auto
    = triton sur CUDA quand Triton est là, torch sinon."""
    pile = w_marlin.dim() == 3
    w = w_marlin if pile else w_marlin.unsqueeze(0)
    s = s_marlin if pile else s_marlin.unsqueeze(0)
    E = w.shape[0]
    assert tuple(w.shape) == (E, K // 16, 2 * N) and tuple(s.shape) == (E, K // 16, N), (tuple(w.shape), tuple(s.shape), K, N)
    g = torch.as_tensor(g_marlin, dtype=torch.float32, device=w.device).reshape(-1)
    if g.numel() == 1 and E > 1:
        g = g.expand(E)
    par_colonne = g.numel() == E * N and N > 1                 # [N] (134, E = 1) ou [E, N] (209, pile à facteur par ligne)
    if par_colonne and E == 1 and noyau not in ("triton", "cuda", "torch") and not (
            noyau == "auto" and w.device.type == "cuda" and (triton is not None or _depaqueter_cuda_disponible())):
        raise ValueError("depaqueter_marlin : échelle globale par colonne servie par les noyaux Triton, CUDA et torch")
    # 209 (a) refusait ici TOUTE échelle par colonne au noyau CUDA, y compris E = 1 (le cas 134/147 que l'extension sert
    # au bit) : deux tests p147 rouges depuis le 25/09 (232, rejeu main = branche). Seule la pile E > 1 lui est fermée.
    if par_colonne and E > 1 and noyau == "cuda":
        raise ValueError("depaqueter_marlin : échelle globale par colonne d'une pile (E > 1) : noyau triton ou torch")
    if out is None:
        out = torch.empty(E, N, K, dtype=torch.bfloat16, device=w.device)
    res = out.view(E, N, K)
    if noyau == "auto":
        noyau = _DEPAQUETAGE if _DEPAQUETAGE in ("cuda", "triton", "torch") else "auto"
    if noyau == "auto":
        # une pile E > 1 à g [E, N] ne va JAMAIS au noyau CUDA (il lirait g[e] comme scalaire, sans erreur) : triton/torch
        if w.device.type == "cuda" and _depaqueter_cuda_disponible() and not (par_colonne and E > 1):
            noyau = "cuda"
        else:
            noyau = "triton" if (w.device.type == "cuda" and triton is not None) else "torch"
    if noyau == "cuda":
        _depaqueter_cuda(w, s, g, res, K, N)
        DEPAQUETAGES["cuda"] += 1
        return res if pile else res[0]
    if noyau == "triton":
        _depaqueter_triton(w, s, g, res, K, N)
        DEPAQUETAGES["triton"] += 1
        return res if pile else res[0]
    if noyau != "torch":
        raise ValueError(f"depaqueter_marlin : noyau {noyau!r}, attendu auto | torch | triton")
    _, idx, perm_e, niveaux = _indices(w.device)
    for e0 in range(0, E, max(1, par)):
        e1 = min(E, e0 + max(1, par))
        octets = w[e0:e1].contiguous().view(torch.uint8).view(e1 - e0, K // 16, N // 64, 512)
        quartets = torch.stack((octets & 0xF, octets >> 4), dim=-1).view(e1 - e0, K // 16, N // 64, 1024)
        codes = quartets[..., idx.view(-1)].view(e1 - e0, K // 16, N // 64, 64, 16).permute(0, 2, 3, 1, 4)   # [e, nt, nl, kt, kl]
        vals = niveaux[codes.to(torch.int32)]                                                             # fp32, -0.0 conservé
        s8 = s[e0:e1].contiguous().view(torch.uint8).view(e1 - e0, K // 16, N // 64, 64)[..., perm_e]     # [e, kt, nt, o]
        s_dec = ((s8.to(torch.int32) << 20) + 0x34800000).view(torch.float32)                             # = s·facteur
        s_dec = torch.where(s8 == 0, torch.zeros_like(s_dec), s_dec)                                      # annulée : 0, pas 2⁻²²
        g_e = (g.view(E, N)[e0:e1].view(e1 - e0, 1, N // 64, 64) if par_colonne             # 209 : g par (expert, colonne)
               else g[e0:e1].view(-1, 1, 1, 1))
        echelle = (s_dec * g_e) * 2.0 ** -119                                                             # = fl(s·g), au bit
        res[e0:e1] = (vals * echelle.permute(0, 2, 3, 1).unsqueeze(-1)).reshape(e1 - e0, N, K).to(torch.bfloat16)
    return res if pile else res[0]


def gemv_marlin_torch(w_marlin: torch.Tensor, s_marlin: torch.Tensor, g_marlin, expert_ids: torch.Tensor,
                      token_ids: torch.Tensor, x: torch.Tensor, K: int, N: int, xscale=None) -> torch.Tensor:
    """Jumeau torch (à sec, chantier C10) de `nvfp4_gemv_marlin` (acvram_kernels.cu,
    `nvfp4_gemv_marlin_kernel<XT, 1>`) : lit la DISPOSITION MARLIN d'une pile
    ([E, K/16, 2N], [E, K/16, N], [E]) aux mêmes places que le noyau (quartets
    `_indices().depaquet`, octets d'échelle `_indices().echelles`) et rend
    [G, N] fp32, ligne g = paire (expert_ids[g], token_ids[g]), dans le
    GROUPEMENT du noyau : par tuile k, Σ_k code·x en fp32, × échelle S0E5M3
    décodée ((b << 20) + 0x34800000 — l'octet 0 vaut 2⁻²² comme dans
    `s0e5m3_octet`, pas 0), Σ sur les tuiles, × g_marlin·2⁻¹¹⁹. Ce qu'il ne
    reproduit pas : l'ordre des FMA du noyau (4 voies, 8 warps) — le juge est
    fp32 par ligne ≤ 2⁻⁷·max|y| (tests/test_gemv_marlin.py), pas le bit.
    Créneau fantôme (expert < 0) : ligne nulle, aucun poids lu. ``xscale`` (pièce 47, [E, ≥ K] bf16) : le
    noyau divise x par la table de l'expert, x et le quotient arrondis en bf16 — reproduit ici tel quel."""
    E = w_marlin.shape[0]
    assert tuple(w_marlin.shape) == (E, K // 16, 2 * N) and tuple(s_marlin.shape) == (E, K // 16, N)
    g2 = torch.as_tensor(g_marlin, dtype=torch.float32, device=w_marlin.device)
    par_colonne = g2.dim() == 2                                       # 209 : g [E, ≥ N] par (expert, colonne)
    g = g2 if par_colonne else g2.reshape(-1)
    _, idx, perm_e, niveaux = _indices(w_marlin.device)
    KT, NT = K // 16, N // 64
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    if xf.shape[1] != K:
        xf = torch.nn.functional.pad(xf, (0, K - xf.shape[1]))
    out = torch.zeros(expert_ids.numel(), N, dtype=torch.float32, device=w_marlin.device)
    cache: dict = {}
    for gi, (e, t) in enumerate(zip(expert_ids.tolist(), token_ids.tolist())):
        if e < 0:
            continue
        if e not in cache:
            octets = w_marlin[e].contiguous().view(torch.uint8).view(KT, NT, 512)
            quartets = torch.stack((octets & 0xF, octets >> 4), dim=-1).view(KT, NT, 1024)
            codes = quartets[:, :, idx.view(-1)].view(KT, NT, 64, 16).permute(1, 2, 0, 3)   # [nt, nl, kt, kl]
            vals = niveaux[codes.to(torch.int32)].reshape(N, KT, 16)
            s8 = s_marlin[e].contiguous().view(torch.uint8).view(KT, NT, 64)[..., perm_e]  # [kt, nt, o]
            s_dec = ((s8.to(torch.int32) << 20) + 0x34800000).view(torch.float32)          # octet 0 → 2⁻²², comme le noyau
            cache[e] = (vals, s_dec.permute(1, 2, 0).reshape(N, KT))
        vals, s_dec = cache[e]
        xt = xf[t]
        if xscale is not None:
            xt = (xt.to(torch.bfloat16).float() / xscale[e, :K].float()).to(torch.bfloat16).float()
        partiel = torch.einsum("nkl,kl->nk", vals, xt.view(KT, 16))                     # Σ_k code·x par tuile, fp32
        out[gi] = (partiel * s_dec).sum(dim=1) * ((g[e, :N] if par_colonne else g[e]) * 2.0 ** -119)
    return out


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


def aligner_blocs_tries(e_sorted: torch.Tensor, comptes: torch.Tensor, block_size: int, num_experts: int):
    """C15-prefill : `aligner_blocs` quand les paires sont DÉJÀ triées par
    expert (``e_sorted`` [n], croissant) et leurs comptes connus (``comptes``
    [E] int64 = bincount) — le prefill Marlin (`_forward_prefill_grouped`)
    l'appelait sur `e_sorted` et retriait une liste triée (argsort 4 passes
    radix + histogramme + scan, 1,0 ms et 7 lancements par prefill 2 047) :
    l'ordre stable d'une liste triée est l'identité, sorted_ids = position de
    chaque rang. AUCUN scalaire hôte : sorties de taille FIXE P = G + E·(bloc − 1)
    arrondi au bloc (comme `aligner_blocs_capturable` et vLLM
    `moe_align_block_size`), sentinelle G au-delà du total dans sorted_ids et
    −1 dans expert_ids ; le noyau lit `num_tokens_post_padded` sur la carte
    (ops.cu:58) et ne visite jamais ces blocs (−1 ignoré, ops.cu:116). Les
    `total` premières entrées sont celles d'`aligner_blocs`, au bit."""
    n = e_sorted.numel()
    dev = e_sorted.device
    P = -(-(n + num_experts * (block_size - 1)) // block_size) * block_size
    rembourres = ((comptes + block_size - 1) // block_size) * block_size
    fin_pad = torch.cumsum(rembourres, 0)                                  # fin (rembourrée) de chaque expert
    debut_pad = fin_pad - rembourres
    debut_tri = torch.cumsum(comptes, 0) - comptes
    rang = torch.arange(n, device=dev)
    pos = debut_pad[e_sorted] + (rang - debut_tri[e_sorted])
    sorted_ids = torch.full((P,), n, dtype=torch.int32, device=dev)
    sorted_ids[pos] = rang.to(torch.int32)
    # expert du bloc b : le premier expert dont la fin rembourrée dépasse b·bloc
    # (les experts vides ont une fin égale à la précédente : sautés) ; −1 au-delà
    # du total
    debuts_blocs = torch.arange(P // block_size, device=dev) * block_size
    e_b = torch.searchsorted(fin_pad, debuts_blocs, right=True)
    expert_ids = torch.where(debuts_blocs < fin_pad[-1], e_b, -1).to(torch.int32)
    return sorted_ids, expert_ids, fin_pad[-1:].to(torch.int32)


def choisir_block_size(M: int, top_k: int, E: int) -> int:
    for b in [8, 16, 32, 48, 64]:
        if M * top_k / E / b < 0.9:
            break
    return b


def espace_travail(device, blocs_par_sm: int = 4) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * blocs_par_sm, dtype=torch.int32, device=device)


def preparer_dense(t, repack=None):
    """Pièce 101 : un poids dense NVFP4 acvram (`NVFP4Tensor` [N, K]) → (w_marlin [K/16, 2N], s_marlin [K/16, N],
    g_marlin) pour `gemm_dense`. Si ``t`` porte `global_scale_rows` (q/k/v empilés, une échelle globale par
    segment), g_marlin est PAR COLONNE [N] — même traitement (×2^119 / facteur, exact : puissances de deux) que
    l'échelle scalaire, appliqué en fp32 dans l'épilogue du noyau porté (marlin_template.h, `gs_par_colonne`) ;
    sinon [1]."""
    bs = t.block_scale
    N = t.qweight.shape[0]
    f_n = None
    if echelles_ecrasees(bs):
        # Pièce 157 : le facteur scalaire écraserait des sous-normales — facteur PAR LIGNE (puissance de 2 ; e4m3 × 2^k ≤
        # 448 exact), repris dans l'échelle globale PAR COLONNE (déjà servie : piles q/k/v, gemm_dense, dépaquetage
        # PAR_COLONNE, v2 × g). Les poids qu'un facteur scalaire n'écrase pas gardent EXACTEMENT la préparation d'avant.
        f_n = _facteur_depuis_max(bs.float().amax(-1)).to(bs.device)
        bs = (bs.float() * f_n[:, None]).to(bs.dtype)
        assert not echelles_ecrasees(bs), "facteur par ligne insuffisant : vérifier marlin_exact avant preparer_dense"
    w, s_, g = preparer_pile(t.qweight[None], bs[None], t.global_scale.reshape(1).float(), repack=repack)
    gsr = getattr(t, "global_scale_rows", None)
    if gsr is not None or f_n is not None:
        facteur = facteur_nvfp4(bs[None].to(torch.bfloat16))
        lignes = (gsr.float().reshape(-1) if gsr is not None
                  else t.global_scale.float().reshape(1).expand(N).to(bs.device))
        if f_n is not None:
            lignes = lignes / f_n
        g = traiter_echelle_globale(lignes, facteur).contiguous()
        assert g.numel() == N, (g.numel(), t.qweight.shape)
    return w[0], s_[0], g


def marlin_exact(t) -> "str | None":
    """Pièce 157 : None si la disposition Marlin de ``t`` (dense) rend exactement ses poids — facteur scalaire, sinon par
    ligne ; sinon la raison (le poids garde le chemin naturel, compté au bilan)."""
    if not echelles_ecrasees(t.block_scale) or not echelles_ecrasees(t.block_scale, par_ligne=True):
        return None
    return f"échelles sous-normales non représentables en Marlin : {echelles_ecrasees(t.block_scale, par_ligne=True)} blocs" 


def gemm_dense(a: torch.Tensor, w_marlin, s_marlin, g_marlin, size_n: int, size_k: int, workspace) -> torch.Tensor:
    """Pièce 101 : GEMM Marlin DENSE NVFP4 (bf16 → bf16), un poids préparé par `preparer_dense` (``g_marlin`` [1]
    scalaire, ou [N] par colonne). Réduction fp32 déterministe (use_fp32_reduce), sans atomique —
    les arguments de `apply_fp4_marlin_linear` de vLLM (marlin_utils_fp4.py:157-219) à atomique coupée."""
    ops = charger()
    return ops.marlin_gemm(a, None, w_marlin, None, s_marlin, None, g_marlin, None, None, None, workspace,
                           _kfe2m1f_id(), a.shape[0], size_n, size_k, True, False, True, False)


def gemm_moe(a: torch.Tensor, w_marlin, s_marlin, g_marlin, sorted_ids, expert_ids, num_post,
             topk_weights, block_size: int, top_k: int, size_m: int, size_n: int, size_k: int,
             workspace, mul_topk_weights: bool = False, c=None) -> torch.Tensor:
    """Une GEMM groupée Marlin (bf16 → bf16), fp32 reduce, sans atomique."""
    ops = charger()
    return ops.moe_wna16_marlin_gemm(
        a, c, w_marlin, None, s_marlin, None, g_marlin, None, None, None, workspace,
        sorted_ids, expert_ids, num_post, topk_weights, block_size, top_k, mul_topk_weights,
        _kfe2m1f_id(), size_m, size_n, size_k, True, False, True, False, -1, -1, -1)


# --- alignement CAPTURABLE (décodage sous graphe, poste7-p1-disposition-unique-18-09) ---

try:
    import triton
    import triton.language as tl
except Exception:                                        # noqa: BLE001
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _aligner_kernel(e_ptr, sorted_ptr, expert_ptr, npost_ptr, G, P, bloc,
                        BG: tl.constexpr, E: tl.constexpr, BP: tl.constexpr):
        """Un programme : tri des G paires par clé composite e·2¹⁶ + paire
        (déterministe), comptes par expert (histogramme), rembourrage de
        chaque expert à un multiple de ``bloc`` (sentinelle G), expert de
        chaque bloc — sorties de taille FIXE, aucun scalaire hôte."""
        i = tl.arange(0, BG)
        masque = i < G
        e = tl.load(e_ptr + i, mask=masque, other=E).to(tl.int32)
        tri = tl.sort(e * 65536 + i)
        rang = tl.arange(0, BG)
        e_tri = tri >> 16
        paire = tri & 65535
        valide = rang < G
        j = tl.arange(0, E)
        h = tl.histogram(tl.where(masque, e, 0), E)
        h0 = tl.sum(tl.where(masque, 0, 1), 0)
        cnt = tl.where(j == 0, h - h0, h)                              # [E]
        rembourre = ((cnt + bloc - 1) // bloc) * bloc
        debut_pad = tl.cumsum(rembourre, 0) - rembourre
        debut_tri = tl.cumsum(cnt, 0) - cnt
        # position de chaque paire triée : début rembourré de son expert + rang dans l'expert
        sel = (j[None, :] == e_tri[:, None]).to(tl.int32)               # [BG, E]
        dp = tl.sum(sel * debut_pad[None, :], 1)
        dt = tl.sum(sel * debut_tri[None, :], 1)
        pos = dp + (rang - dt)
        tl.store(sorted_ptr + pos, paire, mask=valide)
        # expert de chaque bloc : le dernier expert dont debut_pad ≤ b·bloc, parmi ceux qui ont des paires
        b = tl.arange(0, BP)
        deb_b = b * bloc
        total = tl.sum(rembourre, 0)
        masque_b = (b < P // bloc) & (deb_b < total)
        n_le = tl.sum(((debut_pad[None, :] <= deb_b[:, None]) & (rembourre[None, :] > 0)).to(tl.int32), 1)
        # n_le compte les experts non vides commencés avant ou en b·bloc ; l'expert du bloc = le n_le-ième non vide
        rang_nv = tl.cumsum((rembourre > 0).to(tl.int32), 0)            # rang (1-based) des experts non vides
        sel_b = (rang_nv[None, :] == n_le[:, None]).to(tl.int32) * (rembourre[None, :] > 0).to(tl.int32)
        e_b = tl.sum(sel_b * j[None, :], 1)
        tl.store(expert_ptr + b, e_b.to(tl.int32), mask=masque_b)
        tl.store(npost_ptr + tl.arange(0, 1), tl.full((1,), 0, tl.int32) + total)

    @triton.jit
    def _e2m1_valeur(c):
        """Code E2M1 signé (4 bits) → fp32, -0.0 pour le code 8 (comme dequantize_nvfp4)."""
        m = (c & 1).to(tl.float32)
        ex = (c >> 1) & 3
        mult = tl.where(ex == 3, 4.0, tl.where(ex == 2, 2.0, 1.0))
        v = tl.where(ex == 0, 0.5 * m, (1.0 + 0.5 * m) * mult)
        return tl.where((c & 8) != 0, -v, v)

    @triton.jit
    def _bf16_rne(x):
        """fp32 → motif binaire bf16 (int16), arrondi au plus proche pair — ce que
        fait cvt.rn.bf16.f32 sur carte ; explicite ici parce que l'interpréteur
        Triton TRONQUE (`.to(tl.bfloat16)` sous TRITON_INTERPRET=1 : 1,015625
        → 1,0078125), et que le juge est au bit (tests/test_depaqueter_marlin.py)."""
        b = x.to(tl.int32, bitcast=True)
        b = b + 0x7FFF + ((b >> 16) & 1)
        return (b >> 16).to(tl.int16)

    @triton.jit
    def _depaqueter_kernel(w_ptr, s_ptr, g_ptr, out_ptr, KT, NT, K, N, DEUX_MOINS_119: tl.constexpr,
                           PAR_COLONNE: tl.constexpr = False):
        """Un programme par tuile Marlin (expert e, tuile k kt, tuile n nt) :
        512 octets de codes + 64 octets d'échelle → 1 024 poids bf16 [16 k ×
        64 n] écrits à leur place dans out [E, N, K] (vu en int16). Même
        arithmétique que la version torch de depaqueter_marlin : fl(s·g) puis
        × code en fp32, arrondi bf16 au plus proche pair."""
        pid = tl.program_id(0)
        nt = pid % NT
        kt = (pid // NT) % KT
        e = pid // (NT * KT)
        bi = tl.arange(0, 512)
        octets = tl.load(w_ptr + (e * KT + kt) * (NT * 512) + nt * 512 + bi).to(tl.int32)
        mot = bi // 4
        b = bi % 4
        t = mot // 4
        w = mot % 4
        n = w * 16 + t // 4 + 8 * (b & 1)                     # colonne locale du quartet (0..63)
        k0 = (t % 4) * 2 + (b >> 1)                            # k local du quartet bas ; + 8 pour le haut
        q = n // 8
        p = 8 * (n % 8) + ((q & 4) | ((q & 1) << 1) | ((q >> 1) & 1))   # octet d'échelle de la colonne n
        sb = tl.load(s_ptr + (e * KT + kt) * N + nt * 64 + p).to(tl.int32)
        s_dec = ((sb << 20) + 0x34800000).to(tl.float32, bitcast=True)   # = s·facteur (S0E5M3)
        s_dec = tl.where(sb == 0, 0.0, s_dec)
        if PAR_COLONNE:                                       # pièce 134 (E = 1) et 209 (E > 1) : g par (expert, colonne)
            g = tl.load(g_ptr + e * N + nt * 64 + n)
        else:
            g = tl.load(g_ptr + e)
        ech = (s_dec * g) * DEUX_MOINS_119                    # × 2⁻¹¹⁹ (passé par l'hôte) : fl(s·g) au bit
        base = out_ptr + (e * N + nt * 64 + n) * K + kt * 16 + k0
        tl.store(base, _bf16_rne(_e2m1_valeur(octets & 0xF) * ech))
        tl.store(base + 8, _bf16_rne(_e2m1_valeur(octets >> 4) * ech))


# Pièce 147 (L3') : le dépaquetage par lignes entières de l'extension CUDA, au bit du Triton et du torch
# (tests/test_depaqueter_cuda_p147.py). ACVRAM_DEPAQUETAGE = auto (cuda si l'extension l'a) | cuda | triton | torch —
# ne joue que sous la disposition unique (ACVRAM_PROJ_MARLIN=1) : le défaut ne l'appelle jamais.
_DEPAQUETAGE = os.environ.get("ACVRAM_DEPAQUETAGE", "auto")
DEPAQUETAGES: dict = {"cuda": 0, "triton": 0}


def _depaqueter_cuda_disponible() -> bool:
    try:
        from .. import get_extension
        ext = get_extension()
    except Exception:                                        # noqa: BLE001
        return False
    return ext is not None and hasattr(ext, "depaqueter_marlin_cuda")


def _depaqueter_cuda(w: torch.Tensor, s: torch.Tensor, g: torch.Tensor, out: torch.Tensor, K: int, N: int) -> None:
    from .. import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "depaqueter_marlin_cuda"):
        raise RuntimeError("depaqueter_marlin(noyau='cuda') : extension sans depaqueter_marlin_cuda")
    E = w.shape[0]
    assert out.is_contiguous() and tuple(out.shape) == (E, N, K)
    par_colonne = E == 1 and g.numel() == N and N > 1
    ws = w if w.dtype == torch.int32 else w.view(torch.int32)
    ext.depaqueter_marlin_cuda(ws, s.view(torch.uint8) if s.dtype != torch.uint8 else s, g.contiguous().float(), out, K, N, par_colonne)


def _depaqueter_triton(w: torch.Tensor, s: torch.Tensor, g: torch.Tensor, out: torch.Tensor, K: int, N: int) -> None:
    """``w`` [E, K/16, 2N] int32, ``s`` [E, K/16, N] E4M3 (octets), ``g`` [E]
    fp32, ``out`` [E, N, K] bf16 contigu — un lancement, E·K/16·N/64 programmes."""
    if triton is None:
        raise RuntimeError("depaqueter_marlin(noyau='triton') : Triton absent")
    E = w.shape[0]
    KT, NT = K // 16, N // 64
    assert out.is_contiguous() and tuple(out.shape) == (E, N, K)
    par_colonne = g.numel() == E * N and N > 1                  # g [N] de preparer_dense (134) ou [E, N] (209)
    _depaqueter_kernel[(E * KT * NT,)](w.contiguous().view(torch.uint8), s.contiguous().view(torch.uint8),
                                       g.contiguous(), out.view(torch.int16), KT, NT, K, N,
                                       DEUX_MOINS_119=2.0 ** -119, PAR_COLONNE=par_colonne, num_warps=4)


def aligner_blocs_cuda(flat_e: torch.Tensor, block_size: int, num_experts: int, tampons=None):
    """Pièce 62 (23/09) : l aligneur CUDA de vLLM (`moe_align_block_size`, 2 lancements, ≈ 3 µs) — mêmes
    sorties que `aligner_blocs_capturable` (sorted_ids sentinelle G, expert_ids par bloc, num_post),
    tailles FIXES, capturable, ≈ 4 × moins cher que l aligneur Triton (12 µs par couche mesurés en service).
    ``flat_e`` [G] int32 (expert de chaque paire, ordre jeton-majeur, aucun < 0)."""
    ops = charger(compiler=False)
    G = flat_e.numel()
    E = num_experts
    P = G + E * (block_size - 1)
    P = -(-P // block_size) * block_size
    if tampons is None:
        sorted_ids = torch.empty(P, dtype=torch.int32, device=flat_e.device)
        expert_ids = torch.empty(P // block_size, dtype=torch.int32, device=flat_e.device)
        num_post = torch.empty(1, dtype=torch.int32, device=flat_e.device)
    else:
        sorted_ids, expert_ids, num_post = tampons
    ops.moe_align_block_size(flat_e.view(1, -1) if flat_e.dim() == 1 else flat_e, E, block_size, sorted_ids, expert_ids, num_post, None)
    return sorted_ids, expert_ids, num_post


def aligner_blocs_capturable(flat_e: torch.Tensor, block_size: int, num_experts: int, tampons=None):
    """`aligner_blocs` en UN lancement Triton, sorties de taille fixe
    (P_max = G + E·(block−1)) : capturable dans un graphe CUDA. ``flat_e``
    [G] int32 (expert de chaque paire, ordre des jetons). ``tampons`` :
    (sorted_ids, expert_ids, num_post) réutilisés (adresses stables)."""
    G = flat_e.numel()
    E = num_experts
    P = G + E * (block_size - 1)
    P = -(-P // block_size) * block_size
    if tampons is None:
        sorted_ids = torch.empty(P, dtype=torch.int32, device=flat_e.device)
        expert_ids = torch.empty(P // block_size, dtype=torch.int32, device=flat_e.device)
        num_post = torch.empty(1, dtype=torch.int32, device=flat_e.device)
    else:
        sorted_ids, expert_ids, num_post = tampons
    sorted_ids.fill_(G)
    expert_ids.fill_(0)
    BG = 16
    while BG < G:
        BG *= 2
    BP = 16
    while BP < P // block_size:
        BP *= 2
    _aligner_kernel[(1,)](flat_e.to(torch.int32).contiguous(), sorted_ids, expert_ids, num_post, G, P, block_size,
                          BG=BG, E=E, BP=BP, num_warps=4)
    return sorted_ids, expert_ids, num_post
