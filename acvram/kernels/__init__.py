"""Chargement des noyaux fusionnés, avec un repli qui fonctionne en leur absence.

L'extension CUDA est compilée au premier usage par
``torch.utils.cpp_extension`` et mise en cache sous ``~/.cache/acvram/kernels``.
Si nvcc manque, si l'architecture n'est pas gérée ou si la compilation échoue,
chaque point d'entrée retombe sur l'implémentation PyTorch de référence. Ce
repli est lent — il matérialise une copie 16 bits des poids — mais il est
numériquement identique : le moteur tourne donc correctement sur une machine
sans compilateur, et la suite de tests peut vérifier les noyaux face à lui.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import os
import time
import re
import shutil
import sys
import warnings
from typing import Any, Optional

import torch

from ..quant.int4 import INT4Tensor, dequantize_int4
from ..quant.formats import INT8Tensor, _dequantize_int8
from ..quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4

from .cpu import (cpu_build_info, cpu_kernels_available, int4_matmul_cpu,
                  nvfp4_matmul_cpu)
from .fp4_gemm import (fp4_mm_available, fp4_mm_info, nvfp4_mm_tensorcore,
                       nvfp4_mm_w4a8)

__all__ = ["get_extension", "kernels_available", "build_info", "matmul",
           "nvfp4_dequant", "nvfp4_matmul", "int4_dequant", "int4_matmul",
           "int8_dequant", "int8_matmul",
           "cpu_kernels_available", "cpu_build_info",
           "fp4_mm_available", "fp4_mm_info"]

_EXT: Optional[Any] = None
_TRIED = False
_ERROR: str = ""
_SO_HASH: str = ""          # sha256 du .so effectivement charge : a joindre
_SO_PATH: str = ""          # a tout releve, car une empreinte .cu/.so prouve
                            # la coherence, jamais l identite de l arbre

# Blackwell exige CUDA 12.8 ou plus récent ; rien de plus ancien ne sait émettre du sm_120.
_MIN_CUDA_FOR_SM120 = (12, 8)


# Le suffixe « f » (family-specific) est apparu avec CUDA 12.9.
_MIN_CUDA_FOR_FAMILY = (12, 9)


def _arch_flags(nvcc_ver: tuple[int, int] | None = None) -> list[str]:
    """Émet du code pour exactement les architectures présentes, plus un repli PTX.

    Les cibles ``sm_100`` et au-delà sont demandées sous leur forme
    *family-specific* (``sm_120f``) et non générique. Ce n'est pas un détail de
    portabilité : ``cuda_fp8.h`` ne définit ``__CUDA_ARCH_FAMILY_SPECIFIC__``
    que dans ce mode, et sans lui ``__nv_cvt_fp4x2_to_halfraw2`` — la
    conversion E2M1 vers half2 qui décode *chaque poids* d'un modèle NVFP4 —
    retombe sur une émulation arithmetique de vingt-cinq instructions par paire
    de poids, au lieu de l'unique ``cvt.rn.f16x2.e2m1x2`` du materiel. Mesuré
    au desassemblage : 11 LOP3, 6 IMAD, 4 IADD, 2 PRMT, 2 SEL et 1 SHF
    disparaissent d'un coup. Une cible « f » reste compatible avec toute la
    famille (sm_121, sm_128...), contrairement au suffixe « a ».
    """
    archs: set[tuple[int, int]] = set()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            archs.add(torch.cuda.get_device_capability(i))
    if not archs:
        archs = {(8, 6), (12, 0)}
    if nvcc_ver is None:
        nvcc_ver = _nvcc_version(_nvcc_path())
    demande = os.environ.get("ACVRAM_ARCH_FAMILY", "1") != "0"
    family = nvcc_ver >= _MIN_CUDA_FOR_FAMILY and demande
    if demande and not family and any(major >= 10 for major, _ in archs):
        # Sans la forme famille, la conversion E2M1 redevient une émulation de
        # vingt-cinq instructions par paire de poids — et rien ne le disait.
        # Le 9/09/2026 nous avons cru cette perte réelle pendant une heure : le
        # binaire était bon, mais aucun message n'aurait signalé qu'il ne
        # l'était pas. La marge est d'UNE version — le seuil est 12.9 et le
        # toolkit du virtualenv fournit 13.0 ; qu'il disparaisse et la perte
        # revient en silence.
        warnings.warn(
            f"FP4 materiel indisponible : nvcc {nvcc_ver[0]}.{nvcc_ver[1]} < "
            f"{_MIN_CUDA_FOR_FAMILY[0]}.{_MIN_CUDA_FOR_FAMILY[1]}, donc "
            f"sm_120f n'est pas demande et la conversion E2M1 retombe sur "
            f"vingt-cinq instructions par paire de poids au lieu d'une. "
            f"Posez ACVRAM_CUDA_HOME sur un toolkit 12.9 ou plus recent.")
    flags: list[str] = []
    for major, minor in sorted(archs):
        cc = f"{major}{minor}"
        suf = "f" if family and major >= 10 else ""
        flags += [f"-gencode=arch=compute_{cc}{suf},code=sm_{cc}{suf}"]
    # Le repli PTX reste générique : une famille ne se compile pas en PTX portable.
    highest = max(archs)
    flags += [f"-gencode=arch=compute_{highest[0]}{highest[1]},"
              f"code=compute_{highest[0]}{highest[1]}"]
    return flags


def _nvcc_path() -> str:
    """Le nvcc du toolkit, pas celui du PATH.

    Le 11/09/2026, deux sessions ont perdu une heure chacune sur la meme
    cause : `/usr/bin/nvcc` (paquet Debian, 12.0) precede `/usr/local/cuda/
    bin/nvcc` (13.2) dans le PATH d'un shell non interactif. L'extension se
    recompilait sans sm_120f, le noyau nvfp4 rendait None, et le diagnostic
    lu etait « toolkit trop ancien » — faux, le bon toolkit etait la.

    Ordre : CUDA_HOME explicite, puis le toolkit installe sous /usr/local, et
    seulement en dernier ce que le PATH propose. Un toolkit present sur le
    disque vaut mieux qu'un lien de distribution."""
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        return os.path.join(home, "bin", "nvcc")
    # Ubuntu (12/09/2026) installe /usr/local/cuda-13.4 sans lien
    # /usr/local/cuda : on prend le toolkit versionne le plus recent.
    versionnes = sorted(glob.glob("/usr/local/cuda-[0-9]*/bin/nvcc"), reverse=True)
    for cand in ("/usr/local/cuda/bin/nvcc", *versionnes, "/opt/cuda/bin/nvcc"):
        if os.path.exists(cand):
            # torch.utils.cpp_extension ne lit PAS ce module : il consulte
            # CUDA_HOME lui-meme, puis le PATH. Sans cette ligne, notre choix
            # ne vaut que pour notre propre detection, et torch recompile avec
            # le 12.0 du PATH — constate par f2 le 11/09 apres le correctif
            # precedent. Poser la variable rend le choix effectif pour les deux.
            os.environ.setdefault("CUDA_HOME", os.path.dirname(os.path.dirname(cand)))
            return cand
    return shutil.which("nvcc") or ""


def _cuda_version() -> tuple[int, int]:
    v = torch.version.cuda or "0.0"
    try:
        parts = v.split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return (0, 0)


def _nvcc_version(nvcc: str) -> tuple[int, int]:
    """Version de nvcc, ou (0, 0) s'il est introuvable ou muet."""
    import subprocess
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _venv_cuda_home() -> Optional[str]:
    """Racine du toolkit CUDA livré par pip (paquets ``nvidia-cuda-nvcc``).

    Les roues ``cuda-toolkit[nvcc]`` déposent un arbre complet sous
    ``site-packages/nvidia/cuXX``. Il lui manque le ``lib64`` et le
    ``libcudart.so`` non versionné que ``cpp_extension`` attend ; on les pose
    en liens symboliques, ce qui est sans effet s'ils existent déjà.
    """
    import glob
    import sysconfig
    roots = [sysconfig.get_paths().get("purelib", ""),
             os.path.join(sys.prefix, "lib")]
    for root in roots:
        for cand in sorted(glob.glob(os.path.join(root, "**", "nvidia", "cu[0-9]*"),
                                     recursive=True), reverse=True):
            if not os.path.isfile(os.path.join(cand, "bin", "nvcc")):
                continue
            try:
                lib = os.path.join(cand, "lib")
                lib64 = os.path.join(cand, "lib64")
                if os.path.isdir(lib) and not os.path.exists(lib64):
                    os.symlink("lib", lib64)
                so = os.path.join(lib, "libcudart.so")
                if not os.path.exists(so):
                    for versioned in sorted(glob.glob(so + ".*")):
                        os.symlink(os.path.basename(versioned), so)
                        break
            except OSError:
                pass                      # arbre en lecture seule : tant pis
            return cand
    return None


def _ensure_cuda_home(need: tuple[int, int]) -> None:
    """Choisit un nvcc capable d'émettre pour ``need``, sans rien exiger du système.

    Une distribution peut livrer un nvcc plus ancien que la roue torch installée
    — Linux Mint 22.3 fournit CUDA 12.0, qui ignore ``compute_120``. Dans ce
    cas on bascule ``CUDA_HOME`` sur le toolkit du virtualenv.
    """
    if os.environ.get("ACVRAM_CUDA_HOME"):
        os.environ["CUDA_HOME"] = os.environ["ACVRAM_CUDA_HOME"]
        return
    current = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = (os.path.join(current, "bin", "nvcc") if current
            else shutil.which("nvcc") or "")
    if nvcc and _nvcc_version(nvcc) >= need:
        return
    venv = _venv_cuda_home()
    if venv is not None and _nvcc_version(os.path.join(venv, "bin", "nvcc")) >= need:
        os.environ["CUDA_HOME"] = venv
        os.environ["PATH"] = os.path.join(venv, "bin") + os.pathsep + os.environ.get("PATH", "")


def build_info() -> dict:
    caps = []
    if torch.cuda.is_available():
        caps = [f"sm_{a}{b}" for a, b in
                (torch.cuda.get_device_capability(i)
                 for i in range(torch.cuda.device_count()))]
    return {
        "available": kernels_available(),
        "error": _ERROR,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_caps": caps,
        "arch_flags": _arch_flags(),
        "cpu": cpu_build_info(),
        "fp4_tensorcore": fp4_mm_info(),
    }


def _purger_verrou(cache: str, age_max: float = 300.0) -> None:
    """torch.utils.cpp_extension pose un fichier ``lock`` (FileBaton) le temps
    de compiler et l'attend indéfiniment s'il existe déjà. Un processus tué
    pendant une compilation le laisse derrière lui : tout chargement suivant
    restait alors en veille sans un mot. Un verrou plus vieux que
    ``age_max`` secondes est réputé orphelin et retiré."""
    verrou = os.path.join(cache, "lock")
    try:
        age = time.time() - os.path.getmtime(verrou)
    except OSError:
        return
    if age > age_max:
        try:
            os.remove(verrou)
            print(f"[acvram] verrou de compilation orphelin retiré ({age:.0f} s) : {verrou}",
                  file=sys.stderr)
        except OSError:
            pass


class _ExtensionMasquee:
    """L'extension sans certains symboles : `hasattr` rend faux, le moteur
    prend le repli torch de ces noyaux seuls (bissection noyau par noyau,
    Laure geste B 16/09, remonté du scratchpad dans le dépôt le 17/09)."""

    def __init__(self, ext, noms):
        self._ext, self._noms = ext, frozenset(noms)

    def __getattr__(self, n):
        if n in self._noms:
            raise AttributeError(n)
        return getattr(self._ext, n)


def masquer_noyaux(noms) -> None:
    """Cache ces noyaux à l'extension (chargée si besoin). Un nom que
    l'extension n'a pas est une erreur : un masque qui ne masque rien
    fabrique un « sans effet »."""
    global _EXT
    ext = get_extension()
    if ext is None:
        raise RuntimeError("extension absente : rien à masquer")
    base = ext._ext if isinstance(ext, _ExtensionMasquee) else ext
    deja = set(ext._noms) if isinstance(ext, _ExtensionMasquee) else set()
    for n in noms:
        if not hasattr(base, n):
            raise KeyError(f"noyau inconnu : {n}")
    _EXT = _ExtensionMasquee(base, deja | set(noms))


def noyaux_masques() -> list[str]:
    return sorted(_EXT._noms) if isinstance(_EXT, _ExtensionMasquee) else []


def get_extension():
    """Compile une fois, puis rend le module d'extension, ou None."""
    global _EXT, _TRIED, _ERROR
    if _TRIED:
        return _EXT
    _TRIED = True

    if os.environ.get("ACVRAM_DISABLE_KERNELS"):
        _ERROR = "desactive par ACVRAM_DISABLE_KERNELS"
        return None
    if not torch.cuda.is_available():
        _ERROR = "aucun peripherique CUDA"
        return None

    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    if any(c >= (12, 0) for c in caps) and _cuda_version() < _MIN_CUDA_FOR_SM120:
        _ERROR = (f"un peripherique Blackwell est present mais torch a ete "
                  f"compile pour CUDA {torch.version.cuda} ; sm_120 exige 12.8 "
                  f"ou plus recent. Installez une version cu128 ou cu130.")
        warnings.warn(_ERROR)
        return None

    # Le source inclut ``cuda_fp4.h``, qui n'existe qu'a partir de CUDA 12.8 :
    # l'exigence ne depend pas de l'architecture visee. Un poste dont le nvcc
    # systeme est plus ancien (Mint 22.3 livre CUDA 12.0) compilait sans
    # broncher pour sm_86 et echouait sur l'en-tete manquant.
    _ensure_cuda_home(_MIN_CUDA_FOR_SM120)

    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        # UN REPERTOIRE DE COMPILATION PAR ARBRE. Le defaut etait partage par
        # les quatre worktrees, sous un nom de module fixe : deux sessions dont
        # les sources different s'ecrasent le meme .so, et _purger_verrou()
        # retire le verrou d'une compilation qui n'est pas la sienne. Le
        # symptome est exactement celui qu'on avait attribue a ccache — un
        # binaire coherent avec un source qui n'est pas le votre, une date
        # rassurante, aucune erreur. Les deux mecanismes existent ; celui-ci
        # etait invisible.
        # La cle est le chemin du paquet : deux arbres ne peuvent plus se
        # rencontrer, et un meme arbre garde son cache d'une fois sur l'autre.
        cache = os.environ.get("ACVRAM_KERNEL_CACHE")
        if not cache:
            _cle = hashlib.sha256(
                os.path.realpath(here).encode()).hexdigest()[:12]
            cache = os.path.expanduser(f"~/.cache/acvram/kernels-{_cle}")
        os.makedirs(cache, exist_ok=True)
        _purger_verrou(cache)
        # EMPREINTE DU SOURCE, injectee comme option de compilation.
        # `ccache` enveloppe nvcc (build.ninja) et son hachage ne distingue pas
        # toujours deux versions du code *device* : le 9/09/2026 une
        # modification du .cu a rendu un binaire compile en 191 ms qui ne la
        # contenait pas, tout en etant PLUS RECENT que la source. Quatre
        # valeurs d'un parametre ont ainsi donne quatre fois le meme chiffre —
        # le meme binaire — et la conclusion qu'on allait en tirer etait fausse.
        # Une option qui CHANGE avec le contenu interdit structurellement a
        # ccache de rendre un objet perime, sans le desactiver ni perdre son
        # benefice sur les compilations legitimes.
        src = os.path.join(here, "acvram_kernels.cu")
        with open(src, "rb") as fh:
            _SRC_OCTETS = fh.read()
        # TOUT CE QUI ENTRE DANS LA CONSTRUCTION ENTRE DANS LE HASH — et voici
        # exactement contre quoi cela protege, ni plus ni moins.
        #
        # CE QUI N'ETAIT PAS LE DEFAUT, verifie plutot que suppose : ccache et
        # ninja distinguent DEJA les drapeaux. Trois compilations enchainees
        # dans le meme cache, mesurees le 10/09 :
        #     CVD=0     62 s  empreinte 403da9739cb9
        #     CVD=0,1   83 s  empreinte 3eb2234a6864   (reconstruction)
        #     CVD=0      1 s  empreinte 403da9739cb9   (le premier objet revient)
        # Le binaire n'est donc JAMAIS croise : la cle de ccache contient la
        # ligne de commande, et ninja reconstruit quand elle change. Une
        # premiere version de ce commentaire affirmait le contraire ; elle
        # sur-estimait le danger, et une explication trop belle est un defaut.
        #
        # CE QUI ETAIT LE DEFAUT : le repertoire de compilation ne contient
        # qu'UN acvram_kernels.so, reecrit a chaque changement de drapeaux.
        # Deux processus concurrents aux drapeaux differents se le disputent, et
        # l'un peut charger le binaire construit pour l'autre. Le hash aveugle
        # aux drapeaux ne pouvait pas le voir : il declarait coherent un .so
        # construit pour une autre architecture. C'est le controle qui etait
        # partiel, pas la construction qui etait fausse — et une empreinte
        # partielle est pire que pas d'empreinte, parce qu'elle rassure.
        #
        # DEUX chemins y echappaient, pas un :
        #   ACVRAM_GW_WARPS   -> -DGW_WARPS=n, qui sert le MoE groupe ;
        #   ACVRAM_ARCH_FAMILY et les cartes visibles -> _arch_flags(), dont
        #   l'effet est majeur : sans le mode family-specific, la conversion
        #   E2M1 retombe sur vingt-cinq instructions d'emulation par paire de
        #   poids. Deux binaires que tout separe en performance portaient donc
        #   la meme empreinte.
        #
        # On ne les enumere plus : on hache LA LISTE COMPLETE des drapeaux
        # effectivement passes au compilateur. Tout ajout futur y entre de
        # lui-meme, et le controle cesse d'etre partiel. Une empreinte
        # partielle est pire que pas d'empreinte, parce qu'elle rassure.
        _flags_cuda = (["-O3", "--use_fast_math", "-lineinfo"]
                       + ([f"-DGW_WARPS={os.environ['ACVRAM_GW_WARPS']}"]
                          if os.environ.get("ACVRAM_GW_WARPS") else [])
                       + _arch_flags())
        _flags_c = ["-O3"]
        _SRC_HASH = hashlib.sha256(
            _SRC_OCTETS
            + b"\x00FLAGS\x00"
            + "\x00".join(_flags_cuda + _flags_c).encode()
        ).hexdigest()[:16]
        # LE CONTROLE EST-IL SEULEMENT APPLICABLE ? L'absence du marqueur dans
        # le .so a DEUX causes : un binaire perime, ou un source qui n'en porte
        # pas. Ne proposer que la premiere l'a fait accuser a tort, et le
        # controle a coute deux compilations a une autre session avec un
        # message qui designait la mauvaise piste — le dispositif s'etait
        # transporte a moitie, le marqueur etant dans le .cu et le controle
        # dans ce fichier. Un controle qui ne peut pas s'appliquer doit LE
        # DIRE, jamais conclure. Marqueur et controle voyagent ensemble.
        # ...ET LE DIRE AVANT DE COMPILER. Place apres le `load()`, ce refus
        # aurait coute neuf minutes de nvcc pour annoncer que rien ne pouvait
        # les satisfaire. Un controle inapplicable se declare tout de suite.
        if b"acvram_src_hash" not in _SRC_OCTETS:
            _ERROR = (f"le source {src} ne contient pas la variable "
                      f"`acvram_src_hash` : le controle d'empreinte est "
                      f"INAPPLICABLE sur cet arbre, aucun binaire ne pourra le "
                      f"satisfaire — et rien ici ne permet d'accuser un cache "
                      f"de compilation. Reportez le bloc `extern \"C\" ... "
                      f"acvram_src_hash = ACVRAM_SRC_HASH` dans le .cu, ou "
                      f"reprenez la version du noyau qui va avec ce controle.")
            warnings.warn(f"acvram : {_ERROR}")
            _EXT = None
            return None
        _SRC_U64 = int(_SRC_HASH, 16)          # entier : aucun guillemet a echapper
        _EXT = load(
            name="acvram_kernels",
            sources=[src],
            # la meme liste que celle qui a ete hachee, plus l'empreinte
            extra_cuda_cflags=_flags_cuda + [f"-DACVRAM_SRC_HASH={_SRC_U64}ULL"],
            extra_cflags=_flags_c,
            build_directory=cache,
            verbose=bool(os.environ.get("ACVRAM_VERBOSE_BUILD")),
        )
        # LE BINAIRE PORTE-T-IL BIEN CE SOURCE ? Une fois, au chargement,
        # jamais dans le chemin chaud. Comparer les horodatages ne prouve
        # rien : ccache reecrit le .so, donc sa date est bonne et son contenu
        # ancien. Seul le CONTENU repond.
        so = os.path.join(cache, "acvram_kernels.so")
        try:
            with open(so, "rb") as fh:
                octets = fh.read()
                # l'entier est ecrit en little-endian dans le binaire
                porte = _SRC_U64.to_bytes(8, "little") in octets
            # EMPREINTE DU BINAIRE LUI-MEME, a joindre a tout releve. Le
            # controle ci-dessus prouve que le .so est COHERENT avec un .cu ;
            # il ne peut pas voir que le couple entier vient d'un autre arbre —
            # demontre le 10/09, ou PYTHONPATH manquant faisait mesurer le
            # depot principal avec son propre binaire, parfaitement coherent.
            # Une empreinte prouve la coherence, pas l'identite : c'est le sha
            # du .so, joint au chiffre, qui identifie ce qui a tourne.
            global _SO_HASH, _SO_PATH
            _SO_HASH = hashlib.sha256(octets).hexdigest()[:16]
            _SO_PATH = so
        except OSError:
            porte = True                       # pas de .so a inspecter : on n'accuse pas
        if not porte:
            _ERROR = (f"le binaire {so} ne porte pas l'empreinte du source "
                      f"({_SRC_HASH}) alors que ce source PORTE bien un "
                      f"marqueur : le .so ne contient pas vos modifications. "
                      f"Videz {cache} — `CCACHE_DISABLE=1` seul ne suffit "
                      f"pas, il n'entre pas dans la ligne de commande que "
                      f"ninja compare, donc ninja voit son .so a jour et ne "
                      f"recompile rien.")
            warnings.warn(f"acvram : {_ERROR}")
            _EXT = None
            return None
    except Exception as exc:                      # noqa: BLE001 — signaler, pas planter
        _ERROR = f"{type(exc).__name__}: {exc}"
        _EXT = None
        warnings.warn(f"acvram : repli sur les noyaux de reference ({_ERROR})")
    return _EXT


def kernels_available() -> bool:
    return get_extension() is not None


# --------------------------------------------------------------------------
# NVFP4
# --------------------------------------------------------------------------


def nvfp4_dequant(t: NVFP4Tensor, dtype: torch.dtype = torch.bfloat16,
                  gscale_rows: Optional[torch.Tensor] = None,
                  rows_per_group: int = 1) -> torch.Tensor:
    """``gscale_rows`` [M / rows_per_group] (fp32) : une échelle globale par
    groupe de lignes (pile d'experts), à la place de ``t.global_scale``."""
    ext = get_extension()
    if ext is None or not t.qweight.is_cuda:
        if gscale_rows is None:
            return dequantize_nvfp4(t, dtype)
        # Le jumeau torch du noyau : échelle de bloc × échelle globale DE LA
        # LIGNE en fp32, puis × code, un seul arrondi vers dtype — pas une
        # déquantification en bf16 remultipliée ensuite (double arrondi, 1 ulp
        # d'écart avec le noyau, verdict-repli-torch-passage-direct-17-09).
        # Les piles (model.py _pile_bf16) portent global_scale = 1 : gscale_rows
        # est alors l'échelle globale de chaque expert telle quelle.
        gsr = gscale_rows.to(torch.float32).repeat_interleave(int(rows_per_group))
        jumeau = NVFP4Tensor(t.qweight, t.block_scale, t.global_scale, t.shape, t.padded_in,
                             global_scale_rows=gsr)      # remplace t.global_scale, comme le noyau
        return dequantize_nvfp4(jumeau, dtype)
    out = ext.nvfp4_dequant(
        t.qweight.contiguous(),
        t.block_scale.view(torch.uint8).contiguous(),
        t.global_scale_float(),
        t.padded_in, dtype,
        None if gscale_rows is None else gscale_rows.to(torch.float32).contiguous(),
        int(rows_per_group))
    return out[:, : t.shape[-1]] if t.padded_in != t.shape[-1] else out


# Contrairement au seuil INT8, celui-ci est bien placé : le chemin W4A8 ne
# matérialise pas le poids entier à chaque appel.
#
# Le seuil a valu 8 jusqu'au 10/09/2026, sur ce balayage : « dense de 27B, TTFT
# d'une invite de 16 jetons, 194,7 ms à 8, 202,8 à 32, 265,1 à 64 » — une
# mesure à UNE séquence, où le décodage ne franchit jamais le seuil. Elle ne
# disait donc rien du seul régime où il décide : la CONCURRENCE. Les deux
# mesures ne se contredisaient pas, elles ne parlaient pas du même monde.
#
# Refait le 10/09/2026 sous verrou de carte (`outils/carte.sh` — une première
# campagne avait été jetée : une autre session chargeait en même temps, et
# c'est le bras SURVIVANT, pas celui mort en OOM, qui rendait un chiffre que
# rien ne signalait comme faux). ABBA, une valeur par processus, le seuil étant
# lu à l'import. Chaque bras deux fois ; la dispersion INTRA-bras est donnée
# pour que l'écart se lise contre elle.
#
#   modele      regime            seuil 8          seuil 32        ecart
#   Qwen3-4B    12 seq, debit   18,93 18,93 p/s  53,67 53,65     x2,84
#   Qwen3-4B    12 seq, TTFT    1021,2 1013,1 ms  924,6  926,3    -9,4 %
#   Qwen3-4B     1 seq, TTFT     288,7  289,9     227,1  229,1   -21,3 %
#   Qwen3-4B     1 seq, decode  227,11 227,00 p/s 227,41 226,59    nul
#   AWAXIS-31B   1 seq, TTFT     464,0  463,0     433,2  434,6    -6,5 %
#   AWAXIS-31B   1 seq, decode   47,42  47,32     47,40  47,45     nul
#
# Dispersion intra-bras : au plus 8 ms et 0,1 pas/s. Chaque ecart vaut 4 a 200
# fois cette dispersion. A une sequence le decodage ne franchit pas le seuil,
# des deux cotes : sa neutralite est le TEMOIN de la manche.
#
# Et la justesse va dans le meme sens, ce que personne n'avait mesure : contre
# le poids REELLEMENT stocke, le GEMV rend 55,6 dB la ou le chemin W4A8 rend
# 27,9 — +27,8 dB. Au-dessus de huit lignes on payait donc une erreur quatre
# fois plus grande sans l'avoir choisie.
#
# 32 plutôt que plus haut : le croisement mesuré sur six formes du parc va de
# 40 à plus de 64, et un seuil sous le plus petit croisement ne peut pas perdre
# sur une forme non balayée. La variable reste comme échappement.
_NVFP4_GEMV_MAX = int(os.environ.get("ACVRAM_NVFP4_GEMV_MAX", "32"))
PREFILL_REGIMES = ("bf16", "w4a16", "w8a8", "w4a4")


def prefill_regime() -> str:
    """Régime du prefill NVFP4 au-delà du seuil GEMV, lu à chaque appel :
    ``bf16`` (défaut, exact : déquant puis cuBLAS), ``w4a16`` (même
    arithmétique, poids lus en 4 bits dans la tuile — kernels/gemm_groupe,
    B1, sortie = bf16 ± 2⁻⁷), ``w8a8``, ``w4a4``. Un nom inconnu — dont les
    anciens ``a8``/``a4`` — est une erreur, pas un repli silencieux."""
    mode = os.environ.get("ACVRAM_PREFILL", "bf16")
    if mode not in PREFILL_REGIMES:
        raise ValueError(f"ACVRAM_PREFILL={mode!r} : attendu {', '.join(PREFILL_REGIMES)}")
    return mode
# GEMM étroit sur tensor cores (1aj marche 2, 15/09) pour les linéaires int8 et
# NVFP4 à M petit ; ACVRAM_NARROW_GEMM=1 pour l'ouvrir, ACVRAM_NARROW_MIN_M = lot
# minimal (à M=1 le GEMV lit x une fois et reste bon).
_NARROW_GEMM = os.environ.get("ACVRAM_NARROW_GEMM", "0") == "1"
_NARROW_MIN = int(os.environ.get("ACVRAM_NARROW_MIN_M", "2"))
# NVFP4 : le chemin étroit est plus LENT que son GEMV (D b=12 : 13,41 → 14,08 ms,
# 15/09) — décodage E2M1 + une MMA par bloc de 16 ; coupé tant que non repris.
_NARROW_NVFP4 = os.environ.get("ACVRAM_NARROW_NVFP4", "0") == "1"


_NARROW_ROWS = int(os.environ.get("ACVRAM_NARROW_ROWS", "32"))


def _narrow_rows(n_sortie: int) -> int:
    """ACVRAM_NARROW_ROWS (32) lignes par CTA sous 16 384 sorties, 128 au-delà (lm_head)."""
    return _NARROW_ROWS if n_sortie <= 16384 else 128


def nvfp4_matmul(x: torch.Tensor, t: NVFP4Tensor,
                 gemv_threshold: int = 0) -> torch.Tensor:
    """``x @ W.T`` avec W stocké en NVFP4.

    En dessous de ``gemv_threshold`` lignes, le chemin fusionné l'emporte : les
    poids sont lus une fois et jamais réécrits en 16 bits. Au-dessus,
    matérialiser la matrice et la confier à cuBLAS est plus rapide, car le coût
    de déquantification est payé une seule fois pour tout le lot et le produit
    matriciel de cuBLAS est bien mieux réglé que tout ce qu'on écrirait ici.
    """
    if not t.qweight.is_cuda:
        # Étage hôte : on lit les poids empaquetés sur place, plutôt que de les
        # copier vers le GPU ou de les étendre d'abord en 16 bits.
        return nvfp4_matmul_cpu(x, t)

    if gemv_threshold <= 0:
        gemv_threshold = _NVFP4_GEMV_MAX
    ext = get_extension()
    orig_shape = x.shape
    xf = x.reshape(-1, x.shape[-1])
    n = xf.shape[0]

    if ext is not None and n <= gemv_threshold and t.padded_in % 32 == 0:
        # Le noyau lit les blocs par paires (32 poids) ; un K non multiple de
        # 32 — jamais vu sur un vrai modele — prend le chemin dequantifie.
        if t.padded_in != xf.shape[-1]:
            xf = torch.nn.functional.pad(xf, (0, t.padded_in - xf.shape[-1]))
        gsr = getattr(t, "global_scale_rows", None)
        if _DENSE_NVFP4 == "triton" and _DENSE_NVFP4_MIN_M <= n <= 32 and xf.dtype == torch.bfloat16:
            # GEMM dense étroite W4A16 (sage-hybrides-etape1-close-gemm-dense-
            # 17-09 § 2) : les M lignes en registres, les poids lus UNE fois
            # par pas — la boucle GEMV ci-dessous les relit par séquence
            # (Qwen3.8 b=12 : 0,23 To/s). M = 1 garde la GEMV.
            from . import gemm_dense_etroit
            if gemm_dense_etroit.disponible():
                y = gemm_dense_etroit.gemm_dense_etroit(xf, t)
                return y.reshape(*orig_shape[:-1], t.shape[0])
        if (_NARROW_GEMM and _NARROW_NVFP4 and gsr is None and _NARROW_MIN <= n <= 16 and t.padded_in % 64 == 0
                and xf.dtype == torch.bfloat16 and hasattr(ext, "narrow_gemm")):
            # 1aj marche 2 : GEMM étroit tensor cores, poids lus une fois par
            # CTA en étages (le GEMV relisait W par tranche de 8 et ne
            # recouvrait aucune latence : 767 Go/s à b=12).
            y = ext.narrow_gemm(t.qweight.contiguous(), t.block_scale.view(torch.uint8).contiguous(),
                                None, None, xf.contiguous(), t.padded_in, 16, t.global_scale_float(),
                                _narrow_rows(t.shape[0]))
            return y.to(x.dtype).reshape(*orig_shape[:-1], t.shape[0])
        y = ext.nvfp4_gemv(
            t.qweight.contiguous(),
            t.block_scale.view(torch.uint8).contiguous(),
            1.0 if gsr is not None else t.global_scale_float(),
            xf.contiguous(), t.padded_in, gsr)
        return y.to(x.dtype).reshape(*orig_shape[:-1], t.shape[0])

    gsr = getattr(t, "global_scale_rows", None)
    if gsr is not None:
        # Pile à une échelle globale par segment (q, k, v ou q_a, kv_a
        # empilés) : seul le noyau GEMV la lit. Les chemins tensor cores
        # ci-dessous prennent ``t.global_scale``, celle du premier segment,
        # pour toute la pile — sur GLM-4.7 le latent kv en sortait à la
        # mauvaise échelle dès que le prefill dépassait huit jetons, et le
        # modèle répondait « de de de de ». La déquantification, elle, sait
        # appliquer une échelle par ligne.
        w = nvfp4_dequant(t, x.dtype if x.dtype != torch.float32 else torch.bfloat16,
                          gscale_rows=gsr, rows_per_group=1)
        return torch.nn.functional.linear(x, w.to(x.dtype))

    # Prefill. Par défaut ``bf16`` : déquantification exacte puis cuBLAS —
    # W4A16 au sens propre. Les deux autres régimes changent la sortie et se
    # demandent par leur nom (sage-prefill-a8-verdict-17-09) : ``w8a8``
    # requantifie le poids déquantifié en E4M3 par ligne ET l'activation en
    # FP8 (double quantification, 3,6-4,0 % RMS mesurés contre 0,14 % en bf16,
    # verdict-diff-moe-prefill-w4a8-17-09 — ce fut le défaut « a8 » jusqu'au
    # 17/09, et le 1 % perdu contre Marlin sur GLM) ; ``w4a4`` quantifie
    # l'activation en E2M1 bloc 16 sur la MMA FP4. Ni l'un ni l'autre ne
    # retombe sur un régime tiers : indisponible → bf16.
    if n > gemv_threshold:
        mode = prefill_regime()
        if mode == "w4a4":
            tc = nvfp4_mm_tensorcore(x, t)
            if tc is not None:
                return tc
        elif mode == "w8a8":
            tc = nvfp4_mm_w4a8(x, t)
            if tc is not None:
                return tc
        elif mode == "w4a16":
            from . import gemm_groupe
            if gemm_groupe.disponible():
                return gemm_groupe.nvfp4_linear(x, t)

    w = nvfp4_dequant(t, x.dtype if x.dtype != torch.float32 else torch.bfloat16)
    return torch.nn.functional.linear(x, w.to(x.dtype))


# --------------------------------------------------------------------------
# INT4
# --------------------------------------------------------------------------


def int4_dequant(t: INT4Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    ext = get_extension()
    if ext is None or not t.qweight.is_cuda:
        return dequantize_int4(t, dtype)
    out = ext.int4_dequant(
        t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
        t.padded_in, t.group_size, dtype)
    return out[:, : t.shape[-1]] if t.padded_in != t.shape[-1] else out


def int4_matmul(x: torch.Tensor, t: INT4Tensor,
                gemv_threshold: int = 8) -> torch.Tensor:
    if not t.qweight.is_cuda:
        return int4_matmul_cpu(x, t)

    ext = get_extension()
    orig_shape = x.shape
    xf = x.reshape(-1, x.shape[-1])
    n = xf.shape[0]

    if ext is not None and n <= gemv_threshold:
        if t.padded_in != xf.shape[-1]:
            xf = torch.nn.functional.pad(xf, (0, t.padded_in - xf.shape[-1]))
        y = ext.int4_gemv(
            t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
            xf.contiguous(), t.padded_in, t.group_size)
        return y.to(x.dtype).reshape(*orig_shape[:-1], t.shape[0])

    w = int4_dequant(t, x.dtype if x.dtype != torch.float32 else torch.float16)
    return torch.nn.functional.linear(x, w.to(x.dtype))


# --------------------------------------------------------------------------
# INT8
# --------------------------------------------------------------------------


def int8_dequant(t: INT8Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    ext = get_extension()
    if ext is None or not t.qweight.is_cuda:
        return _dequantize_int8(t, dtype)
    out = ext.int8_dequant(
        t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
        t.group_size, dtype)
    k = t.shape[-1]
    return out[:, :k] if out.shape[1] != k else out


_INT8_GEMV_MAX = int(os.environ.get("ACVRAM_INT8_GEMV_MAX", "80"))
# Linéaires INT8 au préfill (n > INT8_GEMV_MAX) : bf16 (défaut jusqu'au scellé
# P0 : déquant entière + cutlass) | a8 (kernels/gemm_w8a8.py : activation int8
# par jeton, tensor cores int8, sans déquant). Scellé : Coder préfill 2 048
# ≥ 11 000 j/s, porte PPL privé ≤ 1,020 (sage-profil-verdict-18-09).
# | cublas (P2, sage-marlin-ouvert-p2-18-09 : poids INT8 SYMÉTRIQUES PAR CANAL
# — un groupe = K, zéro = 128, convertis `-qkvo-i8c` — activation A8 par
# jeton puis `torch._int_mm` cuBLASLt int8 → int32, une échelle par ligne ×
# une par colonne ; un tenseur affine par groupes n'y est pas éligible et
# passe par a8/bf16 comme avant — le chemin ne change la sortie d'aucun
# autre converti).
# DÉFAUT « cublas » depuis sage-p2-au-defaut-19-09 (six lignes P2 tenues : C11
# b=12 1 365 t/s, J 0,993 ×) : seuls les poids éligibles (par canal, zéros 128,
# M > 16, K/N multiples de 8) le prennent ; tout autre poids — les classés
# (g128 affine) — garde la déquant bf16 d'avant, sortie INCHANGÉE au bit
# (tests/test_prefill_int8_cublas.py, converti réel). a8 reste un régime
# distinct : sous cublas, un poids inéligible ne passe JAMAIS par W8A8.
_PREFILL_INT8 = os.environ.get("ACVRAM_PREFILL_INT8", "cublas")
if _PREFILL_INT8 not in ("bf16", "a8", "cublas"):
    raise ValueError(f"ACVRAM_PREFILL_INT8={_PREFILL_INT8!r} : attendu bf16 | a8 | cublas")


def _i8c_poids(t: INT8Tensor):
    """Poids int8 signés (q − 128) [N, K_pad] d'un INT8Tensor symétrique par
    canal, construits UNE fois et gardés sur le tenseur (copie int8 de la
    taille du poids : q/k/v/o de Coder ≈ 0,9 Gio) ; None si le tenseur n'est
    pas par canal symétrique (group_size ≠ K_pad ou un zéro ≠ 128)."""
    cache = t.__dict__.get("_i8c")
    if cache is not None:
        return cache if cache is not False else None
    ok = (t.group_size == t.qweight.shape[1] and t.zeros.shape[1] == 1
          and bool((t.zeros == 128).all()))
    if not ok:
        t.__dict__["_i8c"] = False
        return None
    w = (t.qweight.to(torch.int16) - 128).to(torch.int8).contiguous()
    t.__dict__["_i8c"] = w
    return w


def vue_g128(t: INT8Tensor) -> INT8Tensor:
    """C11 (sage-p2-dec-c11-c6-19-09) : un poids INT8 symétrique PAR CANAL
    (groupe = K, zéro = 128 : convertis -qkvo-i8c) vu comme un poids à
    groupes de 128 — mêmes codes (le tenseur qweight est partagé, aucune
    copie), échelle de la ligne répétée sur K/128 groupes, zéros à 128. Les
    chemins étroits (gemm_etroit Triton, tuile K = 128 ; narrow_gemm CUDA) le
    servent alors comme le g128 du classé, à l'arithmétique près de l'ordre
    des sommes par groupe (× la même échelle). Construite une fois par
    tenseur ([N, K/128] fp16 + uint8 : 4096 × 16 × 3 o = 192 Kio pour q_proj).
    Rend t lui-même s'il n'est pas par canal."""
    cache = t.__dict__.get("_g128")
    if cache is not None:
        return cache if cache is not False else t
    N, k_pad = t.qweight.shape
    if t.group_size == 128 or t.group_size != k_pad or k_pad % 128 or t.zeros.shape[1] != 1 \
            or not bool((t.zeros == 128).all()):
        t.__dict__["_g128"] = False
        return t
    ng = k_pad // 128
    vue = INT8Tensor(t.qweight, t.scales.expand(N, ng).contiguous(), t.zeros.expand(N, ng).contiguous(),
                     128, t.shape, t.format)
    for k in ("etroit",):                       # désignations portées par le tenseur d'origine
        if k in t.__dict__:
            vue.__dict__[k] = t.__dict__[k]
    t.__dict__["_g128"] = vue
    return vue


def gemm_i8c_cublas(x: torch.Tensor, t: INT8Tensor, sortie_fp32: bool = False, a8=None):
    """``x`` [M, K] → [M, N] par `torch._int_mm` sur un poids symétrique par
    canal : y = s_x[m] · s_w[n] · Σ_k a8[m,k]·(q[n,k] − 128), produit entier
    exact, échelles en fp32. None si inéligible (poids affine/groupé, M ≤ 16 :
    cuBLASLt exige M > 16, K et N multiples de 8). ``a8`` : (a8, s_x) déjà
    quantifiés par `quantifier_a8_i8c(x)` (C15-prefill, q/k/v partagent x)."""
    w = _i8c_poids(t)
    M, K = x.shape
    N, k_pad = t.qweight.shape
    if w is None or M <= 16 or k_pad % 8 or N % 8:
        return None
    from .gemm_w8a8 import quantifier_a8, quantifier_a8_torch, disponible as _w8a8_dispo
    if a8 is not None:
        a, sx = a8                               # C15-prefill : A8 quantifiée une fois pour q/k/v
    elif _w8a8_dispo() and x.is_cuda:
        a, sx = quantifier_a8(x)
    else:
        a, sx = quantifier_a8_torch(x)           # à sec : le même arrondi que le noyau, au bit
    if K != k_pad:
        a = torch.nn.functional.pad(a, (0, k_pad - K))
    acc = torch._int_mm(a.contiguous(), w.t())                       # [M, N] int32
    if prefill_compact("epilogue") and _w8a8_dispo() and (x.is_cuda or _interprete()):
        # C15-prefill : f32(acc)·s_x·s_w → dtype en UN noyau (gemm_w8a8.epilogue_i8c),
        # la même chaîne d'arrondis que les quatre noyaux torch ci-dessous, au bit
        from .gemm_w8a8 import epilogue_i8c
        return epilogue_i8c(acc, sx, _i8c_echelles(t), torch.float32 if sortie_fp32 else x.dtype)
    y = acc.to(torch.float32) * sx[:, None] * t.scales[:, 0].to(torch.float32)[None, :]
    return y if sortie_fp32 else y.to(x.dtype)


def _interprete() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _i8c_echelles(t: INT8Tensor) -> torch.Tensor:
    """Échelle par canal en fp32 [N], convertie UNE fois (fp16 → fp32 exact,
    les mêmes valeurs que `t.scales[:, 0].to(torch.float32)` à chaque appel :
    une copie de moins par projection et par couche)."""
    s = t.__dict__.get("_i8c_s32")
    if s is None:
        s = t.scales[:, 0].to(torch.float32).contiguous()
        t.__dict__["_i8c_s32"] = s
    return s


def int8_matmul_partage(x: torch.Tensor, ts: list) -> Optional[list]:
    """C15-prefill : ``[x @ W_i.T for W_i in ts]`` avec l'A8 par jeton quantifiée
    UNE fois — le chemin cublas de `int8_matmul` pour chacun, au bit (même
    quantificateur, même `_int_mm`, même épilogue). None si UN des tenseurs ne
    prendrait pas ce chemin sous le dispatcher (alors chaque projection suit
    `int8_matmul` seule : le témoin) — les conditions sont celles de
    `int8_matmul` (n > seuil GEMV ou sans extension, régime cublas, dtype) et
    de `gemm_i8c_cublas` (par canal symétrique, M > 16, K et N multiples de 8)."""
    xf = x.reshape(-1, x.shape[-1])
    n = xf.shape[0]
    if _PREFILL_INT8 != "cublas" or x.dtype not in (torch.bfloat16, torch.float16) or n <= 16:
        return None
    ext = get_extension()
    if ext is not None and ts[0].qweight.is_cuda and n <= _INT8_GEMV_MAX:
        return None                                  # le dispatcher prendrait le GEMV
    if x.is_cuda and (not ts[0].qweight.is_cuda or _bk.resolve("int8", ts[0].qweight.device)[0].name != "cuda-fusionne"):
        return None                                  # backend masqué : la référence torch
    for t in ts:
        if _i8c_poids(t) is None or t.qweight.shape[1] % 8 or t.qweight.shape[0] % 8:
            return None
    a8 = quantifier_a8_i8c(xf)
    sorties = []
    for t in ts:
        y = gemm_i8c_cublas(xf, t, sortie_fp32=False, a8=a8)
        assert y is not None
        CHEMINS_INT8["cublas_partage"] += 1
        sorties.append(y[:, : t.shape[0]].reshape(*x.shape[:-1], t.shape[0]))
    return sorties


def quantifier_a8_i8c(x: torch.Tensor):
    """L'activation A8 par jeton du chemin cublas, (a8 int8 [M, K], s_x fp32 [M]),
    pour la passer à `gemm_i8c_cublas(..., a8=)` — C15-prefill : q, k et v
    lisent la même ligne normée, la quantifier trois fois rendait trois fois
    les mêmes octets (nsys P2 19/09 : `_quant_a8_kernel` ×4 par couche)."""
    from .gemm_w8a8 import quantifier_a8, quantifier_a8_torch, disponible as _w8a8_dispo
    if _w8a8_dispo() and x.is_cuda:
        return quantifier_a8(x)
    return quantifier_a8_torch(x)


def prefill_int8_regime() -> str:
    return _PREFILL_INT8


def tete_int8_entree_bf16(n_lignes: int) -> bool:
    """La tête INT8 reçoit x en bf16 (GEMV fp32) SEULEMENT sous le seuil GEMV —
    le régime que le moteur sert (tête à n ≤ 12 en service). Au-delà (PPL
    `evaluate.perplexity`, logits sur toute la fenêtre) la tête garde la
    conversion fp32 + déquant, quel que soit le régime : c62e2ef l'avait
    ouverte à a8/cublas et la tête passait en W8A8 à 2 048 lignes (déquant
    fp32 1,245 Gio + E4M3 + sortie fp32 1,245 Gio + log-softmax : pic 4 Gio,
    `sage-p2-ppl-instrument-file-7h-19-09`) — un régime jamais servi ; la
    PPL se borne en découpant la tête par tranches (evaluate.perplexity)."""
    return n_lignes <= _INT8_GEMV_MAX


# Compteur des chemins pris par `int8_matmul` (clé = branche) : une preuve
# lisible par un test — « le chemin cublas a été pris, la déquant non » — au
# lieu d'une déduction depuis une pile d'OOM (19/09).
import collections as _collections
CHEMINS_INT8 = _collections.Counter()
# Linéaires NVFP4 denses à 2 ≤ b ≤ 32 : gemv (défaut, témoin : nvfp4_gemv, les
# poids relus par séquence) | triton (kernels/gemm_dense_etroit.py, poids lus
# une fois par pas) — défaut à basculer sur le scellé de Sage (micro-banc
# ≥ 1,3 To/s, puis Qwen3.8 b=12 ≥ 500 t/s).
# Défaut « triton » depuis verdict-gemm-dense-palier1-situ-17-09 (Laure :
# Qwen3.8 b=12 128 → 349 t/s, J/j ÷ 2,7, b=1 et ppl-decode-kv inchangés) ;
# bascule à M ≥ 4 (Sage : à M = 2 la GEMV gagne au banc, 1,50 contre 1,12 To/s).
_DENSE_NVFP4 = os.environ.get("ACVRAM_DENSE_NVFP4", "triton")
_DENSE_NVFP4_MIN_M = int(os.environ.get("ACVRAM_DENSE_NVFP4_MIN_M", "4"))
# Plafond (octets) du pic de déquantification du repli GEMM d'int8_matmul,
# au-delà duquel la matrice est traitée par tranches de lignes.
_DEQUANT_TRANCHE_MAX = int(os.environ.get("ACVRAM_DEQUANT_TRANCHE_MAX", str(256 * 2**20)))
# Linéaires INT8 à b ≤ 16 (poste C, sage-e-c-verdict-17-09 § 2) : "mixte"
# (défaut : Triton dès b ≥ NARROW_TRITON_MIN_B, CUDA en dessous) | "cuda"
# (narrow_gemm / int8_gemv) | "triton" (kernels/gemm_etroit.py partout) |
# "tete" (Triton pour la tête seule). Le point de bascule est MESURÉ (Laure,
# rejeu de graphe, pas dense Coder 48 couches + tête, ms) :
#     b        1     2     4     8    12
#     cuda   2,69  5,09  4,56  4,37  4,11
#     triton 2,95  2,31  1,96  2,06  2,07
# Triton gagne dès b = 2 (×2,2) et perd à b = 1 (×0,91) : la constante porte
# sa mesure dans tests/test_gemm_etroit.py, sous graphes le lot est le godet.
#
# "mixte" DÉFAUT DE NOUVEAU (verdict-coder-c-mixte-17-09) : cause du plantage
# corrigée (Laurine 186177a, débordement mémoire réel sur les tranches K,
# 186177a) — condition 1 (test cassant sur godet/fantômes/sentinelles) et
# condition 2 (capture godets 1/2/8/16, rondes b=12 997,8 t/s) tenues.
_NARROW_KERNEL = os.environ.get("ACVRAM_NARROW_KERNEL", "mixte")
if _NARROW_KERNEL not in ("mixte", "cuda", "triton", "tete"):
    raise ValueError(f"ACVRAM_NARROW_KERNEL={_NARROW_KERNEL!r} : attendu mixte, cuda, triton ou tete")
_NARROW_TRITON_MIN_B = int(os.environ.get("ACVRAM_NARROW_TRITON_MIN_B", "2"))
MESURE_BASCULE_DENSE = {1: (2.69, 2.95), 2: (5.09, 2.31), 4: (4.56, 1.96), 8: (4.37, 2.06), 12: (4.11, 2.07)}


def narrow_choix(n: int, sortie_fp32: bool = False) -> str:
    """Le noyau des linéaires INT8 pour un lot de ``n`` lignes : "triton" ou "cuda"."""
    if _NARROW_KERNEL == "mixte":
        return "triton" if n >= _NARROW_TRITON_MIN_B else "cuda"
    if _NARROW_KERNEL == "tete":
        return "triton" if sortie_fp32 else "cuda"
    return _NARROW_KERNEL


def narrow_regime() -> str:
    """Pour `regime_ligne()` : ``dense=triton≥2|cuda`` (mixte), ``cuda``, ``triton``, ``tete``."""
    if _NARROW_KERNEL == "mixte":
        return f"triton≥{_NARROW_TRITON_MIN_B}|cuda"
    return _NARROW_KERNEL


def int8_matmul_norme(delta: torch.Tensor, t: INT8Tensor, res: torch.Tensor,
                      w: torch.Tensor, eps: float, mult: float = 1.0):
    """Poste F, fusion (3b) : ``x = res + mult·delta`` puis ``y = W ·
    rmsnorm(x)`` en UN lancement (`int8_gemv_norme`, prologue de norme dans
    `int8_gemv_kernel`). Rend (y [N, M], x) ou None si inéligible — l'appelant
    fait alors `add_norm` puis le GEMV, comme avant."""
    ext = get_extension()
    if ext is None or not hasattr(ext, "int8_gemv_norme") or not t.qweight.is_cuda:
        return None
    if delta.dtype != torch.bfloat16 or res.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return None
    forme = delta.shape
    xf = delta.reshape(-1, forme[-1])
    n, k = xf.shape
    k_pad = t.qweight.shape[1]
    if n < 1 or n > 8 or k != k_pad or k % 16 or n * k * 2 > 32 * 1024 or w.numel() != k:
        return None
    y, x = ext.int8_gemv_norme(t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
                               xf.contiguous(), t.group_size, res.reshape(-1, k).contiguous(),
                               w.contiguous(), float(eps), float(mult))
    return (y[:, : t.shape[0]].reshape(*forme[:-1], t.shape[0]), x.reshape(forme))


def int8_matmul(x: torch.Tensor, t: INT8Tensor,
                gemv_threshold: int = 0, sortie_fp32: bool = False) -> torch.Tensor:
    """``x @ W.T`` avec W stocké en INT8 affine par groupes.

    Sans ce chemin, les tenseurs promus en INT8 par la conversion — quelques
    pour cent du modèle, choisis précisément parce qu'ils sont sensibles —
    étaient rematérialisés en 16 bits par PyTorch à chaque jeton, et dominaient
    le temps de décodage entier.

    Le seuil de bascule vaut ``ACVRAM_INT8_GEMV_MAX`` (80 par défaut). Le noyau
    GEMV traite N activations par lecture de poids et relit W une fois par
    tranche de 8 ; la déquantification, elle, lit W, écrit W en 16 bits et le
    relit — un coût fixe, indépendant du nombre de jetons. Mesuré sur un
    tenseur 5120x5120 par groupes de 128 : le GEMV gagne jusqu'à 64 jetons
    (0,409 ms contre 0,561), les deux se croisent vers 88, et la
    déquantification l'emporte ensuite (256 jetons : 0,617 contre 1,632).
    Le seuil précédent était de 8 : tout prefill interactif — une invite
    courte — payait la déquantification complète des 128 tenseurs INT8 d'un
    27B, soit une centaine de millisecondes pour vingt jetons.
    """
    if gemv_threshold <= 0:
        gemv_threshold = _INT8_GEMV_MAX
    ext = get_extension()
    orig_shape = x.shape
    xf = x.reshape(-1, x.shape[-1])
    n = xf.shape[0]
    k_pad = t.qweight.shape[1]

    if ext is not None and t.qweight.is_cuda and n <= gemv_threshold:
        if n >= 2:
            # C11 : un poids par canal (i8c) prend les chemins étroits par sa vue
            # g128 (mêmes codes, échelle répétée) — sinon 2 ≤ M ≤ 16 n'avait
            # que le repli déquant (P2-déc b=12 1 062 t/s contre 1 334)
            t = vue_g128(t)
        # Poste C (sage-profil-verdict-17-09) : GEMM étroit W8A16 Triton pour
        # b ≤ 16, linéaires denses ET tête (fp32) ; opt-in jusqu'au scellé
        # (dense b=12 ≤ 1,0 ms/pas, sortie = chemin actuel ± 2⁻⁸)
        # "tete" : Triton pour la tête seule — mesuré ×3,03 à b=12 (0,659 →
        # 0,217 ms, exact) là où les linéaires denses sont plus lents (Laure
        # d65e49e : 6,59 contre 4,86 ms/pas, k/v N=512 ×0,4)
        if narrow_choix(n, sortie_fp32) == "triton" and n <= 16 and xf.dtype == torch.bfloat16:
            from . import gemm_etroit
            if gemm_etroit.disponible() and gemm_etroit.eligible(t):
                CHEMINS_INT8["etroit_triton"] += 1
                y = gemm_etroit.gemm_etroit(xf.contiguous(), t, sortie_fp32,
                                            compact=glue_compact("etroit"))[:, : t.shape[0]]
                return y.reshape(*orig_shape[:-1], t.shape[0])
        if k_pad != xf.shape[-1]:
            xf = torch.nn.functional.pad(xf, (0, k_pad - xf.shape[-1]))
        # `t.etroit` : tenseur désigné pour le GEMM étroit indépendamment du
        # réglage global (les projections int8 q/kv/o de l'attention MLA,
        # sage-duel-verdict § 14 (i) : int8_gemv<4,12> = 4,4 ms/pas à b=12)
        etroit = _NARROW_GEMM or getattr(t, "etroit", False)
        if (etroit and not sortie_fp32 and _NARROW_MIN <= n <= 16 and k_pad % 64 == 0
                and t.group_size % 64 == 0 and xf.dtype == torch.bfloat16 and hasattr(ext, "narrow_gemm")):
            CHEMINS_INT8["narrow_cuda"] += 1
            y = ext.narrow_gemm(t.qweight.contiguous(), None, t.scales.contiguous(), t.zeros.contiguous(),
                                xf.contiguous(), k_pad, t.group_size, 1.0, _narrow_rows(t.qweight.shape[0]))
            return y.to(x.dtype).reshape(*orig_shape[:-1], t.qweight.shape[0])
        CHEMINS_INT8["gemv"] += 1
        y = ext.int8_gemv(
            t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
            xf.contiguous(), t.group_size, sortie_fp32)
        y = y[..., : t.shape[0]]
        if sortie_fp32:
            # tete lm_head : x bf16, logits fp32 — pas de retour au dtype de x
            return y.to(torch.float32).reshape(*orig_shape[:-1], t.shape[0])
        return y.to(x.dtype).reshape(*orig_shape[:-1], t.shape[0])

    # P2 : poids symétriques par canal (convertis -qkvo-i8c) → cuBLASLt int8
    # (`torch._int_mm`), ACVRAM_PREFILL_INT8=cublas (défaut) ; un poids non
    # éligible continue ci-dessous vers la déquant bf16, jamais vers a8
    if _PREFILL_INT8 == "cublas" and x.dtype in (torch.bfloat16, torch.float16):
        y = gemm_i8c_cublas(xf, t, sortie_fp32=sortie_fp32)
        if y is not None:
            CHEMINS_INT8["cublas"] += 1
            return y[:, : t.shape[0]].reshape(*orig_shape[:-1], t.shape[0])
    # P0 (sage-profil-verdict-18-09) : au-delà du seuil GEMV, GEMM W8A8 sans
    # déquantification par appel — activation int8 par jeton, poids uint8
    # tels quels, produit entier exact sur les tensor cores ; ACVRAM_PREFILL_INT8=a8
    if (_PREFILL_INT8 == "a8" and x.dtype in (torch.bfloat16, torch.float16)
            and t.qweight.shape[1] % t.group_size == 0):
        from . import gemm_w8a8
        if gemm_w8a8.disponible() and (x.is_cuda or gemm_w8a8.INTERPRETE):
            CHEMINS_INT8["a8"] += 1
            y = gemm_w8a8.gemm_w8a8(xf, t, sortie_fp32=sortie_fp32)[:, : t.shape[0]]
            return y.reshape(*orig_shape[:-1], t.shape[0])
    dt = x.dtype if x.dtype != torch.float32 else torch.float16
    # `ext.int8_dequant` (extension CUDA) est calibré pour group_size=128 :
    # un group_size différent ne plante pas mais retient ~12 Gio au premier
    # prefill au lieu d'échouer proprement (trouvé par observation mémoire,
    # pas par erreur explicite -- bead, pas un chantier). Le repli SANS
    # extension (`_dequantize_int8`, carte absente ou processeur) gère lui
    # n'importe quel group_size -- vu par `test_int8_matmul_tranches.py`
    # (group_size=64, CPU) -- donc le refus ne vaut que pour le chemin CUDA.
    if ext is not None and t.qweight.is_cuda and t.group_size != 128:
        raise NotImplementedError(
            f"int8_matmul (repli GEMM CUDA, prefill n={n} > ACVRAM_INT8_GEMV_MAX="
            f"{gemv_threshold}) : group_size={t.group_size} != 128 non servi. "
            f"GEMV (n <= {gemv_threshold}) n'est pas concerné.")
    # Repli GEMM : déquantifier la matrice ENTIÈRE en bf16 coûtait, sur la
    # tête de Gemma-4-31B (262 144 × 5 376), 5,25 Gio d'un coup au premier
    # préfill — OOM après un chargement juste (Laure 0cf7fe6). Par tranches
    # de lignes de sortie : même arithmétique, mêmes valeurs (chaque tranche
    # est la même matrice restreinte), pic borné par _DEQUANT_TRANCHE_MAX.
    CHEMINS_INT8["dequant"] += 1
    lignes = t.qweight.shape[0]
    par_ligne = t.qweight.shape[1] * (4 + dt.itemsize)     # fp32 intermédiaire + sortie
    if lignes * par_ligne > _DEQUANT_TRANCHE_MAX:
        pas = max(64, (_DEQUANT_TRANCHE_MAX // par_ligne) // 64 * 64)
        out = torch.empty(*x.shape[:-1], t.shape[0], dtype=x.dtype, device=x.device)
        for a in range(0, t.shape[0], pas):
            b = min(a + pas, t.shape[0])
            tr = INT8Tensor(t.qweight[a:b], t.scales[a:b], t.zeros[a:b], t.group_size,
                            (b - a, t.shape[1]), t.format)
            w = int8_dequant(tr, dt)
            out[..., a:b] = torch.nn.functional.linear(x, w.to(x.dtype))
        return out
    w = int8_dequant(t, dt)
    return torch.nn.functional.linear(x, w.to(x.dtype))


# --------------------------------------------------------------------------
# Enregistrement des backends livrés — voir backends.py pour le contrat.
# --------------------------------------------------------------------------

from . import backends as _bk
from ..quant.formats import PlainTensor, dequantize as _dequantize_ref

matmul = _bk.matmul                      # le point d'entrée du moteur


def _cuda_ok(dev: torch.device) -> bool:
    return get_extension() is not None


def _sm100_ok(dev: torch.device) -> bool:
    # La carte est passee a fp4_mm_available : une extinction survenue sur une
    # capacite donnee ne doit valoir que pour elle.
    return (fp4_mm_available(dev)
            and torch.cuda.get_device_capability(dev) >= (10, 0))


def _gemv_or_none(fn):
    """Adapte les wrappers historiques : ils font déjà seuils et replis."""
    def call(x, w):
        return fn(x, w)
    return call


def q3n_matmul(x, w):
    """GEMV Q3N fusionné ; None si l'extension manque (repli référence)."""
    ext = get_extension()
    if ext is None or not hasattr(ext, "q3n_gemv"):
        return None
    return ext.q3n_gemv(w.qweight, w.block_scale.view(torch.uint8),
                        w.global_scale.to(w.qweight.device),
                        w.table_gpu(w.qweight.device), x,
                        w.shape[1], w.block)


def q3n_dequant(w, dt):
    from ..quant.q3n import dequantize_q3n
    return dequantize_q3n(w, dt)


_bk.register(_bk.Backend(
    name="cuda-fusionne", formats=("nvfp4", "int4_awq", "int8", "q3n"),
    device_type="cuda", priority=100, available=_cuda_ok,
    matmul=lambda x, w: {"nvfp4": nvfp4_matmul, "int4_awq": int4_matmul,
                         "int8": int8_matmul, "q3n": q3n_matmul}[w.format](x, w),
    dequant=lambda w, dt: {"nvfp4": nvfp4_dequant, "int4_awq": int4_dequant,
                           "int8": int8_dequant, "q3n": q3n_dequant}[w.format](w, dt),
    note="dequantification + GEMV fusionnes, acvram_kernels.cu"))

# LE BACKEND `fp4-tensorcores` N'EST PLUS ENREGISTRE — mesure du 10/09/2026.
#
# `nvfp4_mm_tensorcore` reste ci-dessus, appelable et testable : le code d'une
# experience ratee vaut d'etre conserve, son inscription au registre non. Ce qui
# suit est la raison, pour que personne ne le reinscrive dans six mois.
#
# Duel bout en bout, ABBA, un bras par processus, sur Qwen3-4B-nvfp4 (5090).
# Prefill seul, une sequence, graphes coupes — le seul regime ou ce chemin
# etait pris :
#
#     invite   ttft avec   ttft sans   il PERD de
#       192      324,0       295,8        8,7 %
#       512      310,8       305,8        1,6 %
#      2048      371,9       367,4        1,2 %
#      4096      528,1       466,9       11,6 %
#
#     ABBA a 4096 :  avec 520,8 / 520,0    sans 466,2 / 465,3
#     dispersion intra-bras 0,8 ms — l'ecart vaut SOIXANTE fois la dispersion.
#
# **Il perd dans le regime pour lequel il avait ete ecrit** : sa note disait
# « prefill W4A4 », et a 4 096 jetons de prefill il coute +11,6 % de TTFT.
#
# En decodage concurrent il etait pire encore : a douze sequences, graphes
# coupes, l'ecarter rend x3,28 de debit et -55 % de TTFT — et le texte produit
# differe, le chemin ecarte etant aussi le plus juste (le GEMV rend 55,6 dB
# contre le poids reellement stocke, la ou ce chemin quantifie l'ACTIVATION en
# FP4, ~9,5 % d'erreur).
#
# Pourquoi il ne pouvait pas gagner, et pourquoi un balayage etroit le cachait :
# ses 385 us mesures « a plat » de 1 a 32 lignes etaient un plateau de FRAIS
# FIXES dans un domaine trop etroit pour voir la pente. La quantification de
# l'activation en FP4 est un travail PROPORTIONNEL a n : un cout fixe s'amortit,
# un cout proportionnel jamais.
#
# Il avait aussi fallu l'exclure explicitement de la capture de graphe pour que
# celle-ci survive. Un chemin qu'on doit interdire la ou l'on veut aller, et qui
# perd la ou il est permis, n'a pas d'endroit ou il est le bon.


_bk.register(_bk.Backend(
    name="cpu-avx2", formats=("nvfp4", "int4_awq"), device_type="cpu",
    priority=50, available=lambda d: cpu_kernels_available(),
    matmul=lambda x, w: {"nvfp4": nvfp4_matmul_cpu,
                         "int4_awq": int4_matmul_cpu}[w.format](x, w),
    note="GEMV C, AVX2 + repli scalaire, ABI ctypes"))


def _ref_matmul(x, w):
    if isinstance(w, PlainTensor):
        return torch.nn.functional.linear(x, w.weight.to(x.dtype))
    return torch.nn.functional.linear(x, _dequantize_ref(w, x.dtype))


for _dev in ("cuda", "cpu"):
    _bk.register(_bk.Backend(
        name=f"reference-{_dev}",
        formats=("nvfp4", "int4_awq", "int8", "bf16", "fp16", "plain", "q3n"),
        device_type=_dev, priority=0, available=lambda d: True,
        matmul=_ref_matmul,
        dequant=lambda w, dt: _dequantize_ref(w, dt),
        note="PyTorch pur ; lent, numeriquement identique, ferme la liste"))


# --------------------------------------------------------------------------
# GEMV groupés MoE — les poids des experts empilés, un lancement par projection
# --------------------------------------------------------------------------


def nvfp4_gemv_grouped(x: torch.Tensor, qw: torch.Tensor, bscale: torch.Tensor,
                       gscales: torch.Tensor, expert_ids: torch.Tensor,
                       token_ids: torch.Tensor, k: int) -> Optional[torch.Tensor]:
    ext = get_extension()
    if ext is None or k % 32 != 0:
        return None
    if x.shape[-1] != k:
        x = torch.nn.functional.pad(x, (0, k - x.shape[-1]))
    return ext.nvfp4_gemv_grouped(qw, bscale, gscales, expert_ids, token_ids,
                                  x.contiguous(), k)


def nvfp4_gemv_grouped_v2(x: torch.Tensor, qw: torch.Tensor, bscale: torch.Tensor,
                          gscales: torch.Tensor, eid_s: torch.Tensor, tok_s: torch.Tensor,
                          ordre: torch.Tensor, k: int) -> Optional[torch.Tensor]:
    """v2 (18/09) : paires triées par expert (``eid_s``, ``tok_s``), ``ordre``
    = place d'origine de chaque paire ; poids lus une fois pour ≤ 4 jetons."""
    ext = get_extension()
    if ext is None or k % 32 != 0:
        return None
    if x.shape[-1] != k:
        x = torch.nn.functional.pad(x, (0, k - x.shape[-1]))
    return ext.nvfp4_gemv_grouped_v2(qw, bscale, gscales, eid_s, tok_s, ordre, x.contiguous(), k)


def nvfp4_gemv_grouped_table(x: torch.Tensor, table_qw: torch.Tensor,
                             table_bs: torch.Tensor, gscales: torch.Tensor,
                             expert_ids: torch.Tensor, token_ids: torch.Tensor,
                             k: int, m: int) -> Optional[torch.Tensor]:
    """Pendant table de `nvfp4_gemv_grouped` (bead pds) : chaque expert lu par
    adresse (résident ou épinglé), pas par une pile contiguë — couche au
    placement hétérogène."""
    ext = get_extension()
    if ext is None or k % 32 != 0:
        return None
    if x.shape[-1] != k:
        x = torch.nn.functional.pad(x, (0, k - x.shape[-1]))
    return ext.nvfp4_gemv_grouped_table(table_qw, table_bs, gscales, expert_ids,
                                        token_ids, x.contiguous(), m, k)


def int4_gemv_grouped(x: torch.Tensor, qw: torch.Tensor, scales: torch.Tensor,
                      zeros: torch.Tensor, expert_ids: torch.Tensor,
                      token_ids: torch.Tensor, k: int,
                      group_size: int) -> Optional[torch.Tensor]:
    ext = get_extension()
    if ext is None:
        return None
    if x.shape[-1] != k:
        x = torch.nn.functional.pad(x, (0, k - x.shape[-1]))
    return ext.int4_gemv_grouped(qw, scales, zeros, expert_ids, token_ids,
                                 x.contiguous(), k, group_size)


# Attention paginée du décodage : "triton" (kernels/attn_paginee.py, poste E,
# une lecture de K/V par groupe GQA, défaut depuis sage-e-c-verdict-17-09 —
# tenu sur Coder-30B (4 cellules) ET le témoin dense Qwen2.5-Coder-14B, PPL
# décodage 8k ±0,002, pas de dérive au long contexte ctx 8k) | "cuda"
# (acvram_kernels.cu, ancien défaut, gardé témoin) ; GLM (MLA) ne passe pas
# par ce chemin
_PAGED_ATTN = os.environ.get("ACVRAM_PAGED_ATTN", "triton")
if _PAGED_ATTN not in ("cuda", "triton"):
    raise ValueError(f"ACVRAM_PAGED_ATTN={_PAGED_ATTN!r} : attendu cuda ou triton")

# C15 niveau 3 (revue/chantier-c15-niveau3-coder-20-09) : nœuds de graphe du
# pas de décodage GQA/MoE réduits par fusions, chacune débranchable par cette
# seule variable (0 = témoin, le chemin d'avant, au bit) : routeur MoE en un
# noyau (logits + top-k, route_prep.route_logits_fusee), GEMM étroit int8 sans
# somme torch des tranches split-K (réduction par le dernier programme,
# gemm_etroit), attention paginée sans second noyau de réduction
# (attn_paginee). Défaut 0 tant que le scellé n'est pas mesuré sur carte.
_GLUE_COMPACT = int(os.environ.get("ACVRAM_GLUE_COMPACT", "1"))   # défaut 1 depuis le 20/09 (C15-3d bis)
if _GLUE_COMPACT not in (0, 1):
    raise ValueError(f"ACVRAM_GLUE_COMPACT={_GLUE_COMPACT!r} : attendu 0 ou 1")
# C15-3b : bissection par fusion — sous GLUE_COMPACT=1, la liste des fusions
# prises (vide = toutes) : routeur | attn | etroit | kv (kv_write par tranches,
# q sans copie, valid une fois par pas). Une fusion absente de la liste suit
# le témoin. Sans effet sous GLUE_COMPACT=0.
GLUE_COMPACT_FUSIONS = ("routeur", "attn", "etroit", "kv")
_GLUE_COMPACT_ITEMS = os.environ.get("ACVRAM_GLUE_COMPACT_ITEMS", "")
for _f in filter(None, _GLUE_COMPACT_ITEMS.split(",")):
    if _f not in GLUE_COMPACT_FUSIONS:
        raise ValueError(f"ACVRAM_GLUE_COMPACT_ITEMS={_GLUE_COMPACT_ITEMS!r} : fusions connues "
                         f"{', '.join(GLUE_COMPACT_FUSIONS)}")


def glue_compact(fusion: str = "") -> bool:
    """Lu à l'appel (pas à l'import) : `regime.masquer` réécrit l'attribut.
    ``fusion`` : nom d'une fusion (GLUE_COMPACT_FUSIONS) — vraie si le niveau
    3 est pris ET que la fusion n'est pas écartée par GLUE_COMPACT_ITEMS."""
    if not _GLUE_COMPACT:
        return False
    if fusion and _GLUE_COMPACT_ITEMS:
        assert fusion in GLUE_COMPACT_FUSIONS, fusion
        return fusion in _GLUE_COMPACT_ITEMS.split(",")
    return True


# C15-prefill (revue/chantier-c15-prefill-20-09) : la glue du PRÉFILL eager
# (L > _MOE_GROUPED_MAX) réduite par fusions au bit, chacune débranchable par
# cette seule variable (0 = témoin, le chemin d'avant) : épilogue des GEMM
# int8 cuBLAS en un noyau (gemm_w8a8.epilogue_i8c : f32(acc)·s_x·s_w → bf16,
# la même chaîne d'arrondis que les quatre noyaux torch), activation A8 par
# jeton quantifiée UNE fois pour q/k/v (Attention._proj), résidu différé
# (x + y absorbé par add_norm de la couche suivante, comme au décodage),
# permutations MoE sans second tri ni conversions (MoEBlock._forward_prefill_
# grouped). DÉFAUT 1 depuis le 20/09 (verdict-c15-prefill-19-09, Manon bcd7a73b, 93b4e586 :
# PPL au bit sur 3 tranches, capture 4/4, prefill Coder servi 22 748 / 22 683 contre
# 17 609 / 16 981 j/s = × 1,29-1,34, noyaux 89,37 contre 101,67 ms) ; sans `norm`
# (rmsnorm un warp par ligne 3,82 ms contre le bloc 2,12 : opt-in ITEMS=…,norm).
_PREFILL_COMPACT = int(os.environ.get("ACVRAM_PREFILL_COMPACT", "1"))
if _PREFILL_COMPACT not in (0, 1):
    raise ValueError(f"ACVRAM_PREFILL_COMPACT={_PREFILL_COMPACT!r} : attendu 0 ou 1")
# bissection par fusion, comme GLUE_COMPACT_ITEMS : vide = toutes
PREFILL_COMPACT_FUSIONS = ("epilogue", "a8", "residu", "permut", "norm", "attn")
# « norm » (rmsnorm un warp par ligne) n'est PAS dans le défaut : mesurée par
# Manon le 20/09 à 3,82 ms contre 2,12 pour le bloc (edd44987, +1,7 ms), le
# noyau reste en opt-in (ITEMS=…,norm) tant qu'il n'a pas battu le bloc
PREFILL_COMPACT_DEFAUT = ("epilogue", "a8", "residu", "permut", "attn")
_PREFILL_COMPACT_ITEMS = os.environ.get("ACVRAM_PREFILL_COMPACT_ITEMS", "")
for _f in filter(None, _PREFILL_COMPACT_ITEMS.split(",")):
    if _f not in PREFILL_COMPACT_FUSIONS:
        raise ValueError(f"ACVRAM_PREFILL_COMPACT_ITEMS={_PREFILL_COMPACT_ITEMS!r} : fusions connues "
                         f"{', '.join(PREFILL_COMPACT_FUSIONS)}")


def prefill_compact(fusion: str = "") -> bool:
    """Lu à l'appel : `regime.masquer` réécrit l'attribut. ``fusion`` : nom
    d'une fusion (PREFILL_COMPACT_FUSIONS) — vraie si le compact est pris ET
    que la fusion n'est pas écartée par PREFILL_COMPACT_ITEMS."""
    if not _PREFILL_COMPACT:
        return False
    if fusion:
        assert fusion in PREFILL_COMPACT_FUSIONS, fusion
        if _PREFILL_COMPACT_ITEMS:
            return fusion in _PREFILL_COMPACT_ITEMS.split(",")
        return fusion in PREFILL_COMPACT_DEFAUT
    return True


def paged_attention(q: torch.Tensor, cache, tables: torch.Tensor,
                    seq_lens: torch.Tensor, n_rep: int,
                    scale: float, q_len: int = 1,
                    window: int = 0) -> Optional[torch.Tensor]:
    """Attention de décodage fusionnée sur le cache paginé INT8, ou None.

    Conditions : extension compilée, cache quantifié en int8, dimension de
    tête instanciée (64/128/256). Le repli — déquantifier puis SDPA — reste
    numériquement la référence ; un test les compare.
    """
    if os.environ.get("ACVRAM_DISABLE_PAGED_ATTN"):
        return None
    ext = get_extension()
    if ext is None or not q.is_cuda:
        return None
    if cache.k_scale is None or cache.cfg.dtype != "int8":
        return None
    d = q.shape[-1]
    if d not in (32, 64, 128, 256, 512):
        return None
    # C5-b : clés par canal (kv_canal) — seule la variante CANAL du noyau CUDA
    # sait lire sc E4M3 par bloc et le bloc courant bf16 ; le noyau Triton du
    # poste E ne le sait pas. Sans le symbole : None → gather_fixed (qui lit
    # la réserve) + decode_attention_fixed, jamais un noyau qui lirait des
    # codes par canal avec des échelles par jeton.
    if getattr(cache, "canal", False):
        if not hasattr(ext, "paged_attention_canal"):
            return None
        return ext.paged_attention_canal(
            q.contiguous(), cache.k, cache.k_scale, cache.v, cache.v_scale,
            cache.k_scale_canal.view(torch.uint8), cache.tampon, cache.tampon_de,
            tables.contiguous(), seq_lens.contiguous(), cache.cfg.num_kv_heads,
            float(scale), int(q_len), int(window))
    # Poste E (sage-b0-et-cause-lm4-17-09) : noyau Triton par groupe GQA,
    # opt-in tant que le scellé (≤ 1,5 ms b=12 ctx 2048 ET ≤ 0,72 ms b=1,
    # sortie = noyau CUDA ± 2⁻⁸) n'est pas mesuré ; décodage q_len = 1 seul.
    if (_PAGED_ATTN == "triton" and q_len == 1 and d in (64, 128)
            and q.shape[1] // cache.cfg.num_kv_heads <= 16):
        from . import attn_paginee
        if attn_paginee.disponible():
            # C15 niveau 3 : le noyau Triton lit q par ses pas (stride_qb,
            # stride_qh) — la tranche q de la projection empilée passe sans
            # copie (un nœud par couche) ; le témoin recopie comme avant.
            compact = glue_compact("attn")
            qq = q if (glue_compact("kv") and q.stride(2) == 1 and q.stride(1) == q.shape[2]) \
                else q.contiguous()
            return attn_paginee.paged_attention(
                qq, cache.k, cache.k_scale, cache.v, cache.v_scale,
                tables.contiguous(), seq_lens.contiguous(), cache.cfg.num_kv_heads,
                float(scale), int(window), compact=compact)
    return ext.paged_attention(
        q.contiguous(), cache.k, cache.k_scale,
        cache.v, cache.v_scale, tables.contiguous(),
        seq_lens.contiguous(), cache.cfg.num_kv_heads, float(scale),
        int(q_len), int(window))
