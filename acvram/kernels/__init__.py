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
    for cand in ("/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"):
        if os.path.exists(cand):
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
        out = dequantize_nvfp4(t, dtype)
        if gscale_rows is not None:
            out = out.view(-1, rows_per_group, out.shape[-1]) \
                * (gscale_rows.to(out.dtype) / float(t.global_scale_float())).view(-1, 1, 1)
            out = out.reshape(-1, out.shape[-1])
        return out
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

    # Prefill. Par défaut, W4A8 : activation FP8 (≈2 % d'erreur contre ≈9,5 %
    # en FP4) sur les tensor cores FP8. ACVRAM_PREFILL=a4 rend le chemin FP4
    # pur (le plus rapide, le moins précis) ; =bf16 force le repli.
    if n > gemv_threshold:
        mode = os.environ.get("ACVRAM_PREFILL", "a8")
        if mode == "a4":
            tc = nvfp4_mm_tensorcore(x, t)
            if tc is not None:
                return tc
        elif mode != "bf16":
            tc = nvfp4_mm_w4a8(x, t)
            if tc is not None:
                return tc
            tc = nvfp4_mm_tensorcore(x, t)
            if tc is not None:
                return tc

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


def int8_matmul(x: torch.Tensor, t: INT8Tensor,
                gemv_threshold: int = 0) -> torch.Tensor:
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
        if k_pad != xf.shape[-1]:
            xf = torch.nn.functional.pad(xf, (0, k_pad - xf.shape[-1]))
        y = ext.int8_gemv(
            t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
            xf.contiguous(), t.group_size)
        y = y[..., : t.shape[0]]
        return y.to(x.dtype).reshape(*orig_shape[:-1], t.shape[0])

    w = int8_dequant(t, x.dtype if x.dtype != torch.float32 else torch.float16)
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
    return ext.paged_attention(
        q.contiguous(), cache.k, cache.k_scale,
        cache.v, cache.v_scale, tables.contiguous(),
        seq_lens.contiguous(), cache.cfg.num_kv_heads, float(scale),
        int(q_len), int(window))
