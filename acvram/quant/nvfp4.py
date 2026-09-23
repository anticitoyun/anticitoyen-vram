"""NVFP4 — virgule flottante 4 bits à échelle par bloc (Blackwell / sm_120).

Disposition, conforme à la définition NVIDIA/OCP que consomment les tensor
cores Blackwell et CUTLASS :

    élément        FP4 E2M1   1 signe + 2 exposant + 1 mantisse
                              magnitudes {0, 0,5, 1, 1,5, 2, 3, 4, 6}
    échelle bloc   FP8 E4M3   une pour 16 éléments consécutifs le long de K
    échelle glob.  FP32       une par tenseur

    w[i] ≈ niveau(q[i]) × e4m3(échelle_bloc[i//16]) × échelle_globale

Coût de stockage par poids :

    4 bits (élément) + 8/16 bits (échelle de bloc) = 4,5 bits par poids

soit ×3,56 plus petit que le BF16. Sur 32 Go de VRAM cela fait environ
56 milliards de paramètres de poids, contre 16 en BF16.

Pourquoi une échelle *globale* par-dessus l'échelle de bloc : l'E4M3 sature à
448, et c'est un diviseur par tenseur qui maintient chaque échelle de bloc
dans la plage représentable, quelle que soit la dynamique du tenseur.

Ce module est en PyTorch pur et tourne sur processeur, ce qui le rend testable
sans GPU ; le chemin CUDA fusionné de ``acvram.kernels`` doit le reproduire au
bit près (voir tests/test_quant.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "E2M1_LEVELS",
    "ECHELLES", "regler_echelle", "echelle_courante",
    "E4M3_MAX",
    "E2M1_MAX",
    "NVFP4Tensor",
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "pack_e2m1",
    "unpack_e2m1",
    "round_to_e2m1",
]

BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0

# code -> magnitude ; l'index est le champ de magnitude sur 3 bits de l'E2M1
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Milieux entre niveaux consécutifs. L'arrondi au pair le plus proche porte sur
# l'index du *code*, d'où l'alternance de comparaisons strictes et larges : une
# égalité doit tomber sur un code pair, donc 0,25 descend vers le code 0 tandis
# que 0,75 monte vers le code 2.
_EGALITE_BAS = (0.25, 1.25, 2.5, 5.0)   # l'égalité descend (code pair en dessous)
_EGALITE_HAUT = (0.75, 1.75, 3.5)       # l'égalité monte   (code pair au-dessus)


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Arrondit les magnitudes sur la grille E2M1 et rend le code 3 bits 0..7.

    Saturant : tout ce qui dépasse 6 est ramené au code 7, l'E2M1 n'ayant pas
    d'infini.
    """
    m = x.abs()
    code = (
        (m > _EGALITE_BAS[0]).to(torch.uint8)
        + (m >= _EGALITE_HAUT[0]).to(torch.uint8)
        + (m > _EGALITE_BAS[1]).to(torch.uint8)
        + (m >= _EGALITE_HAUT[1]).to(torch.uint8)
        + (m > _EGALITE_BAS[2]).to(torch.uint8)
        + (m >= _EGALITE_HAUT[2]).to(torch.uint8)
        + (m > _EGALITE_BAS[3]).to(torch.uint8)
    )
    return code


def _levels_tensor(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(E2M1_LEVELS, device=device, dtype=dtype)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Empaquette des codes 4 bits (dernière dimension, de longueur paire), deux
    par octet.

    L'ordre des quartets place le bas en premier : l'élément 2k occupe les bits
    0 à 3, l'élément 2k+1 les bits 4 à 7. C'est la convention ``e2m1_x2``
    qu'utilisent CUTLASS et TensorRT, si bien qu'un tampon empaqueté peut être
    remis tel quel à un produit matriciel Blackwell.
    """
    if codes.shape[-1] % 2:
        raise ValueError("l'empaquetage e2m1 exige un nombre pair d'éléments")
    codes = codes.to(torch.uint8)
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo & 0x0F) | ((hi & 0x0F) << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Inverse de :func:`pack_e2m1`."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack((lo, hi), dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@dataclass
class NVFP4Tensor:
    """Une matrice de poids stockée en NVFP4.

    ``qweight``      uint8         [sortie, entrée//2]  paires E2M1 empaquetées
    ``block_scale``  float8_e4m3fn [sortie, entrée//16]
    ``global_scale`` float32       scalaire
    ``shape``        forme logique d'origine, avant tout remplissage de K
    """

    qweight: torch.Tensor
    block_scale: torch.Tensor
    global_scale: torch.Tensor
    shape: tuple[int, ...]
    padded_in: int
    # Échelle globale par ligne de sortie, posée seulement sur un tenseur
    # empilé (q, k et v en un) : chaque segment garde la sienne, sans le
    # réarrondi qu'imposerait une échelle commune.
    global_scale_rows: Optional[torch.Tensor] = None
    # Q1 : {"echelle", "blocs", "amax4", "clampes"} posé par `quantize_nvfp4` — part des blocs ayant choisi amax/4
    # et part des blocs dont amax/6 sature l E4M3 (candidats confondus) ; publié au manifeste, jamais un tenseur
    echelle_stats: Optional[dict] = None

    format = "nvfp4"

    def global_scale_float(self) -> float:
        """Le scalaire d'echelle, memorise cote hote.

        ``global_scale`` est un tenseur d'un element qui ne change jamais apres
        la quantification ; le relire par ``.item()`` a chaque GEMV synchronise
        le flux CUDA et dominait le temps de decodage (62 % au profil).
        """
        gs = self.__dict__.get("_gs_f")
        if gs is None:
            gs = float(self.global_scale.item())
            self.__dict__["_gs_f"] = gs
        return gs

    @property
    def nbytes(self) -> int:
        return (
            self.qweight.numel()
            + self.block_scale.numel()          # 1 byte per E4M3
            + 4
        )

    @property
    def bits_per_weight(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return self.nbytes * 8 / max(1, n)

    @classmethod
    def transferts_par_poids(cls) -> int:
        """Nombre de copies distinctes qu'un poids coute pour changer de memoire.

        Compte les champs qui sont des tenseurs, par introspection de la
        dataclasse : c'est exactement ce que ``to()`` deplace, un appel par
        champ. Trois aujourd'hui — qweight, echelles de bloc, echelle globale.

        Cette methode existe pour que le planificateur n'ecrive pas ce nombre
        en dur. Il l'a fait deux fois et s'est trompe deux fois, d'un facteur
        dix-huit puis trois ; et le jour ou les trois tenseurs voyageront dans
        un seul tampon epingle, une constante ecrite ailleurs resterait fausse
        en silence. Ici elle suit la classe.
        """
        import dataclasses
        return sum(1 for f in dataclasses.fields(cls)
                   if f.type in ("torch.Tensor", torch.Tensor))

    def to(self, device, non_blocking: bool = False) -> "NVFP4Tensor":
        return NVFP4Tensor(
            qweight=self.qweight.to(device, non_blocking=non_blocking),
            block_scale=self.block_scale.to(device, non_blocking=non_blocking),
            global_scale=self.global_scale.to(device, non_blocking=non_blocking),
            shape=self.shape,
            padded_in=self.padded_in,
            echelle_stats=self.echelle_stats,
        )

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}qweight": self.qweight,
            # safetensors ne sait pas transporter des métadonnées float8 de
            # façon portable selon les versions : les échelles voyagent donc en
            # octets bruts, et sont réinterprétées à la lecture.
            f"{prefix}block_scale": self.block_scale.view(torch.uint8),
            f"{prefix}global_scale": self.global_scale,
        }

    @staticmethod
    def from_state_dict(sd: dict[str, torch.Tensor], prefix: str,
                        shape: tuple[int, ...]) -> "NVFP4Tensor":
        q = sd[f"{prefix}qweight"]
        bs = sd[f"{prefix}block_scale"].view(torch.float8_e4m3fn)
        gs = sd[f"{prefix}global_scale"]
        return NVFP4Tensor(q, bs, gs, tuple(shape), q.shape[-1] * 2)


def _pad_k(w: torch.Tensor, block: int) -> tuple[torch.Tensor, int]:
    k = w.shape[-1]
    rem = k % block
    if rem == 0:
        return w, k
    pad = block - rem
    return torch.nn.functional.pad(w, (0, pad)), k


# Règle d'échelle de bloc (Maîtresse 21/09, pièce Q1, arXiv 2512.02010 « Four Over Six ») :
#   max6  : échelle = amax/6, le plus grand du bloc tombe sur 6 (classique, tout le parc jusqu'ici) ;
#   4sur6 : par bloc, deux candidats amax/6 et amax/4 (le plus grand tombe sur 4 : pas de saturation, les
#           autres éléments profitent des niveaux fins 0,5/1/1,5/2/3 sur une échelle plus grande) ; le
#           moindre MSE de reconstruction gagne, égalité → max6. Format E2M1 + E4M3 et noyaux inchangés :
#           seule la VALEUR de l'échelle de bloc change, elle reste arrondie en E4M3 avant le choix.
# Le défaut du module est posé par `regler_echelle` (CLI `--echelle`) : la recherche AWQ et la
# quantification finale passent toutes deux par `quantize_nvfp4`, donc par la même règle.
ECHELLES = ("max6", "4sur6")
_CANDIDATS_4SUR6 = (E2M1_MAX, 4.0)
_echelle_defaut = "max6"


def regler_echelle(nom: str) -> str:
    global _echelle_defaut
    if nom not in ECHELLES:
        raise ValueError(f"échelle de bloc inconnue : {nom!r} (attendu : {', '.join(ECHELLES)})")
    _echelle_defaut = nom
    return nom


def echelle_courante() -> str:
    return _echelle_defaut


def _coder(wb: torch.Tensor, bs: torch.Tensor) -> torch.Tensor:
    """Codes E2M1 (3 bits de magnitude) de ``wb`` [sortie, blocs, bloc] pour l échelle effective ``bs`` [sortie, blocs]."""
    safe = bs.clamp(min=torch.finfo(torch.float32).tiny)
    codes = round_to_e2m1(wb / safe.unsqueeze(-1))
    return torch.where((bs.unsqueeze(-1) > 0), codes, torch.zeros_like(codes))


def _mse_bloc(wb: torch.Tensor, codes: torch.Tensor, bs: torch.Tensor) -> torch.Tensor:
    """Erreur quadratique par bloc de la reconstruction |w| ≈ niveau[code] × bs (les signes sont portés à part)."""
    vals = _levels_tensor(wb.device)[codes.long()] * bs.unsqueeze(-1)
    return ((wb.abs() - vals) ** 2).sum(dim=-1)


def quantize_nvfp4(
    weight: torch.Tensor,
    block: int = BLOCK,
    global_scale: Optional[torch.Tensor] = None,
    echelle: Optional[str] = None,
) -> NVFP4Tensor:
    """Quantifie en NVFP4 un poids 2-D ``[sorties, entrées]``.

    Les blocs courent le long des entrées, c'est-à-dire de la dimension de
    réduction, ce qu'attend un produit matriciel orienté K : chaque tranche de
    16 porte sa propre échelle, si bien qu'un unique canal aberrant ne peut pas
    aplatir toute une ligne. ``echelle`` : max6 | 4sur6 (voir ``ECHELLES``),
    défaut = celui du module.
    """
    echelle = echelle or _echelle_defaut
    if echelle not in ECHELLES:
        raise ValueError(f"échelle de bloc inconnue : {echelle!r}")
    if weight.dim() != 2:
        raise ValueError(f"poids 2-D attendu, reçu {tuple(weight.shape)}")
    orig_shape = tuple(weight.shape)
    w = weight.detach().to(torch.float32)
    w, orig_k = _pad_k(w, block)
    out_f, k = w.shape
    wb = w.view(out_f, k // block, block)

    if global_scale is None:
        amax = wb.abs().amax()
        # On choisit g pour que la plus grande échelle de bloc tombe exactement
        # sur le plafond de l'E4M3.
        g = amax / (E2M1_MAX * E4M3_MAX)
        if not torch.isfinite(g) or g <= 0:
            g = torch.tensor(1.0)
        global_scale = g.reshape(())
    gs = global_scale.to(torch.float32).reshape(())

    block_amax = wb.abs().amax(dim=-1)                    # [sortie, k/bloc]
    # On représente l'échelle de bloc en E4M3 : c'est un arrondi réel et avec
    # perte, et le noyau doit utiliser la valeur *arrondie*, jamais `ideal`.
    def candidat(div: float):
        e4m3 = (block_amax / div / gs).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        eff = e4m3.to(torch.float32) * gs                 # échelle effective
        return e4m3, eff, _coder(wb, eff)
    bs_e4m3, bs, codes = candidat(E2M1_MAX)
    stats = {"echelle": echelle, "blocs": int(block_amax.numel()), "amax4": 0, "clampes": 0}
    if echelle == "4sur6":
        # Le bloc dont amax/6 sature déjà l E4M3 (celui qui fixe g) a ses deux candidats clampés à la même
        # valeur : 4sur6 = max6 pour lui, par construction. Le MSE se juge sur l échelle E4M3 arrondie.
        e4, b4, c4 = candidat(_CANDIDATS_4SUR6[1])
        mieux = (_mse_bloc(wb, c4, b4) < _mse_bloc(wb, codes, bs))     # strict : égalité → max6
        bs_e4m3 = torch.where(mieux, e4, bs_e4m3)
        bs = torch.where(mieux, b4, bs)
        codes = torch.where(mieux.unsqueeze(-1), c4, codes)
        # amax4 = blocs où amax/4 est RÉELLEMENT l échelle (candidat non clampé) : c est ce que relit
        # outils/part-amax4.py dans les codes (plus grand code = niveau 4) — un candidat 4 clampé à 448 n est ni l un
        # ni l autre, il compte dans « clampés »
        stats["amax4"] = int((mieux & (e4.to(torch.float32) < E4M3_MAX)).sum())
    # clampés = échelle de bloc finale == E4M3_MAX : les deux candidats y sont confondus ou tronqués, 4sur6 sans effet
    stats["clampes"] = int((bs_e4m3.to(torch.float32) >= E4M3_MAX).sum())
    sign = (wb < 0).to(torch.uint8) << 3
    codes = codes | sign
    codes = codes.reshape(out_f, k)

    return NVFP4Tensor(
        qweight=pack_e2m1(codes),
        block_scale=bs_e4m3,
        global_scale=gs.clone(),
        shape=orig_shape,
        padded_in=k,
        echelle_stats=stats,
    )


def dequantize_nvfp4(t: NVFP4Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Déquantification de référence. Le noyau CUDA doit s'y conformer exactement."""
    codes = unpack_e2m1(t.qweight)                        # [out, k]
    out_f, k = codes.shape
    mag = codes & 0x07
    neg = (codes & 0x08) != 0
    levels = _levels_tensor(codes.device)
    vals = levels[mag.long()]
    vals = torch.where(neg, -vals, vals)
    block = k // t.block_scale.shape[-1]
    vals = vals.view(out_f, k // block, block)
    # `global_scale_rows` porte une echelle globale PAR LIGNE DE SORTIE : la
    # fusion l'y pose parce que q, k et v gardent chacune la leur, et que les
    # ramener a une seule les ferait passer par un arrondi e4m3 a 6 %.
    # L'ignorer ici rendait des nombres FAUX sur tout tenseur fusionne
    # dequantifie — chemin CPU, ou repli quand l'extension manque. Mesure du
    # 9/09/2026 : facteur d'echelle constant de 1,1216 sur la projection k.
    gsr = getattr(t, "global_scale_rows", None)
    gs = (gsr.to(torch.float32).view(-1, 1) if gsr is not None
          else t.global_scale.to(torch.float32))
    # Les échelles de bloc voyagent parfois en OCTETS (safetensors, piles
    # d'experts `_pile_bf16`) : un uint8 converti en float donnerait la valeur
    # de l'octet (0-255) au lieu de l'E4M3 décodée — c'est ce qui rendait une
    # PPL de 10⁸ sous ACVRAM_DISABLE_KERNELS=1 (le noyau CUDA, lui, décode les
    # octets ; le jumeau torch doit faire pareil — REGLES §7).
    bs = t.block_scale
    if bs.dtype == torch.uint8:
        bs = bs.view(torch.float8_e4m3fn)
    scale = bs.to(torch.float32) * gs
    out = vals * scale.unsqueeze(-1)
    out = out.reshape(out_f, k)
    if k != t.shape[-1]:
        out = out[:, : t.shape[-1]]
    return out.to(dtype)
