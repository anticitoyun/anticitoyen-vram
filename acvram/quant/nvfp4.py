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


def quantize_nvfp4(
    weight: torch.Tensor,
    block: int = BLOCK,
    global_scale: Optional[torch.Tensor] = None,
) -> NVFP4Tensor:
    """Quantifie en NVFP4 un poids 2-D ``[sorties, entrées]``.

    Les blocs courent le long des entrées, c'est-à-dire de la dimension de
    réduction, ce qu'attend un produit matriciel orienté K : chaque tranche de
    16 porte sa propre échelle, si bien qu'un unique canal aberrant ne peut pas
    aplatir toute une ligne.
    """
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
    ideal = block_amax / E2M1_MAX                         # échelle exacte par bloc
    # On représente l'échelle de bloc en E4M3 : c'est un arrondi réel et avec
    # perte, et le noyau doit utiliser la valeur *arrondie*, jamais `ideal`.
    bs_e4m3 = (ideal / gs).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    bs = bs_e4m3.to(torch.float32) * gs                   # échelle effective

    safe = bs.clamp(min=torch.finfo(torch.float32).tiny)
    normed = wb / safe.unsqueeze(-1)
    codes = round_to_e2m1(normed)
    codes = torch.where((bs.unsqueeze(-1) > 0), codes, torch.zeros_like(codes))
    sign = (wb < 0).to(torch.uint8) << 3
    codes = codes | sign
    codes = codes.reshape(out_f, k)

    return NVFP4Tensor(
        qweight=pack_e2m1(codes),
        block_scale=bs_e4m3,
        global_scale=gs.clone(),
        shape=orig_shape,
        padded_in=k,
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
