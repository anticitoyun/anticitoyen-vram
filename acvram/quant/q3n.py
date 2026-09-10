"""Q3N — quantiles 3 bits, échelle FP8 par bloc. 3,25 bits par poids en bloc de 32.

Spécification du 8/09/2026 (`docs/FORMAT-3BITS.md`) : huit niveaux issus des
quantiles d'une gaussienne, table **symétrique** — la forme NF4, quatre
niveaux négatifs pour trois positifs, écrêtait d'un tiers les poids positifs
proches du maximum, et le défaut ne s'est vu qu'à une anomalie de mesure.
Pas de zéro exact : à huit niveaux, la symétrie vaut plus qu'un zéro.

Empaquetage : huit valeurs de trois bits dans un mot de 24 bits, faible vers
fort, trois octets. Un bloc de 32 pèse 12 octets de charges plus 1 d'échelle,
soit exactement 3,25 bits par poids.

Raison d'être : la conversion refuse depuis le 8/09 de faire grossir un
modèle ; or un GGUF trois bits (Q3_K_S, 3,4 bits/poids) grossissait d'un
tiers vers NVFP4 (4,5) et son débordement a coûté un facteur quinze au
décodage de Qwen3-Coder-Next. Q3N est le premier format d'acvram sous la
source. Mesuré à la spécification (1 M poids, processeur) : 15,0 dB de SNR
en gaussien et 13,6 en queue lourde au bloc de 32, contre 14,0 et 12,4 pour
des entiers 3 bits au bloc de 16 pourtant plus lourds. **La perplexité sur un
vrai modèle n'est pas encore mesurée**, ni le débit du chemin de calcul.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Table de la spécification d'origine — le REPLI des manifestes qui ne
# déclarent pas la leur. Depuis le 8/09 au soir, la table vit dans le
# manifeste, ajustée par modèle (Lloyd-Max sur échantillon stratifié) ; la
# symétrie et les bornes ±1 sont invariantes, les niveaux ne le sont pas.
TABLE_Q3N = (-1.0000, -0.5783, -0.3186, -0.1025,
             0.1025, 0.3186, 0.5783, 1.0000)

# Table Lloyd-Max stratifiée de Qwen3-Coder-Next (mesure du 8/09 : +1,73 dB
# moyen, zéro perdant sur 288 tenseurs d'évaluation disjoints). Sept niveaux
# logiques, huit entrées physiques — le masque & 7 du dépaquetage exige un
# index toujours valide, la huitième répète la septième.
TABLE_Q3N_LLOYD_CODER_NEXT = (-1.0, -0.5699, -0.2491, 0.0,
                              0.2491, 0.5699, 1.0, 1.0)


def valider_table_q3n(table) -> tuple:
    """Une table de manifeste porte ce qu'on y a écrit ; on vérifie tout.

    L'invariant, arbitré le 8/09 au soir par l'auteur de la spécification :
    huit entrées ; bornes ±1 EXACTES (c'est elles qui interdisent
    l'écrêtage — la symétrie de la première spec n'en était qu'une façon
    commode, réfutée par la mesure : la table asymétrique à zéro est mieux
    centrée en sortie que la symétrique d'origine) ; croissance strictement
    monotone, sauf l'égalité admise entre les deux dernières entrées quand
    la table n'a que sept niveaux logiques ; au plus un zéro. Ni symétrie
    ni parité. Une table non triée rendrait ``bucketize`` incohérent avec
    le dépaquetage et personne ne le verrait avant la perplexité.
    """
    t = tuple(float(v) for v in table)
    if len(t) != 8:
        raise ValueError(f"table q3n : 8 entrées attendues, {len(t)} reçues")
    if abs(t[0] + 1.0) > 1e-6 or abs(t[-1] - 1.0) > 1e-6:
        raise ValueError(f"table q3n : bornes ±1 attendues, {t[0]}..{t[-1]}")
    for i, (a, b) in enumerate(zip(t, t[1:])):
        if b < a or (b == a and i != 6):
            raise ValueError(f"table q3n non strictement croissante : "
                             f"{a} puis {b} (position {i})")
    if sum(1 for v in t if v == 0.0) > 1:
        raise ValueError("table q3n : au plus un zéro")
    return t

BLOC_DEFAUT = 32          # 3,25 bits/poids ; 16 (3,5 bpw) reste ouvert :
                          # +1,48 dB sur la queue lourde, choix à trancher
                          # par une perplexité, pas par ce fichier.


@dataclass
class Q3NTensor:
    qweight: torch.Tensor        # uint8 [sortie, entrée * 3 // 8]
    block_scale: torch.Tensor    # float8_e4m3fn [sortie, entrée // bloc]
    global_scale: torch.Tensor   # float32 scalaire
    block: int
    shape: tuple[int, ...]
    format: str = "q3n"
    # La table ne va PAS dans state_dict : identique pour tout le modèle,
    # elle suit le gabarit dans _rehydrate (comme block et shape) au lieu de
    # traverser _emballer/_decouper à chaque transfert d'expert.
    table: tuple = TABLE_Q3N

    @property
    def nbytes(self) -> int:
        return (self.qweight.numel() + self.block_scale.numel()
                + self.global_scale.numel() * 4)

    @property
    def bits_per_weight(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return self.nbytes * 8 / max(1, n)

    def global_scale_float(self) -> float:
        """Le scalaire d'échelle, mémorisé côté hôte.

        Copié sur ``NVFP4Tensor.global_scale_float`` et pour la même raison,
        déjà mesurée sur ce dépôt : relire ce tenseur par ``.item()`` à chaque
        GEMV synchronise le flux CUDA, et ce seul appel pesait 62 % du temps de
        décodage au profil NVFP4. L'échelle globale ne change plus après la
        quantification ; elle se lit une fois.

        Une synchronisation ici a un second effet, plus silencieux : elle rend
        le noyau **incapturable dans un graphe CUDA**, et acvram en capture.
        """
        gs = self.__dict__.get("_gs_f")
        if gs is None:
            gs = float(self.global_scale.item())
            self.__dict__["_gs_f"] = gs
        return gs

    def to(self, device, non_blocking: bool = False) -> "Q3NTensor":
        return Q3NTensor(self.qweight.to(device, non_blocking=non_blocking),
                         self.block_scale.to(device, non_blocking=non_blocking),
                         self.global_scale.to(device, non_blocking=non_blocking),
                         self.block, self.shape, self.format, self.table)

    def table_gpu(self, device) -> torch.Tensor:
        """La table en float32 sur ``device``, mémorisée — un tenseur de 32
        octets par modèle et par carte, jamais recréé par appel (une
        allocation par GEMV serait un fantôme dans une capture de graphe)."""
        cache = self.__dict__.setdefault("_tables_gpu", {})
        cle = str(device)
        t = cache.get(cle)
        if t is None:
            t = torch.tensor(self.table, device=device, dtype=torch.float32)
            cache[cle] = t
        return t

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}qweight": self.qweight,
                f"{prefix}block_scale": self.block_scale,
                f"{prefix}global_scale": self.global_scale}


def empaqueter_q3(q: torch.Tensor) -> torch.Tensor:
    """Indices 0..7 → octets, huit valeurs sur trois octets, faible vers fort."""
    assert q.shape[-1] % 8 == 0
    g = q.reshape(*q.shape[:-1], -1, 8).to(torch.int32)
    mot = (g[..., 0] | (g[..., 1] << 3) | (g[..., 2] << 6) | (g[..., 3] << 9)
           | (g[..., 4] << 12) | (g[..., 5] << 15) | (g[..., 6] << 18)
           | (g[..., 7] << 21))
    octets = torch.stack([(mot >> 0) & 0xFF, (mot >> 8) & 0xFF,
                          (mot >> 16) & 0xFF], dim=-1)
    return octets.to(torch.uint8).reshape(*q.shape[:-1], -1)


def depaqueter_q3(octets: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse exact de ``empaqueter_q3`` ; ``n`` valeurs par ligne."""
    o = octets.reshape(*octets.shape[:-1], -1, 3).to(torch.int32)
    mot = o[..., 0] | (o[..., 1] << 8) | (o[..., 2] << 16)
    vals = torch.stack([(mot >> (3 * i)) & 0x7 for i in range(8)], dim=-1)
    return vals.reshape(*octets.shape[:-1], -1)[..., :n]


def quantize_q3n(w: torch.Tensor, block: int = BLOC_DEFAUT,
                 table=None) -> Q3NTensor:
    """Quantifie ``w`` [sortie, entrée] ; l'entrée doit être multiple du bloc.

    L'échelle de bloc est arrondie en FP8 **avant** la quantification, comme
    à l'exécution : optimiser les indices contre une échelle qui sera ensuite
    dégradée donnerait un SNR de papier.
    """
    assert w.dim() == 2 and w.shape[1] % block == 0, w.shape
    sortie, entree = w.shape
    wf = w.to(torch.float32)
    g = wf.abs().max().clamp(min=1e-12)
    blocs = wf.reshape(sortie, entree // block, block)
    bs = (blocs.abs().amax(dim=-1) / g).to(torch.float8_e4m3fn)
    echelle = (bs.to(torch.float32) * g).clamp(min=1e-12)
    reduit = blocs / echelle.unsqueeze(-1)
    tab = valider_table_q3n(table) if table is not None else TABLE_Q3N
    # bucketize sur les niveaux DISTINCTS : une table à sept niveaux (doublon
    # final) ne produit alors jamais le code 7, mais le dépaquetage le lirait
    # sans danger — huit entrées physiques toujours.
    niveaux = sorted(set(tab))
    tn = torch.tensor(niveaux, device=w.device, dtype=torch.float32)
    bornes = (tn[1:] + tn[:-1]) / 2
    q = torch.bucketize(reduit.reshape(sortie, entree), bornes)
    return Q3NTensor(empaqueter_q3(q), bs, g.reshape(()),
                     block, (sortie, entree), "q3n", tab)


def dequantize_q3n(t: Q3NTensor, out_dtype: torch.dtype = torch.bfloat16
                   ) -> torch.Tensor:
    sortie, entree = t.shape
    q = depaqueter_q3(t.qweight, entree)
    table = torch.tensor(t.table, device=t.qweight.device, dtype=torch.float32)
    vals = table[q.long()].reshape(sortie, entree // t.block, t.block)
    echelle = t.block_scale.to(torch.float32) * t.global_scale.to(t.qweight.device)
    return (vals * echelle.unsqueeze(-1)).reshape(sortie, entree).to(out_dtype)


def q3n_gemv(x: torch.Tensor, t: Q3NTensor) -> Optional[torch.Tensor]:
    """``x @ W.T`` par déquantification — chemin de référence.

    Un noyau fusionné (déquantification en mémoire partagée, comme la GEMM
    groupée NVFP4) viendra le remplacer ; ce chemin-ci ferme la marche et fixe
    la numérique que le noyau devra égaler.

    Rend ``None`` sur une forme qui ne convient pas, jamais une exception : le
    planificateur essaie un chemin puis retombe sur le suivant, et une
    exception ici fait tomber la requête entière au lieu de replier. Même
    contrat que ``nvfp4_mm_tensorcore``.
    """
    if x.dim() not in (1, 2):
        return None
    if x.shape[-1] != t.shape[1]:
        return None
    dt = x.dtype if x.dtype != torch.float32 else torch.bfloat16
    return torch.nn.functional.linear(x, dequantize_q3n(t, dt).to(x.dtype))
