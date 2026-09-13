"""Rendre 4 bits réellement utilisables : mise à l'échelle AWQ et rotation de
Hadamard.

Un arrondi naïf au plus proche sur 4 bits coûte environ 20 dB de rapport
signal/bruit sur une matrice de poids, ce qui suffit à abîmer visiblement un
modèle. Deux techniques peu coûteuses et orthogonales en récupèrent l'essentiel,
et toutes deux sont *compatibles avec le stockage* des codecs de
:mod:`acvram.quant.nvfp4` et :mod:`acvram.quant.int4` : elles ne changent que ce
qui est quantifié, jamais la façon dont c'est empaqueté.

1. Mise à l'échelle des canaux guidée par les activations (AWQ)
   Les canaux d'entrée saillants — ceux sur lesquels les activations sont
   grandes — méritent une plus grande part du budget de 4 bits. On multiplie la
   colonne j du poids par s_j avant de quantifier, et on divise l'activation par
   s_j à l'exécution : le produit est inchangé, mais la grille de quantification
   atterrit désormais là où cela compte. s = moyenne|x_j| ** alpha, alpha étant
   trouvé par recherche sur grille sur l'erreur en sortie de couche.

2. Rotation de Hadamard aléatoire (famille QuaRot / SpinQuant)
   Multiplier par une matrice de Hadamard orthogonale répartit les valeurs
   aberrantes entre les canaux, transformant une distribution à queue lourde en
   une distribution quasi gaussienne, qu'une grille uniforme sur 4 bits épouse
   bien mieux. Appliquée des deux côtés, elle s'annule exactement :
   y = x W^T = (x H)(W H)^T, puisque H H^T = I.

Toutes deux produisent un ``ChannelScaler`` par couche, que l'exécution applique
à l'activation d'entrée. Le coût au décodage est d'une multiplication terme à
terme (AWQ) et d'une transformée en n log n (Hadamard) par couche linéaire —
négligeable devant le produit matriciel lui-même.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from . import formats

__all__ = ["ChannelScaler", "search_channel_scales", "hadamard_transform",
           "largest_pow2_divisor", "apply_hadamard_weight", "ActStats",
           "quantize_with_calibration"]


# --------------------------------------------------------------------------
# Hadamard
# --------------------------------------------------------------------------


def largest_pow2_divisor(n: int, cap: int = 8192) -> int:
    """Plus grande puissance de deux divisant ``n``, bornée, pour un usage bloc-diagonal."""
    p = 1
    while p * 2 <= min(n, cap) and n % (p * 2) == 0:
        p *= 2
    return p


def hadamard_transform(x: torch.Tensor, block: Optional[int] = None,
                       normalize: bool = True) -> torch.Tensor:
    """Transformée de Walsh-Hadamard rapide sur la dernière dimension.

    Lorsque la dernière dimension n'est pas une puissance de deux, la
    transformée est appliquée par blocs diagonaux sur le plus grand diviseur
    puissance de deux, ce qui reste une application orthogonale exacte — une
    somme directe de matrices de Hadamard — et décorrèle toujours à l'intérieur
    de chaque bloc.
    """
    n = x.shape[-1]
    b = block or largest_pow2_divisor(n)
    if b < 2:
        return x
    if n % b:
        raise ValueError(f"le bloc de Hadamard {b} ne divise pas {n}")
    orig_shape = x.shape
    y = x.reshape(-1, n // b, b).clone()
    h = 1
    while h < b:
        y = y.view(y.shape[0], y.shape[1], b // (2 * h), 2, h)
        a = y[..., 0, :]
        c = y[..., 1, :]
        y = torch.stack((a + c, a - c), dim=-2)
        h *= 2
    y = y.reshape(-1, n // b, b)
    if normalize:
        y = y / math.sqrt(b)
    return y.reshape(orig_shape)


def apply_hadamard_weight(weight: torch.Tensor, block: Optional[int] = None) -> torch.Tensor:
    """Fait tourner un poids selon sa dimension d'entrée : ``W <- W H``."""
    return hadamard_transform(weight, block=block)


# --------------------------------------------------------------------------
# activation statistics
# --------------------------------------------------------------------------


@dataclass
class ActStats:
    """Magnitude d'activation par canal d'entrée, relevée sur des données de calibration."""

    mean_abs: torch.Tensor          # [in_features]
    max_abs: Optional[torch.Tensor] = None
    n_samples: int = 0

    @staticmethod
    def from_inputs(x: torch.Tensor) -> "ActStats":
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        return ActStats(flat.abs().mean(0), flat.abs().amax(0), flat.shape[0])

    def merge(self, other: "ActStats") -> "ActStats":
        n = self.n_samples + other.n_samples
        if n == 0:
            return self
        w1, w2 = self.n_samples / n, other.n_samples / n
        return ActStats(
            self.mean_abs * w1 + other.mean_abs * w2,
            torch.maximum(self.max_abs, other.max_abs)
            if self.max_abs is not None and other.max_abs is not None else None,
            n,
        )


@dataclass
class ChannelScaler:
    """Ce que l'exécution applique à l'activation avant le produit matriciel.

    ``x_eff = hadamard(x) / échelle`` quand la rotation est active, sinon
    ``x_eff = x / échelle``.
    """

    scale: Optional[torch.Tensor]      # [in_features], fp16/bf16
    hadamard_block: int = 0

    @property
    def is_identity(self) -> bool:
        return self.scale is None and self.hadamard_block == 0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.hadamard_block:
            x = hadamard_transform(x, block=self.hadamard_block)
        if self.scale is not None:
            x = x / self._au_dtype(x.dtype)
        return x

    def _au_dtype(self, dtype: torch.dtype) -> torch.Tensor:
        """L'échelle au dtype demandé, convertie UNE FOIS.

        ``self.scale.to(x.dtype)`` s'exécutait à chaque appel de chaque
        projection. Les échelles sont stockées en fp16 et les activations sont
        en bf16 : la conversion est réelle, pas un no-op, et elle lance un
        noyau. Compté sous ncu sur Qwen2.5-Coder-14B en nvfp4 : **337
        `unrolled_elementwise` par pas — exactement sept projections par couche
        sur quarante-huit, plus la tête**. Le bf16, qui n'a pas d'échelle, n'en
        lance aucun.

        Le cache est par dtype et non unique : rien ne garantit qu'un modèle
        n'exécute qu'en un seul type, et convertir au placement supposerait de
        connaître le dtype d'exécution au moment du placement — ce que
        ``to(device)`` ne sait pas.

        Numériquement identique : c'est la même conversion, faite une fois.
        """
        import os
        if os.environ.get("ACVRAM_SCALER_SANS_CACHE") == "1":
            return self.scale.to(dtype)        # temoin de mesure
        cache = self.__dict__.get("_cache_dtype")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_cache_dtype", cache)
        s = cache.get(dtype)
        if s is None:
            s = self.scale.to(dtype)
            cache[dtype] = s
        return s

    def to(self, device, non_blocking: bool = False) -> "ChannelScaler":
        # Un nouvel objet, donc un cache neuf : une echelle mise en cache pour
        # cuda:0 ne doit jamais servir sur cuda:1.
        return ChannelScaler(
            self.scale.to(device, non_blocking=non_blocking)
            if self.scale is not None else None,
            self.hadamard_block,
        )

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """L'échelle part en float32, quel que soit son dtype de calcul.

        Elle était écrite en fp16, dont l'exposant ne couvre que 5 bits : sur
        Qwen2.5-Coder-14B les échelles vont de 8,9e-4 à 4,0e+2, soit une marge
        de x164 au plafond mais seulement **x15 au plancher** des dénormaux
        (6,1e-5). Un modèle dont les échelles seraient quinze fois plus petites
        y tomberait — et ce ne serait plus une perte de précision mais des
        valeurs fausses, sans que rien ne le signale.

        Un garde attraperait ce défaut ; le float32 supprime la classe. Le coût
        est nul des deux côtés : ~4 Mo par modèle sur disque, et zéro à
        l'exécution puisque l'échelle est convertie une seule fois au
        chargement, au dtype des activations.

        Les modèles déjà convertis restent en fp16 et se chargent sans
        changement : le lecteur prend le dtype qu'il trouve.
        """
        out: dict[str, torch.Tensor] = {}
        if self.scale is not None:
            out[f"{prefix}act_scale"] = self.scale.to(torch.float32)
        return out


# --------------------------------------------------------------------------
# AWQ grid search
# --------------------------------------------------------------------------


def _quant_dequant(w: torch.Tensor, fmt: str, group_size: Optional[int]) -> torch.Tensor:
    t = formats.quantize(w, fmt, group_size=group_size)
    return formats.dequantize(t, torch.float32)


def search_channel_scales(
    weight: torch.Tensor,
    stats: Optional[ActStats],
    fmt: str,
    group_size: Optional[int] = None,
    n_grid: int = 20,
    calib_x: Optional[torch.Tensor] = None,
    journal: Optional[dict] = None,
) -> tuple[ChannelScaler, float]:
    """Recherche AWQ sur grille de l'échelle par canal.

    Rend l'échelle retenue et l'erreur relative en sortie qu'elle atteint.
    Lorsque ``calib_x`` est fourni, l'objectif est la véritable erreur en sortie
    de couche sur de vraies activations ; sinon l'activation est approchée par sa
    magnitude moyenne par canal, ce que fait le mode économique d'AWQ.
    """
    w = weight.detach().to(torch.float32)
    device = w.device
    k = w.shape[1]

    if stats is None:
        act = torch.ones(k, device=device)
    else:
        act = stats.mean_abs.to(device).to(torch.float32).clamp(min=1e-6)

    if calib_x is not None:
        x = calib_x.reshape(-1, k).to(torch.float32).to(device)
    else:
        # Substitut : une sonde diagonale pondérée par la magnitude des canaux
        # reproduit la pondération d'erreur par canal d'AWQ sans conserver les
        # activations.
        x = torch.diag(act)

    y_ref = x @ w.t()
    ref_norm = y_ref.norm().clamp(min=1e-12)

    best_err = float("inf")
    best_scale: Optional[torch.Tensor] = None
    # `journal` conserve l'erreur des n_grid+1 valeurs, pas seulement le
    # minimum. Sans elles, le prix d'un exposant COMMUN a un groupe empilable
    # n'est pas calculable : `gate` et `up` lisent la meme entree mais chacun
    # choisit son exposant, et `_scaler_commun` refuse par `torch.equal` — 5
    # empilements sur 64 sur Llama-2-7b-int8, mesures le 10/09, pour un gain de
    # +0,19 % la ou une couverture complete vaudrait +2,43 %. Le prix se lit
    # dans ces erreurs, sans une conversion de plus.
    grille: list[float] = []

    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = act.pow(alpha)
        s = s / s.mean().clamp(min=1e-12)            # garde l'échelle centrée
        s = s.clamp(min=1e-4, max=1e4)
        wq = _quant_dequant(w * s.unsqueeze(0), fmt, group_size)
        y = (x / s) @ wq.t()
        err = ((y - y_ref).norm() / ref_norm).item()
        grille.append(err)
        if err < best_err:
            best_err, best_scale = err, s.clone()

    assert best_scale is not None
    if journal is not None:
        journal["erreurs_grille"] = [round(e, 8) for e in grille]
        journal["alpha_retenu"] = round(grille.index(best_err) / n_grid, 4)
    identity = torch.ones_like(best_scale)
    if torch.allclose(best_scale, identity, atol=1e-3):
        best_scale = None
    # L'échelle reste en float32. Elle était rabattue en fp16 ici, dont
    # l'exposant ne couvre que 5 bits : sous 6,1e-5 les valeurs deviennent
    # dénormales, sous ~6e-8 elles deviennent NULLES. Sur Qwen2.5-Coder-14B la
    # plus petite vaut 8,9e-4, soit une marge de x15 seulement — un modèle aux
    # échelles quinze fois plus petites aurait été écrêté en silence, et une
    # échelle nulle ne dégrade pas la sortie, elle la détruit.
    #
    # Écrire du float32 au manifeste ne suffisait pas : la valeur était déjà
    # perdue ICI, à la recherche. Le coût est un vecteur de quelques milliers
    # de valeurs par tenseur, et zéro à l'exécution depuis que l'échelle est
    # convertie une seule fois au chargement.
    return ChannelScaler(
        best_scale.to(torch.float32) if best_scale is not None else None
    ), best_err


def kld_couche_bits(y_ref: torch.Tensor, y_q: torch.Tensor) -> float:
    """Divergence de Kullback-Leibler entre les sorties de couche, en bits.

    Duck.ai (poste2, 12/09/2026, revue/duck-poste2-12-09.md) : trois modeles a
    recherche web concordent — le KLD au logit final correle mieux que le
    SNR-bloc avec la perte utilisateur reelle (Spearman 0,96-0,97 avec les
    inversions de jeton, methodologie Fireworks). Le mesurer sur le VRAI
    logit final couterait un forward complet du modele par tenseur candidat,
    hors de portee d'une decision prise au fil de la conversion, tenseur par
    tenseur. On calcule donc un proxy : softmax du MEME vecteur de sortie
    deja produit pour `out_snr_db` (x = diag(probe), y = x @ w.T), traite
    comme une loi de probabilite sur ses composantes.

    C'est un OPTION AJOUTEE, pas un remplacement du SNR (consigne chef
    12/09) : le convertisseur continue de decider par `out_snr_db`, ce champ
    est seulement mesure et publie a cote pour le protocole A/B a venir.

    Toujours positif, nul seulement si les deux lois coincident exactement.
    PAS symetrique : kld(ref, q) mesure ce que le format quantifie perd de la
    loi de reference — c'est le sens utilise par la litterature de
    calibration (KL(p_fp16 || p_quant)), donc celui retenu ici.
    """
    p = torch.softmax(y_ref.flatten().to(torch.float64), dim=0)
    q = torch.softmax(y_q.flatten().to(torch.float64), dim=0).clamp(min=1e-12)
    kld_nats = torch.sum(p * (p.clamp(min=1e-12).log() - q.log())).item()
    return kld_nats / math.log(2)


def quantize_with_calibration(
    weight: torch.Tensor,
    fmt: str,
    stats: Optional[ActStats] = None,
    group_size: Optional[int] = None,
    use_hadamard: bool = False,
    use_awq: bool = True,
    n_grid: int = 20,
    garder_grille: bool = False,
    table=None,
    mesurer_kld: bool = False,
) -> tuple[Any, ChannelScaler, dict]:
    """Chaîne complète par couche : tourner, mettre à l'échelle, quantifier.

    L'ordre compte. La rotation de Hadamard vient en premier parce qu'elle
    change les statistiques de canaux sur lesquelles opère la recherche AWQ :
    chercher avant de tourner optimiserait une échelle pour une distribution qui
    n'existe plus.
    """
    w = weight.detach().to(torch.float32)
    had_block = 0
    if use_hadamard:
        had_block = largest_pow2_divisor(w.shape[1])
        if had_block >= 8:
            w = apply_hadamard_weight(w, had_block)
            if stats is not None:
                rotated = hadamard_transform(stats.mean_abs.reshape(1, -1).to(torch.float32),
                                             block=had_block).abs().reshape(-1)
                stats = ActStats(rotated.clamp(min=1e-6), None, stats.n_samples)
        else:
            had_block = 0

    scaler = ChannelScaler(None, had_block)
    journal: Optional[dict] = {} if garder_grille else None
    if use_awq:
        found, _ = search_channel_scales(w, stats, fmt, group_size, n_grid,
                                         journal=journal)
        scaler = ChannelScaler(found.scale, had_block)

    w_eff = w * scaler.scale.to(torch.float32).unsqueeze(0) \
        if scaler.scale is not None else w
    qt = formats.quantize(w_eff, fmt, group_size=group_size,
                          **({"table": table} if fmt == "q3n" else {}))

    deq = formats.dequantize(qt, torch.float32)
    if scaler.scale is not None:
        deq = deq / scaler.scale.to(torch.float32).unsqueeze(0)

    # Deux erreurs différentes, et elles ne varient pas ensemble.
    #
    #   w_err    à quelle distance les poids reconstruits sont des originaux
    #   out_err  à quelle distance est la *sortie de couche*, sur des
    #            activations ressemblant au jeu de calibration
    #
    # AWQ dégrade délibérément w_err pour améliorer out_err : il dépense de la
    # résolution de grille sur les canaux où les activations sont réellement
    # grandes. Juger AWQ sur w_err le rejetterait à chaque fois ; c'est donc
    # out_err que le convertisseur rapporte et sur quoi il classe.
    w_err = ((deq - w).norm() / w.norm().clamp(min=1e-12)).item()

    probe = (stats.mean_abs.to(torch.float32).clamp(min=1e-6)
             if stats is not None else torch.ones(w.shape[1], device=w.device))
    x = torch.diag(probe.to(w.device))
    y_ref = x @ w.t()
    y_q = x @ deq.t()
    # `out_err` est une erreur RELATIVE : le denominateur `||y_ref||` disparait
    # dans le rapport. C'est le bon chiffre pour juger un tenseur CONTRE
    # LUI-MEME — « ce format le degrade-t-il plus que cet autre ? » — et le
    # mauvais pour classer DEUX TENSEURS l'un contre l'autre, ce que fait le sac
    # a dos budgetaire. Deux tenseurs a 20 dB et 30 dB dont les sorties valent
    # 1 et 100 portent des erreurs absolues de 0,1 et 3,16 : le second nuit
    # trente fois plus et le classement par decibels le met second.
    #
    # Ce qui se propage jusqu'a la perte est l'erreur ABSOLUE. Elle est deja
    # calculee ici — c'est le numerateur — et jetee. On la garde, avec l'echelle
    # qui la rend interpretable. Aucun chemin existant ne change : les deux
    # champs sont ajoutes, aucun n'est remplace.
    out_abs_err = (y_q - y_ref).norm().item()
    out_ref_norm = y_ref.norm().clamp(min=1e-12).item()
    out_err = out_abs_err / out_ref_norm

    metrics = {
        "w_rel_err": w_err,
        "w_snr_db": 20 * math.log10(1.0 / max(w_err, 1e-12)),
        "out_rel_err": out_err,
        "out_abs_err": out_abs_err,
        "out_ref_norm": out_ref_norm,
        "out_snr_db": 20 * math.log10(1.0 / max(out_err, 1e-12)),
        "hadamard_block": had_block,
        "awq": scaler.scale is not None,
        "bpw": getattr(qt, "bits_per_weight", 16.0),
    }
    if mesurer_kld:
        # OPTION, pas de remplacement : `out_snr_db` reste le critere de
        # decision du convertisseur (consigne chef 12/09). Ce champ n'est
        # lu par aucun chemin de decision existant.
        metrics["out_kld_bits"] = kld_couche_bits(y_ref, y_q)
    if journal:
        metrics.update(journal)
    return qt, scaler, metrics
