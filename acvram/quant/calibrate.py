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
from .fakequant_activation import fake_quantize_nvfp4_activation

__all__ = ["ChannelScaler", "search_channel_scales",
           "search_channel_scales_commun", "alpha_commun_gate_up",
           "hadamard_transform",
           "fwht_activations",
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


def fwht_activations(x: torch.Tensor, block: int) -> torch.Tensor:
    """La rotation de Hadamard des ACTIVATIONS, l'arithmétique du noyau
    (nvfp4_quant_act, poste7-hadamard-16-09) : étages papillon en fp32 (chaque
    sortie = une somme de deux valeurs), normalisation × (1/√bloc) calculée en
    fp32 RN (√ puis division tenseur/tenseur — tenseur/scalaire multiplierait
    par l'inverse, un ulp d'écart), puis retour au dtype de x. Utilisée par la
    boucle par expert (ChannelScaler.apply) et par les chemins torch de la
    pile : les trois arrondissent pareil, la référence des tests aussi."""
    n = x.shape[-1]
    if block < 2 or n % block:
        raise ValueError(f"le bloc de Hadamard {block} ne divise pas {n}")
    orig_shape, dtype = x.shape, x.dtype
    y = x.reshape(-1, n // block, block).to(torch.float32).clone()
    h = 1
    while h < block:
        v = y.view(y.shape[0], y.shape[1], block // (2 * h), 2, h)
        a, c = v[..., 0, :], v[..., 1, :]
        y = torch.stack((a + c, a - c), dim=-2).reshape(y.shape[0], y.shape[1], block)
        h *= 2
    # 1/√bloc en fp32 RN, calculé UNE fois sur l'hôte (numpy : √ et division
    # correctement arrondies, comme __fsqrt_rn / __fdiv_rn du noyau) — pas de
    # torch.tensor(..., device=cuda) ici : cette copie hôte→carte est interdite
    # pendant une capture de graphe CUDA (poste2, verdict-glm-hadamard-conversion :
    # 0 godet capturé sur le converti tourné). Le scalaire Python est un fp32
    # exact ; y * scalaire multiplie en fp32 RN.
    return (y * _inv_racine_fp32(block)).reshape(orig_shape).to(dtype)


_INV_RACINES: dict = {}


def _inv_racine_fp32(block: int) -> float:
    inv = _INV_RACINES.get(block)
    if inv is None:
        import numpy as np
        inv = float(np.float32(1.0) / np.sqrt(np.float32(block)))
        _INV_RACINES[block] = inv
    return inv


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
            x = fwht_activations(x, self.hadamard_block)
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


def _magnitude_avec_plancher_relatif(mean_abs: torch.Tensor) -> torch.Tensor:
    """`mean_abs`, plancher borné sur l'ÉTENDUE (max/plancher ≤ 4096), pas
    une statistique de position comme la médiane (poste7-awq-plancher-
    median-faute-18-09 : CORRECTIF -- la version précédente de cette
    fonction, `1e-2 × médiane(mean_abs)`, supposait moins de la moitié des
    canaux quasi nuls ; ReLU²(up(x)) (entrée de `down_proj`, nemotron_h,
    `config.py:563`/`model.py:622`) viole cette hypothèse PAR NATURE dès
    qu'une majorité de canaux ne s'active jamais sur le corpus -- la
    médiane elle-même retombe alors près de zéro, et `1e-2 × médiane`
    devient un plancher PLUS BAS que l'ancien `1e-6`, élargissant
    l'étendue au lieu de la borner (mesuré : 6/10 tenseurs Nemotron
    aggravés jusqu'à ×319, ratio de norme 0,29-0,69 → PPL 1,2634 sur un
    converti désormais retiré).

    Borner sur le MAXIMUM du tenseur, jamais sa position centrale, tient
    quelle que soit la fraction de canaux nuls : `plancher = max(1e-6,
    max(mean_abs)/4096)` — l'étendue résultante est AU PLUS 4096 par
    construction, peu importe combien de canaux sont écrasés. Sur une
    activation sans motif creux (Coder/GLM, SiLU à porte), le maximum est
    déjà proche des canaux bas et ce plancher reste sous 1e-6, sans effet."""
    plancher = torch.clamp(mean_abs.max() / 4096.0, min=1e-6)
    return mean_abs.clamp(min=plancher)


def search_channel_scales(
    weight: torch.Tensor,
    stats: Optional[ActStats],
    fmt: str,
    group_size: Optional[int] = None,
    n_grid: int = 20,
    calib_x: Optional[torch.Tensor] = None,
    journal: Optional[dict] = None,
    quantize_activation_nvfp4: bool = False,
) -> tuple[ChannelScaler, float]:
    """Recherche AWQ sur grille de l'échelle par canal.

    Rend l'échelle retenue et l'erreur relative en sortie qu'elle atteint.
    Lorsque ``calib_x`` est fourni, l'objectif est la véritable erreur en sortie
    de couche sur de vraies activations ; sinon l'activation est approchée par sa
    magnitude moyenne par canal, ce que fait le mode économique d'AWQ.

    ``quantize_activation_nvfp4`` (poste7, `poste7-glm-mma0-verdict-16-09.md` § 2,
    16/09) : par défaut la métrique suppose ``x / s`` en pleine précision
    (W4A16) alors que le chemin réel des experts en pile groupée quantifie
    aussi l'activation en NVFP4 après division par l'échelle
    (`nvfp4_quant_act`, `model.py:1022`) — une métrique W4A16 pour un chemin
    W4A4 choisit un alpha qui élargit l'étendue intra-bloc de l'activation
    sans que la recherche le voie. Passer ce drapeau applique
    `fake_quantize_nvfp4_activation` à ``x / s`` avant le produit matriciel
    de la grille, pour que l'alpha retenu minimise l'erreur du chemin
    réellement emprunté.
    """
    w = weight.detach().to(torch.float32)
    device = w.device
    k = w.shape[1]

    if stats is None:
        act = torch.ones(k, device=device)
    else:
        act = _magnitude_avec_plancher_relatif(stats.mean_abs.to(device).to(torch.float32))

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

    # Pièce 23 (22/09) : quand la magnitude par canal est CONSTANTE — c est le
    # cas dès que `stats is None`, donc pour tout expert que le corpus n a pas
    # routé — `s = act^alpha / mean(act^alpha)` vaut 1 pour TOUTE valeur
    # d alpha : les n_grid+1 évaluations sont identiques, et leur résultat est
    # connu d avance (l identité). On en fait UNE, pour l erreur rendue, et la
    # grille est remplie de cette même valeur : sortie au bit, journal
    # identique, n_grid+1 fois moins de quantifications. Sur le 30B-VL du
    # 22/09, 5 235 experts sans statistique sur 6 144, soit ~85 % des tenseurs
    # quantifiés, prenaient ce chemin pour rien.
    plat = bool(torch.allclose(act, act.reshape(-1)[0].expand_as(act)))
    n_eval = 1 if plat else n_grid + 1
    for i in range(n_eval):
        alpha = i / n_grid
        s = act.pow(alpha)
        s = s / s.mean().clamp(min=1e-12)            # garde l'échelle centrée
        s = s.clamp(min=1e-4, max=1e4)
        wq = _quant_dequant(w * s.unsqueeze(0), fmt, group_size)
        xa = x / s
        if quantize_activation_nvfp4:
            xa = fake_quantize_nvfp4_activation(xa)
        y = xa @ wq.t()
        err = ((y - y_ref).norm() / ref_norm).item()
        grille.append(err)
        if err < best_err:
            best_err, best_scale = err, s.clone()
    if plat:
        grille = grille * (n_grid + 1)               # le journal ne change pas

    assert best_scale is not None
    if journal is not None:
        journal["erreurs_grille"] = [round(e, 8) for e in grille]
        journal["alpha_retenu"] = round(grille.index(best_err) / n_grid, 4)
        journal["grille_plate"] = plat
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


def search_channel_scales_commun(
    weights: list[torch.Tensor],
    stats: Optional[ActStats],
    fmt: str,
    group_size: Optional[int] = None,
    n_grid: int = 20,
    calib_x: Optional[torch.Tensor] = None,
    journal: Optional[dict] = None,
    quantize_activation_nvfp4: bool = False,
) -> tuple[ChannelScaler, list[float]]:
    """Comme `search_channel_scales`, mais un SEUL alpha pour plusieurs
    tenseurs qui lisent la MEME entree (gate_proj/up_proj d'un bloc MLP,
    par exemple).

    Item A7 de l'audit poste7 (14/09) : `search_channel_scales` appelee
    separement sur `gate` et `up` choisit deux exposants differents la
    plupart du temps (chacun minimise SA PROPRE erreur), et
    `_scaler_commun` (layers.py) refuse alors de porter le scaler sur la
    pile empilee — 5 empilements sur 64 mesures sur Llama-2-7b-int8 le
    10/09 (`revue/alpha-partage-recuperer-59-fusions.md`). Chercher un
    alpha qui minimise la SOMME des erreurs relatives des deux tenseurs
    rend les deux scalers identiques PAR CONSTRUCTION, sans toucher au
    moteur : `_scaler_commun`/`_try_build_stacks` acceptent deja un
    scaler partage, ils ne le reçoivent simplement jamais.

    Rend l'echelle commune retenue et la liste des erreurs relatives
    INDIVIDUELLES obtenues avec CET alpha (une par tenseur, meme ordre
    que `weights`) — a comparer au meilleur alpha propre de chaque
    tenseur (`search_channel_scales` appele separement) pour connaitre
    le prix de la fusion avant de la choisir (deja mesure le 10/09 :
    sous 2 % a budget 4,50 Gio pour 45/64 groupes).
    """
    if len(weights) < 2:
        raise ValueError("recherche commune : au moins deux tenseurs")
    ws = [w.detach().to(torch.float32) for w in weights]
    k = ws[0].shape[1]
    if any(w.shape[1] != k for w in ws):
        raise ValueError("recherche commune : les tenseurs doivent partager "
                         "le meme nombre de canaux d'entree")
    device = ws[0].device

    if stats is None:
        act = torch.ones(k, device=device)
    else:
        act = _magnitude_avec_plancher_relatif(stats.mean_abs.to(device).to(torch.float32))

    if calib_x is not None:
        x = calib_x.reshape(-1, k).to(torch.float32).to(device)
    else:
        x = torch.diag(act)

    y_refs = [x @ w.t() for w in ws]
    ref_norms = [y.norm().clamp(min=1e-12) for y in y_refs]

    best_somme = float("inf")
    best_scale: Optional[torch.Tensor] = None
    best_erreurs: list[float] = []
    grille_somme: list[float] = []

    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = act.pow(alpha)
        s = s / s.mean().clamp(min=1e-12)
        s = s.clamp(min=1e-4, max=1e4)
        erreurs = []
        xa = x / s
        # Même régime que la recherche par tenseur : les experts MoE sont
        # servis en W4A4 (activation quantifiée par le noyau), et un alpha
        # cherché sans cette quantification est choisi pour un autre chemin
        # que celui qui le consommera (`search_channel_scales`, l.357).
        if quantize_activation_nvfp4:
            xa = fake_quantize_nvfp4_activation(xa)
        for w, y_ref, ref_norm in zip(ws, y_refs, ref_norms):
            wq = _quant_dequant(w * s.unsqueeze(0), fmt, group_size)
            y = xa @ wq.t()
            erreurs.append(((y - y_ref).norm() / ref_norm).item())
        somme = sum(erreurs)
        grille_somme.append(somme)
        if somme < best_somme:
            best_somme, best_scale, best_erreurs = somme, s.clone(), erreurs

    assert best_scale is not None
    if journal is not None:
        journal["erreurs_grille_commune"] = [round(e, 8) for e in grille_somme]
        journal["alpha_commun_retenu"] = round(
            grille_somme.index(best_somme) / n_grid, 4)
    identity = torch.ones_like(best_scale)
    if torch.allclose(best_scale, identity, atol=1e-3):
        best_scale = None
    return ChannelScaler(
        best_scale.to(torch.float32) if best_scale is not None else None
    ), best_erreurs


def alpha_commun_gate_up(
    weights: list[torch.Tensor],
    stats: Optional[ActStats],
    fmt: str,
    group_size: Optional[int] = None,
    use_hadamard: bool = False,
    n_grid: int = 20,
    journal: Optional[dict] = None,
    quantize_activation_nvfp4: bool = False,
) -> tuple[Optional[torch.Tensor], int]:
    """Alpha commun A7, cablage `convert.py` : replique la rotation de
    Hadamard de `quantize_with_calibration` avant `search_channel_scales_commun`,
    pour que l'echelle rendue vive dans le MEME espace que celui ou
    `quantize_with_calibration(..., forced_scale=...)` la consommera —
    sinon un `use_hadamard=True` cote gate/up ferait chercher l'alpha sur des
    poids non tournes et l'appliquer a des poids tournes, silencieusement."""
    ws = [w.detach().to(torch.float32) for w in weights]
    had_block = 0
    if use_hadamard:
        had_block = largest_pow2_divisor(ws[0].shape[1])
        if had_block >= 8:
            ws = [apply_hadamard_weight(w, had_block) for w in ws]
            if stats is not None:
                rotated = hadamard_transform(
                    stats.mean_abs.reshape(1, -1).to(torch.float32),
                    block=had_block).abs().reshape(-1)
                stats = ActStats(rotated.clamp(min=1e-6), None, stats.n_samples)
        else:
            had_block = 0
    scaler, _ = search_channel_scales_commun(
        ws, stats, fmt, group_size, n_grid, journal=journal,
        quantize_activation_nvfp4=quantize_activation_nvfp4)
    return scaler.scale, had_block


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
    forced_scale: Optional[torch.Tensor] = None,
    quantize_activation_nvfp4: bool = False,
    hadamard_block: Optional[int] = None,
    symmetric: bool = False,
) -> tuple[Any, ChannelScaler, dict]:
    """Chaîne complète par couche : tourner, mettre à l'échelle, quantifier.

    L'ordre compte. La rotation de Hadamard vient en premier parce qu'elle
    change les statistiques de canaux sur lesquelles opère la recherche AWQ :
    chercher avant de tourner optimiserait une échelle pour une distribution qui
    n'existe plus.

    ``hadamard_block`` impose la taille du bloc au lieu du plus grand diviseur
    puissance de deux (`largest_pow2_divisor`) — nécessaire quand l'expérience
    veut un bloc PLUS PETIT que celui-ci (poste7, `poste7-hadamard-16-09.md` :
    H_512 sur des tenseurs K=2048, dont le diviseur naturel serait 2048).
    """
    w = weight.detach().to(torch.float32)
    had_block = 0
    if use_hadamard:
        had_block = hadamard_block or largest_pow2_divisor(w.shape[1])
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
    if forced_scale is not None:
        # A7 (alpha commun gate/up) : l'echelle vient d'une recherche jointe
        # faite en amont dans le MEME espace tourne (voir
        # `alpha_commun_gate_up`) — la recherche par tenseur ci-dessous est
        # sautee, pas rejouee.
        scaler = ChannelScaler(forced_scale.to(w.device, torch.float32), had_block)
    elif use_awq:
        found, _ = search_channel_scales(
            w, stats, fmt, group_size, n_grid, journal=journal,
            quantize_activation_nvfp4=quantize_activation_nvfp4)
        scaler = ChannelScaler(found.scale, had_block)

    w_eff = w * scaler.scale.to(torch.float32).unsqueeze(0) \
        if scaler.scale is not None else w
    qt = formats.quantize(w_eff, fmt, group_size=group_size,
                          **({"table": table} if fmt == "q3n" else {}),
                          **({"symmetric": symmetric} if fmt == "int8" else {}))

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
        # poste7-awq-experts-peu-routes-portee-17-09 : ‖poids reconstruit‖ /
        # ‖poids source‖ -- distinct de `w_rel_err` (une erreur ELEMENT PAR
        # ELEMENT peut rester petite alors que la NORME globale s'effondre
        # si l'echelle est mal reglee sur seulement quelques canaux, motif
        # mesure sur Nemotron : 6 experts a 27-29 dB de w_snr_db propre mais
        # ratio_norme 0,29-0,69 apres reechelonnage).
        "ratio_norme": (deq.norm() / w.norm().clamp(min=1e-12)).item(),
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
