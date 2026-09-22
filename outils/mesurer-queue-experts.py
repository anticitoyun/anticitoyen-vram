#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anticitoyen
# SPDX-License-Identifier: Apache-2.0
"""Lourdeur de queue des poids d'experts, et arbitrage bloc 16 / bloc 32.

Pourquoi
--------

La spécification du format q3n (docs/FORMAT-3BITS.md) a été écrite sur des
distributions **synthétiques** : une gaussienne, et une gaussienne modulée par
une lognormale de paramètre σ = 0,5. Le tableau de sensibilité montre que le
résultat en dépend fortement — 13,6 dB à σ = 0,5, 9,5 à σ = 1,0 — et surtout
que l'écart entre bloc 16 et bloc 32 se **creuse** quand la queue s'alourdit :
0,85 dB en gaussien, 2,91 dB à σ = 1,5.

Le choix de la taille de bloc dépend donc d'un chiffre qu'aucun de nous n'a
mesuré : la lourdeur de queue des vrais poids d'experts.

Ce que cet outil mesure, et ce qu'il ne mesure pas
--------------------------------------------------

Deux choses, dont **une seule décide** :

1. Le SNR de q3n en bloc 16 et en bloc 32, sur les poids réels. C'est la
   réponse. Elle ne passe par aucun modèle de distribution.

2. σ estimé, par ``σ² = Var(log|w|) − π²/8`` — la variance de ``log|z|`` pour
   ``z`` gaussien vaut π²/8. Ce chiffre ne sert **qu'à situer** les poids réels
   dans le tableau de la spécification, et à dire si ce tableau était pertinent.
   Il suppose la famille lognormale, que rien ne garantit ; ne pas l'utiliser
   pour décider quand la mesure directe est disponible.

L'entrée est le modèle déjà converti en NVFP4, seule forme des experts présente
sur ce disque. **Il n'y a pas de source bf16**, donc :

- le SNR rendu ici est celui de q3n contre du NVFP4 déquantifié, pas contre le
  modèle d'origine — c'est bien la question posée, puisque la conversion q3n
  part de ce fichier, mais ce n'est pas la qualité absolue du format ;
- l'estimation de σ hérite de l'écrasement de queue *intra-bloc* du NVFP4.
  Elle reste utilisable parce que σ décrit surtout la dispersion des amplitudes
  *entre* blocs, que le NVFP4 conserve dans ses échelles FP8 ; mais elle est un
  minorant, jamais un majorant.

Usage
-----

    outils/mesurer-queue-experts.py [dossier] [--experts N] [--fichier i]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import sys

import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4      # noqa: E402
from acvram.quant.q3n import quantize_q3n, dequantize_q3n          # noqa: E402

VAR_LOG_ABS_NORMALE = math.pi ** 2 / 8      # Var(log|z|), z ~ N(0,1)


def snr_db(reference: torch.Tensor, obtenu: torch.Tensor) -> float:
    r = reference.to(torch.float64)
    e = obtenu.to(torch.float64) - r
    d = e.pow(2).sum()
    return float("inf") if d == 0 else 10.0 * math.log10(float(r.pow(2).sum() / d))


def sigma_inter_blocs(w: torch.Tensor, block: int = 32) -> float:
    """σ de la lognormale, estimé sur les **amplitudes de bloc**.

    Une première version estimait σ sur les poids individuels, par
    ``Var(log|w|) − π²/8``. Elle rendait zéro partout, en contradiction directe
    avec le SNR mesuré, qui correspondait à σ ≈ 0,5. Elle avait tort : la
    source est déjà quantifiée en NVFP4, ses poids ne prennent que huit
    amplitudes distinctes par bloc et un cinquième vaut exactement zéro. La
    variance intra-bloc mesurait la grille du format source, pas la queue.

    L'amplitude maximale par bloc, elle, est portée par l'échelle FP8 du NVFP4
    et survit à la quantification. C'est aussi exactement ce que q3n voit :
    une échelle par bloc. Le témoin gaussien est calculé sur place plutôt que
    tabulé, parce qu'il dépend de la taille de bloc.
    """
    b = w[:, : w.shape[1] // block * block].reshape(w.shape[0], -1, block)
    a = b.abs().amax(-1).reshape(-1)
    a = a[a > 0]
    if a.numel() < 256:
        return float("nan")
    v = torch.log(a.to(torch.float64)).var().item()
    g = torch.randn(256, block * 64, generator=torch.Generator().manual_seed(0))
    ga = g.reshape(256, -1, block).abs().amax(-1).reshape(-1)
    v0 = torch.log(ga.to(torch.float64)).var().item()
    return math.sqrt(max(0.0, v - v0))


def entete(chemin: str) -> dict:
    with open(chemin, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def charger_experts(chemin: str, combien: int):
    """Rend des couples (nom, poids fp32) d'experts, déquantifiés du NVFP4."""
    from safetensors.torch import load_file
    h = entete(chemin)
    prefixes = sorted({k[: -len(".qweight")] for k in h
                       if k.endswith(".qweight") and "expert" in k})[:combien]
    if not prefixes:
        return
    sd = load_file(chemin)
    for p in prefixes:
        q = sd[f"{p}.qweight"]
        # NVFP4 : deux poids par octet. La forme n'est pas dans le fichier ;
        # elle se relit sur qweight, et le remplissage éventuel est repris tel
        # quel — un poids de remplissage vaut zéro et sera écarté du calcul.
        forme = (q.shape[0], q.shape[-1] * 2)
        t = NVFP4Tensor.from_state_dict(sd, p + ".", forme)
        w = dequantize_nvfp4(t, torch.float32)
        if w.shape[1] % 32:
            w = w[:, : w.shape[1] // 32 * 32]
        yield p, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dossier", nargs="?",
                    default=_RACINE + "/qwen3-coder-next-80b")
    ap.add_argument("--experts", type=int, default=200)
    ap.add_argument("--fichier", type=int, default=3,
                    help="indice du safetensors à échantillonner")
    a = ap.parse_args()

    fichiers = sorted(glob.glob(os.path.join(a.dossier, "*.safetensors")))
    if not fichiers:
        print(f"aucun safetensors dans {a.dossier}", file=sys.stderr)
        return 2
    chemin = fichiers[min(a.fichier, len(fichiers) - 1)]
    print(f"Source : {chemin}")
    print(f"         (NVFP4 déjà quantifié — il n'existe pas de bf16 sur ce disque)")

    sigmas, s16, s32, noms = [], [], [], []
    for nom, w in charger_experts(chemin, a.experts):
        sigmas.append(sigma_inter_blocs(w))
        s16.append(snr_db(w, dequantize_q3n(quantize_q3n(w, block=16), torch.float32)))
        s32.append(snr_db(w, dequantize_q3n(quantize_q3n(w, block=32), torch.float32)))
        noms.append(nom)

    if not noms:
        print("aucun expert trouvé", file=sys.stderr)
        return 2

    def med(v):
        t = torch.tensor([x for x in v if math.isfinite(x)], dtype=torch.float64)
        return t.median().item(), t.min().item(), t.max().item()

    ms, mns, mxs = med(sigmas)
    m16, n16, x16 = med(s16)
    m32, n32, x32 = med(s32)
    ecarts = torch.tensor([b - a_ for a_, b in zip(s32, s16)], dtype=torch.float64)

    print(f"\n{len(noms)} tenseurs d'experts, {noms[0].split('.experts.')[0]}\n")
    print(f"  σ inter-blocs   médiane {ms:.3f}   min {mns:.3f}   max {mxs:.3f}")
    print(f"  SNR q3n bloc 32 médiane {m32:.2f} dB   min {n32:.2f}   max {x32:.2f}")
    print(f"  SNR q3n bloc 16 médiane {m16:.2f} dB   min {n16:.2f}   max {x16:.2f}")
    print(f"  écart 16 − 32   médiane {ecarts.median().item():.2f} dB   "
          f"min {ecarts.min().item():.2f}   max {ecarts.max().item():.2f}")

    zeros = 0.0
    for _, w in charger_experts(chemin, 8):
        zeros = max(zeros, float((w == 0).double().mean()))
    print(f"\n  Poids exactement nuls dans la source : {zeros:.1%}")
    print("\n  Lecture : l'écart mesuré est ce qui s'achète avec 0,25 bit de plus")
    print("  (3,50 contre 3,25 bits par poids, soit 7 % de mémoire sur les experts).")
    print("  Le σ n'est là que pour situer ce résultat dans le tableau de la")
    print("  spécification ; c'est l'écart mesuré qui décide, pas lui.")
    print("\n  MISE EN GARDE. Ce SNR compare q3n à du NVFP4 déquantifié, donc")
    print("  deux grilles de quantification l'une contre l'autre — pas q3n aux")
    print("  poids d'origine. Un cinquième des valeurs vaut exactement zéro et")
    print("  chaque bloc ne porte que huit amplitudes distinctes : ces deux")
    print("  propriétés viennent du format source, pas du modèle. Ne rien")
    print("  conclure d'ici sur la TABLE de q3n ; l'écart entre tailles de bloc,")
    print("  lui, reste lisible parce que les deux subissent la même source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
