#!/usr/bin/env python3
"""Banc de prefill MoE : GEMM groupée vs déquant+grouped_mm.

Mesure le débit en jetons/s des deux chemins de prefill MoE pour différentes
longueurs de séquence. Le croisement mesuré se situe entre 64 et 128 jetons
par expert (cf. commentaire model.py:733).

Référence : 6 420 j/s sur Qwen3-Coder-30B (NVFP4, RTX 5090, prefill 512 j).

Prérequis
---------
- SSD des modèles relié (ACVRAM_MODELES ou /media/.../2TO_2023_980PRO1)
- GPU libre (vérifié par outils/carte.sh)
- Extension CUDA compilée avec nvfp4_gemm_grouped

Usage
-----
    source outils/carte.sh prendre 0
    ACVRAM_MODELES=/chemin/vers/modeles python3 outils/banc_prefill_moe.py \\
        --modele Qwen3-Coder-30B-srcHF-nvfp4 \\
        --longueurs 64,128,256,512,1024,2048 \\
        --repetitions 5
    source outils/carte.sh rendre 0

Protocole
---------
1. Charger le modèle (NVFP4, bf16, cuda:0).
2. Pour chaque longueur L :
   a. Construire un lot de prefill de L jetons aléatoires (1 séquence).
   b. Exécuter 2 passes de chauffe (non mesurées).
   c. Mesurer ``repetitions`` passes, synchroniser CUDA entre chaque.
   d. Calculer le médian en jetons/s.
3. Rapporter : longueur, j/s GEMM groupée, j/s grouped_mm, ratio.

Le banc force le chemin via ACVRAM_PREFILL_DEQUANT=1 pour le second.
"""
import argparse
import os
import sys
import time

import torch


def _charger(chemin_modele: str):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    loaded = load_model(chemin_modele, dtype=torch.bfloat16,
                        device_override="cuda:0")
    e = Engine(loaded, None, max_batch_size=1, max_model_len=4096,
               enable_cuda_graphs=False)
    # `enable_cuda_graphs=False` est volontaire ici (prefill n'en profite
    # pas) -- `exiger_regime_nominal` ne le compte pas comme une
    # dégradation, seulement l'exil/le partage sur 2 cartes/les piles.
    _prefill_un(e, 8)                     # un pas reel : resout piles_ok
    from regime import exiger_regime_nominal
    exiger_regime_nominal(e, autoriser_piles_inconnues=False)
    return e, loaded


def _prefill_un(e, longueur: int):
    """Un pas de prefill avec des jetons aléatoires, renvoie la durée."""
    from acvram.engine.sampler import SamplingParams
    prompt = list(range(1, longueur + 1))
    t0 = time.perf_counter()
    seq = e.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=1))
    e.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    for s in list(e.running):
        s.finished = True
    e.running.clear()
    return dt


def _mesurer(e, longueur: int, repetitions: int, chauffe: int = 2):
    """Médian et σ de ``repetitions`` mesures après ``chauffe`` passes."""
    import math
    for _ in range(chauffe):
        _prefill_un(e, longueur)
    durees = []
    for _ in range(repetitions):
        dt = _prefill_un(e, longueur)
        durees.append(dt)
    durees.sort()
    median = durees[len(durees) // 2]
    jps = [longueur / d for d in durees]
    moy = sum(jps) / len(jps)
    sigma = math.sqrt(sum((x - moy) ** 2 for x in jps) / len(jps))
    return longueur / median, sigma


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modele", required=True,
                   help="nom du dossier converti sous ACVRAM_MODELES")
    p.add_argument("--longueurs", default="64,128,256,512,1024,2048",
                   help="longueurs de séquence, séparées par des virgules")
    p.add_argument("--repetitions", type=int, default=5)
    args = p.parse_args()

    import importlib.util, pathlib
    _spec = importlib.util.spec_from_file_location(
        "racine_modeles",
        pathlib.Path(__file__).with_name("racine_modeles.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    racine = _mod.MODELES
    chemin = os.path.join(racine, args.modele)
    if not os.path.isdir(chemin):
        print(f"ERREUR : {chemin} introuvable (ACVRAM_MODELES={racine})",
              file=sys.stderr)
        sys.exit(1)

    longueurs = [int(x) for x in args.longueurs.split(",")]

    print(f"modèle : {args.modele}")
    print(f"longueurs : {longueurs}")
    print(f"répétitions : {args.repetitions}")
    print()

    e, loaded = _charger(chemin)

    print(f"{'L':>6}  {'GEMM grp':>10} {'±σ':>6}  {'grouped_mm':>10} {'±σ':>6}  {'ratio':>6}")
    print("-" * 60)

    for L in longueurs:
        os.environ.pop("ACVRAM_PREFILL_DEQUANT", None)
        jps_gemm, sg = _mesurer(e, L, args.repetitions)

        os.environ["ACVRAM_PREFILL_DEQUANT"] = "1"
        jps_gmm, sm = _mesurer(e, L, args.repetitions)
        os.environ.pop("ACVRAM_PREFILL_DEQUANT", None)

        ratio = jps_gemm / max(jps_gmm, 1e-9)
        print(f"{L:>6}  {jps_gemm:>10.0f} {sg:>5.0f}σ  {jps_gmm:>10.0f} {sm:>5.0f}σ  {ratio:>6.2f}x")

    print()
    print("référence : 6 420 j/s (Qwen3-Coder-30B, NVFP4, RTX 5090, 512 j)")


if __name__ == "__main__":
    main()
