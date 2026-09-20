#!/usr/bin/env python3
"""Le seuil GEMV change-t-il le RESULTAT a douze lignes ?

**Une perplexite ne peut pas repondre, et il faut le dire avant de la lancer.**
`evaluate.perplexity` avance par fenetres de 512 jetons (`window=512,
stride=256`) : chaque passage avant porte 512 lignes, donc au-dessus de 8 comme
de 32. Les deux seuils y prennent le MEME chemin et rendraient le meme chiffre —
un resultat qui ressemblerait a « aucune degradation » alors qu'il ne mesure
rien. Le seuil ne decide qu'entre 9 et 32 lignes.

**La reference est un TIERS**, jamais l'un des deux bras : `ACVRAM_PREFILL=bf16`
dequantifie le poids stocke et fait une `linear` bf16 ordinaire. Elle ne partage
ni le noyau du GEMV, ni la quantification d'activation du chemin W4A8. Deux
chemins peuvent etre proches l'un de l'autre et faux tous les deux.

Un passage avant du modele entier sur douze lignes, logits compares. Le seuil
est lu a l'import : **une valeur par processus**.
"""
import argparse
import os
import sys
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

A = _RACINE


def _snr(ref: torch.Tensor, y: torch.Tensor) -> float:
    ref, y = ref.float(), y.float()
    bruit = ((ref - y) ** 2).sum()
    return float("inf") if bruit == 0 else float(
        10 * torch.log10((ref ** 2).sum() / bruit))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modele", default="Qwen3-4B-srcgguf-nvfp4")
    ap.add_argument("--lignes", type=int, default=12)
    ap.add_argument("--sortie")
    ap.add_argument("--comparer", nargs="+", metavar="FICHIER",
                    help="le PREMIER est la reference")
    ns = ap.parse_args(argv[1:])

    if ns.comparer:
        ref = torch.load(ns.comparer[0])
        print(f"reference : seuil {ref['seuil']}, prefill {ref['prefill']}")
        print("bras\tseuil\tprefill\tSNR vs reference\tmemes argmax")
        for f in ns.comparer[1:]:
            b = torch.load(f)
            ega = int((ref["logits"].argmax(-1) == b["logits"].argmax(-1)).sum())
            n = ref["logits"].shape[0]
            print(f"{os.path.basename(f)}\t{b['seuil']}\t{b['prefill']}\t"
                  f"{_snr(ref['logits'], b['logits']):.2f} dB\t{ega}/{n}")
        return 0

    if torch.cuda.device_count() != 1:
        raise SystemExit("CUDA_VISIBLE_DEVICES=0 obligatoire")

    from acvram.kernels import _NVFP4_GEMV_MAX
    from acvram.engine.loader import load_model

    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    charge = load_model(os.path.join(A, ns.modele), max_model_len=4096)
    m = charge.model
    # Le lot se construit par le MOTEUR : tables de blocs, slots et cache KV
    # sont son travail, et les refaire a la main mesurerait mon montage plutot
    # que le sien. On se contente d intercepter les logits qu il obtient.
    pris = []
    _avant = m.forward

    def _capter(batch):
        y = _avant(batch)
        if not batch.is_prefill and batch.batch_size == ns.lignes and not pris:
            pris.append(y.detach().float().cpu())
        return y

    m.forward = _capter
    # Graphes coupes : une capture rejoue un noyau FIGE au moment ou elle a eu
    # lieu, donc les deux bras rejoueraient peut-etre le meme. On veut le
    # chemin choisi A CHAQUE PAS.
    moteur = Engine(charge, None, max_batch_size=ns.lignes, max_model_len=4096,
                    enable_cuda_graphs=False)
    params = SamplingParams(temperature=0.0, max_tokens=4)
    for i in range(ns.lignes):
        ids = [(i * 7919 + j * 31 + 11) % 30000 + 1 for j in range(64)]
        moteur.add_request(ids, params, request_id=f"r{i}")
    for _ in range(8):
        moteur.step()
        if pris:
            break
    if not pris:
        raise SystemExit("aucun pas de decodage a douze lignes : rien a comparer")
    logits = pris[0]
    obj = {"logits": logits, "seuil": _NVFP4_GEMV_MAX,
           "prefill": os.environ.get("ACVRAM_PREFILL", "bf16")}
    if ns.sortie:
        torch.save(obj, ns.sortie)
    print(f"{_NVFP4_GEMV_MAX}\t{obj['prefill']}\t{tuple(logits.shape)}\t"
          f"{ns.sortie or '(non ecrit)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
