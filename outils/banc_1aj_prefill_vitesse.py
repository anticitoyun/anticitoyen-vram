#!/usr/bin/env python3
"""Bead 1aj, volet A (vitesse seule, qualité ignorée) -- prédiction scellée
dans acvram-memoire/revue/1aj-prefill-w4a4-14-09.md, à lire avant ce script.

Sur ce modèle, q/k/v/o sont TOUS int8 (vérifié 14/09 soir, pas nvfp4 --
plancher délibéré de la conversion pour l'attention). Décision de Jérôme :
tester quand même, en quantifiant CES poids int8 vers nvfp4 EN MÉMOIRE
(pas le fichier converti), pour ce banc « vitesse seule » -- lm_head
exclu (reste int8 dans tous les cas, 151 936 classes).

Chemin ACTUEL : `kernels.matmul` sur le poids int8 réel, inchangé.
Chemin NOUVEAU : poids requantifié nvfp4 en mémoire + `nvfp4_quant_act`
(activation) + `nvfp4_gemm_grouped_mma` E=1 (noyau écrit main, PAS
torch._scaled_mm, cf. la note du 10/09 dans kernels/__init__.py).

    outils/carte.sh .venv/bin/python outils/banc_1aj_prefill_vitesse.py
"""
import os
import sys
import time
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)

import torch

from acvram import kernels
from acvram.engine.loader import load_model
from acvram.engine.model import MoEBlock
from acvram.quant.nvfp4 import quantize_nvfp4
# Pas d'Engine construit ici (appel direct aux noyaux sur les poids charges) :
# exiger_regime_nominal ne s'applique pas, rien a regarder.

MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
G = 2048          # pp2048, comme la mesure de départ (laurine:4346)
REP = 5
BT, ETAGES, KS = 16, 4, 64


def _projections(model):
    """(nom, QuantLinear) pour q/k/v/o de chaque couche -- vérifié sur ce
    modèle (14/09 soir) : TOUS int8 (pas nvfp4), plancher délibéré de la
    conversion pour l'attention. `lm_head` exclu sur consigne de Jérôme
    (151 936 classes, reste int8 dans tous les cas)."""
    for i, layer in enumerate(model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for nom in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, nom, None)
            if lin is None:
                continue
            w = lin.qweight
            if getattr(w, "format", None) == "int8":
                yield f"L{i}.{nom}", lin


def _chrono(fn, rep=REP):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep


def main():
    loaded = load_model(MODEL, dtype=torch.bfloat16, device_override="cuda:0")
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemm_grouped_mma"):
        print("ÉCHEC : extension CUDA indisponible ou nvfp4_gemm_grouped_mma absent")
        sys.exit(1)

    dev = torch.device("cuda:0")
    cnt = torch.tensor([G], dtype=torch.int64, device=dev)
    tiles = MoEBlock._tuiles(cnt, bt=BT)

    n = 0
    t_actuel = t_nouveau = 0.0
    ecarts = []
    for nom, lin in _projections(loaded.model):
        w = lin.qweight                        # int8 reel, chemin actuel inchange
        w_dense = kernels.int8_dequant(w, torch.bfloat16)     # [out, in]
        w_nv = quantize_nvfp4(w_dense)          # requantifie nvfp4 EN MEMOIRE, pour ce banc seul

        x = torch.randn(G, lin.in_features, dtype=torch.bfloat16, device=dev)
        x = x * 0.02  # échelle d'activation réaliste (pas de saturation E2M1)

        dt_a = _chrono(lambda: kernels.matmul(x, w))

        table_qw = torch.tensor([w_nv.qweight.data_ptr()], dtype=torch.int64, device=dev)
        table_bs = torch.tensor([w_nv.block_scale.data_ptr()], dtype=torch.int64, device=dev)
        gscales = torch.tensor([w_nv.global_scale_float()], dtype=torch.float32, device=dev)

        def nouveau():
            xq, xsf, gr = ext.nvfp4_quant_act(x)
            return ext.nvfp4_gemm_grouped_mma(
                table_qw, table_bs, gscales, xq, xsf,
                tiles[0], tiles[1], tiles[2],
                lin.out_features, lin.in_features, BT, ETAGES, KS, grow=gr)

        dt_n = _chrono(nouveau)

        y_a = kernels.matmul(x, w)
        xq, xsf, gr = ext.nvfp4_quant_act(x)
        y_n = ext.nvfp4_gemm_grouped_mma(table_qw, table_bs, gscales, xq, xsf,
                                         tiles[0], tiles[1], tiles[2],
                                         lin.out_features, lin.in_features, BT, ETAGES, KS, grow=gr)
        y_n = y_n[:, :lin.out_features]
        ecart = (y_a - y_n).abs().mean().item() / (y_a.abs().mean().item() + 1e-9)
        ecarts.append(ecart)

        t_actuel += dt_a
        t_nouveau += dt_n
        n += 1
        print(f"  {nom:16s} actuel={dt_a*1000:7.3f} ms  nouveau={dt_n*1000:7.3f} ms  "
              f"ecart_relatif={ecart:.4f}", flush=True)

    print(f"\n{n} projections mesurées (q/k/v/o, int8 -> nvfp4 en mémoire ; lm_head exclu)")
    print(f"SOMME actuel  : {t_actuel*1000:.2f} ms")
    print(f"SOMME nouveau : {t_nouveau*1000:.2f} ms")
    print(f"écart relatif moyen (nouveau vs actuel, sanité seulement, pas la mesure de qualité du volet B) : "
          f"{sum(ecarts)/len(ecarts):.4f}")
    print(f"\nDépart (laurine:4346) : 30 ms")
    print(f"Seuil de preuve : <= 8 ms")
    print(f"Seuil de réfutation : >= 20 ms")
    if t_nouveau * 1000 <= 8:
        verdict = "PREUVE"
    elif t_nouveau * 1000 >= 20:
        verdict = "RÉFUTATION"
    else:
        verdict = "NI L'UN NI L'AUTRE"
    print(f"VERDICT : {verdict}")


if __name__ == "__main__":
    main()
