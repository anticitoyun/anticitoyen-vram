"""Pièce 156 : la combinaison de défauts QUALIFIÉE (129/130/134/142, TPB par forme 142 bis) est figée ici. Le mixte
(PROJ_MARLIN_DOUBLES non vide : exil puis refus, 129) ne doit pas revenir comme défaut ; un modèle à MoEBlock (non mesuré)
ne doit pas passer au Marlin par défaut ; le port absent donne un repli NOMMÉ, jamais un refus, sauf demande explicite."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

RACINE = Path(__file__).resolve().parents[1]
carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _defauts(**pose):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ACVRAM_PROJ_MARLIN", "ACVRAM_GEMV_MARLIN", "ACVRAM_PREFILL"))}
    env.update(CUDA_VISIBLE_DEVICES="", **pose)
    code = ("import acvram.kernels as k; print(k._PROJ_MARLIN, sorted(k._PROJ_MARLIN_DOUBLES), k._GEMV_MARLIN_V2, "
            "k._GEMV_MARLIN_TPB, k._GEMV_MARLIN_S, k._PROJ_MARLIN_PORTEE, k.prefill_regime(), k._tpb_marlin(65536), "
            "k._tpb_marlin(34816))")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(RACINE))
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr


def test_combinaison_de_defauts_qualifiee():
    assert _defauts() == "True [] True 0 0 denses bf16 2 1", "les défauts ne sont plus la configuration qualifiée"


def test_repli_explicite():
    assert _defauts(ACVRAM_PROJ_MARLIN="0").startswith("False "), "ACVRAM_PROJ_MARLIN=0 doit couper le Marlin"


def test_regime_declare_les_memes_defauts():
    from acvram.regime import VARIABLES
    d = {v.nom: v.defaut for v in VARIABLES}
    assert (d["PROJ_MARLIN"], d["PROJ_MARLIN_DOUBLES"], d["GEMV_MARLIN_V2"], d["GEMV_MARLIN_TPB"], d["GEMV_MARLIN_S"],
            d["PROJ_MARLIN_PORTEE"]) == ("1", "", "1", "0", "0", "denses")


class MoEBlockFactice(torch.nn.Module):                 # le nom suffit : la passe reconnaît les MoEBlock* par leur type
    pass


@carte
def test_moe_reste_au_naturel_par_defaut(monkeypatch):
    """Au défaut (portée denses), un modèle à MoEBlock n'est pas converti ET le Marlin paresseux est coupé ; casse si
    un MoE passe au Marlin par défaut."""
    from acvram import kernels
    from acvram.engine.layers import QuantLinear
    from acvram.kernels import marlin_port as MP
    from acvram.quant.nvfp4 import quantize_nvfp4
    if kernels.get_extension() is None or MP.charger(compiler=False) is None:
        pytest.skip("extension ou port Marlin absents")
    monkeypatch.setattr(kernels, "_PROJ_MARLIN", True)                # restauré après le test
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_PORTEE", os.environ.get("ACVRAM_PROJ_MARLIN_PORTEE", "denses"))
    if kernels._PROJ_MARLIN_PORTEE != "denses":
        pytest.skip("ACVRAM_PROJ_MARLIN_PORTEE posée")
    boite = torch.nn.Module()
    boite.proj = QuantLinear(quantize_nvfp4(torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16) * 0.02))
    boite.moe = MoEBlockFactice()
    bilan = kernels.preparer_disposition_marlin(boite)
    assert bilan.get("portee") == "denses:moe-exclu" and bilan["seuls"] == 0
    assert boite.proj.qweight.qweight is not None and not hasattr(boite.proj.qweight, "_marlin_dense")
    assert kernels._PROJ_MARLIN is True, "l'exclusion d'un MoE ne doit pas couper le Marlin du PROCESSUS"
    assert getattr(boite.proj.qweight, "_marlin_interdit", False), "le Marlin paresseux (2 ≤ M ≤ 16) resterait actif"
    assert kernels._marlin_dense(torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16), boite.proj.qweight) is None
