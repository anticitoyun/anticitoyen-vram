"""Pièce 65 (23/09) : le chemin tensor-core est SERVI PAR DÉFAUT aux godets ≥ 8 (MOE_TENSOR_MIN_T, mesuré), avec la glue fusionnée
(reproductible) — jamais la glue A4 (aligneur vLLM par atomiques). À sec : défauts nus, règle statique de forme,
contrôle du parc. Carte : même entrée vingt fois → sortie identique au bit sur le chemin servi (casse si l on
rebranche l aligneur A4 par défaut : `diag-aubit` du 23/09 le montre non reproductible dès deux appels)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _racine_parc() -> str:
    """Racine du parc lue comme les outils (ACVRAM_MODELES, ~/.config/acvram/modeles), jamais en dur."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outils"))
    from racine_modeles import racine_modeles
    return racine_modeles()

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_defauts_nus_a_sec():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "from acvram.engine import moe; print(moe._MOE_TENSOR, moe._MOE_TENSOR_FUSION, moe._MOE_TENSOR_MIN_T)"],
                         env=env, capture_output=True, text=True, timeout=180, cwd=RACINE)
    assert out.stdout.strip().splitlines()[-1] == "True True 8", out.stdout + out.stderr


def test_regle_statique_de_forme():
    from acvram.engine.moe import forme_tensor_refus, TENSOR_E_MAX
    assert forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, True) == ""              # Coder officiel
    assert forme_tensor_refus({"nvfp4"}, 2688, 1856, 2688, 128, True) == ""             # Nemotron (multiples de 64)
    assert "AWQ" in forme_tensor_refus({"nvfp4"}, 2048, 1536, 2048, 64, False)          # GLM calibA
    assert "NVFP4" in forme_tensor_refus({"bf16"}, 2048, 1536, 2048, 64, True)
    assert "NVFP4" in forme_tensor_refus({"int8", "nvfp4"}, 2048, 1536, 2048, 64, True)
    assert "64" in forme_tensor_refus({"nvfp4"}, 2048, 1000, 2048, 64, True)
    assert "aligneur" in forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, TENSOR_E_MAX + 1, True)
    assert "Hadamard" in forme_tensor_refus({"nvfp4"}, 2048, 1536, 2048, 64, True, 512)


def test_controle_du_parc_a_sec():
    parc = Path(os.environ.get("ACVRAM_PARC") or _racine_parc())
    if not parc.is_dir():
        pytest.skip("parc absent")
    out = subprocess.run([sys.executable, str(RACINE / "outils" / "controle-moe-tensor-alias.py"), "--parc", str(parc)],
                         capture_output=True, text=True, timeout=300, cwd=RACINE)
    assert out.returncode == 0, out.stderr
    lignes = out.stdout.splitlines()
    assert any(l.startswith("ACCEPTÉ") and "Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c" in l for l in lignes), out.stdout
    assert any("calibA" in l and "AWQ" in l for l in lignes), out.stdout


def test_sans_port_compile_le_defaut_replie_et_ne_leve_pas(monkeypatch):
    """chef 23/09 : à sec (CI sans carte) le chemin tensor par défaut levait « port non compilé » ; un défaut ne
    lève jamais là où le GEMV marchait — repli statique nommé `port Marlin non compilé` (ligne de régime), et le
    forward reste sur le GEMV (test_c10_marlin_distinct_glm[12] le joue de bout en bout, à sec, b=12 ≥ MIN_T)."""
    from acvram.engine.moe import MoEBlock
    from acvram.kernels import marlin_port as MP
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    monkeypatch.setattr(MoEBlock, "_construire_marlin", lambda self, p, a, h: {n: ("w", "s", "g", p[n][4], p[n][5]) for n in p})
    monkeypatch.setattr(MP, "charger", lambda *a, **k: None)
    bloc = _bloc_moe_jouet(4, 128, 64, 2, dev="cpu")
    assert bloc._try_build_stacks()
    assert "port Marlin non compilé" in bloc._tensor_refus, bloc._tensor_refus
    del bloc.__dict__["_tensor_refus"]                       # piles posées sans _try_build_stacks : calcul paresseux
    assert "port Marlin non compilé" in bloc._raison_tensor()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
def test_chemin_servi_reproductible_au_bit():
    from test_moe_tensor_decodage import _charger, _piles, E, K, I
    from test_moe_tensor_glue_fusee import _eid
    from acvram.engine import moe
    from acvram.engine.moe import gemm_experts_tensor
    dev = torch.device("cuda", 0)
    _, MP, ext, banc = _charger(); marlin = _piles(MP, banc, dev); k, t = 8, 16
    eid = _eid(t, k, E, 650, True).to(dev)                    # expert 7 : 16 paires → deux blocs (le cas non reproductible d A4)
    x = torch.randn(t, K, dtype=torch.bfloat16, device=dev, generator=torch.Generator(dev).manual_seed(65)); x[-1] = 0
    ws = MP.espace_travail(dev, 4); uns = torch.ones(t * k, 1, dtype=torch.float32, device=dev)
    assert moe._MOE_TENSOR_FUSION, "la glue A4 ne doit jamais être le défaut"
    ref = gemm_experts_tensor(MP, ext, x, eid, marlin, k, I, I, 0, ws, uns, {}, {}, fusion=moe._MOE_TENSOR_FUSION)
    for _ in range(20):
        d = gemm_experts_tensor(MP, ext, x, eid, marlin, k, I, I, 0, ws, uns, {}, {}, fusion=moe._MOE_TENSOR_FUSION)
        assert torch.equal(ref, d), "le chemin servi n est pas reproductible au bit"
