"""Pièce 129 (24/09) : disposition Marlin MIXTE, opt-in ACVRAM_PROJ_MARLIN=1 (`kernels.preparer_disposition_marlin`).

À sec : le défaut reste hors du chemin (sous-processus), la réserve du chargeur vaut 0 sans la variable et compte les
rôles doublés avec, la preuve mémoire refuse NOMMÉMENT une capacité KV trop petite.
Carte : un poids en Marlin SEUL rend ce que rend le chemin naturel (≤ 2⁻⁷·max par ligne, reproductible au bit) à
M = 1, 8, 300 — y compris K > 11 264 (deux moitiés de K) et une pile q/k/v à échelle par segment dont les SOURCES
(vues, servies au préfill) passent par la pile ; un rôle doublé garde sa disposition naturelle, et M = 1 y reste au
bit du chemin d'avant."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
RACINE = Path(__file__).resolve().parents[1]


def test_repli_nomme_hors_du_chemin_sous_processus():
    """Pièce 156 : le Marlin est le défaut (test_defaut_marlin_156) ; ACVRAM_PROJ_MARLIN=0 le coupe."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_PROJ_MARLIN")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ACVRAM_PROJ_MARLIN"] = "0"
    r = subprocess.run([sys.executable, "-c", "import acvram.kernels as k; print(k._PROJ_MARLIN)"], env=env,
                       capture_output=True, text=True, cwd=str(RACINE))
    assert r.stdout.strip().splitlines()[-1] == "False", r.stdout + r.stderr


def test_reserve_du_chargeur(monkeypatch):
    from acvram import kernels
    from acvram.engine import loader
    man = {"tensors": {
        "model.layers.0.mlp.gate_proj.weight": {"format": "nvfp4", "shape": [4096, 1024]},
        "model.layers.0.mlp.down_proj.weight": {"format": "nvfp4", "shape": [1024, 4096]},
        "model.layers.0.linear_attn.out.weight": {"format": "nvfp4", "shape": [1024, 2048]},
        "model.layers.0.self_attn.q_proj.weight": {"format": "nvfp4", "shape": [8192, 1024]},
        "mtp.layers.0.mlp.down_proj.weight": {"format": "nvfp4", "shape": [1024, 4096]},
        "model.norm.weight": {"format": "bf16", "shape": [1024]}}}
    o = lambda n, k: n * k // 2 + n * k // 16                                    # noqa: E731
    monkeypatch.setattr(kernels, "_PROJ_MARLIN", False)
    assert loader._octets_marlin(man) == 0
    monkeypatch.setattr(kernels, "_PROJ_MARLIN", True)
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_DOUBLES", frozenset({"mlp.gate_up", "mlp.down", "gdn.out"}))   # le mixte (129)
    # pièce 146 : les doubles seuls (gate‖up, down, gdn.out au défaut des doubles) — plus de « plus gros » transitoire
    assert loader._octets_marlin(man) == o(4096, 1024) + o(1024, 4096) + o(1024, 2048)
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_DOUBLES", frozenset())
    assert loader._octets_marlin(man) == 0, "disposition unique : la réserve ne coûte aucun KV"


def test_preuve_memoire_refus_nomme(monkeypatch):
    from acvram.engine import loader
    from acvram.memory.kvcache import BLOCK_SIZE
    monkeypatch.delenv("ACVRAM_PROJ_MARLIN_CAPACITE", raising=False)
    b = {"doubles": 0, "octets_doubles": 0}
    with pytest.raises(RuntimeError, match="ACVRAM_PROJ_MARLIN=1 : capacité KV"):
        loader._verifier_memoire_marlin([], {"cuda:0": 10}, b)                        # défaut 8 × 8 192
    loader._verifier_memoire_marlin([], {"cuda:0": 10 ** 6}, b)
    loader._verifier_memoire_marlin([], {"cuda:0": 1024 // BLOCK_SIZE}, b, demande=1024)   # ce que le chargement demande
    with pytest.raises(RuntimeError, match="capacité KV"):
        loader._verifier_memoire_marlin([], {"cuda:0": 1024 // BLOCK_SIZE}, b, demande=8 * 8192)
    with pytest.raises(RuntimeError, match="en flux depuis l'hôte"):                  # exil = refus (ABBA b=1, 24/09)
        loader._verifier_memoire_marlin([], {"cuda:0": 10 ** 6}, {"doubles": 1, "octets_doubles": 1, "en_flux": 3})


# ---------------------------------------------------------------- carte

def _pret():
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_marlin") or MP.charger(compiler=False) is None:
        pytest.skip("extension ou port Marlin absents")
    return kernels


def _lin(n, k, graine):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4
    g = torch.Generator(device="cuda").manual_seed(graine)
    return QuantLinear(quantize_nvfp4(torch.randn(n, k, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02))


def _hors(y, ref):
    ref = ref.float()
    return int(((y.float() - ref).abs() > 2 ** -7 * ref.abs().amax(-1, keepdim=True)).sum())


@carte
@pytest.mark.parametrize("n,k", [(2048, 1024), (2048, 17408)])
def test_marlin_seul_egal_naturel(n, k):
    kernels = _pret()
    boite = torch.nn.Module(); boite.proj = _lin(n, k, 7)
    xs = {m: torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for m in (1, 8, 300)}
    avant = {m: boite.proj(x) for m, x in xs.items()}
    bilan = kernels.preparer_disposition_marlin(boite)
    assert bilan["seuls"] == 1 and boite.proj.qweight.qweight is None            # naturelle libérée
    for m, x in xs.items():
        y = boite.proj(x)
        assert y.shape == avant[m].shape
        assert _hors(y, avant[m]) == 0, f"M={m} : hors 2⁻⁷·max"
        if m > 32:                                     # pièce 134 : préfill = arithmétique du défaut, au bit
            assert torch.equal(y, avant[m]), f"M={m} : préfill pas au bit du défaut"
        assert torch.equal(y, boite.proj(x)), f"M={m} : non reproductible"


@carte
def test_pile_a_echelle_par_segment_et_ses_vues():
    kernels = _pret()
    from acvram.engine.layers import stack_nvfp4_linears
    q, kk, v = _lin(4096, 1024, 1), _lin(512, 1024, 2), _lin(512, 1024, 3)
    pile = stack_nvfp4_linears([q, kk, v])
    assert pile is not None and pile.qweight.global_scale_rows is not None
    boite = torch.nn.Module(); boite.qkv_proj, boite.q_proj, boite.k_proj, boite.v_proj = pile, q, kk, v
    xs = {m: torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16) for m in (1, 8, 300)}
    avant = {(nom, m): getattr(boite, nom)(x) for nom in ("qkv_proj", "q_proj", "k_proj", "v_proj") for m, x in xs.items()}
    kernels.preparer_disposition_marlin(boite)
    assert pile.qweight._marlin_unique and kk.qweight._marlin_parent[1:] == (4096, 512)
    kernels.CHEMINS_NVFP4.pop("marlin_depaquete_prefill_vue", None)
    boite.k_proj(xs[300])
    # pièce 134 : la vue d'une pile se déquantifie SEULE au préfill (coût du défaut), pas la pile entière puis découpe
    assert kernels.CHEMINS_NVFP4.get("marlin_depaquete_prefill_vue", 0) == 1, "vue servie par la pile entière"
    for (nom, m), ref in avant.items():
        y = getattr(boite, nom)(xs[m])
        assert _hors(y, ref) == 0, f"{nom} M={m} : hors 2⁻⁷·max"
        if m > 32:                                     # pièce 134 : pile (échelle par colonne) et vues, au bit
            assert torch.equal(y, ref), f"{nom} M={m} : préfill pas au bit du défaut"


@carte
def test_role_double_garde_la_naturelle_au_bit(monkeypatch):
    kernels = _pret()
    from acvram.engine.attention import MLP
    mlp = MLP(_lin(4096, 2048, 4), _lin(4096, 2048, 5), _lin(2048, 4096, 6))     # N ≥ 2 048 partout
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_DOUBLES", frozenset({"mlp.gate_up", "mlp.down", "gdn.out"}))   # 156 : défaut vide
    mlp.fuse()
    x1 = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    avant = mlp(x1)
    bilan = kernels.preparer_disposition_marlin(mlp)
    assert bilan["doubles"] == 2 and bilan["seuls"] == 0                         # gate‖up et down doublés
    assert mlp.gate_up.qweight.qweight is not None and hasattr(mlp.gate_up.qweight, "_marlin_dense")
    assert torch.equal(mlp(x1), avant), "M = 1 d'un rôle doublé : doit rester au bit du GEMV naturel"


class MoEBlockFactice(torch.nn.Module):                 # le nom suffit : la passe reconnaît les MoEBlock* par leur type
    pass


@carte
def test_garde_modele_dense_exclut_les_moe(monkeypatch):
    """Pièce 142 : sous ACVRAM_PROJ_MARLIN_PORTEE=denses, un modèle qui contient un MoEBlock n'est pas converti — casse
    si un MoE passe la garde ; en global (défaut), le même poids est converti."""
    kernels = _pret()
    boite = torch.nn.Module(); boite.proj = _lin(2048, 1024, 11); boite.moe = MoEBlockFactice()
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_PORTEE", "denses")
    bilan = kernels.preparer_disposition_marlin(boite)
    assert bilan.get("portee") == "denses:moe-exclu" and bilan["seuls"] == 0 and boite.proj.qweight.qweight is not None
    dense = torch.nn.Module(); dense.proj = _lin(2048, 1024, 12)
    assert kernels.preparer_disposition_marlin(dense)["seuls"] == 1               # dense : converti sous « denses »
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_PORTEE", "global")
    assert kernels.preparer_disposition_marlin(boite)["seuls"] == 1               # global : le MoE est converti aussi


def test_portee_posee_global_sous_processus():
    """Pièce 156 : le défaut est « denses » (test_defaut_marlin_156) ; « global » reste accessible par la variable."""
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_PROJ_MARLIN_PORTEE"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ACVRAM_PROJ_MARLIN_PORTEE"] = "global"
    r = subprocess.run([sys.executable, "-c", "import acvram.kernels as k; print(k._PROJ_MARLIN_PORTEE)"], env=env,
                       capture_output=True, text=True, cwd=str(RACINE))
    assert r.stdout.strip().splitlines()[-1] == "global", r.stdout + r.stderr
