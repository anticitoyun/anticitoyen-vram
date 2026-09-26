"""Instruments C9 M-HÔTE et M-TRACE (poste1-c9-conception-21-09 § 3) sur un
faux 119B minuscule : mêmes clés `layers.N.experts.E.w1/w3/w2.{weight_packed,
weight_scale,weight_global_scale}` et même disposition compressed-tensors
que le NVFP4 officiel ; trace de routage synthétique au format de
`trace_routage.noter`. Ce qui doit casser : un ordre de quartets ou une
échelle globale faux (contrôle d exactitude → INVALIDE), un verdict qui ne
suit plus les seuils écrits, une prise de trace dont h ne dépend plus du
routage."""
import importlib.util
import os

import pytest
import torch

ICI = os.path.dirname(os.path.abspath(__file__))
MESURE = os.path.join(ICI, "..", "outils", "gpu", "mesure")


def _charger(nom):
    spec = importlib.util.spec_from_file_location(nom.replace("-", "_"), os.path.join(MESURE, nom + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def faux_119b(tmp_path_factory):
    """Deux shards, couche 0 : experts 0-3, formes réduites (K=256, I=64), un
    fp32 aléatoire quantifié par acvram puis écrit dans la disposition
    compressed-tensors (global = 1/g d acvram, comme `nvfp4_direct` le relit)."""
    from safetensors.torch import save_file
    from acvram.quant.nvfp4 import quantize_nvfp4
    torch.manual_seed(1)
    d = tmp_path_factory.mktemp("faux119b")
    shards = [{}, {}]
    for e in range(4):
        for proj, forme in (("w1", (64, 256)), ("w3", (64, 256)), ("w2", (256, 64))):
            t = quantize_nvfp4(torch.randn(*forme) * 0.1)
            base = f"layers.0.experts.{e}.{proj}."
            s = shards[e % 2]
            s[base + "weight_packed"] = t.qweight.contiguous()
            s[base + "weight_scale"] = t.block_scale.contiguous()
            s[base + "weight_global_scale"] = (1.0 / t.global_scale.float()).reshape(1)
            s[base + "input_global_scale"] = torch.ones(1)
    for i, s in enumerate(shards):
        save_file(s, str(d / f"consolidated-0000{i + 1}-of-00002.safetensors"))
    return str(d)


def test_m_hote_mesure_et_verdict(faux_119b):
    m = _charger("c9-m-hote")
    r = m.mesurer(faux_119b, 0, [0, 1, 2, 3], rep=5, chauffe=2)
    assert r["formes"] == {"w1": [64, 256], "w3": [64, 256], "w2": [256, 64]}
    assert r["exactitude_rel_max"] <= 2 ** -6, r["exactitude_rel_max"]
    assert r["octets_par_expert"] == 3 * (64 * 128 + 64 * 16)
    # Pièce 267c (CI GitHub, runner 4 cœurs) : `go_s_effectif` est un débit ARRONDI à 1
    # décimale (`round(octets / (med_e·1e-3) / 1e9, 1)`) — sur un modèle jouet minuscule
    # (27 648 octets) et un processeur lent/virtualisé, le temps mesuré est dominé par le
    # lancement de l'opération torch (pas par le débit mémoire réel), et le débit calculé,
    # bien que réel et positif, arrondit à 0,0. La garde qui a du sens porte sur les
    # quantités NON arrondies : octets_par_expert > 0 (déjà vérifié) et mediane > 0 — leur
    # positivité implique déjà mathématiquement go_s_effectif > 0 au bit près ; l'exiger
    # aussi sur le CHAMP ARRONDI n'a jamais rien prouvé de plus, et casse sur du matériel
    # lent sans qu'aucun calcul ne soit faux.
    assert r["ms_par_expert"]["mediane"] > 0 and r["go_s_effectif"] >= 0
    assert "INVALIDE" not in r["verdict"]


def test_m_hote_verdict_suit_les_seuils():
    m = _charger("c9-m-hote")
    assert m.verdict(0.80, 4, 0.0)["verdict"] == "TENU"
    assert m.verdict(1.60, 8, 0.0)["verdict"] == "TENU"                 # 8 experts en 1,60 → 0,80 les 4
    assert m.verdict(1.30, 4, 0.0)["verdict"].startswith("RÉFUTÉ — arrêt")
    assert m.verdict(1.05, 4, 0.0)["verdict"].startswith("RÉFUTÉ hors bande")
    assert m.verdict(0.50, 4, 0.0)["verdict"].startswith("TENU, mieux")
    assert m.verdict(0.80, 4, 0.1)["verdict"].startswith("INVALIDE")


def test_m_hote_exactitude_casse_sur_quartets_inverses(faux_119b):
    """Un expert dont les quartets sont échangés doit rendre un écart > 2^-6 :
    c est ce contrôle qui protège le chiffre de temps."""
    m = _charger("c9-m-hote")
    ex = m.charger_expert(faux_119b, 0, 0)
    x = (torch.randn(1, 256) * 0.5).to(torch.bfloat16)
    assert m.controle_exactitude(x, ex) <= 2 ** -6
    q = ex["w1"].qweight
    ex["w1"].qweight = ((q & 0xF) << 4) | (q >> 4)
    from acvram.kernels.cpu import nvfp4_matmul_cpu
    from acvram.quant.nvfp4 import dequantize_nvfp4
    ref = torch.nn.functional.linear(x.float(), dequantize_nvfp4(ex["w1"], torch.float32))
    y = nvfp4_matmul_cpu(x, ex["w1"]).float()
    # noyau et déquantification acvram restent d accord entre eux (même relecture)…
    assert ((y - ref).abs().max() / ref.abs().max()).item() <= 2 ** -6
    # …mais plus avec la déquantification compressed-tensors d origine
    from acvram.quant.hfquant import HFQuantCheckpoint
    ct0 = HFQuantCheckpoint._ct_nvfp4(q, ex["w1"].block_scale, 1.0 / ex["w1"].global_scale)
    ct1 = HFQuantCheckpoint._ct_nvfp4(ex["w1"].qweight, ex["w1"].block_scale, 1.0 / ex["w1"].global_scale)
    assert not torch.equal(ct0, ct1)


def _trace(chemin, couches, jetons, experts_par_jeton, E, concentration):
    """Trace synthétique : `concentration` de la masse sur E//4 experts chauds
    par couche, le reste uniforme ; format `jeton couche e1,e2,…`."""
    g = torch.Generator().manual_seed(7)
    with open(chemin, "w") as f:
        for j in range(jetons):
            for c in range(couches):
                chauds = torch.arange(E // 4) + c
                choix = set()
                while len(choix) < experts_par_jeton:
                    if torch.rand((), generator=g).item() < concentration:
                        choix.add(int(chauds[torch.randint(len(chauds), (), generator=g)]) % E)
                    else:
                        choix.add(int(torch.randint(E, (), generator=g)))
                f.write(f"{j} {c} {','.join(map(str, sorted(choix)))}\n")


def test_m_trace_h_suit_le_routage(tmp_path):
    m = _charger("c9-m-trace")
    concentre, uniforme = str(tmp_path / "c.txt"), str(tmp_path / "u.txt")
    _trace(concentre, couches=4, jetons=400, experts_par_jeton=4, E=128, concentration=0.9)
    _trace(uniforme, couches=4, jetons=400, experts_par_jeton=4, E=128, concentration=0.0)
    rc = m.analyser(concentre, 52, 128)
    ru = m.analyser(uniforme, 52, 128)
    assert rc["jetons"] == 400 and rc["couches"] == 4 and len(rc["h_pin_par_couche"]) == 4
    assert rc["h_pin"] > 0.8 and rc["verdict"].startswith("TENU, mieux")
    assert abs(ru["h_pin"] - 52 / 128) < 0.08 and ru["verdict"].startswith("RÉFUTÉ")
    assert "uniforme" in ru["verdict"]
    assert 0.0 <= rc["h_lru"] <= 1.0


def test_m_trace_verdict_suit_les_seuils():
    m = _charger("c9-m-trace")
    assert m.verdict(0.55, 52, 128)["verdict"] == "TENU"
    assert m.verdict(0.46, 52, 128)["verdict"] == "TENU"
    assert m.verdict(0.44, 52, 128)["verdict"].startswith("RÉFUTÉ")
    assert m.verdict(0.70, 52, 128)["verdict"].startswith("TENU, mieux")
