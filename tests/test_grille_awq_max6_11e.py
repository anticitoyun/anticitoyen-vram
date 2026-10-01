"""11e, option A (chef 01/10) : la recherche AWQ tourne toujours en max6, quelle que soit la règle d'échelle réglée ;
seule la quantification finale applique la règle (et l'importance de balayage-w). Deux contrôles :
(1) sur un VRAI tenseur (Qwen2.5-Coder-14B, q_proj couche 0, 512 lignes), les scalers AWQ des quatre règles sont égaux au bit ;
(2) sentinelle : l'échelle passée par la grille à `quantize_nvfp4` est « max6 » à chaque appel — casse si `_quant_dequant`
reprend l'échelle du bras (le défaut B1 reviendrait, et le coût ×10 de B2 avec lui)."""
import os

import pytest
import torch

from acvram.quant import nvfp4
from acvram.quant.calibrate import ActStats, quantize_with_calibration

REEL = "/mnt/4TO_SATACMR_2022/Modeles/models/Qwen2.5-Coder-14B-Instruct/model-00001-of-00006.safetensors"
REGLES = ("max6", "4sur6", "balayage", "balayage-w")


@pytest.fixture(autouse=True)
def _max6_ensuite():
    yield
    nvfp4.regler_echelle("max6")


def _poids():
    if os.path.exists(REEL):
        from safetensors import safe_open
        with safe_open(REEL, "pt", device="cpu") as f:
            w = f.get_slice("model.layers.0.self_attn.q_proj.weight")[:512].to(torch.float32)
        return w, "réel (Qwen2.5-Coder-14B q_proj couche 0, 512 lignes)"
    g = torch.Generator().manual_seed(11)
    return torch.randn(512, 1024, generator=g) * 0.02, "synthétique (tenseur réel absent de ce poste)"


def _stats(k):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(256, k, generator=g) * (1 + 3 * torch.rand(k, generator=g))     # canaux inégaux : l'AWQ a de quoi choisir
    return ActStats.from_inputs(x)


def test_scalers_awq_egaux_au_bit_entre_les_quatre_regles():
    w, origine = _poids()
    stats = _stats(w.shape[1])
    scalers, qts = {}, {}
    for regle in REGLES:
        nvfp4.regler_echelle(regle)
        qt, scaler, _ = quantize_with_calibration(w, "nvfp4", stats, use_awq=True, n_grid=4)
        scalers[regle], qts[regle] = scaler.scale.clone(), qt
    ref = scalers["max6"]
    assert ref is not None and not torch.all(ref == 1.0), "scaler AWQ trivial : le test ne jugerait rien"
    for regle in REGLES[1:]:
        assert torch.equal(scalers[regle], ref), f"{regle} : scaler AWQ ≠ max6 ({origine})"
    # la règle a bien agi sur la quantification FINALE : les codes diffèrent de max6 pour au moins une règle de balayage
    def echelles(qt):
        sd = qt.state_dict()
        return [v for k, v in sd.items() if "scale" in k]
    assert any(any(not torch.equal(a, b) for a, b in zip(echelles(qts[r]), echelles(qts["max6"])))
               for r in ("balayage", "balayage-w")), "aucune règle de balayage n'a changé les échelles finales"
    print(f"\n{origine} : scalers AWQ identiques au bit sur {len(REGLES)} règles")


@pytest.mark.parametrize("regle", ("balayage", "balayage-w"))
def test_la_grille_awq_quantifie_en_max6_quelle_que_soit_la_regle(monkeypatch, regle):
    w, _ = _poids()
    w = w[:64]
    stats = _stats(w.shape[1])
    nvfp4.regler_echelle(regle)
    vus: list = []
    import acvram.quant.formats as F
    vrai = F.quantize

    def espion(weight, fmt, group_size=None, **kw):
        if fmt == "nvfp4":
            vus.append(kw.get("echelle"))
        return vrai(weight, fmt, group_size=group_size, **kw)
    monkeypatch.setattr(F, "quantize", espion)              # `_quant_dequant` et la quantification finale passent par là
    quantize_with_calibration(w, "nvfp4", stats, use_awq=True, n_grid=4)
    grille, finale = vus[:-1], vus[-1]
    assert len(grille) >= 2 and all(e == "max6" for e in grille), f"la grille a quantifié sous {set(grille)} au lieu de max6"
    assert finale in (None, regle), f"la quantification finale devait suivre la règle {regle}, a reçu {finale!r}"
