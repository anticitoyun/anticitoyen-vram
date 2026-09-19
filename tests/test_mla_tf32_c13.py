"""C13-a (poste7-c7-clos-c13-attention-glm-19-09) : TF32 à portée limitée sur les
deux einsum du cœur MLA. À sec, TF32 n'existe pas (CPU) : on l'ÉMULE — entrées
arrondies à 10 bits de mantisse (troncature, borne haute de l'arrondi matériel),
accumulation fp32 — et on borne l'écart au fp32 plein à 2⁻¹⁰ relatif sur des
tenseurs aux formes et échelles de GLM (rank 512 + rope 64, 20 têtes, 2 047
jetons) avec un v_b RÉEL du converti si présent. Le contexte restaure le
drapeau ; identité au défaut."""
import glob
import json
import os
import sys
import pathlib

import pytest
import torch

RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from acvram.engine import mla as MLA                                            # noqa: E402

GLM = "/mnt/2TO_2023_980PRO/Modeles/models_acvram/GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA"


def _tf32(x: torch.Tensor) -> torch.Tensor:
    """fp32 → 10 bits de mantisse (les 13 bits bas à zéro), comme les tensor cores."""
    return (x.float().view(torch.int32) & torch.tensor(-8192, dtype=torch.int32)).view(torch.float32)


def _v_b_reel():
    try:
        man = json.load(open(os.path.join(GLM, "acvram_manifest.json")))
    except OSError:
        return None
    for nom, e in man["tensors"].items():
        if "layers.3." in nom and "kv_b" in nom and e.get("format") in ("bf16", "plain", "float16", "bfloat16"):
            from safetensors import safe_open
            for f in glob.glob(os.path.join(GLM, "*.safetensors")):
                with safe_open(f, "pt") as fh:
                    if nom in fh.keys():
                        return fh.get_tensor(nom)
    return None


def test_tf32_emule_borne_a_2_moins_10_sur_le_coeur():
    torch.manual_seed(13)
    t, nh, r, rope, s = 64, 20, 512, 64, 2047
    q_eff = torch.randn(t, nh, r + rope) * 0.3
    C = torch.randn(s, r + rope) * 0.5
    scale = (r + rope) ** -0.5
    sc = torch.einsum('thr,sr->ths', q_eff, C) * scale
    sc32 = torch.einsum('thr,sr->ths', _tf32(q_eff), _tf32(C)) * scale
    assert (sc32 - sc).norm() / sc.norm() < 2 ** -10
    probs = sc.softmax(-1)
    V = C[:, :r]
    o = torch.einsum('ths,sr->thr', probs, V)
    o32 = torch.einsum('ths,sr->thr', _tf32(probs), _tf32(V))
    assert (o32 - o).norm() / o.norm() < 2 ** -10
    # le v_b réel du converti (hors portée TF32 dans le code : témoin de l'échelle réelle des poids)
    v_b = _v_b_reel()
    if v_b is not None:
        vb = v_b.float().reshape(nh, -1, r) if v_b.numel() % (nh * r) == 0 else None
        if vb is not None:
            y = torch.einsum('hvr,thr->thv', vb, o)
            y32 = torch.einsum('hvr,thr->thv', _tf32(vb), _tf32(o))
            assert (y32 - y).norm() / y.norm() < 2 ** -10
    # témoin cassant : 7 bits de mantisse (bf16) dépasse la borne
    bf = lambda x: x.to(torch.bfloat16).float()
    scb = torch.einsum('thr,sr->ths', bf(q_eff), bf(C)) * scale
    assert (scb - sc).norm() / sc.norm() > 2 ** -10


def test_contexte_tf32_restaure_et_identite_au_defaut(monkeypatch):
    monkeypatch.setattr(MLA, "_MLA_CORE", "fp32")
    with MLA._tf32_coeur() as c:
        assert not c._actif and MLA._dt_coeur() is torch.float32
    monkeypatch.setattr(MLA, "_MLA_CORE", "tf32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    with MLA._tf32_coeur() as c:
        assert c._actif and torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is False              # restauré
    with MLA._tf32_coeur(decode=True) as c:
        assert c._actif                                                 # tf32 couvre aussi le décodage
    monkeypatch.setattr(MLA, "_MLA_CORE", "bf16")
    with MLA._tf32_coeur() as c:
        assert not c._actif and MLA._dt_coeur() is torch.bfloat16      # bf16 : pas de drapeau
    from acvram import regime
    v = {x.env: x for x in regime.VARIABLES}["ACVRAM_MLA_CORE"]
    assert v.defaut == "fp32"
    src = (RACINE / "acvram" / "engine" / "mla.py").read_text()
    assert src.count("with _tf32_coeur():") == 5                        # préfill : scores, o_lat, v_b (chunké 3, non chunké 2)
    assert src.count("with _tf32_coeur(decode=True):") == 4             # décodage : y = v_b · o_lat
    assert src.count("_dt_coeur()") >= 10                               # chaque produit prend le dtype du régime


def test_coeur_bf16_borne_a_2_moins_7_par_ligne_formes_reelles():
    """C13-b à sec : le cœur en bf16 (entrées castées, accumulation fp32 —
    ce que font les tensor cores et le matmul bf16 CPU de torch) contre le
    fp32 plein, formes GLM réelles L=2 047 (20 têtes, rank 512 + rope 64) :
    par LIGNE de sortie, max|Δ| ≤ 2⁻⁷ · max|y_ligne| (le juge de
    test_gemv_marlin), sur scores, o_lat et y ; témoin cassant : casser une
    échelle de v_b ×2 dépasse la borne."""
    torch.manual_seed(1313)
    t, nh, r, rope, L = 64, 20, 512, 64, 2047
    q_eff = torch.randn(t, nh, r + rope) * 0.3
    C = torch.randn(L, r + rope) * 0.5
    scale = (r + rope) ** -0.5

    v_b = _v_b_reel()
    vb = v_b.float().reshape(nh, -1, r) if v_b is not None and v_b.numel() % (nh * r) == 0 else torch.randn(nh, 128, r) * 0.02

    def coeur(dt):
        sc = torch.einsum('thr,sr->ths', q_eff.to(dt), C.to(dt)).float() * scale
        probs = sc.softmax(-1)
        o = torch.einsum('ths,sr->thr', probs.to(dt), C[:, :r].to(dt))
        y = torch.einsum('hvr,thr->thv', vb.to(dt), o.to(dt))
        return sc, o.float(), y.float(), vb

    def borne_ligne(a, b):
        a2 = a.reshape(-1, a.shape[-1]).float(); b2 = b.reshape(-1, b.shape[-1]).float()
        return bool(((a2 - b2).abs() <= 2 ** -7 * b2.abs().amax(1, keepdim=True).clamp_min(1e-6)).all())

    sc32, o32, y32, vb = coeur(torch.float32)
    sc16, o16, y16, _ = coeur(torch.bfloat16)
    assert borne_ligne(sc16, sc32) and borne_ligne(o16, o32) and borne_ligne(y16, y32)
    y_faux = torch.einsum('hvr,thr->thv', (vb * 2).to(torch.bfloat16), o16.to(torch.bfloat16)).float()
    assert not borne_ligne(y_faux, y32)
