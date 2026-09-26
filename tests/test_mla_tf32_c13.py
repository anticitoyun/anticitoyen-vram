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
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../outils'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from acvram.engine import mla as MLA                                            # noqa: E402

GLM = _RACINE + "/GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA"


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


def test_defaut_tf32_prefill_seul_et_portees_opt_in(monkeypatch):
    """poste7-c13a-defaut-19-09 § 1 + addendum : sans variable, MLA_CORE=tf32 pose allow_tf32
    pendant les deux einsum du préfill et le rend après ; le 3e produit (MLA_CORE_VB) et
    le décodage (MLA_CORE_DECODE) restent fp32 quelle que soit la valeur de MLA_CORE."""
    from acvram import regime
    vs = {x.env: x for x in regime.VARIABLES}
    assert vs["ACVRAM_MLA_CORE"].defaut == "tf32"
    assert vs["ACVRAM_MLA_CORE_VB"].defaut == "0"
    assert vs["ACVRAM_MLA_CORE_DECODE"].defaut == "fp32"
    # défaut = ce que mla.py lit sans variable posée
    for nom in ("ACVRAM_MLA_CORE", "ACVRAM_MLA_CORE_VB", "ACVRAM_MLA_CORE_DECODE"):
        monkeypatch.delenv(nom, raising=False)
    src_defauts = (RACINE / "acvram" / "engine" / "mla.py").read_text()
    assert 'os.environ.get("ACVRAM_MLA_CORE", "tf32")' in src_defauts
    assert 'os.environ.get("ACVRAM_MLA_CORE_DECODE", "fp32")' in src_defauts
    assert 'os.environ.get("ACVRAM_MLA_CORE_VB", "0")' in src_defauts
    monkeypatch.setattr(MLA, "_MLA_CORE", "tf32")
    monkeypatch.setattr(MLA, "_MLA_CORE_VB", False)
    monkeypatch.setattr(MLA, "_MLA_CORE_DECODE", "fp32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    with MLA._tf32_coeur() as c:                                        # les deux einsum du préfill
        assert c._actif and torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is False              # rendu après
    with MLA._tf32_coeur(vb=True) as c:                                 # 3e produit : opt-in, fp32
        assert not c._actif and torch.backends.cuda.matmul.allow_tf32 is False
    with MLA._tf32_coeur(decode=True) as c:                             # décodage : fp32 jusqu'au niveau 2
        assert not c._actif and torch.backends.cuda.matmul.allow_tf32 is False
    assert MLA._dt_coeur() is torch.float32 and MLA._dt_coeur(decode=True) is torch.float32
    # bras qui doivent différer : chaque portée ouverte par sa propre variable
    monkeypatch.setattr(MLA, "_MLA_CORE_VB", True)
    with MLA._tf32_coeur(vb=True) as c:
        assert c._actif
    monkeypatch.setattr(MLA, "_MLA_CORE_DECODE", "tf32")
    with MLA._tf32_coeur(decode=True) as c:
        assert c._actif
    assert torch.backends.cuda.matmul.allow_tf32 is False
    monkeypatch.setattr(MLA, "_MLA_CORE_DECODE", "bf16")
    assert MLA._dt_coeur(decode=True) is torch.bfloat16
    with MLA._tf32_coeur(decode=True) as c:
        assert not c._actif                                             # bf16 : pas de drapeau
    # bf16 au préfill n'atteint ni le 3e produit sans VB, ni le décodage
    monkeypatch.setattr(MLA, "_MLA_CORE", "bf16")
    monkeypatch.setattr(MLA, "_MLA_CORE_VB", False)
    monkeypatch.setattr(MLA, "_MLA_CORE_DECODE", "fp32")
    assert MLA._dt_coeur() is torch.bfloat16
    assert MLA._dt_coeur(vb=True) is torch.float32 and MLA._dt_coeur(decode=True) is torch.float32
    monkeypatch.setattr(MLA, "_MLA_CORE_VB", True)
    assert MLA._dt_coeur(vb=True) is torch.bfloat16
    monkeypatch.setattr(MLA, "_MLA_CORE", "fp32")
    with MLA._tf32_coeur() as c:
        assert not c._actif and MLA._dt_coeur() is torch.float32
    src = src_defauts
    assert src.count("with _tf32_coeur(cles=cles):") == 4              # préfill : scores, o_lat (chunké, non chunké), règle ≤ 2 048 clés
    assert src.count("with _tf32_coeur(vb=True, cles=") == 2            # 3e produit du préfill (chunké, non chunké)
    assert src.count("with _tf32_coeur(decode=True):") == 4             # décodage : y = v_b · o_lat
    assert src.count("_dt_coeur(decode=True)") == 8                     # chaque produit du décodage lit SON régime
    assert src.count("_dt_coeur(vb=True, cles=") == 2
    assert "ACVRAM_MLA_TF32" not in src                                 # plus d'alias (addendum : une variable par régime)


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


def test_regime_reduit_seulement_sous_2048_cles_vues(monkeypatch):
    """poste7-c14-defaut-tf32-8k-20-09 addendum 02 h 25 (verdict-tf32-8k : max |Δ| 6,23 % à 8 k) :
    tf32 / bf16 seulement quand le morceau de préfill VOIT ≤ 2 048 clés ; un préfill de 4 096 en
    morceaux de 256 : les 8 premiers morceaux (clés 256..2048) en tf32, les 8 suivants en fp32."""
    monkeypatch.setattr(MLA, "_MLA_CORE", "tf32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    t, passe = 4096, 0
    regimes = []
    for d0 in range(0, t, 256):
        d1 = min(t, d0 + 256); cles = passe + d1                          # la formule du chemin chunké (mla.py)
        with MLA._tf32_coeur(cles=cles) as c:
            regimes.append((cles, c._actif, MLA._dt_coeur(cles=cles)))
    assert [r[1] for r in regimes] == [True] * 8 + [False] * 8, regimes    # tf32 puis fp32
    assert regimes[7][0] == 2048 and regimes[8][0] == 2304
    assert torch.backends.cuda.matmul.allow_tf32 is False
    # bf16 (bras F) : même règle — bf16 jusqu'à 2 048 clés, fp32 au-delà
    monkeypatch.setattr(MLA, "_MLA_CORE", "bf16")
    assert MLA._dt_coeur(cles=2048) is torch.bfloat16 and MLA._dt_coeur(cles=2049) is torch.float32
    with MLA._tf32_coeur(cles=4096) as c:
        assert not c._actif
    # un préfill continué (passe > 0) : les clés vues comptent le passé
    monkeypatch.setattr(MLA, "_MLA_CORE", "tf32")
    with MLA._tf32_coeur(cles=1800 + 256) as c:
        assert not c._actif                                                 # 2 056 clés > 2 048
    # sans `cles` (décodage, 3e produit) la règle ne s'applique pas ; la ligne de régime le nomme
    from acvram import regime
    assert "mla_core=tf32(≤2048 clés)" in regime.regime_ligne()
    src = (RACINE / "acvram" / "engine" / "mla.py").read_text()
    assert src.count("_tf32_coeur(cles=cles)") == 4 and src.count("cles = passe + d1") == 1


def test_regle_des_cles_neutralisee_par_max_cles_zero(monkeypatch):
    """ACVRAM_MLA_CORE_MAX_CLES=0 (bras « règle neutralisée » de la prise à 36 tranches, poste2 20/09) :
    aucune longueur ne bascule en fp32 et la ligne le dit ; 2048 = défaut servi (témoin)."""
    import importlib
    monkeypatch.setenv("ACVRAM_MLA_CORE", "bf16")           # bf16 : le dtype dit le régime à sec (tf32 = fp32 émulé)
    monkeypatch.setenv("ACVRAM_MLA_CORE_MAX_CLES", "0")
    importlib.reload(MLA)
    try:
        assert MLA._dt_coeur(cles=2049) is torch.bfloat16 and MLA._dt_coeur(cles=1 << 20) is torch.bfloat16
        assert MLA.regime_coeur_texte() == "mla_core=bf16(sans règle des clés)"
        monkeypatch.setenv("ACVRAM_MLA_CORE_MAX_CLES", "2048")
        importlib.reload(MLA)
        assert MLA._dt_coeur(cles=2049) is torch.float32 and "≤2048 clés" in MLA.regime_coeur_texte()
    finally:
        monkeypatch.delenv("ACVRAM_MLA_CORE_MAX_CLES", raising=False); monkeypatch.delenv("ACVRAM_MLA_CORE", raising=False)
        importlib.reload(MLA)
