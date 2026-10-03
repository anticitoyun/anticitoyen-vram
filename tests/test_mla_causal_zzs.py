"""zzs (revue/poste5-zzs-scelle-01-10.md) : au préfill par morceaux, le cœur MLA ne traite que les clés que le morceau
VOIT (passe + d1), au lieu de toutes puis masque. Sans carte : la forme (chaque einsum du cœur reçoit passe + d1 clés)
et l'écart au chemin complet, borné par 2 × l'erreur fp32 du chemin complet contre une référence fp64 (E2 amendé).
Sur carte : E1, au bit du chemin complet, au régime servi (tf32 ≤ 2 048 clés vues, fp32 au-delà), formes de Kimi-Linear.
Bras cassant : une troncature décalée de un (la diagonale perdue) doit rendre faux."""
import pathlib
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from acvram.engine import mla as M                                              # noqa: E402


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _module(H, NH, NOPE, ROPE, RANK, DV, dt, dev, graine=5):
    """Forme de Kimi-Linear : ni q en bas rang ni RoPE (mla.py, MLAttention.__init__)."""
    g = torch.Generator().manual_seed(graine)
    lin = lambda o, i: nn.Linear(i, o, bias=False).to(dt).to(dev)               # noqa: E731
    mods = [lin(NH * (NOPE + ROPE), H), lin(RANK + ROPE, H), lin(H, NH * DV)]
    for m in mods:
        m.weight.copy_((torch.randn(m.weight.shape, generator=g) * m.weight.shape[1] ** -0.5).to(dt))
    return M.MLAttention(*mods, (torch.rand(RANK, generator=g) + 0.5).to(dt).to(dev),
                         (torch.randn(NH, RANK, NOPE, generator=g) * 0.1).to(dt).to(dev),
                         (torch.randn(NH, DV, RANK, generator=g) * 0.1).to(dt).to(dev),
                         NH, NOPE, ROPE, RANK, DV, eps=1e-5).to(dev)


def _prefill(mod, x0, x):
    """Préfill de x0 (sans passé) puis de x avec le cache de x0 : rend la sortie de x."""
    _, cache0 = mod.forward(x0, None) if x0 is not None else (None, None)
    y, _ = mod.forward(x, cache0)
    return y


def _bras(monkeypatch, causal):
    monkeypatch.setattr(M, "_MLA_CAUSAL", causal)


@pytest.fixture
def _regime_servi(monkeypatch):
    monkeypatch.setattr(M, "_MLA_CORE", "tf32")
    monkeypatch.setattr(M, "_MLA_CORE_VB", False)
    monkeypatch.setattr(M, "_MLA_CORE_MAX_CLES", 2048)


def test_chaque_morceau_ne_traite_que_les_cles_vues(monkeypatch, _regime_servi):
    H, NH, NOPE, ROPE, RANK, DV = 64, 4, 32, 16, 64, 32
    mod = _module(H, NH, NOPE, ROPE, RANK, DV, torch.float32, "cpu")
    g = torch.Generator().manual_seed(1)
    x0, x = torch.randn(300, H, generator=g), torch.randn(700, H, generator=g)
    vues = {}
    vrai = torch.einsum

    def espion(eq, *ops):
        if eq in ("thr,sr->ths", "ths,sr->thr"):
            vues.setdefault(eq, []).append(ops[1].shape[0])
        return vrai(eq, *ops)
    monkeypatch.setattr(torch, "einsum", espion)
    for causal, attendu in ((True, [300 + d1 for d1 in (256, 512, 700)]), (False, [1000] * 3)):
        _bras(monkeypatch, causal)
        vues.clear()
        _prefill(mod, x0, x)
        # le préfill de x0 (300 jetons, sans passé) passe aussi par le cœur : ses morceaux d'abord
        x0_attendu = [256, 300] if causal else [300, 300]
        assert vues["thr,sr->ths"] == x0_attendu + attendu, (causal, vues)
        assert vues["ths,sr->thr"] == x0_attendu + attendu, (causal, vues)
    assert M.regime_causal_texte() == ""
    _bras(monkeypatch, True)
    assert M.regime_causal_texte() == "mla_causal=1(opt-in)"


def _ecarts(monkeypatch, dev, dt, t0, t, formes, decalage=None, dev_ref="cpu"):
    """y du chemin tronqué (B), du chemin complet (A) et la référence fp64 (même module, tout en fp64).
    Rend (|B − A|, |B − ref|, |A − ref|) au max, et les sorties. Le témoin est |A − ref| : l'erreur d'arrondi fp32
    que le chemin servi a déjà (amendement du scellé : un témoin « morceaux de 128 » est nul, la longueur des sommes
    du chemin complet ne dépendant pas de la taille des morceaux)."""
    H, NH, NOPE, ROPE, RANK, DV = formes
    g = torch.Generator().manual_seed(2)
    x0 = (torch.randn(t0, H, generator=g) * 0.5) if t0 else None
    x = torch.randn(t, H, generator=g) * 0.5
    ys, vrai = {}, M._cles_vues
    for nom, causal in (("A", False), ("B", True)):
        _bras(monkeypatch, causal)
        if nom == "B" and decalage is not None:
            monkeypatch.setattr(M, "_cles_vues", lambda cles, total: cles + decalage)
        mod = _module(H, NH, NOPE, ROPE, RANK, DV, dt, dev)
        ys[nom] = _prefill(mod, None if x0 is None else x0.to(dt).to(dev), x.to(dt).to(dev)).double().cpu()
    monkeypatch.setattr(M, "_cles_vues", vrai)                 # la référence n'est jamais sabotée
    monkeypatch.setattr(M, "_dt_coeur", lambda **k: torch.float64)
    _bras(monkeypatch, False)
    mod = _module(H, NH, NOPE, ROPE, RANK, DV, torch.float64, dev_ref)
    ref = _prefill(mod, None if x0 is None else x0.double().to(dev_ref), x.double().to(dev_ref)).cpu()
    m = lambda a, b: float((a - b).abs().max())                                 # noqa: E731
    return m(ys["B"], ys["A"]), m(ys["B"], ref), m(ys["A"], ref), ys


def test_ecart_au_chemin_complet_sous_le_temoin_fp32(monkeypatch, _regime_servi):
    """E2 à sec : la troncature ne s'éloigne de la référence fp64, ni du chemin complet, de plus de 2 × l'erreur fp32
    du chemin complet (le témoin). Mesuré le 01/10 sur CPU : |B − A| 1,0e-7, |B − ref| = |A − ref| = 1,4e-7."""
    dBA, dBref, dAref, ys = _ecarts(monkeypatch, "cpu", torch.float32, 300, 700, (64, 4, 32, 16, 64, 32))
    assert dAref > 0, "témoin nul : la référence fp64 n'est pas une autre arithmétique"
    assert dBref <= 2 * dAref, (dBref, dAref)
    assert dBA <= 2 * dAref, (dBA, dAref)


def test_bras_cassant_diagonale_perdue(monkeypatch, _regime_servi):
    """Une troncature à passe + d1 − 1 (la dernière requête du morceau perd sa propre clé) doit rendre faux."""
    dBA, dBref, dAref, ys = _ecarts(monkeypatch, "cpu", torch.float32, 300, 700, (64, 4, 32, 16, 64, 32), decalage=-1)
    assert dBref > 2 * dAref and dBA > 2 * dAref, (dBA, dBref, dAref)


carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="E2 aux formes de Kimi : carte requise")


@carte
@pytest.mark.parametrize("t0,t", [(0, 8192), (3000, 1000)])
def test_e2_formes_kimi_sur_carte_et_son_bras_cassant(monkeypatch, _regime_servi, t0, t):
    """Jugement sur carte par E2 sous témoin (REGLES § 4), au régime servi du cœur (tf32 ≤ 2 048 clés vues, fp32 au-delà),
    formes de Kimi-Linear (hidden 2 304, 32 têtes, nope 128, rope 64, rang 512, dv 128) en fp32 : |B − A| et |B − ref|
    ≤ 2 × |A − ref|, la référence étant le même module en fp64 (sur la carte : 8 192 jetons en fp64 sur le processeur
    prendraient des minutes). Le bras cassant (diagonale perdue, passe + d1 − 1) doit rendre faux aux MÊMES formes.

    Pourquoi pas E1 au bit : il est faux sur carte (verdict poste5-zzs-verdict-02-10 : 262 473 éléments ≠ à 8 192, max
    3,9e-3 ; 17 357 ≠ à 3 000 + 1 000, max 2,0e-3 — cuBLAS et le softmax choisissent leurs noyaux selon la longueur), et
    pas même stable sur le processeur (au bit en bf16, 6,7e-8 en fp32 sans passé, 02/10).
    Pourquoi pas en bf16 : le témoin bf16 (arrondi de la sortie) est trop large pour voir le bras cassant — à sec,
    formes réduites, 300 + 700 : |B_cassé − ref| 3,35e-3 contre 2 × témoin 3,42e-3. Un E2 bf16 ne pourrait pas rendre
    faux (REGLES § 3) ; la sortie de service se juge de bout en bout (ABBA, garde PPL du scellé)."""
    formes = (2304, 32, 128, 64, 512, 128)
    dBA, dBref, dAref, ys = _ecarts(monkeypatch, "cuda", torch.float32, t0, t, formes, dev_ref="cuda")
    assert dAref > 0, "témoin nul : la référence fp64 n'est pas une autre arithmétique"
    assert dBref <= 2 * dAref and dBA <= 2 * dAref, (t0, t, dBA, dBref, dAref)
    cBA, cBref, cAref, _ = _ecarts(monkeypatch, "cuda", torch.float32, t0, t, formes, decalage=-1, dev_ref="cuda")
    assert cBref > 2 * cAref and cBA > 2 * cAref, ("bras cassant non vu", t0, t, cBA, cBref, cAref)


def test_la_troncature_est_opt_in_defaut_off():
    """Verdict du 02/10 (chef) : la troncature change la sortie (E1 au bit faux, 2 réponses gloutonnes sur 6 divergent) ;
    elle reste OPT-IN tant que la garde de PPL de décodage à 8 192 + 512 n'est pas passée. Casse si le défaut bascule —
    dans le registre, dans le module, ou dans un processus neuf sans aucune variable ACVRAM_*."""
    import os
    import subprocess
    import sys
    from acvram import regime
    assert next(v for v in regime.VARIABLES if v.nom == "MLA_CAUSAL").defaut == "0"
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "from acvram.engine import mla; print(mla._MLA_CAUSAL, repr(mla.regime_causal_texte()))"],
                         env=env, capture_output=True, text=True, timeout=180)
    assert out.stdout.strip().splitlines()[-1] == "False ''", out.stdout[-500:] + out.stderr[-500:]
