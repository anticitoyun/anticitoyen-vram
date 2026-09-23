"""Le convertisseur consigne ce qui empeche la fusion — et ne decide rien.

Le 9/09/2026 : des groupes q/k/v melent nvfp4, int8 et int4_awq parce que le
plancher de SNR promeut TENSEUR PAR TENSEUR sans regarder les voisins du
groupe, alors que la fusion se consomme PAR GROUPE. Uniformiser semblait
gratuit — la qualite ne baisse pas, l'AWQ ne bouge pas — mais les octets
DOUBLENT, et sur un decodage lie a la memoire ils se paient a chaque pas.

Mesure : +43 us/pas sur un modele, -539 et -553 sur deux autres. Le signe
depend de la largeur. Le diagnostic accumule donc le cout sans appliquer de
regle : son terme de gain manque et ne doit pas etre invente.
"""
from acvram.quant.convert import _diagnostic_fusion


def _t(fmt, shape, had=0):
    return {"format": fmt, "shape": shape, "hadamard_block": had}


def _groupe(couche, fmts, hads=None):
    hads = hads or [0] * len(fmts)
    noms = ["q", "k", "v"] if len(fmts) == 3 else ["gate", "up"]
    pref = "self_attn" if len(fmts) == 3 else "mlp"
    tailles = {"q": [2048, 2048], "k": [256, 2048], "v": [256, 2048],
               "gate": [4096, 2048], "up": [4096, 2048]}
    return {f"model.layers.{couche}.{pref}.{n}_proj.weight":
            _t(f, tailles[n], h) for n, f, h in zip(noms, fmts, hads)}


def test_un_groupe_homogene_n_apparait_pas():
    ts = _groupe(0, ["nvfp4"] * 3)
    d = _diagnostic_fusion(ts)
    assert d["groupes_totaux"] == 1
    assert d["groupes_non_fusionnables"] == 0
    assert d["octets_ajoutes_si_uniformise"] == 0


def test_un_groupe_a_formats_melanges_est_chiffre():
    """q en nvfp4, k et v en int8 : uniformiser promeut q, le PLUS GROS."""
    ts = _groupe(0, ["nvfp4", "int8", "int8"])
    d = _diagnostic_fusion(ts)
    assert d["groupes_non_fusionnables"] == 1
    # q fait 2048x2048 et passe de 0,5625 a 1,0625 octet par poids
    attendu = int(2048 * 2048 * (1.0625 - 0.5625))
    assert d["octets_ajoutes_si_uniformise"] == attendu
    assert d["detail"][0]["formats"] == ["int8", "nvfp4"]


def test_un_hadamard_different_est_signale_sans_cout_d_octets():
    """Une rotation differente interdit la fusion mais ne se paie pas en
    octets : le cout doit rester nul, pas etre invente."""
    ts = _groupe(0, ["nvfp4"] * 3, hads=[2048, 0, 0])
    d = _diagnostic_fusion(ts)
    assert d["groupes_non_fusionnables"] == 1
    assert d["octets_ajoutes_si_uniformise"] == 0
    assert d["detail"][0]["hadamard_blocks"] == [0, 2048]


def test_le_gain_reste_absent_et_aucune_regle_n_est_appliquee():
    """Le terme de gain demande une mesure de fusion sur un modele reel.
    L'inventer ici le rendrait invisible et durable dans chaque poids
    converti — un poids ne se relit pas pour savoir d'ou venait sa constante."""
    d = _diagnostic_fusion(_groupe(0, ["nvfp4", "int8", "int8"]))
    assert d["gain_microsecondes"] is None
    assert "decision" not in d and "promouvoir" not in d


def test_le_detecteur_sait_compter_plusieurs_couches():
    ts = {}
    for c in range(3):
        ts.update(_groupe(c, ["nvfp4", "int8", "int8"]))
        ts.update(_groupe(c, ["nvfp4", "nvfp4"]))
    d = _diagnostic_fusion(ts)
    assert d["groupes_totaux"] == 6
    assert d["groupes_non_fusionnables"] == 3
