"""Garde falaise (pièce 295, option C de la 294) : un exil qui fait tomber le débit prévu sous 25 % du résident
se DIT au chargement (avertissement nommé, ligne de régime, contexte sans exil), sans refus.

PRÉDICTION ÉCRITE AVANT LA MESURE (règle 4), plan simulé Devstral 24B à 32 768 (pièce 294 : 10/40 MLP exilés,
réserve de préfill 13,41 Gio) : MLP int8 3·5120·32768 = 0,469 Gio, attention 0,049 Gio, VRAM 1050 Go/s, lien
26,8 Go/s → transfert 10 × 0,469 Gio = 188 ms/jeton, pas résident 21,2 ms → débit ≤ 10 % du résident (mesuré le
28/09 : 7,9 / 94,5 = 8 %, graphes coupés en plus). Contexte sans exil ≈ 32 768 × (13,41 − 4,69) / 13,41 ≈ 21 300.
Issues nommées : une seule couche exilée (débit ≈ 53 %) ne doit PAS déclencher ; sans exil, rien.
"""

from acvram.engine.loader import _contexte_sans_exil, _signaler_falaise
from acvram.engine.runner import _falaise_texte
from acvram.memory.tiering import LayerPlacement, Plan, Tier, estimer_cout_exil

_GIB = 2 ** 30
_MLP = 3 * 5120 * 32768                       # int8, 0,469 Gio
_ATTN = int(0.049 * _GIB)
_RESERVE_32K = int(13.41 * _GIB)
_FIXE = 2 * _GIB                              # tampons denses : ne dépend pas du contexte


def _reserve(n: int) -> int:
    return _FIXE + _RESERVE_32K * n // 32768


def _plan_devstral(n_exiles: int) -> Plan:
    tier = Tier(name="cuda:0", kind="gpu", device_index=0, capacity=32 * _GIB, weight_format="int8",
                kv_format="fp8", read_bandwidth=1050.0, link_bandwidth=26.8)
    layers = [LayerPlacement(index=i, exec_device="cuda:0", attn_storage="cuda:0",
                             mlp_storage="cpu" if i >= 40 - n_exiles else "cuda:0", fmt="int8",
                             attn_bytes=_ATTN, mlp_bytes=_MLP, mlp_active_bytes=_MLP)
              for i in range(40)]
    return Plan(model="devstral-24b", tiers=[tier], layers=layers)


def test_devstral_32k_dix_exiles_est_une_falaise_nommee(monkeypatch, capsys):
    monkeypatch.delenv("ACVRAM_SEUIL_FALAISE", raising=False)
    plan = _plan_devstral(10)
    c = estimer_cout_exil(plan)
    assert 0.08 < c["debit_relatif"] < 0.12
    f = _signaler_falaise(plan, 32768, _reserve)
    assert f is plan.falaise and f["n_couches_exilees"] == 10 and f["couches_total"] == 40
    assert 21000 <= f["ctx_sans_exil"] <= 21500
    msg = [w for w in plan.warnings if w.startswith("FALAISE D'EXIL")]
    assert len(msg) == 1 and "10/40" in msg[0] and f"--max-model-len {f['ctx_sans_exil']}" in msg[0]
    assert "ATTENTION — FALAISE D'EXIL" in capsys.readouterr().out
    assert _falaise_texte(plan.falaise) == f"falaise=10%<25%(ctx_sans_exil={f['ctx_sans_exil']}) "


def test_le_contexte_rendu_est_le_plus_grand_qui_libere_l_exil():
    plan = _plan_devstral(10)
    n = _contexte_sans_exil(plan, 32768, _reserve)
    cible = _reserve(32768) - 10 * _MLP
    assert _reserve(n) <= cible < _reserve(n + 1)


def test_une_couche_exilee_ne_declenche_pas_et_sans_exil_rien():
    # règle 5 : le contrôle doit pouvoir rendre « faux »
    for n_exiles in (0, 1):
        plan = _plan_devstral(n_exiles)
        assert _signaler_falaise(plan, 32768, _reserve) is None
        assert plan.falaise is None and not any("FALAISE" in w for w in plan.warnings)
    assert _falaise_texte(None) == ""


def test_seuil_reglable(monkeypatch):
    monkeypatch.setenv("ACVRAM_SEUIL_FALAISE", "0.05")
    assert _signaler_falaise(_plan_devstral(10), 32768, _reserve) is None


def test_exil_plus_gros_que_la_reserve_rend_aucun_contexte():
    plan = _plan_devstral(40)
    f = _signaler_falaise(plan, 32768, _reserve)
    assert f["ctx_sans_exil"] is None
    assert "aucun contexte" in plan.warnings[-1]
    assert "ctx_sans_exil=aucun" in _falaise_texte(f)


def test_load_model_appelle_la_garde():
    # cliquet : retirer l'appel du chemin de chargement doit casser
    import inspect
    from acvram.engine import loader
    assert "_signaler_falaise(plan, max_model_len" in inspect.getsource(loader.load_model)
