"""Pièce 227 (chef, sur constat 226/poste6) : le banc de décodage `banc-llamacpp-16-09.py` ne
doit plus tirer les invites en génération libre sur un modèle MoE — la 217 (poste3) mesurait une
« régression » du Coder qui était en réalité un routage d'experts divergent entre les deux bras,
provoqué par des continuations dégénérées et différentes issues d'invites tirées. `invite_pour`
bascule vers un texte réel fixe (`invite_reelle`) pour un MoE, garde `invite()` (tirée) pour un
modèle dense. Testé en isolation, sans HTTP ni carte, comme `test_banc_llamacpp_cartes.py`."""
import importlib.util
import pathlib
import sys

import pytest

_OUTIL = pathlib.Path(__file__).resolve().parents[1] / "scratchpad" / "banc-llamacpp-16-09.py"


def _charger():
    avant = list(sys.path)
    spec = importlib.util.spec_from_file_location("banc_llamacpp_227", _OUTIL)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = avant
    return mod


class _SpecFabriquee:
    def __init__(self, num_experts):
        self.num_experts = num_experts


class _TokFabrique:
    def __init__(self, ids):
        self._ids = ids

    def encode(self, texte):
        return list(self._ids)


def test_est_moe_lit_num_experts_du_manifeste(monkeypatch):
    mod = _charger()
    import acvram.engine.config as config

    monkeypatch.setattr(config, "load_model_spec", lambda p: _SpecFabriquee(8))
    assert mod.est_moe("/faux/moe-227a") is True

    monkeypatch.setattr(config, "load_model_spec", lambda p: _SpecFabriquee(0))
    assert mod.est_moe("/faux/dense-227a") is False


def test_est_moe_refuse_silencieusement_si_le_manifeste_est_illisible(monkeypatch):
    mod = _charger()
    import acvram.engine.config as config

    def leve(p):
        raise FileNotFoundError(p)

    monkeypatch.setattr(config, "load_model_spec", leve)
    assert mod.est_moe("/faux/illisible-227a") is False


def test_invite_reelle_repete_et_tronque_sur_la_longueur_tokenisee(monkeypatch):
    mod = _charger()
    import acvram.server.chat as chat

    monkeypatch.setattr(chat, "load_tokenizer", lambda p: _TokFabrique([10, 20, 30]))
    assert mod.invite_reelle("/faux/moe-227b", 3) == [10, 20, 30]
    assert mod.invite_reelle("/faux/moe-227b", 7) == [10, 20, 30, 10, 20, 30, 10]
    assert mod.invite_reelle("/faux/moe-227b", 1) == [10]


def test_invite_reelle_refuse_sans_tokenizer():
    """Le REFUS demandé par la 227 : pas d'invites tirées de secours sur un MoE sans tokenizer —
    mieux vaut arrêter le banc que mesurer un routage faussé."""
    mod = _charger()
    import acvram.server.chat as chat
    import pytest as _pytest

    with _pytest.MonkeyPatch().context() as mp:
        mp.setattr(chat, "load_tokenizer", lambda p: None)
        with pytest.raises(RuntimeError, match="REFUS"):
            mod.invite_reelle("/faux/moe-227c", 5)


def test_invite_pour_bascule_reelle_sur_moe_tiree_sur_dense(monkeypatch):
    mod = _charger()
    appels = []
    monkeypatch.setattr(mod, "est_moe", lambda g: g == "/faux/moe-227d")
    monkeypatch.setattr(mod, "invite_reelle", lambda g, n: appels.append(("reelle", g, n)) or [1, 2, 3])
    monkeypatch.setattr(mod, "invite", lambda k, n: appels.append(("tiree", k, n)) or [9, 9, 9])

    assert mod.invite_pour("/faux/moe-227d", 42, 3) == [1, 2, 3]
    assert mod.invite_pour("/faux/dense-227d", 42, 3) == [9, 9, 9]
    assert appels == [("reelle", "/faux/moe-227d", 3), ("tiree", 42, 3)]


def test_cassant_sans_le_correctif_moe_et_dense_seraient_indistinguables(monkeypatch):
    """Preuve que le test précédent rend FAUX si on revient au comportement d'avant la 227 :
    `invite_pour` qui appelait `invite()` sans regarder `est_moe`."""
    mod = _charger()
    monkeypatch.setattr(mod, "est_moe", lambda g: g == "/faux/moe-227e")
    monkeypatch.setattr(mod, "invite_reelle", lambda g, n: [1, 2, 3])
    monkeypatch.setattr(mod, "invite", lambda k, n: [9, 9, 9])

    def ancien_invite_pour(gguf, k, n):   # le code d'avant la 227, ré-injecté ici seulement
        return mod.invite(k, n)

    assert ancien_invite_pour("/faux/moe-227e", 42, 3) == [9, 9, 9]        # bug reproduit : tirée sur un MoE
    assert mod.invite_pour("/faux/moe-227e", 42, 3) == [1, 2, 3]           # corrigé : réelle sur ce MoE
