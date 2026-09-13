"""Le banc doit REFUSER de publier un debit quand le plan a ete rejoue.

`acvram bench --what decode` annoncait 20,04 jetons/s la ou le decodage vaut
175 : deux defauts distincts, tous deux corriges ici et eprouves dans les deux
sens.

1. Le chronometre englobait prefill, allocation et capture des graphes sur 64
   jetons — le cout fixe dominait. On lit desormais `stats.decode_seconds`
   apres chauffe, et l'on rend la mediane avec sa dispersion.
2. Le banc laissait les deux cartes visibles quand le serveur les epingle
   depuis e5cafc0. Un avertissement n'arretait rien : le chiffre produit ne
   portait plus sur la configuration demandee.
"""
import types

import pytest

from acvram import bench


class _PlanRejoue:
    replanifie_cartes = (["cuda:0"], ["cuda:0", "cuda:1"])
    est_decode_tok_s = 687.6


class _PlanNet:
    est_decode_tok_s = 687.6


class _Modele:
    nbytes = 4 * 2 ** 30


class _Charge:
    def __init__(self, plan):
        self.plan = plan
        self.model = _Modele()


def _charger(plan):
    return lambda *a, **k: _Charge(plan)


def test_refus_quand_le_plan_a_ete_rejoue(monkeypatch):
    """Le defaut par defaut est le refus : pas de debit, et la raison avec."""
    monkeypatch.delenv("ACVRAM_BANC_ACCEPTE_REPLAN", raising=False)
    monkeypatch.setattr("acvram.engine.loader.load_model", _charger(_PlanRejoue()))
    d = bench.bench_decode("/inexistant")
    assert "refus" in d, "le banc a publie un debit sur un plan rejoue"
    assert "cuda:1" in d["refus"], "le refus doit nommer la carte en trop"
    assert "decode_tok_s" not in d, "aucun debit ne doit sortir d'un refus"


def test_le_refus_se_laisse_forcer_explicitement(monkeypatch):
    """Une garde qu'on ne peut pas lever bloque un usage legitime.

    Le forcage doit etre EXPLICITE : sans lui, le silence vaut refus.
    """
    monkeypatch.setenv("ACVRAM_BANC_ACCEPTE_REPLAN", "1")
    monkeypatch.setattr("acvram.engine.loader.load_model", _charger(_PlanRejoue()))
    # Le chargement passe la garde ; la suite echoue faute de vrai moteur, ce
    # qui suffit a prouver que le refus n'a PAS ete rendu.
    with pytest.raises(Exception) as exc:
        bench.bench_decode("/inexistant")
    assert "refus" not in str(exc.value).lower()


def test_la_sonde_de_cartes_ne_plante_jamais():
    """Une sonde qui leve fait echouer la mesure qu'elle devait proteger."""
    assert bench._plusieurs_cartes() in (True, False)


def test_l_exil_force_invalide_l_estimation():
    """`_forcer_exil` deplace des MLP APRES que l'estimation a ete posee.

    Le plan annoncait 687,6 jetons/s pour un debit reel de 24,0 a seize
    couches exilees — un facteur 28,7. L'estimateur, lui, voit le placement
    (624,8 -> 30,1 quand la VRAM simulee tombe a 4,8 Gio) : c'est bien
    l'estimation figee qui ment, pas le modele de cout.
    """
    from acvram.engine.loader import _forcer_exil

    class _Couche:
        def __init__(self):
            self.mlp_storage = "gpu"
            self.mlp_exec = "gpu"

    class _Plan:
        def __init__(self):
            self.layers = [_Couche() for _ in range(8)]
            self.est_decode_tok_s = 687.6
            self.est_bytes_per_token = 1_745_879_040

    p = _Plan()
    _forcer_exil(p, 4)
    assert sum(1 for l in p.layers if l.mlp_storage == "cpu") == 4
    assert p.est_decode_tok_s == 0.0, "un debit prevu perime reste lisible"
    assert getattr(p, "estimation_perimee", None), "la raison doit etre dite"


def test_l_exil_par_expert_invalide_aussi_l_estimation():
    """Pendant de `_forcer_exil` au grain de l'expert (bead pds, point 4) :
    même garde — l'estimation figée avant coup ne vaut plus rien après."""
    from acvram.engine.loader import _forcer_exil_experts

    class _Couche:
        def __init__(self, index):
            self.index = index
            self.mlp_storage = "gpu"

    class _Plan:
        def __init__(self):
            self.layers = [_Couche(i) for i in range(4)]
            self.est_decode_tok_s = 687.6

    manifest = {"tensors": {
        f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight": None
        for i in range(4) for e in range(8)
    }}
    p = _Plan()
    _forcer_exil_experts(p, manifest, 0.5)
    assert all(l.experts_residents == 4 for l in p.layers)
    assert p.est_decode_tok_s == 0.0, "un debit prevu perime reste lisible"
    assert getattr(p, "estimation_perimee", None), "la raison doit etre dite"


def test_refus_quand_la_generation_s_interrompt(monkeypatch):
    """Un EOS des le premier jeton faisait publier un debit de demarrage.

    Sur nemotron-lightning-heretic, l'invite artificielle du banc ([1] repete)
    faisait emettre un EOS immediat : UN jeton produit sur 256 demandes, et le
    banc annoncait 2,85 jetons/s la ou le serveur en rend 211. `SamplingParams`
    n'a ni `ignore_eos` ni `min_tokens` — on refuse donc de publier plutot que
    de mesurer autre chose que ce qui a ete demande.
    """
    from acvram import bench as b

    etat = {"p": 0, "d": 0.0, "n": 0}

    class _Stats:
        def to_dict(self):
            return {"decode_seconds": etat["d"], "decode_tokens": etat["n"],
                    "prefill_tokens": etat["p"], "cached_prompt_tokens": 0}

    class _Eng:
        stats = _Stats()
        graphs = None

        def __init__(self, *a, **k):
            pass

        def generate(self, *a, **k):
            # L'invite est prefillee EN ENTIER : seule la generation
            # s'interrompt, sans quoi l'autre garde parlerait a sa place.
            etat["p"] += 128
            etat["n"] += 1
            etat["d"] += 0.001
            yield 1                     # UN seul jeton, puis EOS

    class _Plan:
        est_decode_tok_s = 687.6

    class _Charge:
        plan = _Plan()
        model = type("M", (), {"nbytes": 2**30})()
        spec = type("S", (), {"vocab_size": 32000})()

    monkeypatch.setattr("acvram.engine.loader.load_model",
                        lambda *a, **k: _Charge())
    monkeypatch.setattr("acvram.engine.runner.Engine", _Eng)
    d = b.bench_decode("/inexistant", n_tokens=256, prompt_len=128)
    assert "refus" in d, "un debit calcule sur 1 jeton a ete publie"
    assert d["generated"] == 1
    assert "decode_tok_s" not in d


def test_refus_quand_le_cache_de_prefixe_sert_l_invite(monkeypatch):
    """Une invite repetee fait prefiller 16 jetons sur 512.

    Le debit publie devient celui du reliquat, insensible a la taille du
    prompt — c'est ainsi qu'un « plateau de prefill a 560 j/s » a ete mesure
    et rapporte, alors que le vrai debit vaut 3 485 j/s a 512 jetons et 6 049
    a 2048. Le moteur exposait pourtant `cached_prompt_tokens` depuis
    toujours : la source pouvait parler, personne ne l'interrogeait.
    """
    from acvram import bench as b

    etat = {"p": 0, "c": 0, "d": 0.0, "n": 0}

    class _Stats:
        def to_dict(self):
            return {"decode_seconds": etat["d"], "decode_tokens": etat["n"],
                    "prefill_tokens": etat["p"],
                    "cached_prompt_tokens": etat["c"]}

    class _Eng:
        stats = _Stats()
        graphs = None

        def __init__(self, *a, **k):
            pass

        def generate(self, *a, **k):
            etat["p"] += 16          # 16 jetons prefilles...
            etat["c"] += 112         # ...112 servis par le cache
            for _ in range(256):
                etat["n"] += 1
                etat["d"] += 0.001
                yield 1

    class _Charge:
        plan = type("P", (), {"est_decode_tok_s": 687.6})()
        model = type("M", (), {"nbytes": 2 ** 30})()
        spec = type("S", (), {"vocab_size": 32000})()

    monkeypatch.setattr("acvram.engine.loader.load_model",
                        lambda *a, **k: _Charge())
    monkeypatch.setattr("acvram.engine.runner.Engine", _Eng)
    d = b.bench_decode("/inexistant", n_tokens=256, prompt_len=128)
    assert "refus" in d, "un debit a ete publie sur un prefill de 16/128"
    assert d["prefill_tokens_reels"] == 16
    assert d["prefill_tokens_caches"] == 112
    assert "decode_tok_s" not in d
