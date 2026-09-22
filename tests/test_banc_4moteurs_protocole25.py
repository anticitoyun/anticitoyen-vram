"""Protocole 2.5 (`outils/gpu/mesure/banc-4moteurs.py`, qr.md § 2.5) : les fonctions pures
(_agreger_fenetre, _joules_net, _valider_*) sur des traces SYNTHÉTIQUES — aucun accès carte,
pas de verrou requis. `mesurer_fenetre_protocole25`/`comparer_alternee` (E/S GPU) ne sont pas
testées ici : leur logique propre est nulle, elles appellent ces fonctions."""
import importlib.util
import os
import pytest

_CHEMIN = os.path.join(os.path.dirname(__file__), "..", "outils", "gpu", "mesure", "banc-4moteurs.py")
_spec = importlib.util.spec_from_file_location("banc_4moteurs", _CHEMIN)
banc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(banc)


def _sous_passage(n=100, duree=10.0, joules=500.0, horloges=(2700, 2700), temperatures=(60, 61),
                  bridages=frozenset(), debit=10.0):
    return {"n": n, "duree": duree, "joules": joules, "horloges": list(horloges),
            "temperatures": list(temperatures), "bridages": set(bridages), "debit": debit}


def test_agreger_fenetre_additive():
    sp = [_sous_passage(n=100, duree=10.0, joules=500.0, debit=10.0),
          _sous_passage(n=110, duree=11.0, joules=550.0, debit=10.0)]
    agg = banc._agreger_fenetre(sp)
    assert agg["n"] == 210
    assert agg["duree"] == pytest.approx(21.0)
    assert agg["joules"] == pytest.approx(1050.0)
    assert agg["t_s"] == pytest.approx(210 / 21.0, rel=1e-6)
    assert agg["sd_pct"] == pytest.approx(0.0)        # deux débits identiques
    assert agg["throttle"] is False


def test_agreger_fenetre_horloge_mediane_et_temp_max():
    sp = [_sous_passage(horloges=(2600, 2650), temperatures=(58, 59)),
          _sous_passage(horloges=(2700, 2750), temperatures=(60, 62))]
    agg = banc._agreger_fenetre(sp)
    assert agg["horloge_med"] == 2675.0          # médiane de [2600,2650,2700,2750]
    assert agg["temp_max"] == 62


def test_agreger_fenetre_throttle_si_un_seul_sous_passage_bride():
    sp = [_sous_passage(bridages=frozenset()), _sous_passage(bridages=frozenset({"SwPowerCap"}))]
    agg = banc._agreger_fenetre(sp)
    assert agg["throttle"] is True


def test_agreger_fenetre_ignore_horloges_negatives():
    sp = [_sous_passage(horloges=(-1, 2700), temperatures=(-1, 60))]
    agg = banc._agreger_fenetre(sp)
    assert agg["horloge_med"] == 2700
    assert agg["temp_max"] == 60


def test_agreger_fenetre_vide_rend_moins_un():
    agg = banc._agreger_fenetre([])
    assert agg["horloge_med"] == -1 and agg["temp_max"] == -1 and agg["n"] == 0
    assert agg["t_s"] == 0.0


def test_joules_net_soustrait_le_repos():
    agg = {"joules": 500.0, "duree": 20.0}
    assert banc._joules_net(agg, watts_repos=10.0) == pytest.approx(300.0)   # 500 - 10*20


def test_joules_net_ne_descend_pas_sous_zero():
    agg = {"joules": 10.0, "duree": 20.0}
    assert banc._joules_net(agg, watts_repos=10.0) == 0.0                    # 10 - 200 < 0, borne à 0


def test_sd_relatif_seuil_10_pourcent():
    # débits [9.4, 10.6] -> moyenne 10, pstdev 0.6 -> sd% = 6 % : sous le seuil
    sp = [_sous_passage(debit=9.4), _sous_passage(debit=10.6)]
    agg = banc._agreger_fenetre(sp)
    assert agg["sd_pct"] == pytest.approx(6.0, abs=0.05)
    assert banc._valider_fenetre(agg) == []


def test_sd_relatif_rejette_au_dessus_de_10_pourcent():
    # débits [8, 12] -> moyenne 10, pstdev 2 -> sd% = 20 %
    sp = [_sous_passage(debit=8.0), _sous_passage(debit=12.0)]
    agg = banc._agreger_fenetre(sp)
    assert agg["sd_pct"] == pytest.approx(20.0, abs=0.05)
    raisons = banc._valider_fenetre(agg)
    assert len(raisons) == 1 and "sd" in raisons[0]


def test_valider_fenetre_throttle_rejette_meme_avec_sd_nul():
    sp = [_sous_passage(debit=10.0, bridages=frozenset({"HwSlowdown"})),
          _sous_passage(debit=10.0)]
    agg = banc._agreger_fenetre(sp)
    raisons = banc._valider_fenetre(agg)
    assert any("throttle" in r for r in raisons)


def test_valider_fenetre_cumule_sd_et_throttle():
    sp = [_sous_passage(debit=8.0, bridages=frozenset({"HwSlowdown"})),
          _sous_passage(debit=12.0)]
    agg = banc._agreger_fenetre(sp)
    raisons = banc._valider_fenetre(agg)
    assert len(raisons) == 2


def test_valider_charge_sous_seuil():
    assert banc._valider_charge(4.9) == []


def test_valider_charge_au_dessus_du_seuil():
    raisons = banc._valider_charge(5.1)
    assert len(raisons) == 1 and "5.1" in raisons[0]


def test_valider_charge_none_ne_rejette_pas():
    assert banc._valider_charge(None) == []


def test_valider_paire_ecart_sous_3_pourcent():
    assert banc._valider_paire(2700, 2700 * 1.02) == []


def test_valider_paire_ecart_au_dessus_de_3_pourcent():
    raisons = banc._valider_paire(2700, 2700 * 1.05)
    assert len(raisons) == 1 and "écart horloge" in raisons[0]


def test_valider_paire_horloge_indisponible():
    assert banc._valider_paire(-1, 2700) != []
    assert banc._valider_paire(None, 2700) != []


def test_comparer_alternee_ordre_et_tsv(tmp_path, monkeypatch):
    """Substitue `mesurer_fenetre_protocole25` (E/S GPU) par une trace synthétique
    ordonnée A/B/A/B/A/B : vérifie l'alternance, l'appariement horloge, et le TSV."""
    appels = []

    def _faux(moteur, fenetre_s=20.0):
        appels.append(moteur)
        i = len(appels)
        horloge = 2700 if moteur == "acvram" else 2650   # écart 1,85 %, sous le seuil 3 %
        return {"moteur": moteur, "n": 1000, "duree": 20.0, "joules": 400.0, "joules_net": 300.0,
                "t_s": 50.0 + i, "sd_pct": 1.0, "horloge_med": horloge, "temp_max": 60,
                "throttle": False, "charge_pct": None, "raisons": []}

    monkeypatch.setattr(banc, "mesurer_fenetre_protocole25", _faux)
    sortie = tmp_path / "protocole25.tsv"
    fenetres = banc.comparer_alternee(["acvram", "vllm"], sha="abc1234", n_fenetres=6,
                                      sortie_tsv=str(sortie))
    assert appels == ["acvram", "vllm", "acvram", "vllm", "acvram", "vllm"]
    assert all(not f["raisons"] for f in fenetres)          # écart d'horloge 1,85 % : aucune paire rejetée
    lignes = sortie.read_text().splitlines()
    assert lignes[0] == "moteur\tsha\thorloge\ttemp\tcharge\tJ\tt_s\tdate\tduree\tvalide\traisons"
    assert len(lignes) == 7                                  # en-tête + 6 fenêtres
    assert lignes[1].split("\t")[0] == "acvram" and lignes[1].split("\t")[1] == "abc1234"


def test_comparer_alternee_paire_invalidee_par_horloge(tmp_path, monkeypatch):
    def _faux(moteur, fenetre_s=20.0):
        horloge = 2700 if moteur == "acvram" else 2500      # écart 7,4 %, au-dessus du seuil
        return {"moteur": moteur, "n": 1000, "duree": 20.0, "joules": 400.0, "joules_net": 300.0,
                "t_s": 50.0, "sd_pct": 1.0, "horloge_med": horloge, "temp_max": 60,
                "throttle": False, "charge_pct": None, "raisons": []}

    monkeypatch.setattr(banc, "mesurer_fenetre_protocole25", _faux)
    sortie = tmp_path / "protocole25.tsv"
    fenetres = banc.comparer_alternee(["acvram", "vllm"], sha="abc1234", n_fenetres=6,
                                      sortie_tsv=str(sortie))
    assert all(f["raisons"] for f in fenetres)               # toutes les paires invalidées (écart constant)
    lignes = sortie.read_text().splitlines()
    assert all(l.split("\t")[-2] == "non" for l in lignes[1:])   # colonne "valide" = non partout


def test_comparer_alternee_refuse_moins_de_6_fenetres():
    with pytest.raises(AssertionError):
        banc.comparer_alternee(["acvram", "vllm"], sha="x", n_fenetres=4)
