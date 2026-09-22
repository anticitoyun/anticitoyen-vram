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
                  bridages=frozenset(), debit=10.0, load1_pic=None, nproc=None):
    return {"n": n, "duree": duree, "joules": joules, "horloges": list(horloges),
            "temperatures": list(temperatures), "bridages": set(bridages), "debit": debit,
            "load1_pic": load1_pic, "nproc": nproc}


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


def test_agreger_fenetre_throttle_si_un_seul_sous_passage_bride_thermique():
    """Bridage THERMIQUE (nom réel `energie.py` : "thermique_materiel") : un
    seul sous-passage suffit à rejeter toute la fenêtre — une carte qui dérive
    dérive pour tout le monde, pas seulement le sous-passage où on l'a vue."""
    sp = [_sous_passage(bridages=frozenset()), _sous_passage(bridages=frozenset({"thermique_materiel"}))]
    agg = banc._agreger_fenetre(sp)
    assert agg["throttle"] is True


def test_agreger_fenetre_bridage_puissance_seul_ne_declenche_pas_throttle():
    """Contrôle de conformité qr.md § 2.5 (chef 22/09, 2e passage) : le
    bridage PUISSANCE (SwPowerCap, nom réel "puissance") est l'état normal
    d'un moteur rapide au plafond 400 W -- il ne doit JAMAIS déclencher
    `throttle`, à la différence des bridages thermiques."""
    sp = [_sous_passage(bridages=frozenset({"puissance"})),
          _sous_passage(bridages=frozenset({"puissance"}))]
    agg = banc._agreger_fenetre(sp)
    assert agg["throttle"] is False
    assert agg["bridage_puissance"] is True


def test_valider_fenetre_ne_rejette_pas_le_bridage_puissance():
    """Le test qui casse si une fenêtre bridée PUISSANCE est rejetée à tort."""
    sp = [_sous_passage(debit=10.0, bridages=frozenset({"puissance"})),
          _sous_passage(debit=10.0, bridages=frozenset({"puissance"}))]
    agg = banc._agreger_fenetre(sp)
    assert banc._valider_fenetre(agg) == []


def test_valider_fenetre_rejette_le_bridage_thermique():
    """Le test qui casse si une fenêtre bridée THERMIQUE passe à tort."""
    sp = [_sous_passage(debit=10.0, bridages=frozenset({"thermique_logiciel"})),
          _sous_passage(debit=10.0)]
    agg = banc._agreger_fenetre(sp)
    raisons = banc._valider_fenetre(agg)
    assert any("throttle" in r for r in raisons)


def test_agreger_fenetre_watts_moy():
    sp = [_sous_passage(duree=10.0, joules=500.0), _sous_passage(duree=10.0, joules=300.0)]
    agg = banc._agreger_fenetre(sp)
    assert agg["watts_moy"] == pytest.approx(800.0 / 20.0)  # 40.0 W


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
    sp = [_sous_passage(debit=10.0, bridages=frozenset({"thermique_materiel"})),
          _sous_passage(debit=10.0)]
    agg = banc._agreger_fenetre(sp)
    raisons = banc._valider_fenetre(agg)
    assert any("throttle" in r for r in raisons)


def test_valider_fenetre_cumule_sd_et_throttle():
    sp = [_sous_passage(debit=8.0, bridages=frozenset({"thermique_materiel"})),
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
    assert lignes[0] == "moteur\tsha\thorloge\ttemp\tcharge\tJ\tt_s\tdate\tduree\tvalide\twatts_moy\traisons"
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
    assert all(l.split("\t")[-3] == "non" for l in lignes[1:])   # colonne "valide" = non partout


def test_comparer_alternee_refuse_moins_de_6_fenetres():
    with pytest.raises(AssertionError):
        banc.comparer_alternee(["acvram", "vllm"], sha="x", n_fenetres=4)


# --- contrôle de conformité qr.md § 2.5 (chef 22/09) -------------------
# SHA du modèle, dérive thermique, charge étrangère mesurée, médiane publiée
# par moteur : quatre points absents ou non câblés à l'audit, ajoutés ici.

def test_sha256_modele_stable_et_deterministe(tmp_path):
    d = tmp_path / "modele"
    d.mkdir()
    (d / "model-00001-of-00002.safetensors").write_bytes(b"a" * 1000)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"b" * 2000)
    (d / "README.md").write_text("ignoré : pas un poids")
    sha1 = banc.sha256_modele(str(d))
    sha2 = banc.sha256_modele(str(d))
    assert sha1 == sha2 and len(sha1) == 64


def test_sha256_modele_change_si_un_shard_change_de_taille(tmp_path):
    d = tmp_path / "modele"
    d.mkdir()
    (d / "w.gguf").write_bytes(b"x" * 1000)
    avant = banc.sha256_modele(str(d))
    (d / "w.gguf").write_bytes(b"x" * 1001)
    apres = banc.sha256_modele(str(d))
    assert avant != apres


def test_sha256_modele_independant_de_l_ordre_de_decouverte(tmp_path):
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "a.safetensors").write_bytes(b"1" * 500)
    (d1 / "b.safetensors").write_bytes(b"2" * 700)
    (d2 / "b.safetensors").write_bytes(b"2" * 700)
    (d2 / "a.safetensors").write_bytes(b"1" * 500)
    assert banc.sha256_modele(str(d1)) == banc.sha256_modele(str(d2))


def test_sha256_modele_dossier_absent_ou_vide():
    assert banc.sha256_modele("/n/existe/pas/vraiment") == ""


def test_valider_derive_thermique_sous_le_seuil():
    assert banc._valider_derive_thermique(60, 64) == []          # 4 °C <= 5 °C


def test_valider_derive_thermique_au_dessus_du_seuil():
    raisons = banc._valider_derive_thermique(58, 66)              # 8 °C > 5 °C
    assert len(raisons) == 1 and "dérive thermique" in raisons[0]


def test_valider_derive_thermique_non_mesuree_ne_rejette_pas():
    assert banc._valider_derive_thermique(-1, 70) == []
    assert banc._valider_derive_thermique(60, -1) == []


def test_agreger_fenetre_expose_temp_debut_fin_dans_l_ordre():
    sp = [_sous_passage(temperatures=(58, 59)), _sous_passage(temperatures=(60, 66))]
    agg = banc._agreger_fenetre(sp)
    assert agg["temp_debut"] == 58 and agg["temp_fin"] == 66


def test_valider_fenetre_rejette_la_derive_thermique():
    sp = [_sous_passage(temperatures=(58,)), _sous_passage(temperatures=(70,))]
    agg = banc._agreger_fenetre(sp)
    raisons = banc._valider_fenetre(agg)
    assert any("dérive thermique" in r for r in raisons)


def test_agreger_fenetre_charge_pic_pct_absent_sans_mesure():
    sp = [_sous_passage(), _sous_passage()]
    agg = banc._agreger_fenetre(sp)
    assert agg["charge_pic_pct"] is None


def test_agreger_fenetre_charge_pic_pct_calcule_le_max():
    sp = [_sous_passage(load1_pic=4.0, nproc=32), _sous_passage(load1_pic=20.0, nproc=32)]
    agg = banc._agreger_fenetre(sp)
    assert agg["charge_pic_pct"] == pytest.approx(100 * 20.0 / 32, abs=0.05)


def test_mesurer_fenetre_protocole25_lit_le_pic_charge_automatiquement(monkeypatch):
    """Câblage manquant à l'audit : `charge_pct` restait toujours None faute
    de lecture du pic `Energie.load1_max`/`nproc` -- `_valider_charge` était
    donc un no-op muet en pratique, jamais un test alimenté."""
    class _FauxEnergie:
        duree = 10.0
        joules = 100.0
        horloges = [2700]
        temperatures = [60]
        bridages = set()
        load1_max = 20.0
        nproc = 32

    def _faux_generer(moteur):
        return 50, 0.1, 5.0, _FauxEnergie(), "texte", 50, "moteur"

    def _faux_repos(secondes):
        class _Base:
            moyenne = 10.0
        return _Base()

    monkeypatch.setattr(banc, "generer", _faux_generer)
    monkeypatch.setattr(banc, "repos", _faux_repos)
    agg = banc.mesurer_fenetre_protocole25("acvram", fenetre_s=0.01)
    assert agg["charge_pct"] == pytest.approx(100 * 20.0 / 32, abs=0.05)
    assert any("charge étrangère" in r for r in agg["raisons"])   # 62,5 % > 5 %


def test_resumer_moteur_mediane_sur_fenetres_valides_seulement():
    fenetres = [
        {"moteur": "acvram", "joules_net": 300.0, "n": 1000, "t_s": 1500.0, "raisons": []},
        {"moteur": "acvram", "joules_net": 320.0, "n": 1000, "t_s": 1520.0, "raisons": []},
        {"moteur": "acvram", "joules_net": 999.0, "n": 1000, "t_s": 1.0, "raisons": ["sd 20 % > 10 %"]},
        {"moteur": "vllm", "joules_net": 250.0, "n": 1000, "t_s": 1600.0, "raisons": []},
    ]
    r = banc.resumer_moteur(fenetres, "acvram")
    assert r["n_fenetres_valides"] == 2
    assert r["j_par_jeton_med"] == pytest.approx((0.30 + 0.32) / 2, abs=1e-6)
    assert r["t_s_med"] == pytest.approx((1500.0 + 1520.0) / 2)


def test_resumer_moteur_aucune_fenetre_valide_rend_none():
    fenetres = [{"moteur": "acvram", "joules_net": 300.0, "n": 1000, "t_s": 1500.0,
                "raisons": ["throttle actif"]}]
    assert banc.resumer_moteur(fenetres, "acvram") is None


def test_comparer_alternee_ecrit_le_resume_par_moteur(tmp_path, monkeypatch):
    def _faux(moteur, fenetre_s=20.0):
        return {"moteur": moteur, "n": 1000, "duree": 20.0, "joules": 400.0, "joules_net": 300.0,
                "t_s": 1500.0, "sd_pct": 1.0, "horloge_med": 2700, "temp_max": 60,
                "throttle": False, "charge_pct": None, "raisons": []}

    monkeypatch.setattr(banc, "mesurer_fenetre_protocole25", _faux)
    sortie = tmp_path / "protocole25.tsv"
    banc.comparer_alternee(["acvram", "vllm"], sha="abc1234", n_fenetres=6, sortie_tsv=str(sortie))
    resume = (tmp_path / "protocole25.tsv.resume.tsv").read_text().splitlines()
    assert resume[0] == "moteur\tn_fenetres_valides\tj_par_jeton_med\tt_s_med"
    lignes = {l.split("\t")[0]: l for l in resume[1:]}
    assert lignes["acvram"].split("\t")[1] == "3"          # 3 fenêtres sur 6, alternance stricte
    assert lignes["vllm"].split("\t")[1] == "3"
