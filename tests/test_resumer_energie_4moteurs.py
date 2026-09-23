"""resumer-energie-4moteurs.py (chaîne llama.cpp, qr.md § 2.5, 3e passage Maîtresse 22/09) :
le bridage PUISSANCE (SwPowerCap) est l'état normal d'un moteur rapide au plafond 400 W, il
ne doit pas rejeter une fenêtre -- seul le bridage THERMIQUE dérive réellement. Même défaut
et même correctif que outils/gpu/mesure/banc-4moteurs.py, mais CE script juge la chaîne
llama.cpp (banc-llamacpp-16-09.py) : `bridages` y est une CHAÎNE jointe par virgules
(`e.resume()`), pas un set — testé ici séparément."""
import importlib.util
import os

_CHEMIN = os.path.join(os.path.dirname(__file__), "..", "scratchpad", "laurine-b12-21-09",
                       "resumer-energie-4moteurs.py")
_spec = importlib.util.spec_from_file_location("resumer_energie_4moteurs", _CHEMIN)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _resultat(bridages="aucun", jetons_s=1500.0, horloge=2700, temp=75, watts=390.0,
             j_par_jeton=0.2, duree=20.0, invalidations="aucune"):
    n = round(jetons_s * duree)
    return {"moteur": "acvram", "jetons_s": jetons_s, "horloge_moy": horloge, "temp_max": temp,
           "watts": watts, "joules_net": round(j_par_jeton * n, 1), "j_par_jeton_net": j_par_jeton,
           "duree_mesure_s": duree, "bridages": bridages, "invalidations": invalidations}


def test_bridages_set_parse_la_chaine_jointe_par_virgules():
    assert mod.bridages_set({"bridages": "aucun"}) == set()
    assert mod.bridages_set({"bridages": "puissance"}) == {"puissance"}
    assert mod.bridages_set({"bridages": "puissance,thermique_materiel"}) == \
        {"puissance", "thermique_materiel"}


def test_charge_raison_invalidations_bridage_puissance_seul_ignore():
    """5e passage, Manon : charge_raison() (pas throttle()) juge la chaîne via
    e.invalidations (energie.py:428-429), qui ajoute TOUJOURS une entrée
    bridage même pour puissance seul -- le test qui casse si cette entrée
    reste rejetante."""
    d = _resultat(invalidations="bridage pendant la fenêtre : puissance")
    assert mod.charge_raison(d) is None


def test_charge_raison_invalidations_bridage_thermique_rejette():
    d = _resultat(invalidations="bridage pendant la fenêtre : thermique_materiel")
    assert mod.charge_raison(d) == "bridage pendant la fenêtre : thermique_materiel"


def test_charge_raison_invalidations_bridage_puissance_et_autre_raison():
    """Le bridage puissance est retiré, l'AUTRE raison (ici : processus changés)
    doit rester -- charge_raison() ne doit pas tout effacer d'un coup."""
    d = _resultat(invalidations="carte 0 : les processus ont changé ; "
                                 "bridage pendant la fenêtre : puissance")
    assert mod.charge_raison(d) == "carte 0 : les processus ont changé"


def test_calculer_raisons_invalidations_bridage_puissance_seul_fenetre_valide():
    fenetres = {"acvram-b12-1": _resultat(
        invalidations="bridage pendant la fenêtre : puissance")}
    raisons = mod.calculer_raisons(fenetres)
    assert raisons["acvram-b12-1"] == []


def test_throttle_bridage_puissance_seul_ne_rejette_pas():
    """Le test qui casse si une fenêtre bridée puissance est rejetée à tort."""
    assert mod.throttle(_resultat(bridages="puissance")) is False


def test_throttle_bridage_thermique_rejette():
    """Le test qui casse si une fenêtre bridée thermique passe à tort."""
    assert mod.throttle(_resultat(bridages="thermique_materiel")) is True
    assert mod.throttle(_resultat(bridages="thermique_logiciel")) is True


def test_throttle_bridage_puissance_et_thermique_rejette():
    assert mod.throttle(_resultat(bridages="puissance,thermique_materiel")) is True


def test_calculer_raisons_bridage_puissance_seul_fenetre_valide():
    fenetres = {"acvram-b12-1": _resultat(bridages="puissance")}
    raisons = mod.calculer_raisons(fenetres)
    assert raisons["acvram-b12-1"] == []


def test_calculer_raisons_bridage_thermique_fenetre_invalide():
    fenetres = {"acvram-b12-1": _resultat(bridages="thermique_materiel")}
    raisons = mod.calculer_raisons(fenetres)
    assert any("throttle" in r for r in raisons["acvram-b12-1"])


def test_ecrire_protocole25_publie_watts_moy(tmp_path):
    fenetres = {"acvram-b12-1": _resultat(bridages="puissance", watts=391.2)}
    raisons = mod.calculer_raisons(fenetres)
    mod.ecrire_protocole25(str(tmp_path), "shatest", fenetres, raisons)
    lignes = (tmp_path / "protocole25.tsv").read_text().splitlines()
    assert lignes[0].split("\t") == ["moteur", "sha", "horloge", "temp", "charge", "J", "t_s",
                                     "date", "duree", "valide", "j_par_jeton_net",
                                     "j_par_jeton_par_mhz", "watts_moy", "raisons"]
    champs = lignes[1].split("\t")
    assert champs[0] == "acvram"
    assert champs[9] == "oui"          # valide : bridage puissance seul n'invalide pas
    assert champs[12] == "391.2"       # watts_moy publié


def test_ecrire_protocole25_fenetre_ratee_watts_moy_point_dinterrogation(tmp_path):
    fenetres = {"acvram-b1-2": None}
    raisons = mod.calculer_raisons(fenetres)
    mod.ecrire_protocole25(str(tmp_path), "shatest", fenetres, raisons)
    lignes = (tmp_path / "protocole25.tsv").read_text().splitlines()
    champs = lignes[1].split("\t")
    assert champs[9] == "non" and champs[12] == "?"


def test_main_bout_en_bout_sur_fichiers_synthetiques(tmp_path):
    (tmp_path / "fenetre-acvram-b12-1.log").write_text(
        "RESULTAT " + __import__("json").dumps(_resultat(bridages="puissance")) + "\n")
    mod.main(str(tmp_path), "shabidon")
    assert (tmp_path / "protocole25.tsv").exists()
    assert (tmp_path / "protocole25.tsv.resume.tsv").exists()
    resume = (tmp_path / "protocole25.tsv.resume.tsv").read_text().splitlines()
    assert resume[1].split("\t")[0] == "acvram"
    assert resume[1].split("\t")[1] == "1"   # une fenêtre valide malgré le bridage puissance
