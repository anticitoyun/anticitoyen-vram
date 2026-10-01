"""e50.3 § 3 (poste6, scellé) : barème étoiles ↔ S, gardes par palier, « format » prioritaire.
Cas repris directement de la méthode : un ★★★★★ du menu qui tombe ★★★ n'est pas corrigé,
publié tel quel (§ 3, dernier paragraphe) — ce test vérifie le CALCUL, pas une mesure réelle."""
import importlib.util
import os

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "qualite_e50_bareme", os.path.join(ICI, "outils", "qualite-e50-bareme.py"))
bareme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bareme)
etoile = bareme.etoile


def test_bornes_centrales():
    assert etoile(0.85, 0.8, 0.8, 0.8) == "★★★★★"
    assert etoile(0.80, 0.8, 0.8, 0.8) == "★★★★★"   # borne basse incluse
    assert etoile(0.799, 0.8, 0.8, 0.8) == "★★★★"
    assert etoile(0.70, 0.7, 0.7, 0.7) == "★★★★"
    assert etoile(0.65, 0.7, 0.7, 0.7) == "★★★★"
    assert etoile(0.649, 0.6, 0.6, 0.6) == "★★★"
    assert etoile(0.55, 0.5, 0.5, 0.5) == "★★★"
    assert etoile(0.50, 0.5, 0.5, 0.5) == "★★★"
    assert etoile(0.499, 0.5, 0.4, 0.5) == "★★"
    assert etoile(0.40, 0.4, 0.4, 0.4) == "★★"
    assert etoile(0.35, 0.35, 0.35, 0.35) == "★★"
    assert etoile(0.349, 0.3, 0.3, 0.4) == "★"
    assert etoile(0.0, 0.0, 0.0, 0.0) == "★"


def test_garde_5_etoiles_descend_dune_etoile_pas_plus():
    # S ≥ 0,80 mais une tâche < 0,60 : descend à ★★★★, jamais ★★★ ni plus bas
    assert etoile(0.85, 0.55, 0.9, 0.9) == "★★★★"
    assert etoile(0.90, 0.95, 0.95, 0.55) == "★★★★"


def test_garde_4_etoiles_descend_dune_etoile():
    # S dans [0,65 ; 0,80) mais une tâche < 0,40 : descend à ★★★
    assert etoile(0.70, 0.30, 0.9, 0.9) == "★★★"


def test_pas_de_garde_sous_trois_etoiles():
    # ★★★, ★★, ★ n'ont pas de garde par tâche (§ 3) : seul S compte
    assert etoile(0.55, 0.0, 1.0, 0.65) == "★★★"
    assert etoile(0.40, 0.0, 1.0, 0.20) == "★★"


def test_format_prioritaire_sur_le_calcul_de_s():
    # même un S excellent ne compte pas si l'instrument n'a pas lu le modèle
    assert bareme.etoile(0.95, 0.9, 0.9, 1.0, invalid_mmlu=50, n_mmlu=90) == "format"
    assert bareme.etoile(0.95, 0.9, 0.9, 1.0, reponses_vides=90, n_total=170) == "format"


def test_format_sous_le_seuil_de_moitie_ne_declenche_pas():
    assert etoile(0.70, 0.7, 0.7, 0.7) != "format"   # S normal, pas de format
    assert bareme.etoile(0.70, 0.7, 0.7, 0.7, invalid_mmlu=44, n_mmlu=90) == "★★★★"  # 48,9 % < 50 %


def test_exemple_de_la_methode_coder_30b():
    # § 6, E3 : S 0,75-0,88 → ★★★★ ou ★★★★★, HumanEval >= 0,80
    assert etoile(0.80, 0.78, 0.82, 0.80) == "★★★★★"
    assert etoile(0.78, 0.75, 0.80, 0.80) == "★★★★"
