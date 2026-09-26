"""Pièce 236c (poste2, à sec, demande utilisateur) : `packaging/langues/_source.json` (fichier de base
monolingue pour Weblate, gabarit `locale/_source.json` d'animematrix) doit être EXACTEMENT
`{clé: clé}` pour chaque clé de `_cles.json` -- ni plus, ni moins, sinon Weblate propose une clé qui
n'existe plus dans le script, ou en oublie une qui existe. `test_langues_gui.py` garde déjà
`_cles.json` synchronisé avec les appels `T(...)` du script ; ce test garde `_source.json` synchronisé
avec `_cles.json` (transitivement, avec le script)."""
import json
import pathlib

RACINE = pathlib.Path(__file__).resolve().parent.parent
LANGUES = RACINE / "packaging" / "langues"


def test_source_json_est_l_identite_des_cles():
    cles = set(json.load(open(LANGUES / "_cles.json", encoding="utf-8")))
    source = json.load(open(LANGUES / "_source.json", encoding="utf-8"))
    manquent = sorted(cles - source.keys())
    orphelines = sorted(source.keys() - cles)
    assert not manquent and not orphelines, {
        "manquent dans _source.json": [k[:50] for k in manquent[:6]],
        "orphelines (plus dans _cles.json)": [k[:50] for k in orphelines[:6]]}
    faux = sorted(k for k, v in source.items() if v != k)
    assert not faux, {"valeur ≠ clé (pas une identité)": [k[:50] for k in faux[:6]]}


def test_le_test_sait_dire_faux(tmp_path, monkeypatch):
    """Une clé de _source.json qui diverge de son propre nom (faute de frappe, valeur traduite par
    erreur) rend rouge -- copie modifiée, pas le fichier réel."""
    source = json.load(open(LANGUES / "_source.json", encoding="utf-8"))
    copie = dict(source)
    une_cle = next(iter(copie))
    copie[une_cle] = copie[une_cle] + " (altérée pour le test)"
    faux = sorted(k for k, v in copie.items() if v != k)
    assert faux == [une_cle]
