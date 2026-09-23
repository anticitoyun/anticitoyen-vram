"""sage-expert-partage-cle-portee-17-09, REGLES §4 : « toute mesure porte
son régime dans son en-tête, pas dans un nom de fichier ou une mémoire ».
Un correctif du convertisseur (par exemple `0e4ef38`, calibration
nemotron_h) change le régime des convertis qu'il produit -- deux
manifestes au même nom peuvent venir de deux commits différents, et rien
avant ce champ ne le distinguait.
"""
import json
import os

from acvram.quant.convert import _convertisseur_commit


def test_le_manifeste_porte_le_commit_du_convertisseur(converted):
    """Un changement qui doit casser : retirer la clé `convertisseur` de la
    construction du manifeste (`convert_checkpoint`) fait rougir ce test."""
    manifeste = json.loads(open(os.path.join(converted, "acvram_manifest.json")).read())
    assert "convertisseur" in manifeste
    info = manifeste["convertisseur"]
    # Ce dépôt EST une extraction git (sinon `_convertisseur_commit` rendrait
    # None, un repli honnête -- voir le second test) : le commit doit donc
    # être un vrai sha, pas un défaut deviné.
    assert info is not None
    assert len(info["commit"]) == 40
    assert isinstance(info["arbre_modifie"], bool)


def test_hors_extraction_git_le_champ_est_none_pas_devine(tmp_path, monkeypatch):
    """Repli honnête (REGLES §10) : une installation figée (pas de `.git`)
    ne doit PAS inventer un commit -- `None`, jamais une valeur par défaut
    qui aurait l'air d'une mesure."""
    import acvram.quant.convert as conv

    faux_paquet = tmp_path / "faux_acvram" / "acvram" / "quant"
    faux_paquet.mkdir(parents=True)
    monkeypatch.setattr(conv, "__file__", str(faux_paquet / "convert.py"))
    assert _convertisseur_commit() is None
