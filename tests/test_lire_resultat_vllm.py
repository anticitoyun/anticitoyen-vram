"""Défaut du 22/09 (Manon) : sortie vLLM tronquée systématique sur 3/3 fenêtres, silencieusement
ignorée par un `grep` sans validation. Preuve à sec que le lecteur casse au lieu de continuer."""
import importlib.util
import json
import os

import pytest

_CHEMIN = os.path.join(os.path.dirname(__file__), "..", "scratchpad", "laurine-b12-21-09",
                       "lire-resultat-vllm.py")
_spec = importlib.util.spec_from_file_location("lire_resultat_vllm_mod", _CHEMIN)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _bon_dict():
    return {"moteur": "vllm", "slots": 12, "jetons_s": 1596.1, "lots": 3,
            "prefill_jetons": 9216, "n_jetons_decodes": 12288}


def test_ligne_valide_se_lit(tmp_path):
    log = tmp_path / "fenetre.log"
    log.write_text("[bras] regime ...\nRESULTAT " + json.dumps(_bon_dict()) + "\n")
    r = mod.lire_resultat_vllm(str(log))
    assert r["jetons_s"] == 1596.1 and r["lots"] == 3


def test_ligne_tronquee_exactement_comme_le_22_09_casse(tmp_path):
    """Reproduit le défaut observé : coupé net après '"lots": 3, "prefill' — sans valeur ni
    fermeture. Le lecteur DOIT lever ResultatTronque, jamais rendre un dict partiel ou vide."""
    log = tmp_path / "fenetre.log"
    log.write_text('[bras] regime ...\nRESULTAT {"moteur": "vllm", "slots": 12, "lots": 3, "prefill')
    with pytest.raises(mod.ResultatTronque, match="JSON invalide"):
        mod.lire_resultat_vllm(str(log))


def test_aucune_ligne_resultat_casse_nomme(tmp_path):
    log = tmp_path / "fenetre.log"
    log.write_text("[bras] regime ...\n[bras] warm_graphs 5\n")   # processus tué avant le print
    with pytest.raises(mod.ResultatTronque, match="aucune ligne"):
        mod.lire_resultat_vllm(str(log))


def test_log_absent_casse_nomme(tmp_path):
    with pytest.raises(mod.ResultatTronque, match="absent"):
        mod.lire_resultat_vllm(str(tmp_path / "n-existe-pas.log"))


def test_derniere_ligne_resultat_retenue_si_plusieurs(tmp_path):
    """Un run à plusieurs SLOTS écrit plusieurs lignes RESULTAT : la dernière est celle du run
    en cours (comportement de `banc-llamacpp-16-09.py`/scripts voisins, cohérent)."""
    log = tmp_path / "fenetre.log"
    d1 = {"slots": 1, "jetons_s": 300.0}
    d2 = {"slots": 12, "jetons_s": 1596.1}
    log.write_text("RESULTAT " + json.dumps(d1) + "\nRESULTAT " + json.dumps(d2) + "\n")
    r = mod.lire_resultat_vllm(str(log))
    assert r["slots"] == 12 and r["jetons_s"] == 1596.1


def test_sidecar_prefere_au_print_meme_si_print_tronque(tmp_path):
    """Le sidecar atomique (os.replace côté script) est préféré à la ligne print du log — même
    si CETTE DERNIÈRE est tronquée, le sidecar (complet ou absent, jamais partiel) prime."""
    log = tmp_path / "fenetre.log"
    log.write_text('RESULTAT {"slots": 12, "lots": 3, "prefill')          # tronqué, comme le 22/09
    base = str(tmp_path / "sortie")
    sidecar = tmp_path / "sortie.12.json"
    sidecar.write_text(json.dumps({"slots": 12, "jetons_s": 1596.1, "lots": 3}))
    r = mod.lire_resultat_vllm(str(log), base_sidecar=base, slots=12)
    assert r["jetons_s"] == 1596.1                     # vient du sidecar, pas de la ligne tronquée


def test_sidecar_absent_retombe_sur_le_log(tmp_path):
    log = tmp_path / "fenetre.log"
    log.write_text("RESULTAT " + json.dumps(_bon_dict()) + "\n")
    r = mod.lire_resultat_vllm(str(log), base_sidecar=str(tmp_path / "jamais-ecrit"), slots=12)
    assert r["jetons_s"] == 1596.1


def test_sidecar_lui_meme_vide_ne_masque_pas_le_defaut(tmp_path):
    """Un sidecar présent mais JSON invalide (écriture concurrente incomplète malgré
    os.replace, cas pathologique) doit aussi casser, pas retomber silencieusement sur le log."""
    log = tmp_path / "fenetre.log"
    log.write_text("RESULTAT " + json.dumps(_bon_dict()) + "\n")
    base = str(tmp_path / "sortie")
    (tmp_path / "sortie.12.json").write_text("{pas du json")
    with pytest.raises(json.JSONDecodeError):
        mod.lire_resultat_vllm(str(log), base_sidecar=base, slots=12)
