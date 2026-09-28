"""Pièce 295 (chef, 28/09, sur la 0.7.11 à venir) : la garde 74 de
sortir-version.sh (README.md doit porter la version qu'on sort) est à refaire
à CHAQUE sortie sur 32 fichiers (README.md + 31 traductions) — un outil
plutôt qu'un geste manuel répété. `outils/bumper-version-readme.py` bascule
le numéro de version dans la section État de chaque fichier, RIEN d'autre :
jamais un chiffre ancré (dates, débits, contexte), jamais le mot « Version »
qui reste dans la langue de chaque traduction (testé sur un mot traduit,
« Versión »), jamais une occurrence de l'ancienne version HORS de la section
État (les notes « Résultats mesurés » citent une version d'époque, à
laisser intacte)."""
import importlib.util
from pathlib import Path

DEPOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("bumper_version_readme", DEPOT / "outils" / "bumper-version-readme.py")
bvr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bvr)


FR = """# titre

<a id="etat"></a>

## État

Version 0.7.10. Tout tourne sur la 5090.

6 752 tests.

---

<a id="credits"></a>

## Crédits
"""

ES = """# titulo

<a id="etat"></a>

## Estado

Versión 0.7.10. Todo funciona en la 5090.

6 752 pruebas.

---

<a id="credits"></a>

## Créditos
"""


def _ecrire(tmp_path, nom, contenu):
    f = tmp_path / nom
    f.write_text(contenu, encoding="utf-8")
    return f


def test_bascule_le_numero_partout_garde_le_mot_traduit(tmp_path):
    fr = _ecrire(tmp_path, "README.md", FR)
    es = _ecrire(tmp_path, "README.es.md", ES)
    plan, refus = bvr.plan_bascule("0.7.10", "0.7.11", chemins=[fr, es])
    assert not refus, refus
    assert "Version 0.7.11." in plan[fr]
    assert "Versión 0.7.11." in plan[es]      # le mot espagnol intact, seul le chiffre bouge
    assert "6 752" in plan[fr] and "6 752" in plan[es]   # un autre chiffre ancré, jamais touché


def test_essai_a_blanc_ne_touche_a_rien(tmp_path):
    fr = _ecrire(tmp_path, "README.md", FR)
    avant = fr.read_text(encoding="utf-8")
    plan, refus = bvr.plan_bascule("0.7.10", "0.7.11", chemins=[fr])
    assert not refus
    assert fr.read_text(encoding="utf-8") == avant     # plan_bascule seul n'écrit jamais


def test_ne_touche_pas_une_ancienne_version_hors_section_etat():
    texte = FR.replace(
        "Tout tourne sur la 5090.",
        "Tout tourne sur la 5090.\n\n---\n\n<a id=\"resultats\"></a>\n\n## Résultats\n\n"
        "acvram 0.7.10 (note historique, jamais la version courante).",
    )
    # « 0.7.10 » apparaît maintenant DEUX fois dans le fichier, mais UNE seule dans
    # la section État (la note « Résultats » est HORS section, entre 2 ancres
    # suivantes) — la coupure au bloc État doit l'ignorer.
    import tempfile
    d = Path(tempfile.mkdtemp())
    f = d / "README.md"
    f.write_text(texte, encoding="utf-8")
    plan, refus = bvr.plan_bascule("0.7.10", "0.7.11", chemins=[f])
    assert not refus, refus
    assert "Version 0.7.11." in plan[f]
    assert "acvram 0.7.10 (note historique" in plan[f]   # note d'époque intacte


def test_refuse_si_absente_de_la_section_etat(tmp_path):
    f = _ecrire(tmp_path, "README.md", FR.replace("0.7.10", "0.7.9"))
    plan, refus = bvr.plan_bascule("0.7.10", "0.7.11", chemins=[f])
    assert f not in plan
    assert len(refus) == 1 and "absente" in refus[0][1]


def test_refuse_si_ambigue_deux_occurrences_dans_etat(tmp_path):
    doublee = FR.replace(
        "Version 0.7.10. Tout tourne sur la 5090.",
        "Version 0.7.10. Tout tourne sur la 5090 (déjà 0.7.10 au démarrage).",
    )
    f = _ecrire(tmp_path, "README.md", doublee)
    plan, refus = bvr.plan_bascule("0.7.10", "0.7.11", chemins=[f])
    assert f not in plan
    assert len(refus) == 1 and "2 fois" in refus[0][1]


def test_un_fichier_en_defaut_n_empeche_pas_les_autres(tmp_path):
    fr = _ecrire(tmp_path, "README.md", FR)
    casse = _ecrire(tmp_path, "README.zz.md", FR.replace("0.7.10", "0.7.9"))
    plan, refus = bvr.plan_bascule("0.7.10", "0.7.11", chemins=[fr, casse])
    assert fr in plan and "Version 0.7.11." in plan[fr]
    assert casse not in plan
    assert len(refus) == 1 and refus[0][0] == casse


# ---- journal (chef, 28/09, sur la 0.7.11 réelle) : jamais dans l'arbre du dépôt --------------

def test_journal_par_defaut_hors_de_larbre(tmp_path):
    """CLI réel, versions introuvables (0.0.0/0.0.1 ne matchent rien) : chaque
    fichier réel refuse en « absente », donc rien n'est jamais écrit — seul le
    JOURNAL est sous contrôle ici. HOME simulé pour ne jamais écrire dans le
    vrai ~/.cache de la session."""
    import os
    import subprocess

    faux_home = tmp_path / "home"
    r = subprocess.run(
        ["python3", str(DEPOT / "outils" / "bumper-version-readme.py"), "0.0.0", "0.0.1"],
        cwd=DEPOT, capture_output=True, text=True, timeout=30,
        env={**os.environ, "HOME": str(faux_home)},
    )
    assert r.returncode == 1, r.stdout + r.stderr   # 32 refus (« absente »), aucun fichier basculé
    journal = faux_home / ".cache" / "acvram" / "bumper-version-readme.journal.tsv"
    assert journal.exists(), r.stdout + r.stderr
    assert not (DEPOT / "outils" / "bumper-version-readme.journal.tsv").exists(), (
        "le journal est retombé dans l'arbre du dépôt")


def test_journal_option_surchargeable(tmp_path):
    import os
    import subprocess

    cible = tmp_path / "ailleurs" / "j.tsv"
    r = subprocess.run(
        ["python3", str(DEPOT / "outils" / "bumper-version-readme.py"), "0.0.0", "0.0.1",
         "--journal", str(cible)],
        cwd=DEPOT, capture_output=True, text=True, timeout=30,
        env={**os.environ, "HOME": str(tmp_path / "home-inutilise")},
    )
    assert cible.exists(), r.stdout + r.stderr
