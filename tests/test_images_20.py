"""Le corpus d images synthétiques (20 png, non suivis) se régénère au bit près par outils/generer-images-20.py :
sha256 == images-20.sha256 scellé, descriptions.tsv identique. Sauté sans la police DejaVuSans-Bold (le rendu du texte
en dépend)."""
import os, subprocess, sys, hashlib, pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ICI, "scratchpad", "corpus-prive", "images-20")
POLICE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@pytest.mark.skipif(not os.path.exists(POLICE), reason="police DejaVuSans-Bold absente")
def test_regeneration_au_bit(tmp_path):
    pytest.importorskip("PIL")
    r = subprocess.run([sys.executable, os.path.join(ICI, "outils", "generer-images-20.py"), str(tmp_path)],
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    attendu = dict(l.split()[::-1] for l in open(os.path.join(CORPUS, "images-20.sha256")) if l.strip())
    for nom, h in attendu.items():
        assert hashlib.sha256(open(tmp_path / nom, "rb").read()).hexdigest() == h, nom
    assert (tmp_path / "descriptions.tsv").read_text() == open(os.path.join(CORPUS, "descriptions.tsv")).read()


def test_refus_nomme_si_l_environnement_differe(tmp_path, monkeypatch):
    """poste2 21/09 : 3/20 sha avec un autre venv — le bit près dépend de Pillow/FreeType/zlib/police, pas du script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen", os.path.join(ICI, "outils", "generer-images-20.py"))
    monkeypatch.setattr(sys, "argv", ["gen", "--environnement"])
    gen = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gen)
    except SystemExit:
        pass
    monkeypatch.setitem(gen.ATTENDU, "zlib", "0.0.0")
    assert gen.ecarts() == {"zlib": ("0.0.0", gen.environnement()["zlib"])}
    faux = tmp_path / "PIL"; faux.mkdir()
    (faux / "__init__.py").write_text("__version__ = '0.0.1'\nfrom . import features\n")
    (faux / "features.py").write_text("def version(n): return '0'\n")
    (faux / "Image.py").write_text("")
    (faux / "ImageDraw.py").write_text("")
    (faux / "ImageFont.py").write_text("")
    r = subprocess.run([sys.executable, os.path.join(ICI, "outils", "generer-images-20.py"), str(tmp_path / "out")],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 3 and "REFUS" in r.stdout and "Pillow attendu 12.3.0 trouvé 0.0.1" in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / "out").exists()
