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
