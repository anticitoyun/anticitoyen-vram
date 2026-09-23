"""Pièce 39 (`part-tete-sampler.py`) à sec : sur une trace synthétique de
6 pas × 2 couches (tête `cutlass…wmma` puis trois noyaux d échantillonnage),
la part de la tête et de l échantillon dans le pas, les noyaux nommés, et le
refus quand la trace est trop courte ou sans tête."""
import csv
import importlib.util
import json
import os
import subprocess
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
OUTIL = os.path.join(ICI, "..", "outils", "gpu", "mesure", "part-tete-sampler.py")


def _trace(chemin, pas=8, couches=2, tete_us=130, ech=(20, 8, 4), routeur_cutlass=True):
    """Chaque couche porte AUSSI un `cutlass…wmma` (le GEMM du routeur, 2,7 µs) :
    c est ce qui a fait prendre le premier pour la tête (Manon 63767f45)."""
    t, lignes = 0, []
    for _ in range(pas):
        debut = t
        for _ in range(couches):
            for nom, d in (("_route_fusee_kernel", 7_000),
                           *((("void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16>(...)", 2_700),) if routeur_cutlass else ()),
                           ("void marlin_moe_wna16::Marlin<...>", 50_000)):
                lignes.append((t, d, nom)); t += d
        lignes.append((t, tete_us * 1000, "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16>(...)")); t += tete_us * 1000
        for i, d in enumerate(ech):
            lignes.append((t, d * 1000, f"void at::native::reduce_kernel<{i}>(...)")); t += d * 1000
        t = debut + 1_000_000
    with open(chemin, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start (ns)", "Duration (ns)", "Name"])
        for s, d, n in lignes:
            w.writerow([s, d, n])


def _lancer(trace, sortie=None):
    cmd = [sys.executable, OUTIL, trace, "--couches", "2"] + (["--json", sortie] if sortie else [])
    return subprocess.run(cmd, capture_output=True, text=True)


def test_part_tete_et_echantillon(tmp_path):
    t = str(tmp_path / "trace.csv")
    _trace(t)
    j = str(tmp_path / "part.json")
    r = _lancer(t, j)
    assert r.returncode == 0, r.stderr
    d = json.load(open(j))
    assert d["pas_juges"] == 4 and d["us"]["total"] == 1000.0      # 8 pas → 7 fenêtres, 2 de tête et 1 de queue exclues
    # la tête = le DERNIER cutlass du pas (les couches en portent chacune un : le routeur)
    assert d["us"]["tete"] == 130.0 and abs(d["part_tete"] - 0.13) < 1e-6
    assert d["us"]["echantillon"] == 32.0 and abs(d["part_echantillon"] - 0.032) < 1e-6
    assert d["lancements_apres_tete"] == 3 and len(d["noyaux_apres_tete"]) == 3


def test_refus_sans_tete_et_trace_courte(tmp_path):
    t = str(tmp_path / "sans_tete.csv")
    _trace(t, ech=())
    import re
    lignes = [l for l in open(t).read().splitlines() if "cutlass" not in l]
    open(t, "w").write("\n".join(lignes))
    r = _lancer(t)
    assert r.returncode != 0 and "aucune tête" in r.stdout + r.stderr
    court = str(tmp_path / "court.csv")
    _trace(court, pas=3)
    r = _lancer(court)
    assert r.returncode != 0 and "trace trop courte" in r.stdout + r.stderr
