"""bd 076 : correctif TOPK_DUMP de colibri (`outils/colibri/topk-dump.patch`), vérifié à sec. Le bloc C est extrait du
patch lui-même et compilé seul avec une boucle qui reproduit `tf_nll` (qwen36.c:2447 à fd93c41), puis comparé à une
référence numpy : ids du top-K (y compris l'ordre à égalité), logprobs, logprob du jeton de référence, et le contrôle
« −moyenne(logprob_ref) = TF-NLL » de `outils/colibri/topk_dump.py`. Aucune carte, aucun modèle ; le clone colibri n'est
que LU (git show), pour vérifier que le patch s'applique à fd93c41."""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ICI = Path(__file__).resolve().parent.parent
PATCH = ICI / "outils" / "colibri" / "topk-dump.patch"
CLONE = Path(os.environ.get("ACVRAM_COLIBRI_CLONE", "/mnt/AI_GENERATOR/colibri"))
_spec = importlib.util.spec_from_file_location("topk_dump", ICI / "outils" / "colibri" / "topk_dump.py")
topk_dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(topk_dump)

gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc absent")

BOUCLE = r'''
int main(int argc, char **argv) {
    int V = atoi(argv[3]), n = atoi(argv[4]), pos0 = atoi(argv[5]);
    float *L = malloc((size_t)n * V * sizeof(float)); int32_t *R = malloc((size_t)n * sizeof(int32_t));
    FILE *f = fopen(argv[1], "rb"); if (fread(L, sizeof(float), (size_t)n * V, f) != (size_t)n * V) return 2; fclose(f);
    f = fopen(argv[2], "rb"); if (fread(R, sizeof(int32_t), (size_t)n, f) != (size_t)n) return 2; fclose(f);
    TopkDump td; topk_dump_ouvrir(&td, V, n);
    double nll = 0;
    for (int i = 0; i < n; i++) {                       /* même arithmétique que tf_nll (qwen36.c:2461-2463) */
        const float *logit = L + (size_t)i * V;
        float mx = logit[0]; for (int v = 1; v < V; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < V; v++) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[R[i]] - mx - log(Z));
        topk_dump_ecrire(&td, logit, V, pos0 + i, R[i], mx, Z);
    }
    topk_dump_fermer(&td);
    printf("TF-NLL: %.4f nats/token over %d tokens\n", nll / n, n);
    return 0;
}
'''


def _bloc_c(patch_texte: str) -> str:
    lignes, dedans = [], False
    for l in patch_texte.splitlines():
        if l.startswith("+/* --- TOPK_DUMP"):
            dedans = True
        if dedans:
            assert l.startswith("+"), l
            lignes.append(l[1:])
        if l.startswith("+/* --- fin TOPK_DUMP"):
            break
    assert lignes and lignes[-1].startswith("/* --- fin TOPK_DUMP"), "bloc TOPK_DUMP introuvable dans le patch"
    return "\n".join(lignes)


def _harnais(tmp_path, bloc=None):
    src = tmp_path / "h.c"
    src.write_text("#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <stdint.h>\n"
                   + (bloc or _bloc_c(PATCH.read_text())) + BOUCLE)
    exe = tmp_path / "h"
    r = subprocess.run(["gcc", "-O2", "-std=gnu11", "-Wall", "-Werror", "-o", str(exe), str(src), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def _jouer(tmp_path, exe, logits, refs, pos0, env=None):
    n, V = logits.shape
    (tmp_path / "L.f32").write_bytes(logits.astype("<f4").tobytes())
    (tmp_path / "R.i32").write_bytes(refs.astype("<i4").tobytes())
    return subprocess.run([str(exe), str(tmp_path / "L.f32"), str(tmp_path / "R.i32"), str(V), str(n), str(pos0)],
                          capture_output=True, text=True, env={**os.environ, **(env or {})})


def _logits(n=24, V=3000, graine=0):
    rng = np.random.default_rng(graine)
    # arrondis au dixième : beaucoup d'égalités, y compris dans le top-K (la règle « id le plus petit d'abord » est jouée)
    x = np.round(rng.normal(0, 3, size=(n, V)), 1).astype(np.float32)
    x[3, [7, 11, V - 1]] = x[3].max() + 1.0                      # trois maxima égaux sur une position
    return x, rng.integers(0, V, size=n).astype(np.int32)


def _reference(logits, k):
    out = []
    for row in logits.astype(np.float64):
        mx = np.float32(row.max())
        lz = np.log(np.sum(np.exp(row - mx)))
        ordre = np.lexsort((np.arange(row.size), -row))[:k]
        out.append((ordre, row[ordre] - mx - lz, mx, lz))
    return out


def test_le_patch_s_applique_a_fd93c41(tmp_path):
    if not (CLONE / ".git").exists():
        pytest.skip(f"clone colibri absent ({CLONE})")
    src = subprocess.run(["git", "-C", str(CLONE), "show", "fd93c41:c/qwen36.c"], capture_output=True)
    if src.returncode != 0:
        pytest.skip("fd93c41 absent du clone")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "qwen36.c").write_bytes(src.stdout)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    r = subprocess.run(["git", "-C", str(tmp_path), "apply", "--check", "-p1", str(PATCH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # la ligne de la NLL n'est pas touchée : sans TOPK_DUMP la TF-NLL est la même au bit
    assert "-        nll += " not in PATCH.read_text()


@gcc
def test_top_k_logprobs_et_ordre_a_egalite_contre_numpy(tmp_path):
    logits, refs = _logits()
    exe = _harnais(tmp_path)
    r = _jouer(tmp_path, exe, logits, refs, 512, {"TOPK_DUMP": str(tmp_path / "d.bin"), "TOPK_K": "20"})
    assert r.returncode == 0, r.stderr
    d = topk_dump.lire(str(tmp_path / "d.bin"))
    assert (d["k"], d["vocab"], d["n"]) == (20, logits.shape[1], logits.shape[0])
    assert list(d["pos"]) == list(range(512, 512 + logits.shape[0])) and list(d["ref"]) == list(refs)
    for i, (ids, lps, mx, lz) in enumerate(_reference(logits, 20)):
        assert list(d["ids"][i]) == list(ids), i
        np.testing.assert_allclose(d["lps"][i], lps, atol=2e-6)
        assert abs(d["lp_ref"][i] - (logits[i, refs[i]] - mx - lz)) < 2e-6
    assert list(d["ids"][3][:3]) == [7, 11, logits.shape[1] - 1]
    tf = float(r.stdout.split("TF-NLL:")[1].split()[0])
    assert topk_dump.verifier(d, tf), (topk_dump.nll(d), tf)


@gcc
def test_sans_topk_dump_aucun_fichier_et_meme_tf_nll(tmp_path):
    logits, refs = _logits(graine=1)
    exe = _harnais(tmp_path)
    a = _jouer(tmp_path, exe, logits, refs, 0)
    b = _jouer(tmp_path, exe, logits, refs, 0, {"TOPK_DUMP": str(tmp_path / "d.bin")})
    assert a.returncode == b.returncode == 0 and a.stdout == b.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["L.f32", "R.i32", "d.bin", "h", "h.c"]


@gcc
def test_chemin_illisible_ou_k_hors_bornes_arrete_le_moteur(tmp_path):
    logits, refs = _logits(n=4, V=100)
    exe = _harnais(tmp_path)
    assert _jouer(tmp_path, exe, logits, refs, 0, {"TOPK_DUMP": str(tmp_path / "absent" / "d.bin")}).returncode == 1
    for k in ("0", "257", "101"):
        r = _jouer(tmp_path, exe, logits, refs, 0, {"TOPK_DUMP": str(tmp_path / "d.bin"), "TOPK_K": k})
        assert r.returncode == 1 and "TF-NLL" not in r.stdout, k


@gcc
def test_un_tri_fautif_est_refuse_par_le_lecteur(tmp_path):
    """Le contrôle doit pouvoir rendre faux : on retourne la comparaison de l'insertion (top-K croissant)."""
    bloc = _bloc_c(PATCH.read_text())
    assert bloc.count("d->vals[j - 1] < x") == 1
    exe = _harnais(tmp_path, bloc.replace("d->vals[j - 1] < x", "d->vals[j - 1] > x"))
    logits, refs = _logits(n=6, V=500)
    assert _jouer(tmp_path, exe, logits, refs, 0, {"TOPK_DUMP": str(tmp_path / "d.bin")}).returncode == 0
    with pytest.raises(ValueError):
        topk_dump.lire(str(tmp_path / "d.bin"))


def test_lecteur_refuse_tronque_et_en_tete(tmp_path):
    import struct
    k, n = 2, 3
    corps = b"".join(struct.pack("<iif", p, 1, -0.5) + struct.pack("<ifif", 1, -0.5, 0, -1.0) for p in range(n))
    bon = b"CTK1" + struct.pack("<4i", 1, k, 10, n) + corps
    (tmp_path / "ok").write_bytes(bon)
    d = topk_dump.lire(str(tmp_path / "ok"))
    assert d["n"] == 3 and abs(topk_dump.nll(d) - 0.5) < 1e-7
    assert topk_dump.verifier(d, 0.5) and not topk_dump.verifier(d, 0.5002)
    for nom, octets in (("court", bon[:-4]), ("tete", b"XTK1" + bon[4:])):
        (tmp_path / nom).write_bytes(octets)
        with pytest.raises(ValueError):
            topk_dump.lire(str(tmp_path / nom))
