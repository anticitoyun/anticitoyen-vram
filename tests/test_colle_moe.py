"""Colle MoE en deux lancements (`kernels/colle_moe.py`, P0) contre torch :
tri des paires = argsort stable (même `ordre`, même `e_tri`, même `cnt`),
tuiles = `MoEBlock._tuiles` (mêmes (e, t0, n)), sur un routage Coder
(T 2 048 × top_k 8 sur 128 experts, dont des experts vides) et un petit.
Bras cassant : un compte faux donne des tuiles différentes."""
import pytest
import torch

from acvram.engine.model import MoEBlock
from acvram.kernels import colle_moe as C

pytestmark = pytest.mark.skipif(not C.disponible(), reason="Triton absent")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _routage(T, k, E, seed, vides=0):
    g = torch.Generator().manual_seed(seed)
    topi = torch.stack([torch.randperm(E - vides, generator=g)[:k] for _ in range(T)])
    return topi.reshape(-1).to(torch.int64).to(DEV)


@pytest.mark.parametrize("T,k,E,vides", [(2048, 8, 128, 3), (37, 4, 64, 10), (1, 8, 128, 0)])
def test_le_tri_en_un_lancement_egale_argsort_stable(T, k, E, vides):
    flat_e = _routage(T, k, E, seed=T, vides=vides)
    ordre, e_tri, cnt = C.trier_paires(flat_e, E)
    assert torch.equal(ordre, torch.argsort(flat_e, stable=True))
    assert torch.equal(e_tri, flat_e[ordre]) and torch.equal(cnt, torch.bincount(flat_e, minlength=E))


@pytest.mark.parametrize("bt", [16, 128])
def test_les_tuiles_en_un_lancement_egalent_moeblock_tuiles(bt):
    flat_e = _routage(2048, 8, 128, seed=7, vides=5)
    cnt = torch.bincount(flat_e, minlength=128)
    t_max = -(-flat_e.numel() // bt) + 128
    attendu = MoEBlock._tuiles(cnt, bt, t_max=t_max)
    te, t0, tn = C.tuiles(cnt, bt, t_max)
    for a, b in zip((te, t0, tn), attendu):
        assert torch.equal(a, b.to(DEV))


def test_bras_cassant_compte_faux():
    flat_e = _routage(512, 8, 128, seed=11)
    cnt = torch.bincount(flat_e, minlength=128)
    faux = cnt.clone(); faux[3] += 1; faux[4] -= 1
    t_max = -(-flat_e.numel() // 16) + 128
    a = C.tuiles(cnt, 16, t_max); b = C.tuiles(faux, 16, t_max)
    assert not all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.a_sec
def test_la_tuile_de_creneaux_est_bornee_et_la_grille_couvre_t_max():
    """T4 20/09 : BT_MAX = 2 048 constexpr dans un seul programme faisait boucler LLVM (9 min) ; le noyau est
    borné à BT_BLOC ≤ 512 créneaux par programme et la grille couvre t_max — un bt = 16 sur 16 384 paires
    (t_max 1 152) reste un noyau de 5 programmes de 256, pas un déroulé de 2 048 × 128."""
    import torch
    C = pytest.importorskip("acvram.kernels.colle_moe")
    assert 16 <= C.BT_BLOC <= 512
    cnt = torch.tensor([300, 5, 0, 1000] + [0] * 124, dtype=torch.int32)
    for bt, t_max in ((16, 1152), (128, 140), (16, 1)):
        te, t0, tn = C.tuiles(cnt, bt, t_max)
        attendu = MoEBlock._tuiles(cnt, bt, t_max=t_max)
        assert te.shape[0] == t_max and all(torch.equal(a.to(torch.int32), b) for a, b in zip(attendu, (te, t0, tn)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compilation Triton réelle : carte requise")
def test_la_compilation_du_noyau_tuiles_tient_en_60_s():
    """T4 : l'ancien constexpr BT_MAX = 2 048 sur un seul programme ne sortait pas de make_llir (9 min) ; avec
    BT_BLOC = 256 la compilation à bt = 16 / t_max = 1 152 doit tenir en 60 s — sinon faux (thread de garde)."""
    import threading, time
    cnt = torch.tensor([300, 5, 0, 1000] + [0] * 124, dtype=torch.int32, device="cuda")
    fini = threading.Event(); erreur = []

    def compile_et_lance():
        try:
            C.tuiles(cnt, 16, 1152); torch.cuda.synchronize()
        except Exception as e:                       # noqa: BLE001
            erreur.append(e)
        fini.set()
    t0 = time.perf_counter(); threading.Thread(target=compile_et_lance, daemon=True).start()
    assert fini.wait(60), f"compilation/lancement > 60 s (T4 : LLVM ne sort pas d'un déroulé BT_MAX constexpr)"
    assert not erreur, erreur
    print(f"compilation + lancement : {time.perf_counter() - t0:.1f} s")
