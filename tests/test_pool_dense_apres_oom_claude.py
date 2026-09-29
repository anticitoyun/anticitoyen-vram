"""Pièce claude (poste6, 29/09) — acvram-gemma-4-12b-it-bf16-vision : « ExpertPool saturé : 4 emplacements, tous
distribués et non rendus » au démarrage (23 Go bf16 + vision, 21/48 couches exilées, pool DENSE). Cause : après un OOM
de chauffe, les emplacements pris par `prefetch()` (couche suivante) restaient distribués et `_pending_slot` gardé ;
au pas suivant de la dichotomie `prefetch()` REPRENAIT un emplacement sans rendre le précédent — quatre essais et le
pool était plein. Correctifs : `prefetch()` ne reprend pas ; `ExpertPool.rendre_tout()` ; `oublier_precharges(model)`
appelé par `_apres_oom_de_chauffe`."""
import pytest
import torch

from acvram.engine import layers as L


class _EvenementFactice:
    def record(self, *a): pass
    def wait(self, *a): pass


def _pool_sync(monkeypatch, n=4):
    monkeypatch.setattr(torch.cuda, "Event", _EvenementFactice)
    monkeypatch.setattr(L.ExpertPool, "SYNC", True)
    return L.ExpertPool(torch.device("cpu"), n, dense=True)


def test_rendre_tout_libere_un_pool_sature(monkeypatch):
    pool = _pool_sync(monkeypatch)
    plat = torch.zeros(64, dtype=torch.uint8)
    decoupe = {"w": (0, (64,), torch.uint8, 64)}
    for _ in range(4):
        pool.copier(plat, decoupe)
    with pytest.raises(RuntimeError, match="sature"):
        pool.copier(plat, decoupe)
    assert pool.rendre_tout() == 4
    pool.copier(plat, decoupe)                                   # de nouveau possible
    assert pool.rendre_tout() == 1 and pool.rendre_tout() == 0


class _Streamed:
    """StreamedWeight témoin : compte les prises, retient les rendus."""
    def __init__(self, pool):
        self.pool = pool
        self.prises = 0
        self.rendus = []

    def prefetch(self):
        self.prises += 1
        return self.prises

    def release(self, slot):
        self.rendus.append(slot)


class _Pool:
    dense = True

    def __init__(self):
        self.rendu_tout = 0

    def rendre_tout(self):
        self.rendu_tout += 1
        return 2


def _quant_linear(streamed):
    q = L.QuantLinear.__new__(L.QuantLinear)
    torch.nn.Module.__init__(q)
    q.streamed = streamed
    q._pending_slot = None
    return q


def test_prefetch_ne_reprend_pas_un_emplacement_deja_pris(monkeypatch):
    monkeypatch.delenv("ACVRAM_SANS_PRECHARGE", raising=False)
    st = _Streamed(_Pool())
    q = _quant_linear(st)
    q.prefetch(); q.prefetch(); q.prefetch()
    assert st.prises == 1 and q._pending_slot == 1


def test_oublier_precharges_rend_les_emplacements_et_les_pools(monkeypatch):
    monkeypatch.delenv("ACVRAM_SANS_PRECHARGE", raising=False)
    pool = _Pool()
    a, b = _Streamed(pool), _Streamed(pool)
    m = torch.nn.Module()
    m.a, m.b = _quant_linear(a), _quant_linear(b)
    m.a.prefetch()                                               # b n'a rien pris
    n = L.oublier_precharges(m)
    assert a.rendus == [1] and b.rendus == [] and m.a._pending_slot is None
    assert pool.rendu_tout == 1 and n == 1 + 2                  # un pool partagé : rendu une seule fois
    m.a.prefetch()
    assert a.prises == 2                                         # après l'oubli, la précharge repart


def test_apres_oom_de_chauffe_appelle_oublier_precharges(monkeypatch):
    from acvram.engine.contexte import ChauffeContexte
    appels = []
    monkeypatch.setattr(L, "oublier_precharges", lambda model: appels.append(model) or 0)

    class _Alloc:
        num_blocks, enable_prefix_cache = 1, False

    class _R:
        def __init__(self):
            self.running, self.waiting, self.model, self.allocator = [1], [2], torch.nn.Module(), _Alloc()
            self.oublis = 0

        def _finish(self, seq, raison):
            pass

        def _oublier_la_chauffe(self):
            self.oublis += 1

    r = _R()
    ChauffeContexte._apres_oom_de_chauffe(r)
    assert appels == [r.model] and r.oublis == 1 and r.running == [] and r.waiting == []
