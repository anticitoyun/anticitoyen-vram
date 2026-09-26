"""La durée d'une fenêtre Energie est relevée avant d'attendre le fil de
sonde. Le 15/09, cinq bras certifiés donnaient 25,04 s et 26,04 s exactement :
`join` attendait la fin du `sleep(periode)` et la durée (ms/pas, jetons/s, W)
portait jusqu'à +1 s d'erreur, imputée à un noyau qui coûtait 70 µs."""
import pathlib, sys, time, types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


class _Nvml:
    cartes = [(0, object())]

    def energie_mj(self, h): return 0
    def pids(self, h): return ()
    def bridages(self, h): return set()
    def horloge_sm(self, h): return 0
    def temperature(self, h): return 0
    def puissance_w(self, h): return 0.0


def test_la_duree_ne_porte_pas_le_sommeil_du_fil(monkeypatch):
    from outils.gpu.mesure import energie as E
    monkeypatch.setattr(E, "nvml", lambda: _Nvml())
    with E.Energie(periode=1.0) as e:
        time.sleep(0.15)                      # le fil dort encore 0,85 s
    assert 0.14 <= e.duree < 0.40, f"durée {e.duree:.3f} s : le join du fil est compté"
