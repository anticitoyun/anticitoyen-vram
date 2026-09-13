"""14/09, poste7 : `mesurer_pendant`/`mesurer_idle` rendaient la MEDIANE des
releves de puissance -- sous-estime sur une charge par rafales (le +51 %
mesure entre la mediane et le compteur NVML TotalEnergyConsumption sur une
fenetre de 2 s). La moyenne (ponderee par le temps entre releves, ici
uniforme donc une moyenne simple suffit) est desormais le champ primaire ;
la mediane reste disponible en champ secondaire.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from outils.puissance_nvml import _Echantillonneur, mesurer_pendant  # noqa: E402


def test_moyenne_rafale_diverge_de_la_mediane():
    """Un jeu de releves ou la mediane et la moyenne different nettement --
    releves en rafale (beaucoup de pics, peu de creux), le cas reel signale
    par poste7."""
    releves = [50.0] * 8 + [400.0] * 2  # 8 creux, 2 pics
    ech = object.__new__(_Echantillonneur)
    ech.releves = releves

    def fn():
        return "ok"

    # reproduit le corps de mesurer_pendant sans le thread reel
    import statistics
    resultat = fn()
    moyenne = statistics.mean(ech.releves)
    mediane = statistics.median(ech.releves)
    assert abs(moyenne - 120.0) < 1e-9
    assert abs(mediane - 50.0) < 1e-9
    assert moyenne != mediane, "le jeu de test ne distingue pas les deux mesures"


def test_mesurer_pendant_rend_moyenne_puis_mediane_en_secondaire(monkeypatch):
    """mesurer_pendant() elle-meme : verifie l'ORDRE du tuple rendu."""
    import threading
    import outils.puissance_nvml as mod

    releves_simules = [50.0, 50.0, 50.0, 400.0]
    it = iter(releves_simules)
    tous_consommes = threading.Event()

    def fausse_instantanee(gpu=0):
        try:
            return next(it)
        except StopIteration:
            tous_consommes.set()
            raise

    monkeypatch.setattr(mod, "puissance_instantanee_w", fausse_instantanee)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)

    def attendre_les_releves():
        tous_consommes.wait(timeout=2.0)
        return "fait"

    resultat, moyenne, n, mediane = mesurer_pendant(attendre_les_releves, gpu=0, pas=0.0)
    assert resultat == "fait"
    assert n == len(releves_simules)
    assert moyenne == sum(releves_simules) / len(releves_simules)
    assert mediane == 50.0  # mediane de [50,50,50,400]
    assert moyenne != mediane
