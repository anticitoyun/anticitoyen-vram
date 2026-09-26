"""Échantillonnage de puissance GPU (NVML power.draw.instant, via nvidia-smi)
pour le duel A2 (bead audit poste7) : « même NVML power.draw.instant, idle
soustrait, même fenêtre » pour les quatre moteurs — protocole de chef,
14/09/2026.

Nom en underscore pour rester IMPORTABLE (comme outils/racine_modeles.py) :
utilisé par outils/banc-4moteurs.py ET par un futur script de campagne."""
import statistics
import subprocess
import threading
import time


def puissance_instantanee_w(gpu: int = 0) -> float:
    """Un seul relevé, en watts."""
    sortie = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=power.draw.instant",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return float(sortie.stdout.strip())


def mesurer_idle(gpu: int = 0, secondes: float = 3.0, pas: float = 0.2) -> float:
    """Moyenne de puissance GPU au repos — À MESURER JUSTE AVANT chaque
    régime (l'idle d'une carte que quelqu'un d'autre a chauffée n'est pas
    l'idle de la carte froide).

    Moyenne, pas médiane (poste7, 14/09) : sur une charge par rafales, la
    médiane sous-estime l'énergie réellement consommée — seule la moyenne
    (pondérée par le temps entre relevés) correspond à un total de joules."""
    releves = []
    fin = time.time() + secondes
    while time.time() < fin:
        releves.append(puissance_instantanee_w(gpu))
        time.sleep(pas)
    return statistics.mean(releves)


class _Echantillonneur:
    """Thread qui releve la puissance en continu pendant qu'une fonction
    tourne — la fenetre EST la duree de la fonction, pas une duree fixe
    devinee a l'avance."""

    def __init__(self, gpu: int, pas: float):
        self.gpu = gpu
        self.pas = pas
        self.releves: list = []
        self._arret = threading.Event()
        self._fil = threading.Thread(target=self._boucle, daemon=True)

    def _boucle(self):
        while not self._arret.is_set():
            try:
                self.releves.append(puissance_instantanee_w(self.gpu))
            except (subprocess.CalledProcessError, ValueError):
                pass
            time.sleep(self.pas)

    def __enter__(self):
        self._fil.start()
        return self

    def __exit__(self, *exc):
        self._arret.set()
        self._fil.join(timeout=2.0)


def mesurer_pendant(fn, gpu: int = 0, pas: float = 0.1):
    """Exécute ``fn()`` en échantillonnant la puissance en parallèle.

    Rend (résultat_de_fn, watts_moyen_pendant, nombre_de_releves,
    watts_median_pendant). La MOYENNE est le champ à utiliser pour toute
    conversion en joules (P moyenne × durée = énergie) — la médiane,
    gardée en champ secondaire, sous-estime sur une charge par rafales
    (poste7, 14/09 : +51 % d'écart mesuré entre médiane et compteur NVML
    TotalEnergyConsumption sur une fenêtre de 2 s). Un seul relevé
    (fenêtre trop courte) rend ce relevé pour les deux champs, pas une
    erreur — signalé par ``nombre_de_releves == 1`` pour que l'appelant
    juge s'il fait confiance à la valeur."""
    with _Echantillonneur(gpu, pas) as ech:
        resultat = fn()
    if not ech.releves:
        raise RuntimeError("aucun releve de puissance pendant la fenetre "
                           "— fn() trop rapide pour le pas d'echantillonnage")
    return (resultat, statistics.mean(ech.releves), len(ech.releves),
            statistics.median(ech.releves))
