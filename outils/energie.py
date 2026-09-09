"""Mesure d'énergie d'une carte, par le compteur NVML.

POURQUOI PAS UNE MOYENNE DE PUISSANCES
--------------------------------------
Le banc échantillonnait ``nvidia-smi --query-gpu=power.draw`` toutes les
200 ms et en prenait la moyenne. Trois défauts, mesurés le 8 septembre 2026 :

1. un sous-processus toutes les 200 ms coûte à la machine qu'on mesure ;
2. la moyenne de N échantillons n'est pas l'énergie de la fenêtre — elle
   suppose que le pas est régulier et qu'aucune pointe ne tombe entre deux
   lectures. NVML tient un compteur d'énergie monotone : la différence de
   deux lectures EST l'énergie, sans hypothèse ;
3. la carte 0 seule était lue, alors qu'un modèle exilé travaille sur deux.

CE QUE NVIDIA-SMI NE SAIT PAS FAIRE ICI
--------------------------------------
``--query-gpu=total_energy_consumption`` répond « is not a valid field to
query » sur cette machine, et ``pynvml`` n'y est pas installé. La
bibliothèque, elle, répond : on l'appelle par ctypes, sans dépendance.

CE QUE LA MESURE NE CONTIENT PAS
--------------------------------
L'énergie du reste de la machine : processeur, mémoire, alimentation. NVML ne
voit que les cartes. Un moteur qui déporte du travail sur le processeur
paraîtra donc plus économe qu'il ne l'est. Sous 10 % d'écart entre deux
moteurs, cette part cesse d'être négligeable et il faut un wattmètre à la
prise — dit ici plutôt qu'ignoré.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time

# nvmlClocksThrottleReasons : les deux qui changent le régime d'une carte
BRIDAGE_PUISSANCE = 0x0000000000000004   # SW Power Cap
BRIDAGE_THERMIQUE = 0x0000000000000040   # HW Thermal Slowdown
BRIDAGE_THERMIQUE_SW = 0x0000000000000020
_NOMS_BRIDAGE = [(BRIDAGE_PUISSANCE, "puissance"),
                 (BRIDAGE_THERMIQUE, "thermique_materiel"),
                 (BRIDAGE_THERMIQUE_SW, "thermique_logiciel")]

_HORLOGE_SM = 1
_TEMPERATURE_CARTE = 0


class NvmlAbsent(RuntimeError):
    """NVML n'est pas utilisable : la mesure d'énergie n'a pas lieu."""


class _Nvml:
    """Accès minimal à NVML, chargé une fois."""

    def __init__(self) -> None:
        try:
            self.l = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as e:                       # pragma: no cover - machine sans pilote
            raise NvmlAbsent(str(e)) from e
        if self.l.nvmlInit_v2() != 0:
            raise NvmlAbsent("initialisation refusée")
        n = ctypes.c_uint()
        self.l.nvmlDeviceGetCount_v2(ctypes.byref(n))
        # NVML IGNORE CUDA_VISIBLE_DEVICES (etabli le 8/09). Sans ce filtre,
        # toute mesure agrege les cartes que la campagne n'utilise pas : le
        # 9/09, `plafond_W` valait 875 W — soit 500 (5090) + 375 (3080 Ti) —,
        # la puissance publiee incluait les services permanents de la 3080 Ti,
        # et un drapeau de bridage venait de la carte qui ne participait pas.
        # L'energie par jeton, qui EST l'objectif du projet, etait donc fausse.
        visibles = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        garder = None
        if visibles:
            garder = {int(x) for x in visibles.split(",") if x.strip().isdigit()}
        self.cartes = []
        for i in range(n.value):
            if garder is not None and i not in garder:
                continue
            h = ctypes.c_void_p()
            if self.l.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0:
                self.cartes.append((i, h))
        if not self.cartes:
            raise NvmlAbsent(f"aucune carte (CUDA_VISIBLE_DEVICES={visibles!r})")

    def _u32(self, fn, h, *a) -> int:
        v = ctypes.c_uint()
        return v.value if fn(h, *a, ctypes.byref(v)) == 0 else -1

    def _u64(self, fn, h) -> int:
        v = ctypes.c_ulonglong()
        return v.value if fn(h, ctypes.byref(v)) == 0 else -1

    def energie_mj(self, h) -> int:
        return self._u64(self.l.nvmlDeviceGetTotalEnergyConsumption, h)

    def puissance_w(self, h) -> float:
        v = self._u32(self.l.nvmlDeviceGetPowerUsage, h)
        return v / 1000.0 if v >= 0 else -1.0

    def plafond_w(self, h) -> float:
        v = self._u32(self.l.nvmlDeviceGetPowerManagementLimit, h)
        return v / 1000.0 if v >= 0 else -1.0

    def temperature(self, h) -> int:
        return self._u32(self.l.nvmlDeviceGetTemperature, h, _TEMPERATURE_CARTE)

    def horloge_sm(self, h) -> int:
        return self._u32(self.l.nvmlDeviceGetClockInfo, h, _HORLOGE_SM)

    def bridages(self, h) -> set[str]:
        v = self._u64(self.l.nvmlDeviceGetCurrentClocksThrottleReasons, h)
        if v <= 0:
            return set()
        return {nom for bit, nom in _NOMS_BRIDAGE if v & bit}

    def pids(self, h) -> tuple[int, ...]:
        """La LISTE des processus, pas leur mémoire.

        L'alarme avait d'abord porté sur (pid, mémoire) et criait à chaque
        mesure : la mémoire d'un serveur grossit normalement pendant un
        décodage. Ce qui invalide une fenêtre, c'est qu'un processus
        apparaisse ou disparaisse.
        """
        class _Info(ctypes.Structure):
            _fields_ = [("pid", ctypes.c_uint), ("mem", ctypes.c_ulonglong)]
        n = ctypes.c_uint(64)
        tab = (_Info * 64)()
        if self.l.nvmlDeviceGetComputeRunningProcesses(h, ctypes.byref(n), tab) != 0:
            return ()
        return tuple(sorted(tab[k].pid for k in range(n.value)))


_nvml: _Nvml | None = None


def nvml() -> _Nvml:
    global _nvml
    if _nvml is None:
        _nvml = _Nvml()
    return _nvml


class Energie:
    """Fenêtre de mesure : énergie exacte, régime de la carte, invalidations.

    S'utilise comme l'ancien ``Watt`` :

        with Energie() as e:
            ...          # la génération
        e.joules, e.moyenne, e.invalidations
    """

    def __init__(self, periode: float = 1.0) -> None:
        self.periode = periode
        self.debut: dict[int, int] = {}
        self.fin: dict[int, int] = {}
        self.pids_debut: dict[int, tuple[int, ...]] = {}
        self.pids_fin: dict[int, tuple[int, ...]] = {}
        self.bridages: set[str] = set()
        self.horloges: list[int] = []
        self.temperatures: list[int] = []
        self.puissances: list[float] = []
        self.duree = 0.0
        self._stop = False
        self._t: threading.Thread | None = None
        self.indisponible: str | None = None

    # -- collecte ---------------------------------------------------------
    def __enter__(self) -> "Energie":
        try:
            n = nvml()
        except NvmlAbsent as e:
            self.indisponible = str(e)
            return self
        self._t0 = time.time()
        for i, h in n.cartes:
            self.debut[i] = n.energie_mj(h)
            self.pids_debut[i] = n.pids(h)

        def boucle():
            while not self._stop:
                for i, h in n.cartes:
                    self.bridages |= n.bridages(h)
                    self.horloges.append(n.horloge_sm(h))
                    self.temperatures.append(n.temperature(h))
                    self.puissances.append(n.puissance_w(h))
                time.sleep(self.periode)

        self._t = threading.Thread(target=boucle, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a) -> None:
        self._stop = True
        if self._t:
            self._t.join(timeout=3)
        if self.indisponible:
            return
        n = nvml()
        self.duree = time.time() - self._t0
        for i, h in n.cartes:
            self.fin[i] = n.energie_mj(h)
            self.pids_fin[i] = n.pids(h)

    # -- lecture ----------------------------------------------------------
    @property
    def joules(self) -> float:
        """Énergie de la fenêtre, toutes cartes, en joules."""
        if self.indisponible:
            return 0.0
        total = 0
        for i, e0 in self.debut.items():
            e1 = self.fin.get(i, -1)
            if e0 >= 0 and e1 >= 0:
                total += e1 - e0
        return total / 1000.0

    @property
    def moyenne(self) -> float:
        """Puissance moyenne de la fenêtre, en watts — énergie sur durée."""
        return self.joules / self.duree if self.duree > 0 else 0.0

    @property
    def plafond(self) -> float:
        if self.indisponible:
            return 0.0
        return sum(nvml().plafond_w(h) for _, h in nvml().cartes)

    @property
    def invalidations(self) -> list[str]:
        """Ce qui, mesuré, interdit de conclure. Vide si la fenêtre tient."""
        raisons = []
        if self.indisponible:
            return [f"NVML indisponible : {self.indisponible}"]
        for i, e0 in self.debut.items():
            e1 = self.fin.get(i, -1)
            if e0 < 0 or e1 < 0:
                raisons.append(f"carte {i} : compteur d'énergie non supporté")
            elif e1 == e0:
                raisons.append(f"carte {i} : le compteur d'énergie n'a pas avancé")
        for i, avant in self.pids_debut.items():
            apres = self.pids_fin.get(i, ())
            if avant != apres:
                raisons.append(f"carte {i} : les processus ont changé "
                               f"({list(avant)} -> {list(apres)})")
        if self.bridages:
            raisons.append("bridage pendant la fenêtre : " + ", ".join(sorted(self.bridages)))
        return raisons

    def resume(self) -> dict:
        h = [v for v in self.horloges if v >= 0]
        t = [v for v in self.temperatures if v >= 0]
        return {
            "joules": round(self.joules, 1),
            "watts": round(self.moyenne, 1),
            "duree_s": round(self.duree, 2),
            "plafond_w": round(self.plafond, 0),
            "horloge_min": min(h) if h else -1,
            "horloge_max": max(h) if h else -1,
            "temp_max": max(t) if t else -1,
            "bridages": ",".join(sorted(self.bridages)) or "aucun",
            "invalidations": " ; ".join(self.invalidations) or "aucune",
        }


def repos(secondes: float = 10.0, periode: float = 1.0) -> Energie:
    """Ligne de base, prise APRÈS la mesure, serveur chargé mais inactif.

    Prise avant, elle dérive : sur une séance d'appareil photo du 7 septembre,
    la ligne de base du début a fait attribuer à un calcul une baisse qui
    n'était que la dérive de la base. Elle se prend donc après chaque cas, et
    de préférence de même durée que la fenêtre mesurée.
    """
    with Energie(periode=periode) as e:
        time.sleep(secondes)
    return e
