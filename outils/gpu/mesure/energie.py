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
import re
import subprocess
import sys
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


# -- charge processeur : REGLES §2, "18/09, load 70,8 : trois pytest de
# pairs pendant une fenêtre HTTP" -- une charge CPU non annoncée fausse une
# mesure HTTP (le serveur partage le processeur avec l'appelant) exactement
# comme une charge GPU non verrouillée fausse un ms/pas. Le contrôle est le
# même que celui de charge-gpu.py : refuser plutôt qu'ajouter une consigne
# à se rappeler.
PORTS_SERVICES_PERMANENTS = (8081, 8082, 8083)   # llama-server, embeddings, reranker (REGLES §2)
_JIFFIES_HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100.0


def _pids_services_permanents() -> set[int]:
    """PID des trois services permanents -- légitimes, exclus du contrôle
    de charge (un seul appel `ss`, PAS dans la boucle d'une fenêtre)."""
    pids: set[int] = set()
    try:
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True,
                             timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return pids
    for port in PORTS_SERVICES_PERMANENTS:
        for ligne in out.splitlines():
            if f":{port} " in ligne:
                m = re.search(r"pid=(\d+)", ligne)
                if m:
                    pids.add(int(m.group(1)))
    return pids


def _temps_cpu_jiffies(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            champs = fh.read().rsplit(")", 1)[1].split()
        return int(champs[11]) + int(champs[12])   # utime + stime, en jiffies
    except (OSError, ValueError, IndexError):
        return None


def relever_charge(intervalle: float = 0.1) -> dict:
    """Un instantané : load1/load5, nproc, et les PID hors services
    permanents dont l'usage CPU dépasse 100 % (un coeur plein) sur
    `intervalle` secondes. Coûte `intervalle` s -- appelé une fois au
    début et une fois à la fin d'une fenêtre, jamais dans une boucle."""
    load1, load5, _ = os.getloadavg()
    nproc = os.cpu_count() or 1
    exclus = _pids_services_permanents()
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        pids = []
    pids = [p for p in pids if p not in exclus]
    avant = {p: _temps_cpu_jiffies(p) for p in pids}
    time.sleep(intervalle)
    lourds = []
    for p in pids:
        t0, t1 = avant.get(p), _temps_cpu_jiffies(p)
        if t0 is None or t1 is None:
            continue
        pct = 100.0 * (t1 - t0) / (_JIFFIES_HZ * intervalle)
        if pct > 100.0:
            lourds.append((p, round(pct, 1)))
    return {"load1": round(load1, 2), "load5": round(load5, 2), "nproc": nproc,
           "processus_charges": sorted(lourds, key=lambda x: -x[1])}


def _marquer_charge_deliberee() -> None:
    """Écrit CHARGE-DELIBEREE dans `$VERROU.qui`, comme `charge-gpu.py` --
    un observateur qui lit `qui_tient()` (carte.sh) voit une charge connue,
    pas un intrus à chercher pendant vingt minutes (10/09, REGLES §2)."""
    carte = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "0").split(",")[0]
    if not carte.isdigit():
        carte = "0"
    info = os.environ.get("ACVRAM_VERROU", f"/tmp/acvram-carte-{carte}.lock") + ".qui"
    try:
        with open(info) as fh:
            p, t, n, ty = fh.read().split()
    except (OSError, ValueError):
        return
    if n.endswith("-CHARGE-DELIBEREE"):
        return
    try:
        with open(info, "w") as fh:
            fh.write(f"{p} {t} {n}-CHARGE-DELIBEREE {ty}\n")
    except OSError:
        pass


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
        self.charge_avant: dict | None = None
        self.charge_apres: dict | None = None
        # sage-profil-verdict-18-09 §3 (précision) : ni "fin seulement" ni un
        # second instrument -- `load1` rejoint la boucle d'échantillons PAR
        # TIC qui existe déjà pour horloge/température (gratuit,
        # `/proc/loadavg`) ; jugé sur le MAX de toute la fenêtre, pas sur deux
        # points. Le relevé des processus >100 % CPU, lui, coûteux en continu,
        # reste début+fin (`relever_charge`, inchangé).
        self.charges1: list[float] = []
        self.nproc = os.cpu_count() or 1

    # -- collecte ---------------------------------------------------------
    def __enter__(self) -> "Energie":
        # Relevé AVANT `_t0` : son propre `intervalle` (0,1 s par défaut) ne
        # doit pas entrer dans `self.duree`, sous peine de biaiser la fenêtre
        # avec l'instrument même qui doit garantir qu'elle n'est pas biaisée.
        self.charge_avant = relever_charge()
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
                self.charges1.append(os.getloadavg()[0])
                time.sleep(self.periode)

        self._t = threading.Thread(target=boucle, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a) -> None:
        # Relever AVANT d'arrêter le fil : `join` attend la fin de son
        # `sleep(periode)`, jusqu'à 1 s — la durée le portait, donc ms/pas,
        # jetons/s et W aussi (15/09 : cinq bras à 25,04 / 26,04 s exactement,
        # +1,000 s attribués à tort à un noyau qui coûtait 70 µs).
        if not self.indisponible:
            n = nvml()
            self.duree = time.time() - self._t0
            for i, h in n.cartes:
                self.fin[i] = n.energie_mj(h)
                self.pids_fin[i] = n.pids(h)
        # Après `self.duree` : même raison qu'à l'entrée, son coût ne doit
        # pas s'ajouter à la fenêtre mesurée.
        self.charge_apres = relever_charge()
        pic = self._load1_max()
        if pic is not None and pic > self.nproc / 2 and os.environ.get("ACVRAM_CHARGE_OK") == "1":
            _marquer_charge_deliberee()
        self._stop = True
        if self._t:
            self._t.join(timeout=3)

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
            # `.get(i, ())` faisait passer une carte ABSENTE du releve de fin
            # pour une carte SANS PROCESSUS. Quand `avant` etait vide aussi,
            # les deux tuples etaient egaux et la fenetre etait declaree
            # valide — alors que le releve manquait. Une absence lue comme un
            # resultat, la faute que ce module est cense attraper.
            if i not in self.pids_fin:
                raisons.append(f"carte {i} : aucun releve de processus en fin "
                               f"de fenetre — la carte a disparu du recensement")
                continue
            apres = self.pids_fin[i]
            if avant != apres:
                raisons.append(f"carte {i} : les processus ont changé "
                               f"({list(avant)} -> {list(apres)})")
        # Une carte APPARUE en cours de fenetre n'etait pas regardee non plus :
        # la boucle ne parcourt que les cles du debut.
        for i in self.pids_fin:
            if i not in self.pids_debut:
                raisons.append(f"carte {i} : apparue en cours de fenetre, "
                               f"absente du recensement initial")
        if self.bridages:
            raisons.append("bridage pendant la fenêtre : " + ", ".join(sorted(self.bridages)))
        # Troisième garde, 14/09 (Sage) : campagne-20s-vllm-14-09.py ne
        # posait pas CUDA_VISIBLE_DEVICES — energie.py:80-83 (nvml() IGNORE
        # cette variable pour un usage normal, mais la respecte quand elle
        # EST posée) a alors agrégé la 3080 Ti au repos (~28,5 W) avec la
        # 5090 mesurée, gonflant tout le brut vLLM sans le dire. Une mesure
        # (`ACVRAM_TYPE=mesure`) qui couvre plus d'une carte sans que
        # l'appelant l'ait choisi explicitement est invalidée : le champ
        # `cartes` de `resume()` rend visible ce qui a été sommé.
        if os.environ.get("ACVRAM_TYPE") == "mesure" and len(self.debut) > 1:
            raisons.append(
                f"mesure sur {len(self.debut)} cartes ({sorted(self.debut)}) : "
                f"le brut agrège plusieurs cartes sans le dire — poser "
                f"CUDA_VISIBLE_DEVICES pour restreindre a une seule")
        # Deux gardes ajoutées le 14/09 (Sage/Jérôme, après le +51 % de
        # puissance_nvml.py:76 sur une fenêtre de 2 s) : une moyenne
        # au-dessus du plafond matériel est un signe que la fenêtre est
        # trop courte pour que le limiteur ait pu agir, pas que la carte a
        # dépassé sa limite ; une fenêtre de moins de 10 s n'a jamais assez
        # de marge pour que cette moyenne soit fiable. `getattr` : les
        # fenêtres construites à la main pour d'autres tests (avant cette
        # garde) ne posent pas toujours `duree`.
        duree = getattr(self, "duree", 0.0)
        if 0 < duree < 10.0:
            raisons.append(f"fenêtre trop courte pour une moyenne fiable : "
                           f"{duree:.2f} s < 10 s")
        # Le contrôle de plafond n'a de sens QUE sur une fenêtre déjà jugée
        # assez longue — en dessous, `self.plafond` (un appel NVML) n'a pas
        # à être sollicité du tout.
        if duree >= 10.0:
            plafond = self.plafond
            if plafond > 0 and self.moyenne > plafond:
                raisons.append(f"puissance moyenne {self.moyenne:.1f} W > "
                               f"plafond {plafond:.0f} W — fenêtre trop "
                               f"courte pour que le limiteur ait pu agir")
        # REGLES §2, "18/09, load 70,8 : trois pytest de pairs pendant une
        # fenêtre HTTP" : une charge CPU non annoncée fausse une mesure HTTP
        # comme une charge GPU non verrouillée fausse un ms/pas. Jugée sur le
        # MAX de load1 pendant TOUTE la fenêtre (boucle par tic, comme le
        # bridage), pas seulement à ses deux bouts -- une charge lancée à
        # mi-fenêtre serait sinon passée entre deux relevés. Refuse par
        # défaut ; ACVRAM_CHARGE_OK=1 l'accepte ET l'annonce (__exit__ écrit
        # CHARGE-DELIBEREE dans le verrou), comme charge-gpu.py.
        nproc = getattr(self, "nproc", None) or (os.cpu_count() or 1)
        pic = self._load1_max()
        if pic is not None and pic > nproc / 2 and os.environ.get("ACVRAM_CHARGE_OK") != "1":
            processus = (getattr(self, "charge_apres", None) or {}).get("processus_charges") \
                      or (getattr(self, "charge_avant", None) or {}).get("processus_charges")
            raisons.append(
                f"charge pendant la fenêtre : load1 max {pic:.2f} > nproc/2={nproc / 2} "
                f"({processus or 'aucun PID identifié >100% hors services permanents'}) "
                f"-- ACVRAM_CHARGE_OK=1 si délibérée")
        return raisons

    def _load1_max(self) -> float | None:
        """Le pic de `load1` pendant la fenêtre — la boucle par tic
        (`charges1`) s'il a tourné, sinon les deux bouts (`charge_avant`/
        `charge_apres`, fenêtres construites à la main comme `object.__new__
        (Energie)` dans des tests existants, ou NVML absent)."""
        pics = list(getattr(self, "charges1", None) or [])
        for c in (getattr(self, "charge_avant", None), getattr(self, "charge_apres", None)):
            if c:
                pics.append(c["load1"])
        return max(pics) if pics else None

    def resume(self) -> dict:
        h = [v for v in self.horloges if v >= 0]
        t = [v for v in self.temperatures if v >= 0]
        return {
            "joules": round(self.joules, 1),
            "watts": round(self.moyenne, 1),
            "duree_s": round(self.duree, 2),
            "cartes": sorted(self.debut),
            "plafond_w": round(self.plafond, 0),
            "horloge_min": min(h) if h else -1,
            "horloge_moy": round(sum(h) / len(h)) if h else -1,   # en-tête de toute cellule b=12 (Sage, sage-c16bis-puissance-mesure1-19-09)
            "horloge_max": max(h) if h else -1,
            "temp_max": max(t) if t else -1,
            "bridages": ",".join(sorted(self.bridages)) or "aucun",
            "load1_max": self._load1_max(),
            "charge_avant": getattr(self, "charge_avant", None),
            "charge_apres": getattr(self, "charge_apres", None),
            "invalidations": " ; ".join(self.invalidations) or "aucune",
            # le régime des noyaux (ACVRAM_PREFILL, MOE_MMA…) fait partie de la
            # mesure : sage-prefill-a8-verdict-17-09, un chiffre sans lui n'entre
            # plus dans INDEX
            "regime": _regime_noyaux(),
        }


def _regime_noyaux() -> str:
    try:
        import acvram
        return acvram.regime_ligne()
    except Exception as e:                       # noqa: BLE001 — energie.py sert aussi sans acvram
        return f"indisponible ({type(e).__name__}: {e})"


def repos(secondes: float = 30.0, periode: float = 1.0) -> Energie:
    """Ligne de base, prise APRÈS la mesure, serveur chargé mais inactif.

    Prise avant, elle dérive : sur une séance d'appareil photo du 7 septembre,
    la ligne de base du début a fait attribuer à un calcul une baisse qui
    n'était que la dérive de la base. Elle se prend donc après chaque cas, et
    de préférence de même durée que la fenêtre mesurée.

    ``secondes`` sous 30 s (14/09, duck.ai Jérôme) : un bruit sur la moyenne
    de repos se multiplie par la durée de la fenêtre mesurée dans
    ``joules_net`` — une ligne de base courte propage plus de bruit qu'elle
    n'en économise de temps. Averti, pas refusé : certaines campagnes
    (fenêtres elles-mêmes courtes) n'ont pas le choix.
    """
    if secondes < 30.0:
        print(f"[energie] repos({secondes:.1f} s) < 30 s : ligne de base "
              f"plus bruitée que recommandé (14/09, duck.ai Jérôme)",
              file=sys.stderr, flush=True)
    with Energie(periode=periode) as e:
        time.sleep(secondes)
    return e
