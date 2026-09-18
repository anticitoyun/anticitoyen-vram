"""Relevé de tous les capteurs de la machine — pour la console et l'interface.

Trois sources, chacune lue directement, sans démon intermédiaire :

* ``/sys/class/hwmon`` — ce que le noyau expose : températures, ventilateurs,
  tensions, puissances, courants, avec leurs seuils quand la puce les donne.
  C'est la même source que ``sensors`` (lm-sensors), lue sans passer par lui.
* ``nvidia-smi`` — les cartes NVIDIA : températures GPU et mémoire, ventilateur,
  horloges, puissance, et surtout les **raisons de bridage** — la seule
  information qui dise *pourquoi* une carte tourne moins vite qu'elle ne peut.
* ``psutil`` — processeur, mémoire, disques, réseau. Optionnel : sans lui, ces
  sections manquent, le reste tient.

POURQUOI LIRE hwmon PLUTÔT QUE ``sensors`` : ``sensors`` formate pour l'œil et
perd les seuils ; hwmon donne la valeur brute (millidegrés, millivolts,
microwatts) et les fichiers ``_max``, ``_crit``, ``_min`` à côté. Une jauge sans
seuil est un chiffre ; une jauge avec seuil est une alerte.

Tout est en unités lisibles à la sortie : °C, tr/min, V, W, A.
"""
from __future__ import annotations

import collections
import ctypes
import glob
import os
import subprocess
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover — dépendance facultative
    psutil = None

# Noms lisibles pour les puces hwmon les plus courantes. Un nom inconnu passe
# tel quel : mieux vaut « nct6798 » que « puce inconnue ».
_NOMS_PUCES = {
    "coretemp": "processeur (cœurs)",
    "k10temp": "processeur",
    "zenpower": "processeur",
    "nvme": "disque NVMe",
    "acpitz": "zone thermique ACPI",
    "iwlwifi_1": "Wi-Fi",
    "corsairpsu": "alimentation Corsair",
    "nct6798": "carte mère (nct6798)",
    "nct6775": "carte mère",
    "it8688": "carte mère",
    "asus": "carte mère (asus)",
    "amdgpu": "GPU AMD",
    "drivetemp": "disque SATA",
    "spd5118": "barrette DDR5",
    "jc42": "barrette mémoire",
}

# Facteur de conversion par famille de fichier hwmon : la valeur brute est
# toujours un entier dans une sous-unité fixe.
_FAMILLES = {
    "temp":  ("°C",     1000.0),
    "fan":   ("tr/min", 1.0),
    "in":    ("V",      1000.0),
    "power": ("W",      1_000_000.0),
    "curr":  ("A",      1000.0),
    "humidity": ("%",   1000.0),
}
_SEUILS = ("max", "crit", "min", "lcrit", "emergency", "alarm")


def _lire(chemin: str) -> str | None:
    try:
        with open(chemin) as f:
            return f.read().strip()
    except OSError:
        return None


def _nombre(chemin: str, div: float) -> float | None:
    v = _lire(chemin)
    if v is None:
        return None
    try:
        return int(v) / div
    except ValueError:
        return None


def _puce_hwmon(dossier: str) -> dict | None:
    nom = _lire(os.path.join(dossier, "name")) or os.path.basename(dossier)
    # Le chemin du périphérique distingue quatre NVMe qui portent tous le nom
    # « nvme » : on prend le dernier segment (nvme0, nvme1…).
    dev = os.path.realpath(os.path.join(dossier, "device"))
    ident = os.path.basename(dev) if os.path.isdir(dev) else ""
    lectures: list[dict] = []
    for fichier in sorted(os.listdir(dossier)):
        if not fichier.endswith("_input"):
            continue
        base = fichier[: -len("_input")]                # temp1, fan2, in0…
        famille = base.rstrip("0123456789")
        if famille not in _FAMILLES:
            continue
        unite, div = _FAMILLES[famille]
        valeur = _nombre(os.path.join(dossier, fichier), div)
        if valeur is None:
            continue
        # Un ventilateur à 0 tr/min sur une prise vide n'est pas une panne :
        # la carte mère en expose sept, la moitié n'ont rien de branché.
        if famille == "fan" and valeur == 0:
            continue
        # Meme chose pour une temperature a 0,0 exactement : la nct6798 expose
        # des entrees PCH_* que rien n'alimente sur cette carte mere. 0 °C
        # dans une piece a 20 est une absence, pas une lecture.
        if famille == "temp" and valeur == 0:
            continue
        etiquette = _lire(os.path.join(dossier, base + "_label")) or base
        seuils = {}
        for s in _SEUILS:
            if s == "alarm":
                a = _lire(os.path.join(dossier, base + "_alarm"))
                if a is not None:
                    seuils["alarme"] = a == "1"
                continue
            x = _nombre(os.path.join(dossier, base + "_" + s), div)
            if x is not None:
                seuils[s] = round(x, 3)
        lectures.append({
            "id": base, "etiquette": etiquette, "famille": famille,
            "valeur": round(valeur, 3), "unite": unite, **seuils,
        })
    if not lectures:
        return None
    return {
        "puce": nom, "ident": ident,
        "nom": _NOMS_PUCES.get(nom, nom) + (f" · {ident}" if ident and nom == "nvme" else ""),
        "lectures": lectures,
    }


def hwmon() -> list[dict]:
    puces = []
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        p = _puce_hwmon(d)
        if p:
            puces.append(p)
    return puces


# Les raisons de bridage, bit par bit, d'après nvidia-smi --help-query-gpu.
# C'est la valeur la plus utile de toute la page : une carte à 70 % de sa
# fréquence sans raison affichée est un mystère, avec la raison c'est un fait.
_BRIDAGES = {
    0x0001: "inactive",
    0x0002: "réglage applicatif",
    0x0004: "limite de puissance logicielle",
    0x0008: "limite matérielle (PCIe/alim)",
    0x0010: "synchronisation",
    0x0020: "limite thermique logicielle",
    0x0040: "limite thermique matérielle",
    0x0080: "puissance matérielle",
    0x0100: "horloge affichage",
}


def nvidia() -> list[dict]:
    champs = ["index", "name", "temperature.gpu", "temperature.memory",
              "fan.speed", "clocks.sm", "clocks.max.sm", "clocks.mem",
              "power.draw", "power.limit", "utilization.gpu",
              "utilization.memory", "memory.used", "memory.total",
              "pcie.link.gen.current", "pcie.link.width.current",
              "clocks_throttle_reasons.active", "encoder.stats.sessionCount"]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(champs)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    cartes = []
    for ligne in out.strip().splitlines():
        c = [x.strip() for x in ligne.split(",")]
        if len(c) < len(champs):
            continue
        d = dict(zip(champs, c))

        def n(k):
            try:
                return float(d[k])
            except ValueError:
                return None            # « N/A » (temperature.memory sur GeForce)
        try:
            masque = int(d["clocks_throttle_reasons.active"], 16)
        except ValueError:
            masque = 0
        raisons = [v for b, v in _BRIDAGES.items() if masque & b and b != 0x0001]
        cartes.append({
            "index": int(d["index"]), "nom": d["name"],
            "temp_gpu": n("temperature.gpu"), "temp_mem": n("temperature.memory"),
            "ventilateur_pct": n("fan.speed"),
            "horloge_sm": n("clocks.sm"), "horloge_sm_max": n("clocks.max.sm"),
            "horloge_mem": n("clocks.mem"),
            "watts": n("power.draw"), "watts_max": n("power.limit"),
            "occupation": n("utilization.gpu"), "occupation_mem": n("utilization.memory"),
            "mio_pris": n("memory.used"), "mio_total": n("memory.total"),
            "pcie_gen": n("pcie.link.gen.current"), "pcie_largeur": n("pcie.link.width.current"),
            "bridages": raisons,
            "encodeurs": n("encoder.stats.sessionCount"),
        })
    return cartes


_nvml_lib: object = None   # None = pas encore essayé, False = essayé et absent


def _nvml():
    """Charge NVML une fois (ctypes, sans dépendance ``pynvml``).

    Ajout GUI n°3 (poste7-gui-ajouts-18-09) : `nvidia-smi
    --query-gpu=total_energy_consumption` échoue sur cette machine ; NVML
    tient le compteur d'énergie monotone que `/metrics` a besoin
    d'intégrer sur sa fenêtre glissante. `outils/gpu/mesure/energie.py`
    fait la même chose mais n'est pas empaqueté dans le `.deb` — ce
    lecteur est donc son propre code, minimal, pas un import croisé.
    """
    global _nvml_lib
    if _nvml_lib is None:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit_v2 refusé")
        except OSError:
            _nvml_lib = False
        else:
            _nvml_lib = lib
    return _nvml_lib or None


def energie_mj() -> dict[int, int]:
    """{index carte: millijoules totaux depuis le démarrage du pilote}.

    Vide si NVML est absent — jamais une valeur inventée à la place."""
    lib = _nvml()
    if lib is None:
        return {}
    n = ctypes.c_uint()
    lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
    out: dict[int, int] = {}
    for i in range(n.value):
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
            continue
        v = ctypes.c_ulonglong()
        if lib.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(v)) == 0:
            out[i] = v.value
    return out


class AnneauEnergie:
    """Fenêtre glissante de J/jeton, alimentée par un tic d'arrière-plan.

    Correction (poste7-metrics-energie-fenetre-18-09) : la première version
    de `/metrics` consommait son propre échantillon précédent à chaque
    appel — deux lecteurs concurrents (deux clients qui pollent `/metrics`
    en même temps) se volaient donc le delta l'un l'autre, et rendaient
    DEUX valeurs différentes sur la MÊME fenêtre réelle (trouvé par poste3
    le 18/09 lors du contrôle GUI). Ici le tic alimente un anneau
    d'échantillons partagé ; `j_par_jeton()` ne fait que LIRE le premier
    et le dernier échantillon de la fenêtre — appelable par autant de
    lecteurs qu'on veut, sans état par client, sans jamais consommer."""

    def __init__(self, fenetre_s: float = 10.0, tic_s: float = 1.0,
                 lecteur=None) -> None:
        self.fenetre_s = fenetre_s
        self.tic_s = tic_s
        self._lecteur = lecteur if lecteur is not None else energie_mj
        self._echantillons: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._decode_tokens = None

    def start(self, decode_tokens) -> None:
        """`decode_tokens` : fonction sans argument rendant le compteur
        courant de jetons décodés (`engine.stats.decode_tokens`)."""
        if self._thread is not None:
            return
        self._decode_tokens = decode_tokens
        self._thread = threading.Thread(target=self._run,
                                        name="acvram-energie", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tic(self._decode_tokens())
            self._stop.wait(self.tic_s)

    def tic(self, decode_tokens: int) -> None:
        """Un échantillon : énergie totale (toutes cartes sommées, comme
        `energie.py`) + jetons décodés à l'instant T. Appelable directement
        (tests) ou depuis le fil de fond (serveur réel)."""
        mj = self._lecteur()
        total_mj = sum(mj.values()) if mj else None
        maintenant = time.time()
        with self._lock:
            self._echantillons.append((maintenant, total_mj, decode_tokens))
            limite = maintenant - self.fenetre_s
            while len(self._echantillons) > 1 and self._echantillons[0][0] < limite:
                self._echantillons.popleft()

    def jetons_fenetre(self) -> int:
        """Jetons décodés couverts par la fenêtre actuelle — 0 si moins de
        deux échantillons ou si le compteur n'a pas bougé. poste7
        (poste7-metrics-energie-fenetre-18-09) : `j_par_jeton()` rend None
        SEULEMENT quand ce compte vaut 0, jamais pour une autre raison
        silencieuse — les deux se lisent toujours ensemble."""
        with self._lock:
            echantillons = list(self._echantillons)
        if len(echantillons) < 2:
            return 0
        delta_tok = echantillons[-1][2] - echantillons[0][2]
        return delta_tok if delta_tok > 0 else 0

    def j_par_jeton(self) -> float | None:
        """Lecture seule, jamais 0 : None si NVML absent, fenêtre trop
        jeune (< 2 échantillons), ou aucun jeton décodé dans la fenêtre
        (`jetons_fenetre() == 0`)."""
        with self._lock:
            echantillons = list(self._echantillons)
        if len(echantillons) < 2:
            return None
        _, mj0, tok0 = echantillons[0]
        _, mj1, tok1 = echantillons[-1]
        delta_tok = tok1 - tok0
        if delta_tok <= 0:
            return None
        if mj0 is None or mj1 is None:
            return None
        delta_mj = mj1 - mj0
        if delta_mj < 0:
            return None
        return round(delta_mj / 1000.0 / delta_tok, 4)


def systeme() -> dict:
    """Processeur, mémoire, disques, réseau — via psutil s'il est là."""
    if psutil is None:
        return {"disponible": False}
    freq = psutil.cpu_freq(percpu=False)
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    disques = []
    for part in psutil.disk_partitions(all=False):
        if part.fstype in ("squashfs", "tmpfs", "devtmpfs", "overlay"):
            continue
        try:
            u = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        disques.append({
            "montage": part.mountpoint, "dev": part.device, "fs": part.fstype,
            "gio_total": round(u.total / 2**30, 1), "gio_pris": round(u.used / 2**30, 1),
            "pct": u.percent,
        })
    charge = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
    return {
        "disponible": True,
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_par_coeur": psutil.cpu_percent(interval=None, percpu=True),
        "cpu_mhz": round(freq.current) if freq else None,
        "cpu_mhz_max": round(freq.max) if freq and freq.max else None,
        "charge_1_5_15": [round(x, 2) for x in charge],
        "ram_gio_total": round(vm.total / 2**30, 1),
        "ram_gio_prise": round((vm.total - vm.available) / 2**30, 1),
        "ram_pct": vm.percent,
        "swap_gio_total": round(sw.total / 2**30, 1),
        "swap_gio_pris": round(sw.used / 2**30, 1),
        "disques": disques,
        "processus": len(psutil.pids()),
        "depuis": time.time() - psutil.boot_time(),
    }


def relever() -> dict:
    """Tout, en un appel. Chaque source est indépendante : l'absence de l'une
    ne masque pas les autres."""
    return {
        "horodatage": time.time(),
        "hwmon": hwmon(),
        "nvidia": nvidia(),
        "systeme": systeme(),
    }


if __name__ == "__main__":         # relevé brut, pour vérifier à la main
    import json
    print(json.dumps(relever(), indent=1, ensure_ascii=False))
