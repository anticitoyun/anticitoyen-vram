"""Profils matériels déclarés.

Un profil permet de bâtir un plan de placement pour une machine devant laquelle
on n'est pas assis. Le profil de la machine cible ci-dessous sert à préparer le
déploiement avant que le matériel ne soit joignable ; dès qu'``acvram detect``
tourne sur la vraie machine, il l'emporte toujours sur le profil.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from .detect import Cpu, Gpu, HostMemory, Rig

GB = 1024 ** 3

_BUILTIN: dict[str, dict] = {
    # ------------------------------------------------------------------
    # La machine cible. Les nombres sont des valeurs de plaque signalétique ;
    # tout ce que rapporte le vrai `acvram detect` les remplace.
    # ------------------------------------------------------------------
    "rig-14900k-5090-3080ti": {
        "description": "i9-14900K / ROG Maximus Z790 Dark Hero / 96 GB DDR5 / "
                       "RTX 5090 Astral LC OC 32 GB + RTX 3080 Ti 12 GB",
        "cpu": {
            "model": "Intel(R) Core(TM) i9-14900K",
            "physical_cores": 24,      # 8 P + 16 E
            "logical_cores": 32,
            "performance_cores": 8,
            "efficiency_cores": 16,
            "p_core_cpuset": "0-15",   # sur Raptor Lake, les jumeaux SMT des cœurs P viennent en premier
        },
        "host": {
            "total": 96 * GB,
            "available": 88 * GB,
            "numa_nodes": 1,
        },
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 5090",
                "total_mem": 32 * GB,
                "sm": 120,
                "pci_bus_id": "00000000:01:00.0",
                # Le port 1 de la Dark Hero est en PCIe 5.0 x16 relié au processeur.
                "pcie_gen_max": 5, "pcie_gen_cur": 5,
                "pcie_width_max": 16, "pcie_width_cur": 16,
                "power_limit_w": 600.0,
            },
            {
                "index": 1,
                "name": "NVIDIA GeForce RTX 3080 Ti",
                "total_mem": 12 * GB,
                "sm": 86,
                "pci_bus_id": "00000000:02:00.0",
                # Le second port passe par le chipset et est bien plus étroit.
                # Valeur pessimiste à dessein : le planificateur ne doit pas
                # supposer un trafic hôte-GPU1 bon marché. Les vraies valeurs
                # viennent de la détection.
                "pcie_gen_max": 4, "pcie_gen_cur": 4,
                "pcie_width_max": 4, "pcie_width_cur": 4,
                "power_limit_w": 350.0,
            },
        ],
        # Aucune des deux cartes n'a de NVLink, et NVIDIA désactive le pair-à-pair PCIe sur GeForce.
        "p2p_matrix": [[True, False], [False, True]],
        "driver_version": "580.00",
        "cuda_version": "13.0",
        "distro": "Linux Mint 22.3",
    },
}


def list_profiles() -> dict[str, str]:
    out = {k: v.get("description", "") for k, v in _BUILTIN.items()}
    for path in _user_profile_paths():
        try:
            data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
        out[name] = data.get("description", f"(from {path})")
    return out


def _user_profile_paths() -> list[str]:
    roots = [
        os.path.join(os.path.dirname(__file__), "..", "config"),
        os.path.expanduser("~/.config/acvram/profiles"),
    ]
    paths = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for fn in sorted(os.listdir(root)):
            if fn.endswith((".yaml", ".yml")):
                paths.append(os.path.join(root, fn))
    return paths


def _from_dict(data: dict, name: str) -> Rig:
    rig = Rig(source=f"profile:{name}")
    rig.cpu = Cpu(**data.get("cpu", {}))
    rig.host = HostMemory(**data.get("host", {}))
    rig.gpus = [Gpu(**g) for g in data.get("gpus", [])]
    rig.p2p_matrix = data.get("p2p_matrix") or [
        [i == j for j in range(len(rig.gpus))] for i in range(len(rig.gpus))
    ]
    rig.driver_version = data.get("driver_version", "")
    rig.cuda_version = data.get("cuda_version", "")
    rig.distro = data.get("distro", "")
    rig.kernel = data.get("kernel", "")
    return rig


def load_profile(name: str) -> Rig:
    if name in _BUILTIN:
        return _from_dict(_BUILTIN[name], name)
    for path in _user_profile_paths():
        try:
            data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        pname = data.get("name") or os.path.splitext(os.path.basename(path))[0]
        if pname == name or path == name:
            return _from_dict(data, pname)
    if os.path.isfile(name):
        data = yaml.safe_load(open(name, "r", encoding="utf-8")) or {}
        return _from_dict(data, os.path.basename(name))
    known = ", ".join(list_profiles())
    raise KeyError(f"unknown profile {name!r}; known profiles: {known}")
