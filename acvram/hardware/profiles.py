"""Declared hardware profiles.

A profile lets you build a placement plan for a machine you are not sitting
in front of. The target rig profile below is used to prepare the deployment
before the hardware is reachable; once ``acvram detect`` runs on the real box
it always wins over the profile.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from .detect import Cpu, Gpu, HostMemory, Rig

GB = 1024 ** 3

_BUILTIN: dict[str, dict] = {
    # ------------------------------------------------------------------
    # The target machine. Numbers are nameplate values; anything the real
    # `acvram detect` reports supersedes them.
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
            "p_core_cpuset": "0-15",   # P-core SMT siblings come first on RPL
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
                # Slot 1 on the Dark Hero is CPU-attached PCIe 5.0 x16.
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
                # Second slot is chipset-attached and much narrower. This is a
                # pessimistic default on purpose: the planner must not assume
                # cheap host<->GPU1 traffic. Real values come from detection.
                "pcie_gen_max": 4, "pcie_gen_cur": 4,
                "pcie_width_max": 4, "pcie_width_cur": 4,
                "power_limit_w": 350.0,
            },
        ],
        # No NVLink on either board, and NVIDIA disables PCIe P2P on GeForce.
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
