"""Hardware discovery and per-device capability resolution.

Deliberately dependency-light: this module must work on a machine with no
torch and no CUDA (e.g. while planning a deployment from a laptop), falling
back through NVML -> nvidia-smi -> torch -> a declared static profile.

The output of :func:`detect_rig` is what every other subsystem consumes to
decide *which numeric format a given weight is stored in*, because the two
GPUs in the target rig do not have the same tensor-core capabilities:

    RTX 5090   GB202  sm_120  FP4 + FP8 tensor cores  -> NVFP4 weights
    RTX 3080Ti GA102  sm_86   no FP8, no FP4          -> INT4 weight-only
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

__all__ = [
    "GpuCaps",
    "Gpu",
    "HostMemory",
    "Cpu",
    "Rig",
    "detect_rig",
    "capabilities_for_sm",
]

_SMI_FIELDS = [
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "memory.total",
    "memory.used",
    "compute_cap",
    "pcie.link.gen.max",
    "pcie.link.gen.current",
    "pcie.link.width.max",
    "pcie.link.width.current",
    "driver_version",
    "power.max_limit",
    "clocks.max.memory",
]


# --------------------------------------------------------------------------
# capability model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuCaps:
    """What a given compute capability can actually do in *hardware*.

    ``fp4_tensor_core`` is the discriminator that splits the target rig: it is
    true only for Blackwell (sm_100/103/120+). Everything else has to reach a
    4-bit footprint through *weight-only* quantization, i.e. store 4 bits and
    dequantize to a format the tensor cores do support.
    """

    sm: int
    fp4_tensor_core: bool          # NVFP4 / MXFP4 mma (Blackwell)
    fp8_tensor_core: bool          # E4M3 / E5M2 mma (Ada 89+, Hopper, Blackwell)
    int4_tensor_core: bool         # s4 mma (Turing 75 .. Ampere 86, removed after)
    int8_tensor_core: bool
    bf16: bool
    tf32: bool
    async_copy: bool               # cp.async (sm_80+)
    tma: bool                      # tensor memory accelerator (sm_90+)
    # native weight format chosen for this device
    weight_format: str
    kv_format: str

    @property
    def arch_name(self) -> str:
        return {
            75: "Turing",
            80: "Ampere-A100",
            86: "Ampere",
            89: "Ada Lovelace",
            90: "Hopper",
            100: "Blackwell-DC",
            103: "Blackwell-DC",
            120: "Blackwell",
        }.get(self.sm, f"sm_{self.sm}")


def capabilities_for_sm(sm: int) -> GpuCaps:
    """Map a compute capability (e.g. 120 for sm_120) to a capability set.

    The weight/KV format choice encoded here is the single place where the
    "one format per GPU" policy lives.
    """
    fp4 = sm >= 100                       # Blackwell family (100, 103, 120, 121)
    fp8 = sm >= 89                        # Ada onward
    # s4 mma exists on Turing and Ampere only; nvcc dropped it for sm_90+.
    int4_mma = 75 <= sm <= 89
    int8_mma = sm >= 75
    bf16 = sm >= 80
    tf32 = sm >= 80
    async_copy = sm >= 80
    tma = sm >= 90

    # KV cache defaults to INT8 even where FP8 exists. Measured on this
    # codebase (tests/test_kvcache.py): with one scale per (token, head), INT8
    # reaches ~44 dB against ~32 dB for FP8 E4M3 at identical size, because
    # per-head scaling already supplies the dynamic range that FP8 spends
    # exponent bits on, while INT8 keeps a uniform grid. FP8 is still worth
    # selecting when an attention kernel can consume it without dequantizing
    # -- that is a throughput argument, not an accuracy one -- so it stays
    # available through --kv-format fp8_e4m3.
    if fp4:
        weight_format, kv_format = "nvfp4", "int8"
    elif fp8:
        # Ada: no FP4 tensor cores, but FP8 storage + FP8 mma is available.
        weight_format, kv_format = "int4_awq", "int8"
    elif bf16:
        # Ampere consumer (3080 Ti): weight-only INT4, dequantized to FP16 on
        # the fly, then a normal FP16 tensor-core GEMM.
        weight_format, kv_format = "int4_awq", "int8"
    else:
        weight_format, kv_format = "int8", "fp16"

    return GpuCaps(
        sm=sm,
        fp4_tensor_core=fp4,
        fp8_tensor_core=fp8,
        int4_tensor_core=int4_mma,
        int8_tensor_core=int8_mma,
        bf16=bf16,
        tf32=tf32,
        async_copy=async_copy,
        tma=tma,
        weight_format=weight_format,
        kv_format=kv_format,
    )


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass
class Gpu:
    index: int
    name: str
    total_mem: int                      # bytes
    used_mem: int = 0
    uuid: str = ""
    pci_bus_id: str = ""
    sm: int = 0
    pcie_gen_max: int = 0
    pcie_gen_cur: int = 0
    pcie_width_max: int = 0
    pcie_width_cur: int = 0
    driver_version: str = ""
    power_limit_w: float = 0.0
    mem_clock_max_mhz: int = 0
    caps: Optional[GpuCaps] = None

    def __post_init__(self) -> None:
        if self.caps is None and self.sm:
            self.caps = capabilities_for_sm(self.sm)

    # -- derived ---------------------------------------------------------
    @property
    def free_mem(self) -> int:
        return max(0, self.total_mem - self.used_mem)

    @property
    def host_link_gbps(self) -> float:
        """Effective host<->device bandwidth in GB/s for the *current* link.

        PCIe raw rates per lane, 128b/130b encoded, one direction:
            gen3 ~0.985, gen4 ~1.969, gen5 ~3.938 GB/s per lane.
        A 0.85 factor approximates real achievable copy throughput.
        """
        per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.877}
        gen = self.pcie_gen_cur or self.pcie_gen_max or 4
        width = self.pcie_width_cur or self.pcie_width_max or 16
        return per_lane.get(gen, 1.969) * width * 0.85

    @property
    def vram_bandwidth_gbps(self) -> float:
        """Rough on-package bandwidth, used only to rank tiers."""
        table = {
            "5090": 1792.0, "5080": 960.0, "4090": 1008.0, "4080": 717.0,
            "3090": 936.0, "3080 ti": 912.0, "3080": 760.0, "3060": 360.0,
        }
        low = self.name.lower()
        for key, bw in table.items():
            if key in low:
                return bw
        return 500.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.caps:
            d["caps"] = asdict(self.caps)
            d["caps"]["arch_name"] = self.caps.arch_name
        d["host_link_gbps"] = round(self.host_link_gbps, 1)
        d["vram_bandwidth_gbps"] = self.vram_bandwidth_gbps
        return d


@dataclass
class HostMemory:
    total: int = 0
    available: int = 0
    hugepages_total: int = 0
    hugepage_size: int = 0
    numa_nodes: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Cpu:
    model: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    performance_cores: int = 0     # P-cores on hybrid Intel
    efficiency_cores: int = 0      # E-cores
    p_core_cpuset: str = ""        # e.g. "0-15" -- what we pin worker threads to

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Rig:
    gpus: list[Gpu] = field(default_factory=list)
    host: HostMemory = field(default_factory=HostMemory)
    cpu: Cpu = field(default_factory=Cpu)
    driver_version: str = ""
    cuda_version: str = ""
    kernel: str = ""
    distro: str = ""
    p2p_matrix: list[list[bool]] = field(default_factory=list)
    source: str = "unknown"        # nvml | nvidia-smi | torch | profile

    @property
    def total_vram(self) -> int:
        return sum(g.total_mem for g in self.gpus)

    def gpu_by_index(self, i: int) -> Gpu:
        for g in self.gpus:
            if g.index == i:
                return g
        raise KeyError(f"no GPU with index {i}")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "kernel": self.kernel,
            "distro": self.distro,
            "cpu": self.cpu.to_dict(),
            "host": self.host.to_dict(),
            "gpus": [g.to_dict() for g in self.gpus],
            "p2p_matrix": self.p2p_matrix,
            "total_vram": self.total_vram,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 10.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _probe_gpus_smi() -> tuple[list[Gpu], str]:
    out = _run(["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}",
                "--format=csv,noheader,nounits"])
    if not out:
        return [], ""
    gpus: list[Gpu] = []
    driver = ""
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_SMI_FIELDS):
            continue
        rec = dict(zip(_SMI_FIELDS, parts))

        def _i(key: str, default: int = 0) -> int:
            try:
                return int(float(rec[key]))
            except (ValueError, KeyError):
                return default

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(rec[key])
            except (ValueError, KeyError):
                return default

        cc = rec.get("compute_cap", "")
        sm = 0
        m = re.match(r"^(\d+)\.(\d+)$", cc)
        if m:
            sm = int(m.group(1)) * 10 + int(m.group(2))
        driver = rec.get("driver_version", driver)
        gpus.append(Gpu(
            index=_i("index"),
            name=rec.get("name", "unknown"),
            uuid=rec.get("uuid", ""),
            pci_bus_id=rec.get("pci.bus_id", ""),
            total_mem=_i("memory.total") * 1024 * 1024,
            used_mem=_i("memory.used") * 1024 * 1024,
            sm=sm,
            pcie_gen_max=_i("pcie.link.gen.max"),
            pcie_gen_cur=_i("pcie.link.gen.current"),
            pcie_width_max=_i("pcie.link.width.max"),
            pcie_width_cur=_i("pcie.link.width.current"),
            driver_version=rec.get("driver_version", ""),
            power_limit_w=_f("power.max_limit"),
            mem_clock_max_mhz=_i("clocks.max.memory"),
        ))
    return gpus, driver


def _probe_gpus_torch() -> tuple[list[Gpu], str]:
    try:
        import torch
    except Exception:
        return [], ""
    if not torch.cuda.is_available():
        return [], ""
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(Gpu(
            index=i,
            name=p.name,
            total_mem=p.total_memory,
            sm=p.major * 10 + p.minor,
            pci_bus_id=f"{getattr(p, 'pci_bus_id', 0)}",
        ))
    return gpus, torch.version.cuda or ""


def _probe_p2p(n: int) -> list[list[bool]]:
    """Peer-to-peer reachability.

    Consumer GeForce boards have no NVLink and NVIDIA disables PCIe P2P on
    them, so this is expected to be all-false on the target rig -- which is
    why the runner stages inter-GPU tensors through pinned host memory.
    """
    if n < 2:
        return [[True]] if n == 1 else []
    try:
        import torch
        if torch.cuda.is_available():
            return [[bool(i == j or torch.cuda.can_device_access_peer(i, j))
                     for j in range(n)] for i in range(n)]
    except Exception:
        pass
    return [[i == j for j in range(n)] for i in range(n)]


def _probe_host_memory() -> HostMemory:
    hm = HostMemory()
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            info = {}
            for line in fh:
                key, _, rest = line.partition(":")
                info[key.strip()] = rest.strip()
        def kb(key: str) -> int:
            v = info.get(key, "0 kB").split()[0]
            return int(v) * 1024
        hm.total = kb("MemTotal")
        hm.available = kb("MemAvailable")
        hm.hugepages_total = int(info.get("HugePages_Total", "0").split()[0])
        hm.hugepage_size = kb("Hugepagesize")
    except (OSError, ValueError, IndexError):
        pass
    try:
        hm.numa_nodes = len([d for d in os.listdir("/sys/devices/system/node")
                             if d.startswith("node")]) or 1
    except OSError:
        hm.numa_nodes = 1
    return hm


def _probe_cpu() -> Cpu:
    cpu = Cpu()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return cpu
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    if m:
        cpu.model = m.group(1).strip()
    cpu.logical_cores = len(re.findall(r"^processor\s*:", text, re.M))
    cores = set(re.findall(r"^core id\s*:\s*(\d+)", text, re.M))
    cpu.physical_cores = len(cores) or cpu.logical_cores

    # Hybrid Intel: P-cores expose SMT siblings, E-cores do not. Reading the
    # per-CPU topology is the only reliable discriminator.
    p_cpus: list[int] = []
    e_cpus: list[int] = []
    base = "/sys/devices/system/cpu"
    try:
        for entry in sorted(os.listdir(base)):
            m = re.fullmatch(r"cpu(\d+)", entry)
            if not m:
                continue
            n = int(m.group(1))
            sib_path = os.path.join(base, entry, "topology/thread_siblings_list")
            try:
                with open(sib_path, "r", encoding="utf-8") as fh:
                    sibs = fh.read().strip()
            except OSError:
                continue
            multi = ("," in sibs) or ("-" in sibs)
            (p_cpus if multi else e_cpus).append(n)
    except OSError:
        pass
    if p_cpus and e_cpus:
        cpu.performance_cores = len(p_cpus) // 2
        cpu.efficiency_cores = len(e_cpus)
        cpu.p_core_cpuset = _ranges(p_cpus)
    else:
        cpu.performance_cores = cpu.physical_cores
        cpu.p_core_cpuset = _ranges(list(range(cpu.logical_cores)))
    return cpu


def _ranges(nums: list[int]) -> str:
    if not nums:
        return ""
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def _probe_cuda_version() -> str:
    out = _run(["nvcc", "--version"])
    if out:
        m = re.search(r"release (\d+\.\d+)", out)
        if m:
            return m.group(1)
    try:
        import torch
        return torch.version.cuda or ""
    except Exception:
        return ""


def _probe_distro() -> str:
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.platform()


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def detect_rig(profile: Optional[str] = None) -> Rig:
    """Detect the local machine, or load a declared profile.

    ``profile`` is the escape hatch used to plan a deployment from a machine
    that is not the target (``acvram plan --profile rig-14900k-5090-3080ti``).
    """
    if profile:
        from .profiles import load_profile
        return load_profile(profile)

    rig = Rig()
    gpus, driver = _probe_gpus_smi()
    rig.source = "nvidia-smi"
    if not gpus:
        gpus, cuda = _probe_gpus_torch()
        rig.source = "torch" if gpus else "none"
    rig.gpus = gpus
    rig.driver_version = driver
    rig.cuda_version = _probe_cuda_version()
    rig.host = _probe_host_memory()
    rig.cpu = _probe_cpu()
    rig.kernel = platform.release()
    rig.distro = _probe_distro()
    rig.p2p_matrix = _probe_p2p(len(gpus))
    return rig
