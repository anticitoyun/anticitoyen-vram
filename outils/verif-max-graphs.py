#!/usr/bin/env python3
"""Vérification ACVRAM_CHRONO_SYNC : bind/fill/replay vs pas_total.

Usage :
    python outils/verif-max-graphs.py MODEL_DIR SLOTS N_TOURS CTX SORTIE [MAX_MODEL_LEN]

ACVRAM_HYBRID_SLOTS est positionné d'après SLOTS.
ACVRAM_CHRONO_SYNC contrôle la synchronisation — setter à 1 pour le bras B.
"""
import json
import os
import sys
import threading
import time

import torch

MODEL_DIR = sys.argv[1]
SLOTS = int(sys.argv[2])
N_TOURS = int(sys.argv[3])
CTX = int(sys.argv[4])
SORTIE = sys.argv[5]
MAX_MODEL_LEN = int(sys.argv[6]) if len(sys.argv) > 6 else CTX

os.environ["ACVRAM_HYBRID_SLOTS"] = str(SLOTS)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# OOM-safe : afficher max_memory_reserved avant propagation
def _excepthook_oom(exc_type, exc_val, exc_tb):
    if issubclass(exc_type, torch.cuda.OutOfMemoryError):
        try:
            gio = torch.cuda.max_memory_reserved(0) / 1024 ** 3
            print(f"[OOM] max_memory_reserved={gio:.4f} Gio", flush=True)
        except Exception:
            pass
    import traceback
    traceback.print_exception(exc_type, exc_val, exc_tb)


sys.excepthook = _excepthook_oom

# Filtrage par uuid : ne jamais filtrer sans uuid (piège du 10/09)
UUID_5090 = "GPU-7985099b-7af0-3742-eb7a-f16284734f37"


class VeilleurFrequence:
    """Thread de surveillance GPU : occupants étrangers toutes les 50 ms."""

    def __init__(self, device_uuid: str, intervalle_s: float = 0.05):
        self.uuid = device_uuid
        self.intervalle = intervalle_s
        self._stop = threading.Event()
        self._handle = None
        self.pollution: list = []  # (pid, mem_mio, epoch)
        self._thread: threading.Thread | None = None
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                if pynvml.nvmlDeviceGetUUID(h) == device_uuid:
                    self._handle = h
                    self._pynvml = pynvml
                    break
        except Exception:
            pass

    def _boucle(self) -> None:
        mon_pid = os.getpid()
        while not self._stop.is_set():
            if self._handle is not None:
                try:
                    procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(
                        self._handle
                    )
                    epoch = time.time()
                    for p in procs:
                        if p.pid != mon_pid:
                            self.pollution.append(
                                (p.pid, p.usedGpuMemory // 1024 ** 2, epoch)
                            )
                except Exception:
                    pass
            self._stop.wait(self.intervalle)

    def start(self) -> "VeilleurFrequence":
        self._thread = threading.Thread(target=self._boucle, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


def median(lst: list) -> float | None:
    if not lst:
        return None
    s = sorted(lst)
    return s[len(s) // 2]


# --- chargement ---

from acvram.engine.loader import load_model  # noqa: E402
from acvram.engine.runner import Engine  # noqa: E402
from acvram.engine.sampler import SamplingParams  # noqa: E402

print(f"[INFO] chargement {MODEL_DIR}  max_model_len={MAX_MODEL_LEN}", flush=True)
t_charge0 = time.perf_counter()
loaded = load_model(MODEL_DIR, dtype=torch.bfloat16, max_model_len=MAX_MODEL_LEN)
load_s = time.perf_counter() - t_charge0
print(f"[INFO] chargé en {load_s:.1f} s", flush=True)

engine = Engine(loaded, None, max_batch_size=SLOTS, max_model_len=MAX_MODEL_LEN)

# warm_graphs : capture des graphes CUDA (allocation paresseuse terminée)
print(f"[INFO] warm_graphs slots={SLOTS}", flush=True)
n_captures = engine.warm_graphs()
print(f"[INFO] {n_captures} graphes capturés", flush=True)

# RESET PIC VRAM — après chauffe, avant mesures.
# Sans ce reset, le pic reflète le chargement des poids, pas l'allocation
# des créneaux ; slots=4 et slots=12 montreraient la même valeur.
torch.cuda.reset_peak_memory_stats(0)
print("[INFO] reset_peak_memory_stats effectué", flush=True)

# --- mesures ---

veilleur = VeilleurFrequence(UUID_5090).start()

vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
params = SamplingParams(temperature=0.0, max_tokens=MAX_MODEL_LEN - 32)
chrono_sync = bool(os.environ.get("ACVRAM_CHRONO_SYNC"))


def _invite(k: int) -> list[int]:
    return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(32)]


resultats: list[dict] = []

for tour in range(N_TOURS):
    # Réinitialiser les listes de chronométrie par tour
    if engine.graphs is not None:
        engine.graphs.temps_bind = []
        engine.graphs.temps_fill = []
        engine.graphs.temps_replay = []
    engine.temps_pas_total = []
    engine.temps_pas_voie = []

    avant = engine.stats.to_dict()
    d0 = (avant.get("decode_seconds") or 0.0)
    n0 = (avant.get("decode_tokens") or 0)
    p0 = (avant.get("prefill_seconds") or 0.0)

    epoch_debut = time.time()
    t_debut = time.perf_counter()
    for _ in engine.generate(_invite(tour), params):
        pass
    t_fin = time.perf_counter()
    epoch_fin = time.time()

    apres = engine.stats.to_dict()
    decode_s = (apres.get("decode_seconds") or 0.0) - d0
    decode_n = (apres.get("decode_tokens") or 0) - n0
    prefill_s = (apres.get("prefill_seconds") or 0.0) - p0

    bind = engine.graphs.temps_bind if engine.graphs is not None else []
    fill = engine.graphs.temps_fill if engine.graphs is not None else []
    replay = engine.graphs.temps_replay if engine.graphs is not None else []
    pas_total = engine.temps_pas_total
    pas_voie = engine.temps_pas_voie

    n_graphe = sum(1 for v in pas_voie if v == "graphe")
    n_eager = sum(1 for v in pas_voie if v == "eager")
    part_graphe = n_graphe / len(pas_voie) if pas_voie else None

    res = {
        "tour": tour,
        "epoch_debut": epoch_debut,
        "epoch_fin": epoch_fin,
        "mur_s": t_fin - t_debut,
        "decode_tokens": decode_n,
        "decode_seconds": decode_s,
        "decode_tok_s": decode_n / decode_s if decode_s > 0 else None,
        "prefill_seconds": prefill_s,
        "bind_median_ms": median(bind),
        "fill_median_ms": median(fill),
        "replay_median_ms": median(replay),
        "pas_total_median_ms": median(pas_total),
        "n_pas": len(pas_total),
        "n_graphe": n_graphe,
        "n_eager": n_eager,
        "part_graphe": part_graphe,
        "epoch_debut_tour": epoch_debut,
        "epoch_fin_tour": epoch_fin,
    }
    print(
        f"[TOUR {tour:02d}] "
        f"decode={decode_n}j {decode_s:.3f}s "
        f"prefill_s={prefill_s:.3f} "
        f"replay_med={res['replay_median_ms']:.2f}ms "
        f"pas_med={res['pas_total_median_ms']:.2f}ms "
        f"graphe={n_graphe}/{len(pas_voie)}",
        flush=True,
    )
    resultats.append(res)

veilleur.stop()

# Pic VRAM après reset — ce que les créneaux ont coûté, sans le chargement
max_vram_gio = torch.cuda.max_memory_reserved(0) / 1024 ** 3

# --- résumés console ---

bind_meds = [r["bind_median_ms"] for r in resultats if r.get("bind_median_ms") is not None]
fill_meds = [r["fill_median_ms"] for r in resultats if r.get("fill_median_ms") is not None]
replay_meds = [r["replay_median_ms"] for r in resultats if r.get("replay_median_ms") is not None]
pas_meds = [r["pas_total_median_ms"] for r in resultats if r.get("pas_total_median_ms") is not None]


def med(lst: list) -> float:
    if not lst:
        return float("nan")
    s = sorted(lst)
    return s[len(s) // 2]


print(f"\n=== ETAPES GRAPHE (médiane des médianes, {len(resultats)} tours) ===")
print(f"  bind   : {med(bind_meds):.3f} ms")
print(f"  fill   : {med(fill_meds):.3f} ms")
print(f"  replay : {med(replay_meds):.3f} ms")
print(f"\n=== PAS TOTAL ===")
print(f"  pas_total : {med(pas_meds):.3f} ms")
print(f"\n=== VRAM PEAK (après reset) ===")
print(f"  max_memory_reserved : {max_vram_gio:.4f} Gio")
print(f"\n=== CONFIG ===")
print(f"  slots={SLOTS}  ctx={CTX}  max_model_len={MAX_MODEL_LEN}")
print(f"  ACVRAM_CHRONO_SYNC={'1' if chrono_sync else '0'}")

# --- sauvegarde JSON ---

sortie_json = {
    "modele": MODEL_DIR,
    "slots": SLOTS,
    "ctx": CTX,
    "max_model_len": MAX_MODEL_LEN,
    "n_tours": N_TOURS,
    "acvram_chrono_sync": chrono_sync,
    "max_memory_reserved_gio": max_vram_gio,
    "load_seconds": load_s,
    "n_graphes_captures": n_captures,
    "pollution": veilleur.pollution,
    "tours": resultats,
}

with open(SORTIE, "w", encoding="utf-8") as fh:
    json.dump(sortie_json, fh, indent=2)
print(f"[INFO] résultats → {SORTIE}", flush=True)
