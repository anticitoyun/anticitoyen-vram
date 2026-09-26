"""Réglages hôte génériques du moteur (0.6.31, poste7-tests-rapides-cloture § 2 via chef 09 h 52).

UNE fonction, appelée à l'import du paquet (avant torch, qui lit OMP_NUM_THREADS à l'import)
ET à la construction de `Engine` (idempotente) — donc par `acvram serve` comme par
certifie / duel / ppl-decode-kv, jamais par la CLI seule (REGLES § 4 : la faute
HYBRID_SLOTS à l'envers, un réglage posé par un lanceur et pas par les instruments).

* `THP_MEM_ALLOC_ENABLE=1` (madvise ≥ 1 Mio pour les tenseurs PyTorch) et `OMP_NUM_THREADS=8`
  par `os.environ.setdefault` : un réglage déjà posé par la session (environment.d) gagne ;
* `ACVRAM_CPUS` (optionnel, ex. `0-15` ou `0-3,8`) → `os.sched_setaffinity` ; défaut : aucune
  affinité (lot poste du 20/09 : les P-cores 0-15 par environment.d, pas par le paquet).

La ligne de régime porte `hote=thp,omp8[,cpus0-15]` avec les valeurs EFFECTIVES relues
(`os.environ` après setdefault, `torch.get_num_threads()` si torch est chargé,
`os.sched_getaffinity(0)`), jamais les demandées. Prédiction (poste7) : OMP 8 ne bouge aucune
cellule de plus de ± 1 % ; si le prefill Coder perd > 3 % à T1, OMP est le premier suspect.
"""
from __future__ import annotations

import os
import sys

THP_DEFAUT = "1"
OMP_DEFAUT = "8"


def _cpus(spec: str) -> set[int]:
    """`0-15`, `0-3,8,10-11` → ensemble d'indices ; vide ou illisible → ValueError."""
    out: set[int] = set()
    for morceau in spec.split(","):
        m = morceau.strip()
        if not m:
            continue
        if "-" in m:
            a, b = m.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(m))
    if not out:
        raise ValueError(f"ACVRAM_CPUS={spec!r} : aucun cœur")
    return out


def _plages(cpus: set[int]) -> str:
    """{0,1,2,3,8} → `0-3,8`."""
    s = sorted(cpus); out = []; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        out.append(f"{s[i]}-{s[j]}" if j > i else f"{s[i]}")
        i = j + 1
    return ",".join(out)


def regler_hote() -> dict:
    """Pose les réglages (idempotent) et rend l'état EFFECTIF relu."""
    os.environ.setdefault("THP_MEM_ALLOC_ENABLE", THP_DEFAUT)
    os.environ.setdefault("OMP_NUM_THREADS", OMP_DEFAUT)
    torch = sys.modules.get("torch")
    if torch is not None:
        # torch a pu être importé avant ce paquet (instruments) : OMP_NUM_THREADS n'a alors pas été lu
        try:
            voulu = int(os.environ["OMP_NUM_THREADS"])
            if torch.get_num_threads() != voulu:
                torch.set_num_threads(voulu)
        except (ValueError, RuntimeError):
            pass
    spec = os.environ.get("ACVRAM_CPUS", "").strip()
    if spec and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, _cpus(spec))
        except (ValueError, OSError) as exc:
            print(f"[hote] ACVRAM_CPUS={spec!r} refusé : {exc}", file=sys.stderr, flush=True)
    return etat_hote()


def etat_hote() -> dict:
    """Les valeurs EFFECTIVES, relues — jamais celles demandées."""
    torch = sys.modules.get("torch")
    omp = None
    if torch is not None:
        try:
            omp = int(torch.get_num_threads())
        except Exception:                                  # noqa: BLE001
            omp = None
    if omp is None:
        try:
            omp = int(os.environ.get("OMP_NUM_THREADS", "0")) or None
        except ValueError:
            omp = None
    affinite = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
    return {"thp": os.environ.get("THP_MEM_ALLOC_ENABLE", "") == "1", "omp": omp,
            "cpus": _plages(affinite) if affinite else "", "cpus_demandes": os.environ.get("ACVRAM_CPUS", "")}


def hote_texte() -> str:
    """`hote=thp,omp8` ; `hote=thp,omp8,cpus0-15` quand ACVRAM_CPUS est posé (affinité effective relue) ;
    `hote=sans-thp,omp16` dit ce qui a gagné contre le défaut."""
    e = etat_hote()
    parts = ["thp" if e["thp"] else "sans-thp", f"omp{e['omp']}" if e["omp"] else "omp?"]
    if e["cpus_demandes"]:
        parts.append(f"cpus{e['cpus']}")
    return "hote=" + ",".join(parts)
