#!/usr/bin/env python3
"""carte_rendue — attendre qu'un serveur arrêté ait RENDU sa VRAM avant d'en démarrer un autre (kwh, 30/09).

edz 30/09 00:48 : 10 OOM en 3 min au changement de modèle. Un PID mort quitte `nvidia-smi --query-compute-apps` AVANT
que le pilote ait libéré sa mémoire, et un processus en sortie y figure encore : attendre le PID ne suffit pas, un
seuil fixe de mémoire libre (l'ancien « libre > 24 000 Mio » de vllm-serveur et llamacpp-serveur) attend à tort quand
un processus VIVANT occupe la carte (ComfyUI, appoint 8081) et laisse partir trop tôt un gros modèle.

Sur chaque carte servie, on attend : aucun PID de calcul en sortie (disparu, zombie, PF_EXITING), aucun PID remplacé
encore vivant, et au plus ACVRAM_ORPHELINS_MIO de VRAM sans processus (utilisée − Σ compute-apps). Un processus vivant
ne fait jamais attendre, et ce module ne tue rien. Délai ACVRAM_ATTENTE_VRAM (s, défaut 60 ; 0 = coupé), puis refus
nommé : rc 1, motif sur stderr — plutôt qu'un OOM qui accuse la taille du modèle.

Usage (les trois lanceurs du parc) : carte_rendue.py [--cartes 0,1] [--remplace "PID PID"]... [--nom acvram]
Cartes par défaut : CUDA_VISIBLE_DEVICES (indices nvidia-smi, CUDA_DEVICE_ORDER=PCI_BUS_ID), sinon 0."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def smi(carte: str, q: str) -> str:
    return subprocess.run(["nvidia-smi", f"--id={carte}", q, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20, check=True).stdout


def _stat(pid: int) -> list[str] | None:
    """Champs de /proc/<pid>/stat à partir de l'état (champ 3) ; None si disparu."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()
    except OSError:
        return None


def sortant(pid: int) -> bool:
    """Disparu, zombie, ou en sortie (PF_EXITING = 0x4, champ 9 de /proc/<pid>/stat)."""
    ch = _stat(pid)
    return ch is None or ch[0] in ("Z", "X") or bool(int(ch[6]) & 0x4)


def du_remplace(pid: int, remplaces: set[int]) -> bool:
    """Le PID est un remplacé, ou de sa session/son groupe (champs 5-6) : carte.sh lance chaque service sous
    `setsid`, donc le PID noté (serveur.pid, .qui) est chef de session, et le moteur qui tient la carte (EngineCore
    de vLLM) en hérite même après la mort du chef."""
    ch = _stat(pid)
    return pid in remplaces or (ch is not None and (int(ch[2]) in remplaces or int(ch[3]) in remplaces))


def etat(carte: str, remplaces: set[int]) -> tuple[int, list[int]]:
    """(Mio sans processus, PID qui bloquent) sur une carte. Mémoire par PID illisible ([N/A]) : orphelins à 0."""
    orph = int(smi(carte, "--query-gpu=memory.used").split()[0])
    apps, lisible = [], True
    for l in smi(carte, "--query-compute-apps=pid,used_memory").splitlines():
        p, _, m = (x.strip() for x in l.partition(","))
        if p.isdigit():
            apps.append(int(p))
            orph -= int(m) if m.isdigit() else 0
            lisible &= m.isdigit()
    # un remplacé ne bloque que s'il tient la carte : un PID périmé de serveur.pid, réattribué à un autre
    # processus, ne fait pas attendre (et un processus vivant étranger non plus)
    bloq = [p for p in apps if sortant(p) or du_remplace(p, remplaces)]
    return (orph if lisible else 0), bloq


def attendre(cartes: list[str], remplaces: set[int], delai: float, seuil: int, pas: float) -> tuple[str, float]:
    """(« » si rendue, « muet » si nvidia-smi ne répond pas, sinon le motif du refus ; secondes attendues)."""
    t0, attendu = time.monotonic(), False
    while True:
        try:
            lu = {c: etat(c, remplaces) for c in cartes}
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return "muet", 0.0
        dt = time.monotonic() - t0
        genant = {c: (o, b) for c, (o, b) in lu.items() if b or o > seuil}
        if not genant:
            return "", dt if attendu else 0.0
        if dt >= delai:
            motif = []
            for c, (o, b) in genant.items():
                m = ([f"PID {b} en sortie"] if b else []) + ([f"{o} Mio sans processus"] if o > seuil else [])
                motif.append(f"carte {c} : {' ; '.join(m)}")
            return f"carte non rendue en {delai:.0f} s — {' | '.join(motif)}", dt
        time.sleep(pas)
        attendu = True


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--cartes", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    a.add_argument("--remplace", action="append", default=[], help="PID arrêtés par le lanceur (liste, espaces ou virgules)")
    a.add_argument("--nom", default="parc")
    o = a.parse_args(argv)
    delai = float(os.environ.get("ACVRAM_ATTENTE_VRAM", "60"))
    if delai <= 0:
        return 0
    cartes = [c.strip() for c in o.cartes.split(",") if c.strip()]
    remplaces = {int(x) for r in o.remplace for x in r.replace(",", " ").split() if x.isdigit()}
    motif, dt = attendre(cartes, remplaces, delai, int(os.environ.get("ACVRAM_ORPHELINS_MIO", "1024")),
                         float(os.environ.get("ACVRAM_ATTENTE_PAS", "1")))
    if motif == "muet":
        print(f"{o.nom} : nvidia-smi muet — attente de carte inactive")
    elif motif:
        print(f"{o.nom} : refusé : {motif} (un serveur précédent tient encore la carte)", file=sys.stderr)
        return 1
    elif dt:
        print(f"{o.nom} : carte rendue en {dt:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
