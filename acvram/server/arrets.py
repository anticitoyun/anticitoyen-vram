"""Journal des arrêts de moteurs — pièce ECONNREFUSED (poste6, 29/09, bd 5xw).

Deux services edz sont morts d'un SIGTERM propre pendant la première requête de Claude Code (« Shutting down »
sans trace, 28/09 17:45:58 et 29/09 09:41:17) et rien ne disait QUI l'avait envoyé : ni le lanceur (pas de GET
/v1/models étranger), ni liberer-vram (journal), ni carte.sh (journal), ni Claude Code (strace à sec : aucun kill).
Restent la console (`POST /moteurs/arreter`, un clic) ou un `kill` à la main — invisibles. Ce journal rend chaque
arrêt ATTRIBUABLE : le tueur écrit sa ligne (console, lanceur acvram-serveur), la victime écrit la sienne à
l'arrêt du serveur ; à la prochaine occurrence, une ligne de tueur précède la ligne de victime — ou aucune, et
l'arrêt est extérieur au dépôt. Format TSV : date, pid, signal, ligne de commande (160 car.), motif.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def chemin_journal() -> Path:
    return Path(os.environ.get("ACVRAM_JOURNAL_ARRETS")
                or os.path.join(os.path.expanduser("~"), ".cache", "acvram", "arrets.journal"))


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()[:160]
    except OSError:
        return "?"


def journal_arret(pid: int, motif: str, signal: str = "SIGTERM") -> None:
    """Une ligne, jamais une exception : un journal qui fait tomber un arrêt serait pire que l'absence de journal."""
    try:
        p = chemin_journal()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write("\t".join((time.strftime("%Y-%m-%dT%H:%M:%S"), str(pid), signal, cmdline(pid),
                                motif.replace("\t", " ").replace("\n", " "))) + "\n")
    except Exception:                                                    # noqa: BLE001
        pass
