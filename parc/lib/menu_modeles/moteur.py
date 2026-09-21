"""Commun claude-modeles / kimi-modeles — moteurs d'inférence.

Un serveur d'inférence (Moteur) et les fonctions qui l'interrogent : clé d'API,
modèle servi, VRAM, propriétaire d'un port. Les constantes de chemins (SECRETS,
CONFIG) vivent dans `config` ; on y accède au moment de l'appel (`config.SECRETS`)
et non à l'import, pour ne pas fermer la boucle config ↔ moteur au chargement.
"""
import re
import subprocess
import tomllib
import urllib.request
from pathlib import Path

from . import config


class Moteur:
    """Un serveur d'inférence : son port, sa clé, et par où on lit le modèle servi."""

    def __init__(self, cle_config, nom, port, route, var_cle, cle_defaut, journal=None, tokens=""):
        self.cle_config = cle_config
        self.nom = nom
        self.port = port
        self.route = route
        self.var_cle = var_cle
        self.cle_defaut = cle_defaut
        self.journal = Path(journal) if journal else None
        self.tokens = tokens   # api_tokens.yml (TabbyAPI), depuis parc.toml


def secrets():
    """Clés d'API du fichier unique en 0600. Jamais affichées, jamais journalisées."""
    d = {}
    try:
        for ligne in config.SECRETS.read_text().splitlines():
            ligne = ligne.strip()
            if ligne and not ligne.startswith("#") and "=" in ligne:
                k, v = ligne.split("=", 1)
                d[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return d


def cle_moteur(m, coffre=None):
    coffre = coffre if coffre is not None else secrets()
    if m.cle_config == "tabby" and not coffre.get(m.var_cle) and m.tokens:
        try:
            for ligne in Path(m.tokens).read_text().splitlines():
                if ligne.startswith("api_key:"):
                    return ligne.split(":", 1)[1].strip()
        except OSError:
            return ""
    if m.cle_config == "yals" and not coffre.get(m.var_cle):
        try:
            return tomllib.load(config.CONFIG.open("rb")).get("providers", {}).get("yals", {}).get("api_key", "")
        except Exception:
            return ""
    return coffre.get(m.var_cle, m.cle_defaut)


def modele_servi(m, coffre=None, delai=1.5):
    """Identifiant du modèle actuellement chargé par ce moteur, ou None s'il dort."""
    req = urllib.request.Request(f"http://127.0.0.1:{m.port}{m.route}")
    c = cle_moteur(m, coffre)
    if c:
        req.add_header("Authorization", f"Bearer {c}")
    try:
        with urllib.request.urlopen(req, timeout=delai) as r:
            import json

            d = json.load(r)
    except Exception:
        return None
    if isinstance(d, dict) and "data" in d:
        liste = d.get("data") or []
        return liste[0].get("id") if liste else None
    return (d or {}).get("id") or None


def vram():
    """(utilisée, totale) en Mio, tous GPU confondus ; (None, None) sans nvidia-smi."""
    try:
        s = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        u = t = 0
        for ligne in s.stdout.strip().splitlines():
            a, b = (x.strip() for x in ligne.split(","))
            u += int(a)
            t += int(b)
        return (u, t) if t else (None, None)
    except Exception:
        return (None, None)


def pid_du_port(port):
    """Vrai propriétaire du port : les fichiers .pid des lanceurs mentent parfois
    (setsid dans un shell non interactif enregistre le pid du wrapper)."""
    try:
        s = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    for ligne in s.stdout.splitlines():
        if f"127.0.0.1:{port}" in ligne:
            m = re.search(r"pid=(\d+)", ligne)
            if m:
                return int(m.group(1))
    return None
