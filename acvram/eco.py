"""Mode éco d'horloge (poste7-e1-eco-tenu-19-09 § 2) : `acvram eco {2700|2100|off|etat}`.

`nvidia-smi -lgc` verrouille l'horloge SM de la carte ; à b=12 le mode éco met
acvram devant llama.cpp en énergie ET en vitesse. Le réglage est root et hors
processus : ce n'est pas un défaut du moteur, aucune variable d'environnement
ne bouge — et un chiffre éco passerait pour un chiffre défaut si la ligne de
régime ne portait pas l'horloge. `regime_ligne()` la lit ici, sous la forme
``horloge=lgc2692 | libre | ?``.

Trois gestes, tous en lecture seule sauf `regler` :

* ``lire_horloge(index)`` : `nvidia-smi --query-gpu=…` sans sudo ; ne lève
  jamais, ``verrou`` None quand on ne sait pas.
* ``etiquette_horloge(h)`` : ``lgc<MHz>`` sous verrou, ``libre`` sinon, ``?``.
* ``regler(mode, index)`` : `sudo -n nvidia-smi -lgc m,m` ou `-rgc`. Refuse
  sans droit sudo (code 3) et quand un pair tient la carte (code 4) : un
  `-lgc` pendant la manche d'un autre change son régime en silence (10/09,
  28 % d'écart sur le même bras), donc un réglage d'horloge se fait sous le
  verrou outils/carte.sh, comme une mesure (REGLES § 1).
"""
from __future__ import annotations

import os
import subprocess
import sys

MODES = ("2700", "2100", "off")
CHAMPS = "clocks.sm,clocks.max.sm,clocks_event_reasons.applications_clocks_setting"
# fichier d'information du verrou de carte.sh : « PID pris_a nom type »
VERROU_QUI = "/tmp/acvram-carte-{index}.lock.qui"


def index_carte() -> int:
    """Index nvidia-smi de la carte que le moteur appelle cuda:0 : le premier
    champ de CUDA_VISIBLE_DEVICES quand c'est un entier (carte.sh l'exporte
    depuis le premier index de la carte réservée), 0 sinon."""
    premier = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(premier) if premier.isdigit() else 0


def lire_horloge(index: int = 0, sortie: str | None = None) -> dict:
    """Horloge SM de la carte `index`, lue sans sudo. `sortie` : chaîne
    simulée à la place de nvidia-smi (tests). Ne lève jamais : sans
    nvidia-smi, ou sur une sortie illisible, ``verrou`` vaut None et
    ``erreur`` dit pourquoi. ``verrou`` True = « Active » dans le champ
    applications_clocks_setting, ce que nvidia-smi rend sous `-lgc`."""
    if sortie is None:
        cmd = ["nvidia-smi", "-i", str(index), f"--query-gpu={CHAMPS}",
               "--format=csv,noheader,nounits"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"verrou": None, "erreur": f"{type(exc).__name__}: {exc}", "brut": ""}
        if r.returncode != 0:
            return {"verrou": None, "brut": r.stdout.strip(),
                    "erreur": (r.stderr.strip() or r.stdout.strip()
                               or f"nvidia-smi : code {r.returncode}")}
        sortie = r.stdout
    brut = sortie.strip()
    champs = [c.strip() for c in brut.splitlines()[0].split(",")] if brut else []
    try:
        sm, max_sm, etat = int(champs[0]), int(champs[1]), champs[2]
    except (IndexError, ValueError):
        return {"verrou": None, "erreur": f"sortie nvidia-smi illisible : {brut!r}", "brut": brut}
    return {"sm_mhz": sm, "max_sm_mhz": max_sm, "verrou": etat == "Active", "brut": brut}


def etiquette_horloge(h: dict) -> str:
    """``lgc<MHz>`` sous verrou, ``libre`` sinon, ``?`` quand on ne sait pas."""
    if h.get("verrou") is None:
        return "?"
    return f"lgc{h['sm_mhz']}" if h["verrou"] else "libre"


def _tenue_par_un_pair(index: int) -> int | None:
    """PID vivant d'un autre processus qui tient la carte `index` (fichier
    .qui de carte.sh), None si personne — ou si le détenteur est notre propre
    chaîne (carte.sh exporte ACVRAM_CARTE_TENUE=son PID à ce qu'il lance)."""
    try:
        with open(VERROU_QUI.format(index=index)) as f:
            pid = int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    if pid <= 0 or str(pid) == os.environ.get("ACVRAM_CARTE_TENUE", ""):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None            # détenteur disparu : information périmée
    except PermissionError:
        pass                   # vivant, sous un autre utilisateur
    return pid


def regler(mode: str, index: int = 0, executer=None) -> int:
    """Applique `mode` : 2700 | 2100 → `sudo -n nvidia-smi -lgc m,m`,
    off → `-rgc`. Rend 0 ; 2 mode inconnu ; 3 sudo sans droit ou nvidia-smi
    en échec (rien n'a été appliqué) ; 4 carte tenue par un pair. `executer`
    remplace subprocess.run (tests). Après un réglage, relit l'horloge et
    imprime son étiquette : c'est elle que portera la ligne de régime."""
    if mode not in MODES:
        print(f"acvram eco : mode inconnu {mode!r} (attendu : {' | '.join(MODES)})",
              file=sys.stderr)
        return 2
    pair = _tenue_par_un_pair(index)
    if pair is not None:
        print(f"acvram eco : REFUS — carte tenue par PID {pair} : un réglage d'horloge "
              "se fait sous le verrou outils/carte.sh (REGLES § 1)", file=sys.stderr)
        return 4
    executer = subprocess.run if executer is None else executer
    reglage = ["-rgc"] if mode == "off" else ["-lgc", f"{mode},{mode}"]
    cmd = ["sudo", "-n", "nvidia-smi", "-i", str(index), *reglage]
    try:
        r = executer(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"acvram eco : REFUS — `{' '.join(cmd)}` : {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 3
    texte = f"{r.stdout or ''}\n{r.stderr or ''}".strip()
    if r.returncode != 0 or "password" in texte.lower():
        print(f"acvram eco : REFUS — `{' '.join(cmd)}` a échoué (code {r.returncode}) : "
              f"{texte or 'sans message'}\n"
              "  sudo -n ne demande jamais de mot de passe : il faut une règle sudoers "
              "NOPASSWD pour nvidia-smi. Réglage non appliqué.", file=sys.stderr)
        return 3
    h = lire_horloge(index)
    print(f"carte {index} : nvidia-smi {' '.join(reglage)} → horloge={etiquette_horloge(h)}")
    return 0
