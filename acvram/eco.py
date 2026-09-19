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
# clocks_event_reasons.applications_clocks_setting reste « Not Active » sous -lgc sur ce
# pilote (poste2, 19/09 23 h : SM 2 692 relevé, champ Not Active, « Applications Clocks »
# dépréciées dans -q) : il ne dit RIEN du verrou. On lit l'horloge et la raison « idle »,
# et le verrou vient de l'état écrit par qui l'a posé (ETAT_ECO), vérifié contre clocks.sm.
CHAMPS = "clocks.sm,clocks.max.sm,clocks_event_reasons.gpu_idle"
# état posé par ce poste (acvram eco / le serveur) : {"mode", "pid", "depuis"} — la seule
# mémoire du verrou d'horloge lisible sans sudo ; vérifié contre clocks.sm à chaque lecture
ETAT_ECO = "/tmp/acvram-eco-{index}.json"
TOLERANCE_MHZ = 30          # -lgc 2700 se lit 2 692 (pas de 15 MHz du pilote)
# fichier d'information du verrou de carte.sh : « PID pris_a nom type »
VERROU_QUI = "/tmp/acvram-carte-{index}.lock.qui"


def index_carte() -> int:
    """Index nvidia-smi de la carte que le moteur appelle cuda:0 : le premier
    champ de CUDA_VISIBLE_DEVICES quand c'est un entier (carte.sh l'exporte
    depuis le premier index de la carte réservée), 0 sinon."""
    premier = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(premier) if premier.isdigit() else 0


def _lire_etat(index: int) -> dict | None:
    import json
    try:
        with open(ETAT_ECO.format(index=index), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and str(d.get("mode")) in MODES else None
    except (OSError, ValueError):
        return None


def _ecrire_etat(index: int, mode: str | None) -> None:
    """mode None ou off : l'état est retiré (horloge rendue)."""
    import json, time
    chemin = ETAT_ECO.format(index=index)
    if mode in (None, "off"):
        try:
            os.remove(chemin)
        except OSError:
            pass
        return
    try:
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "pid": os.getpid(), "depuis": time.time()}, f)
    except OSError as exc:
        print(f"acvram eco : état non écrit ({chemin}) : {exc}", file=sys.stderr)


def lire_horloge(index: int = 0, sortie: str | None = None, etat: dict | None = None) -> dict:
    """Horloge SM de la carte `index`, lue sans sudo. `sortie` : chaîne
    simulée à la place de nvidia-smi (tests) ; `etat` : état posé simulé.
    Ne lève jamais : sans nvidia-smi, ou sur une sortie illisible, ``verrou``
    vaut None et ``erreur`` dit pourquoi. ``verrou`` True quand un état posé
    (ETAT_ECO) existe ET que clocks.sm est à ± TOLERANCE_MHZ de son mode ;
    ``mode`` = ce mode. Sans état posé : ``verrou`` False (libre) — sauf carte
    oisive à > 1 000 MHz (signature d'un -lgc posé hors de ce poste), verrou
    True et mode = l'horloge lue, dit ``incertain``."""
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
        sm, max_sm, oisive = int(champs[0]), int(champs[1]), champs[2] == "Active"
    except (IndexError, ValueError):
        return {"verrou": None, "erreur": f"sortie nvidia-smi illisible : {brut!r}", "brut": brut}
    e = _lire_etat(index) if etat is None else etat
    if e:
        cible = int(e["mode"])
        tenu = abs(sm - cible) <= TOLERANCE_MHZ
        return {"sm_mhz": sm, "max_sm_mhz": max_sm, "verrou": tenu, "mode": str(cible),
                "pid": e.get("pid"), "brut": brut,
                **({} if tenu else {"erreur": f"état posé {cible} MHz mais horloge lue {sm}"})}
    if oisive and sm > 1000:
        return {"sm_mhz": sm, "max_sm_mhz": max_sm, "verrou": True, "mode": str(sm),
                "incertain": True, "brut": brut}
    return {"sm_mhz": sm, "max_sm_mhz": max_sm, "verrou": False, "brut": brut}


def etiquette_horloge(h: dict) -> str:
    """``lgc<MHz>`` sous verrou, ``libre`` sinon, ``?`` quand on ne sait pas."""
    if h.get("verrou") is None:
        return "?"
    if not h["verrou"]:
        return "libre" if not h.get("erreur") else f"libre({h['sm_mhz']}≠{h.get('mode')})"
    return f"lgc{h.get('mode', h.get('sm_mhz'))}" + ("?" if h.get("incertain") else "")


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
    _ecrire_etat(index, mode)
    h = lire_horloge(index)
    print(f"carte {index} : nvidia-smi {' '.join(reglage)} → horloge={etiquette_horloge(h)}")
    return 0


# --------------------------------------------------------------------------
# Éco par défaut (poste7-eco-2700-defaut-19-09 § 1, décision utilisateur 19/09
# 20 h 22 : « oui, éco 2700 par défaut »). Le processus qui SERT pose l'horloge
# (`-lgc`) après le verrou de carte et avant le premier chargement, et la rend
# (`-rgc`) à `Engine.fermer`, à SIGTERM/SIGINT et en atexit : le verrou
# d'horloge suit la vie du serveur, pas celle du système. Pas de relâchement à
# l'oisiveté par défaut (C8 reste opt-in). Sans droit sudo on sert quand même,
# bruyamment : `eco=<demandé>(<effectif>)` sur la ligne de régime, et les
# instruments refusent de publier une cellule où demandé ≠ effectif.
# --------------------------------------------------------------------------
CONFIG = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
                      "acvram", "config.json")
ECO_DEFAUT = "2700"


def charger_config(chemin: str | None = None) -> dict:
    """`~/.config/acvram/config.json` (dict), {} s'il manque ou s'il est illisible."""
    import json
    try:
        with open(chemin or CONFIG, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def ecrire_config(cles: dict, chemin: str | None = None) -> str:
    """Fusionne `cles` dans config.json (créé au besoin) ; rend le chemin."""
    import json
    chemin = chemin or CONFIG
    d = charger_config(chemin); d.update(cles)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2); f.write("\n")
    os.replace(tmp, chemin)
    return chemin


def mode_demande(env: dict | None = None, chemin: str | None = None) -> str:
    """Le mode éco demandé : `ACVRAM_ECO` (bras A/B des instruments, jamais le
    service) sinon `config.json["eco"]` sinon 2700. Une valeur hors MODES
    vaut le défaut, dite sur stderr (jamais un refus silencieux)."""
    v = (os.environ.get("ACVRAM_ECO") if env is None else env.get("ACVRAM_ECO")) \
        or str(charger_config(chemin).get("eco", ECO_DEFAUT))
    if v not in MODES:
        print(f"acvram eco : valeur inconnue {v!r} (attendu {' | '.join(MODES)}) : défaut {ECO_DEFAUT}",
              file=sys.stderr)
        return ECO_DEFAUT
    return v


class Horloge:
    """L'horloge posée par CE processus : `poser()` une fois, `rendre()`
    idempotent, enregistré en atexit et sur SIGTERM/SIGINT. `etat` est nommé :
    effectif | libre (off) | refus sudo | non pris | sans carte | inconnu.
    `executer` et `lire` remplacent subprocess.run et lire_horloge (tests)."""

    def __init__(self, mode: str, index: int = 0, executer=None, lire=None):
        self.mode, self.index = mode, index
        self._executer = subprocess.run if executer is None else executer
        self._lire = lire_horloge if lire is None else lire
        self.effectif: str | None = None
        self.etat = "inconnu"
        self.posee = False
        self.rendue = False
        self._anciens = {}

    # -- lecture --------------------------------------------------------
    def relire(self) -> str:
        """effectif = l'horloge SM LUE (clocks.sm), jamais la mémoire d'un réglage."""
        h = self._lire(self.index)
        if h.get("verrou") is None or "sm_mhz" not in h:
            self.effectif = "?"
        elif h["verrou"] or h.get("mode") is not None:     # verrou tenu, ou état posé non tenu : le chiffre
            self.effectif = str(h["sm_mhz"])
        else:
            self.effectif = "libre"
        return self.effectif

    @property
    def conforme(self) -> bool:
        """demandé == effectif : ce qu'un instrument exige avant de publier."""
        if self.mode == "off":
            return self.effectif == "libre"
        try:
            return self.effectif is not None and abs(int(self.effectif) - int(self.mode)) <= TOLERANCE_MHZ
        except ValueError:
            return False

    def etiquette(self) -> str:
        """``eco=2700(2700)`` conforme ; ``eco=2700(libre: refus sudo)`` sinon."""
        eff = self.effectif if self.effectif is not None else "?"
        return f"eco={self.mode}({eff})" if self.conforme else f"eco={self.mode}({eff}: {self.etat})"

    # -- pose / rendu ---------------------------------------------------
    def _nvidia_smi(self, *reglage: str) -> tuple[bool, str]:
        cmd = ["sudo", "-n", "nvidia-smi", "-i", str(self.index), *reglage]
        try:
            r = self._executer(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        texte = f"{r.stdout or ''}\n{r.stderr or ''}".strip()
        if r.returncode != 0 or "password" in texte.lower():
            return False, texte or f"code {r.returncode}"
        return True, texte

    def poser(self) -> str:
        """Pose `-lgc mode,mode` (rien sous `off`) puis relit ; rend `etat`."""
        if self.mode == "off":
            self.etat = "libre"; self.relire(); return self.etat
        ok, msg = self._nvidia_smi("-lgc", f"{self.mode},{self.mode}")
        if not ok:
            self.etat = "refus sudo"; self.relire()
            print(f"acvram eco : {self.etiquette()} — `sudo -n nvidia-smi -i {self.index} -lgc "
                  f"{self.mode},{self.mode}` refusé ({msg.splitlines()[0] if msg else 'sans message'}) ; "
                  "le service continue à l'horloge libre, aucune cellule n'est publiable "
                  "(sudoers NOPASSWD sur nvidia-smi : `acvram doctor`).", file=sys.stderr)
            return self.etat
        self.posee = True
        _ecrire_etat(self.index, self.mode)
        self.relire()
        self.etat = "effectif" if self.conforme else "non pris"
        if self.etat == "non pris":
            print(f"acvram eco : {self.etiquette()} — `-lgc` accepté mais l'horloge lue ne suit pas",
                  file=sys.stderr)
        self._armer()
        return self.etat

    def rendre(self) -> bool:
        """`-rgc` une seule fois, seulement si ce processus a posé ; True s'il l'a fait."""
        if not self.posee or self.rendue:
            return False
        self.rendue = True
        ok, msg = self._nvidia_smi("-rgc")
        _ecrire_etat(self.index, None)
        if not ok:
            print(f"acvram eco : `-rgc` refusé ({msg}) — l'horloge reste à {self.mode} MHz : "
                  f"`sudo nvidia-smi -i {self.index} -rgc` à la main", file=sys.stderr)
        self.effectif = None
        return ok

    def _armer(self) -> None:
        """atexit + SIGTERM/SIGINT : rendre puis laisser l'ancien gestionnaire agir
        (SIGINT → KeyboardInterrupt, SIGTERM → sortie 143 par défaut)."""
        import atexit, signal
        atexit.register(self.rendre)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                ancien = signal.getsignal(sig)
            except (ValueError, OSError):
                continue
            self._anciens[sig] = ancien

            def gestionnaire(num, cadre, ancien=ancien, sig=sig):
                self.rendre()
                if callable(ancien) and ancien not in (signal.SIG_IGN, signal.SIG_DFL):
                    ancien(num, cadre)
                elif sig == signal.SIGINT:
                    raise KeyboardInterrupt
                else:
                    signal.signal(sig, signal.SIG_DFL); os.kill(os.getpid(), sig)
            try:
                signal.signal(sig, gestionnaire)
            except (ValueError, OSError):     # hors fil principal : atexit seul
                pass


_HORLOGE: Horloge | None = None


def poser_pour_ce_processus(index: int | None = None, executer=None, lire=None) -> Horloge:
    """Le geste du serveur et des instruments : une fois par processus, mode
    demandé (`mode_demande()`), carte servie. Sans carte (CUDA invisible) :
    horloge « sans carte », rien n'est exécuté."""
    global _HORLOGE
    if _HORLOGE is not None:
        return _HORLOGE
    mode = mode_demande()
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        h = Horloge(mode, 0, executer, lire); h.etat = "sans carte"; h.effectif = "?"
        _HORLOGE = h; return h
    h = Horloge(mode, index_carte() if index is None else index, executer, lire)
    h.poser()
    _HORLOGE = h
    return h


def horloge_du_processus() -> Horloge | None:
    return _HORLOGE


def rendre_horloge() -> bool:
    """`Engine.fermer` : rend l'horloge posée par ce processus, s'il y en a une."""
    return _HORLOGE.rendre() if _HORLOGE is not None else False


def etat_eco() -> dict:
    """Pour `regime()` et les instruments : demandé, effectif, état, conforme."""
    h = _HORLOGE
    if h is None:
        return {"demande": mode_demande(), "effectif": None, "etat": "non posé", "conforme": False}
    return {"demande": h.mode, "effectif": h.effectif, "etat": h.etat, "conforme": h.conforme}
