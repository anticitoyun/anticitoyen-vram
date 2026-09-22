"""Mode éco d'horloge (sage-e1-eco-tenu-19-09 § 2) : `acvram eco {2700|2100|off|etat}`.

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
import threading
import sys

MODES = ("2700", "2100", "off")
# clocks_event_reasons.applications_clocks_setting reste « Not Active » sous -lgc sur ce
# pilote (Manon, 19/09 23 h : SM 2 692 relevé, champ Not Active, « Applications Clocks »
# dépréciées dans -q) : il ne dit RIEN du verrou. On lit l'horloge et la raison « idle »,
# et le verrou vient de l'état écrit par qui l'a posé (ETAT_ECO), vérifié contre clocks.sm.
CHAMPS = "clocks.sm,clocks.max.sm,clocks_event_reasons.gpu_idle"
# état posé par ce poste (acvram eco / le serveur) : {"mode", "pid", "depuis"} — la seule
# mémoire du verrou d'horloge lisible sans sudo ; vérifié contre clocks.sm à chaque lecture
ETAT_ECO = "/tmp/acvram-eco-{index}.json"
TOLERANCE_MHZ = 60          # -lgc 2700 se lit 2 655-2 692 sous charge (Manon, 19/09 : pas de 15 MHz, palier bas)
VERROUS_CONNUS = ("2700", "2100")   # les consignes que ce poste pose : bandes reconnues sous charge
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


def lire_horloge(index: int = 0, sortie: str | None = None, etat: dict | None = None,
                 sous_charge: bool = False) -> dict:
    """Horloge SM de la carte `index`, lue sans sudo. `sortie` : chaîne
    simulée à la place de nvidia-smi (tests) ; `etat` : état posé simulé.
    Ne lève jamais : sans nvidia-smi, ou sur une sortie illisible, ``verrou``
    vaut None et ``erreur`` dit pourquoi. ``verrou`` True quand un état posé
    (ETAT_ECO) existe ET que clocks.sm est à ± TOLERANCE_MHZ de son mode ;
    ``mode`` = ce mode. Sans état posé : ``verrou`` False (libre) — sauf, lue
    SOUS CHARGE (`sous_charge=True`, cf. `_charge_cuda`), une horloge dans la
    bande d'une consigne connue (2 700 | 2 100 ± TOLERANCE_MHZ : un -lgc posé
    hors de ce poste), verrou True, mode = la consigne, dit ``incertain``. Au
    repos rien n'est concluant (225 MHz verrouillé ou non)."""
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
        # état posé : tenu ssi l'horloge ne dépasse pas la consigne (un verrou est un plafond ;
        # sous charge le bridage de puissance peut la tenir bien en dessous) ; au repos (< 1 000)
        # rien n'est concluant, l'état posé fait foi
        tenu = sm <= cible + TOLERANCE_MHZ
        return {"sm_mhz": sm, "max_sm_mhz": max_sm, "verrou": tenu, "mode": str(cible),
                "pid": e.get("pid"), "brut": brut,
                **({} if tenu else {"erreur": f"état posé {cible} MHz mais horloge lue {sm} au-dessus"})}
    if sous_charge:
        bande = next((v for v in VERROUS_CONNUS if abs(sm - int(v)) <= TOLERANCE_MHZ), None)
        if bande is not None:
            return {"sm_mhz": sm, "max_sm_mhz": max_sm, "verrou": True, "mode": bande,
                    "incertain": True, "brut": brut}
    return {"sm_mhz": sm, "max_sm_mhz": max_sm, "verrou": False, "oisive": oisive, "brut": brut}


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
# Éco par défaut (sage-eco-2700-defaut-19-09 § 1, décision utilisateur 19/09
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


def _charge_cuda(duree: float = 0.15, device: str = "cuda:0") -> None:
    """Occupe LÉGÈREMENT la carte `duree` s (matmuls bf16 1024², synchronisés) :
    une carte verrouillée OISIVE lit 225 MHz et ne monte à sa consigne qu'au
    premier contexte — l'effectif se lit pendant une charge, jamais au repos
    (Manon, verdict-verif-eco-defaut-19-09). La charge est légère à dessein :
    sous une charge lourde une carte LIBRE tombe au plafond de puissance
    (2 524 MHz moyens à 400 W, c16bis) et se confond avec un -lgc 2700 ; légère,
    elle monte au boost (≥ 2 900) et s'en distingue. Le pilote 595 n'expose
    aucun état de verrou (`-q -d CLOCK` : Applications Clocks dépréciées, Max
    Clocks 3 135 verrouillé ou non ; applications_clocks_setting Not Active
    sous -lgc) : l'horloge sous charge légère est le seul instrument. Ne lève
    jamais : sans torch/CUDA, rien."""
    import time
    try:
        import torch
        if not torch.cuda.is_available():
            return
        a = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
        t0 = time.time()
        while time.time() - t0 < duree:
            a = a @ a
            a = a / (a.norm() + 1.0)
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
    except Exception as exc:                                     # noqa: BLE001
        print(f"acvram eco : charge de lecture impossible ({type(exc).__name__}: {exc})", file=sys.stderr)


def lire_sous_charge(index: int = 0, lire=None, charge=None, attente: float = 0.12,
                     plafond: float = 2.5, stable_mhz: int = 30, minimum: float = 0.8,
                     cible: int | None = None) -> dict:
    """`lire(index)` pendant que `charge()` occupe la carte dans un fil, **jusqu'à
    un régime** : une carte froide monte par paliers (225 → 1 102 → 1 980 → 2 977,
    Manon 19/09) et deux lectures égales sur un palier de montée ne sont pas un
    régime. Stable = après ≥ `minimum` s de charge (0,8 s : la montée de la
    5090 dure ≈ 1 s), deux lectures consécutives à ± `stable_mhz` et ≥ 1 000 MHz
    (un palier tenu AVANT le minimum n'est pas cru) ; ou,
    quand une `cible` est demandée, deux lectures consécutives dans sa bande
    (≥ cible − TOLERANCE_MHZ : la carte verrouillée est arrivée). Plafond
    `plafond` s (stable False, dit). `stable` et `lectures` sont rendus."""
    import threading, time
    lire = lire_horloge if lire is None else lire
    charge = _charge_cuda if charge is None else charge
    arret = threading.Event()

    def boucle():
        while not arret.is_set():
            charge()
    fil = threading.Thread(target=boucle, daemon=True); fil.start()
    t0 = time.time(); lectures = []; h = {"verrou": None}
    try:
        while True:
            time.sleep(attente)
            try:
                h = lire(index, sous_charge=True)
            except TypeError:                               # lecteur de test sans le paramètre
                h = lire(index)
            sm = h.get("sm_mhz")
            if sm is None:
                break
            lectures.append(sm)
            ecoule = time.time() - t0
            if cible is not None and len(lectures) >= 2 and all(
                    abs(v - cible) <= TOLERANCE_MHZ for v in lectures[-2:]):
                h["stable"] = True; break                   # arrivée sur la consigne
            # sans consigne : TROIS lectures consécutives à ± stable_mhz (une montée qui traverse
            # une bande — 2 062, 2 062 puis 2 977 — n'est pas un palier : Manon 858298c, lu
            # « lgc2100? » à tort) ; ≥ 1 000 et après le minimum de charge
            if (ecoule >= minimum and len(lectures) >= 3 and sm >= 1000
                    and abs(lectures[-1] - lectures[-2]) <= stable_mhz
                    and abs(lectures[-2] - lectures[-3]) <= stable_mhz):
                h["stable"] = True; break                   # ne monte plus
            if ecoule > plafond:
                h["stable"] = False; break
    finally:
        arret.set(); fil.join(timeout=5)
    h["lectures"] = lectures
    return h


class Horloge:
    """L'horloge posée par CE processus : `poser()` une fois, `rendre()`
    idempotent, enregistré en atexit et sur SIGTERM/SIGINT. `etat` est nommé :
    effectif | libre (off) | refus sudo | non pris | sans carte | inconnu.
    `executer` et `lire` remplacent subprocess.run et lire_horloge (tests)."""

    def __init__(self, mode: str, index: int = 0, executer=None, lire=None, charge=None):
        self.mode, self.index = mode, index
        self._executer = subprocess.run if executer is None else executer
        self._lire = lire_horloge if lire is None else lire
        self._charge = charge                     # None : matmuls CUDA (_charge_cuda) ; tests : fonction
        self.effectif: str | None = None
        self.etat = "inconnu"
        self.posee = False
        self.rendue = False
        self._anciens = {}

    # -- lecture --------------------------------------------------------
    def relire(self, sous_charge: bool = True) -> str:
        """effectif = l'horloge SM LUE (clocks.sm) PENDANT une charge, jamais la
        mémoire d'un réglage ni une lecture au repos (225 MHz sous verrou oisif).
        Chiffre toujours ; « libre » seulement quand rien n'est lisible."""
        cible = None if self.mode == "off" else int(self.mode)
        h = lire_sous_charge(self.index, self._lire, self._charge, cible=cible) if sous_charge else self._lire(self.index)
        self._lecture = h
        self.effectif = str(h["sm_mhz"]) if "sm_mhz" in h else "?"
        if sous_charge and h.get("stable") is False:
            print(f"acvram eco : horloge non stabilisée en 2,5 s sous charge (lectures {h.get('lectures')}) — "
                  f"effectif {self.effectif} pris tel quel", file=sys.stderr)
        return self.effectif

    def _dans_une_bande(self) -> str | None:
        """la consigne connue (2700 | 2100) à ± TOLERANCE_MHZ de l'effectif, sinon None."""
        try:
            sm = int(self.effectif)
        except (TypeError, ValueError):
            return None
        return next((v for v in VERROUS_CONNUS if abs(sm - int(v)) <= TOLERANCE_MHZ), None)

    @property
    def conforme(self) -> bool:
        """demandé == effectif : ce qu'un instrument exige avant de publier."""
        if self.mode == "off":
            # libre = sous charge l'horloge n'est dans aucune bande de verrou connue
            # (un -lgc posé à la main hors de ce processus se voit ici : Manon, bras faux)
            return self.effectif not in (None, "?") and self._dans_une_bande() is None
        # (a) sage-mesure1-ter-c17-ecrit-c14-juge-20-09 : -lgc est un PLAFOND — le limiteur de
        # puissance passe SOUS le verrou (Marlin 1 950-2 265 MHz à 400 W sous -lgc 2700) et jamais
        # au-dessus ; le verrou est effectif ssi l'horloge lue sous charge ne DÉPASSE pas la consigne
        # (+ tolérance) ; en dessous, c'est le bridage, pas l'absence de verrou. (b) MHz et W sont
        # publiés, jamais critères.
        try:
            return self.effectif is not None and 1000 <= int(self.effectif) <= int(self.mode) + TOLERANCE_MHZ
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
            self.relire()
            bande = self._dans_une_bande()
            self.etat = "libre" if bande is None else f"verrou {bande} posé hors processus"
            if bande is not None:
                print(f"acvram eco : {self.etiquette()} — la carte est verrouillée par un autre "
                      f"(`sudo nvidia-smi -i {self.index} -rgc` pour la rendre)", file=sys.stderr)
            return self.etat
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
            print(f"acvram eco : {self.etiquette()} — `-lgc` accepté mais l'horloge lue sous charge ne suit pas",
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


def poser_pour_ce_processus(index: int | None = None, executer=None, lire=None, charge=None) -> Horloge:
    """Le geste du serveur et des instruments : une fois par processus, mode
    demandé (`mode_demande()`), carte servie. Sans carte (CUDA invisible) :
    horloge « sans carte », rien n'est exécuté."""
    global _HORLOGE
    if _HORLOGE is not None:
        return _HORLOGE
    mode = mode_demande()
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        h = Horloge(mode, 0, executer, lire, charge); h.etat = "sans carte"; h.effectif = "?"
        _HORLOGE = h; return h
    h = Horloge(mode, index_carte() if index is None else index, executer, lire, charge)
    h.poser()
    _HORLOGE = h
    return h


def horloge_du_processus() -> Horloge | None:
    return _HORLOGE


def rendre_horloge() -> bool:
    """`Engine.fermer` : rend l'horloge posée par ce processus, s'il y en a une."""
    return _HORLOGE.rendre() if _HORLOGE is not None else False


def etat_eco(relire: bool = False) -> dict:
    """Pour `regime()` et les instruments : demandé, effectif, état, conforme.
    `relire=True` : relecture sous charge maintenant (le moteur est chargé,
    la carte répond) — `Engine.regime()` le fait, pas la ligne à sec."""
    h = _HORLOGE
    if relire and h is not None and h.etat not in ("sans carte", "refus sudo"):
        h.relire()
        if h.mode != "off":
            h.etat = "effectif" if h.conforme else "non pris"
    if h is None:
        return {"demande": mode_demande(), "effectif": None, "etat": "non posé", "conforme": False}
    return {"demande": h.mode, "effectif": h.effectif, "etat": h.etat, "conforme": h.conforme}


# ---------------------------------------------------------------------------
# Régime SOUS LA CHARGE MESURÉE (sage-niveau2-clos-retrait-regles-20-09 § 2) : sous
# -lgc 2700 le b=12 Coder tourne à 387-401 W, médiane 2 550 MHz, 95 % des échantillons
# sous 2 650, seule raison `sw_power_cap` — le plafond de puissance mord sous le plafond
# d'horloge. `eco=2700(2685)` lit une horloge à vide : la ligne de régime d'une mesure
# porte la médiane d'horloge PENDANT la fenêtre et la raison de bridage, relevées par
# l'instrument, pas au chargement.
CHAMPS_CHARGE = ("clocks.sm,power.draw,clocks_event_reasons.sw_power_cap,"
                 "clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown,"
                 "clocks_event_reasons.hw_thermal_slowdown")
RAISONS = ("sw_power_cap", "hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown")


def _lire_charge(index: int = 0) -> list[str] | None:
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(index), f"--query-gpu={CHAMPS_CHARGE}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    champs = [c.strip() for c in out.strip().split(",")]
    return champs if len(champs) == 6 else None


class SondeCharge:
    """Échantillonne horloge SM, puissance et raisons de bridage pendant une
    fenêtre de mesure (fil, une lecture par `pas` s). ``bilan()`` → médiane
    d'horloge, puissance médiane, raison dominante et sa part ; ``etiquette``
    → ``eco=2700(plafond 400 W : méd. 2 550 MHz)`` — le régime réellement
    tenu, à mettre sur la ligne de régime de la cellule. `lire` : lecteur
    simulé (tests)."""

    def __init__(self, index: int = 0, pas: float = 0.5, lire=None):
        self.index, self.pas, self._lire = index, pas, lire or (lambda: _lire_charge(index))
        self.releves: list[list[str]] = []
        self._stop = threading.Event()
        self._fil: threading.Thread | None = None

    def _boucle(self) -> None:
        while not self._stop.is_set():
            r = self._lire()
            if r is not None:
                self.releves.append(r)
            self._stop.wait(self.pas)

    def __enter__(self):
        self._fil = threading.Thread(target=self._boucle, daemon=True)
        self._fil.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._fil is not None:
            self._fil.join(timeout=self.pas + 5)

    def bilan(self) -> dict | None:
        rel = [r for r in self.releves if r[0].isdigit()]
        if not rel:
            return None
        mhz = sorted(int(r[0]) for r in rel)
        watts = sorted(float(r[1]) for r in rel if r[1].replace(".", "", 1).isdigit())
        parts = {nom: sum(1 for r in rel if r[2 + i] == "Active") / len(rel) for i, nom in enumerate(RAISONS)}
        raison, part = max(parts.items(), key=lambda kv: kv[1])
        return {"n": len(rel), "sm_mediane": mhz[len(mhz) // 2], "sm_p95": mhz[min(len(mhz) - 1, int(0.95 * len(mhz)))],
                "watts_mediane": watts[len(watts) // 2] if watts else None,
                "raison": raison if part > 0 else "aucune", "part": part}

    def etiquette(self, mode: str | None) -> str:
        """``eco=<mode>(<raison> : méd. <MHz> MHz)`` ; sans relevé ``eco=<mode>(sous charge : ?)``."""
        b = self.bilan()
        tete = f"eco={mode or 'off'}"
        if b is None:
            return f"{tete}(sous charge : ?)"
        if b["raison"] == "sw_power_cap":
            raison = f"plafond {b['watts_mediane']:.0f} W" if b["watts_mediane"] else "plafond de puissance"
        elif b["raison"] == "aucune":
            raison = "sans bridage"
        else:
            raison = b["raison"]
        return f"{tete}({raison} : méd. {b['sm_mediane']} MHz, {int(b['part'] * 100)} % des {b['n']} relevés)"
