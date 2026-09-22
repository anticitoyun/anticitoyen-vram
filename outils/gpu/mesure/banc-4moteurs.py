#!/usr/bin/env python3
"""Comparatif des quatre moteurs sur le même parc, au même protocole.

Pour chaque modèle servi par plusieurs moteurs, on lance le moteur par son
lanceur officiel (le même que les menus), on envoie un prompt unique en
streaming, 200 jetons, température 0, deux fois, et on garde la meilleure
mesure. On note le débit de décodage, le temps au premier jeton, la puissance
tirée sur la carte pendant la génération et les jetons par kilojoule.

    acvram    acvram-serveur <dossier>           port 8090
    llamacpp  llamacpp-serveur <dossier> <ctx>   port 8080   (llama-server)
    vllm      vllm-serveur <dossier> <ctx>       port 8000
    tabby     TabbyAPI main.py + /v1/model/load  port 5000   (exllamav3)

Un seul moteur occupe la carte à la fois ; le llama-server d'appoint du port
8081 (3080 Ti) n'est jamais touché. Aucune clé n'est affichée : elles sont lues
dans ~/.config/ia-secrets.env et dans api_tokens.yml de TabbyAPI.

    banc-4moteurs.py --simuler                  la table des couples
    banc-4moteurs.py [--moteurs a,b] [--modeles motif] [--ctx 8192] [--sortie f.tsv]

Reprenable : les couples déjà présents dans le TSV de sortie sont sautés.
"""
from collections import namedtuple
import argparse, hashlib, json, os, re, statistics, subprocess, sys, threading, time, tomllib, urllib.request

KIMI = os.path.expanduser("~/.kimi-code")
BIN = os.path.expanduser("~/.local/bin")
SECRETS = os.path.expanduser("~/.config/ia-secrets.env")
TABBY_DIR = "/opt/ia/TabbyAPI"
VLLM_DIR = "/opt/ia/vLLM"
PORTS = {"acvram": 8090, "llamacpp": 8080, "vllm": 8000, "tabby": 5000}
PORT_INTERDIT = 8081
PROMPT = ("Explique en détail, en français et en plusieurs paragraphes, comment "
          "fonctionne la mémoire virtuelle d'un système d'exploitation moderne : "
          "pages, tables de pages, TLB, défauts de page, et ce qui se passe quand "
          "la mémoire physique est pleine.")
MAX_TOKENS = 200
# Trois passages et non deux : avec deux, la « dispersion » publiee est un
# ecart entre deux nombres, dont on ne peut rien conclure. Avec trois, elle
# devient un ecart-type utilisable, et le critere « l'ecart annonce doit valoir
# au moins trois fois la dispersion » a un sens. Le prix est une serie une fois
# et demie plus longue ; pour les grandes series on reduit le nombre de
# modeles, jamais la rigueur par modele. Decision du 8 septembre 2026.
#
# Le PREMIER passage est froid — allocateur, noyaux compiles a la volee, carte
# a temperature de repos. On publie le meilleur des trois, ce qui l'ecarte de
# fait ; sa valeur reste comptee dans la dispersion, et c'est voulu : elle dit
# aussi ce que coute le demarrage a froid.
# Nombre de passages par couple. Trois suffit pour un ecart-type utilisable ;
# CINQ est exige par la serie determinisme (docs/SERIE-DETERMINISME.md) parce
# qu'avec trois, un premier passage froid et deux passages proches donnent la
# meme image qu'une bimodalite, et on ne les distingue pas. Paramétrable par
# `--passages` : on ne reduit jamais la rigueur par modele, on reduit le nombre
# de modeles.
MESURES = 3
FORCER_EXIL = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energie import Energie, repos            # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------
# secrets : lus, jamais affichés
# --------------------------------------------------------------------------
def secret(nom):
    try:
        for l in open(SECRETS):
            if l.startswith(nom + "="):
                return l.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def cle_tabby(admin=False):
    try:
        for l in open(os.path.join(TABBY_DIR, "api_tokens.yml")):
            if l.startswith("admin_key:" if admin else "api_key:"):
                return l.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


CLES = {"acvram": "", "llamacpp": secret("CLE_LLAMACPP"), "vllm": secret("CLE_VLLM"),
        "tabby": cle_tabby()}


# --------------------------------------------------------------------------
# le parc : modèle → {moteur: (alias, dossier, ctx)}
# --------------------------------------------------------------------------
def lire_tsv(nom):
    out = []
    p = os.path.join(KIMI, nom)
    if not os.path.exists(p):
        return out
    for l in open(p):
        c = l.rstrip("\n").split("\t")
        if len(c) >= 3 and c[1].startswith("/"):
            out.append((c[0], c[1], int(c[2]) if c[2].isdigit() else 32768))
    return out


def nom_source_acvram(dossier):
    try:
        m = json.load(open(os.path.join(dossier, "acvram_manifest.json")))
        n = m["model"]["name"]
        return n[:-5] if n.endswith(".gguf") else n
    except Exception:
        return os.path.basename(dossier)


def parc():
    modeles = {}
    for alias, d, ctx in lire_tsv("acvram-chemins.tsv"):
        if os.path.isdir(d):
            for cle in {nom_source_acvram(d), os.path.basename(d)}:
                modeles.setdefault(cle, {})["acvram"] = (alias, d, ctx)
    for alias, d, ctx in lire_tsv("gguf-chemins.tsv"):
        if os.path.isdir(d):
            modeles.setdefault(os.path.basename(d), {})["llamacpp"] = (alias, d, ctx)
    for alias, d, ctx in lire_tsv("vllm-chemins.tsv"):
        if os.path.isdir(d):
            modeles.setdefault(os.path.basename(d), {})["vllm"] = (alias, d, ctx)
    conf = tomllib.load(open(os.path.join(KIMI, "config.toml"), "rb"))
    model_dir = "/mnt/4TO_SATACMR_2022/Modeles/models_exl3"
    try:
        for l in open(os.path.join(TABBY_DIR, "config.yml")):
            if l.strip().startswith("model_dir:"):
                model_dir = l.split(":", 1)[1].strip()
    except OSError:
        pass
    for alias, m in conf.get("models", {}).items():
        if m.get("provider") == "tabby":
            d = os.path.join(model_dir, m["model"])
            if os.path.isdir(d):
                modeles.setdefault(m["model"], {})["tabby"] = (alias, d, m.get("max_context_size", 32768))
    # fusion des entrées acvram doublées (nom source et nom de dossier)
    fusion = {}
    for nom, moteurs in modeles.items():
        fusion.setdefault(nom, {}).update(moteurs)
    return {n: m for n, m in fusion.items() if len(m) >= 1}


# --------------------------------------------------------------------------
# serveurs
# --------------------------------------------------------------------------
def pid_port(port):
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    for l in out.splitlines():
        if f":{port} " in l:
            m = re.search(r"pid=(\d+)", l)
            if m:
                return int(m.group(1))
    return None


def http(port, chemin, cle="", data=None, delai=5, methode=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{chemin}",
                                 data=json.dumps(data).encode() if data is not None else None,
                                 method=methode)
    req.add_header("Content-Type", "application/json")
    if cle:
        req.add_header("Authorization", f"Bearer {cle}")
    return urllib.request.urlopen(req, timeout=delai)


def pret(moteur):
    port = PORTS[moteur]
    try:
        if moteur == "tabby":
            return b"healthy" in http(port, "/health", delai=3).read()
        return http(port, "/v1/models", CLES[moteur], delai=3).status == 200
    except Exception:
        return False


def arreter(moteur):
    """Arrêt par PID du port — jamais pkill -f, jamais le port 8081."""
    port = PORTS[moteur]
    assert port != PORT_INTERDIT
    if moteur == "tabby":
        try:
            http(port, "/v1/model/unload", cle_tabby(True), delai=30, methode="POST").read()
            time.sleep(2)
        except Exception:
            pass
        return
    pid = pid_port(port)
    if pid:
        subprocess.run(["kill", str(pid)])
        for _ in range(30):
            if pid_port(port) is None:
                break
            time.sleep(1)
        time.sleep(3)


def arreter_tout_sauf(moteur):
    for m in PORTS:
        if m != moteur and (m == "tabby" or pid_port(PORTS[m])):
            arreter(m)


MOTIFS = {"acvram": "acvram serve", "llamacpp": f"llama-server", "vllm": "vllm"}


def tuer_orphelin(moteur, age_max):
    """Un serveur qui n'a pas ouvert son port au timeout n'a pas de PID de
    port : ``arreter()`` ne le voit pas, il continue à charger et fait manquer
    de mémoire les modèles suivants (NEO-CODE, 6/09/2026). On le cherche par
    son motif de commande et son âge, jamais par ``pkill -f``, et jamais le
    llama-server permanent du port 8081."""
    motif = MOTIFS.get(moteur)
    if not motif:
        return
    ps = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout
    for l in ps.splitlines()[1:]:
        c = l.split(None, 2)
        if len(c) < 3 or motif not in c[2] or str(PORT_INTERDIT) in c[2]:
            continue
        if int(c[1]) <= age_max + 60:
            log(f"           serveur {moteur} orphelin (PID {c[0]}, {c[1]} s) : tué")
            subprocess.run(["kill", c[0]])


def attendre(moteur, secondes):
    t0 = time.time()
    while time.time() - t0 < secondes:
        if pret(moteur):
            return time.time() - t0
        time.sleep(2)
    tuer_orphelin(moteur, secondes)
    raise RuntimeError(f"{moteur} n'a pas démarré en {secondes} s")


def memoire_libre():
    """Memoire libre par carte, en Mio, dans l'ordre des index physiques."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return [int(l.strip()) for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def _meminfo(champ):
    try:
        for l in open("/proc/meminfo", encoding="utf-8"):
            if l.startswith(champ + ":"):
                return int(l.split()[1]) // 1024
    except OSError:
        pass
    return 0


def ram_hote_libre_mio():
    """RAM hote reellement disponible (MemAvailable), en Mio."""
    return _meminfo("MemAvailable")


def ram_verrouillee_mio():
    """Memoire que le noyau ne peut ni evincer ni swapper, en Mio.

    `Unevictable` et non `Mlocked` : releve d'Oceane le 8/09/2026 sur cette
    machine, au repos, sans rien charger — Unevictable 960 664 kio contre
    Mlocked 132. `Mlocked` ne compte que le mlock() classique ; la memoire
    epinglee par le pilote CUDA ne passe pas par la. Diagnostiquer avec lui
    ferait lire « rien n'est verrouille » sur des dizaines de gigaoctets qui
    le sont.
    """
    return _meminfo("Unevictable")


def ram_totale_mio():
    return _meminfo("MemTotal")


def poids_mio(dossier):
    """Taille des poids sur disque, en Mio. 0 si on ne sait pas."""
    total = 0
    try:
        for rep, _, fichiers in os.walk(dossier):
            for f in fichiers:
                if f.endswith((".safetensors", ".gguf", ".bin")):
                    try:
                        total += os.path.getsize(os.path.join(rep, f))
                    except OSError:
                        pass
    except OSError:
        return 0
    return total // (1024 * 1024)


def attendre_memoire(secondes=90, marge=200):
    """Attend que la memoire des cartes cesse de bouger avant de charger.

    Mesure du 8 septembre 2026 : le MEME modele, charge deux fois de suite
    sous le meme alias, a rendu 15,8 puis 152,2 t/s — facteur 9,6. Le journal
    du serveur donne la cause : au premier chargement le plan exilait 5 puis 6
    MLP de plus en RAM hote (« poids reels 8.7 Gio pour 11.1 Gio »), au second
    aucun. Le serveur precedent n'avait pas fini de rendre sa memoire quand le
    suivant a calcule son plan. Le port etait ferme, la memoire non.

    Sans cette attente, le placement — donc le debit — est tire au sort a
    chaque chargement, et aucune comparaison entre modeles ne veut rien dire.
    """
    precedent = None
    t0 = time.time()
    while time.time() - t0 < secondes:
        libre = memoire_libre()
        if not libre:
            return []
        if precedent and all(abs(a - b) <= marge for a, b in zip(libre, precedent)):
            return libre
        precedent = libre
        time.sleep(3)
    return memoire_libre()


class MemoireInsuffisante(RuntimeError):
    pass


def verifier_place(dossier, libre_vram, ram_libre=None, reserve_mio=8192,
                   part_tampons=0.20, verrouille_mio=None, total_mio=None,
                   reserve_systeme_mio=16384):
    """Refuse de charger ce qui ne tiendrait pas en RAM hote.

    Attendre que la memoire cesse de bouger ne suffit pas : le 8 septembre
    2026 la garde a rendu la main sur 1951 Mio de VRAM libre — stables — et
    un modele de 44 Gio a ete charge quand meme. Ce qui ne tient pas dans la
    VRAM part en RAM hote ; la RAM a sature et la machine a redemarre,
    emportant le worktree et les mesures en cours. Attendre la STABILITE
    n'est pas verifier la DISPONIBILITE.

    Ce qui est mesure : le poids sur disque, la VRAM libre, MemAvailable au
    moment du chargement. Ce qui ne l'est PAS : les deux constantes.
    `reserve_mio` (8 Gio pour le reste du systeme) et `part_tampons` (20 % du
    poids pour le cache KV, les tampons et les copies de chargement) sont des
    choix d'ingenierie, pas des mesures. Ce qui les etablirait : relever le
    MemAvailable MINIMUM pendant un chargement dont l'exil est connu, sur
    trois tailles de modele — la difference avec l'exil donne les tampons, et
    le plancher tolerable donne la reserve. Tant que cette mesure n'est pas
    faite, ces deux chiffres sont prudents et arbitraires, et le refus qu'ils
    provoquent doit pouvoir etre leve : c'est le role de --forcer-exil.
    """
    poids = poids_mio(dossier)
    if not poids or not libre_vram:
        return
    exil = poids - sum(libre_vram)
    if exil <= 0:
        return
    ram = ram_hote_libre_mio() if ram_libre is None else ram_libre
    besoin = exil + int(part_tampons * poids) + reserve_mio
    if ram and besoin > ram:
        raise MemoireInsuffisante(
            f"{poids} Mio de poids, {sum(libre_vram)} Mio de VRAM libre : "
            f"{exil} Mio partiraient en RAM hôte, soit {besoin} Mio avec les "
            f"tampons et la réserve, pour {ram} Mio disponibles. "
            f"Chargement refusé (--forcer-exil pour passer outre).")

    # Second test, et il ne fait PAS double emploi avec le premier. Les poids
    # exiles vivent en memoire EPINGLEE : le noyau ne peut ni les evincer ni
    # les swapper, donc il chasse tout le reste — le 8/09/2026, le bureau et
    # le navigateur sont partis en swap et la machine est devenue inutilisable
    # alors que `free` annoncait 83 Go disponibles.
    #
    # `MemAvailable` repond a « puis-je prendre cela maintenant » ; il ne dit
    # rien de ce qui restera au systeme APRES. Un chargement peut passer le
    # premier test et porter le total verrouille a un niveau ou le reste de la
    # machine n'a plus de quoi vivre. On borne donc aussi le verrouillage
    # total, rapporte a la RAM du systeme et non a ce qui est libre.
    verrouille = ram_verrouillee_mio() if verrouille_mio is None else verrouille_mio
    total = ram_totale_mio() if total_mio is None else total_mio
    if total and verrouille + exil > total - reserve_systeme_mio:
        raise MemoireInsuffisante(
            f"{verrouille} Mio déjà verrouillés + {exil} Mio à épingler = "
            f"{verrouille + exil} Mio sur {total} Mio de RAM totale, il ne "
            f"resterait pas {reserve_systeme_mio} Mio au reste du système. "
            f"Chargement refusé (--forcer-exil pour passer outre).")


def demarrer(moteur, dossier, ctx):
    """Lance le moteur et rend le temps de chargement, lanceur compris (les
    lanceurs attendent eux-mêmes le port avant de rendre la main)."""
    libre = attendre_memoire()
    if libre:
        log("           mémoire libre avant chargement : "
            + " / ".join(f"{m} Mio" for m in libre)
            + f" ; RAM hôte {ram_hote_libre_mio()} Mio")
    if not FORCER_EXIL:
        verifier_place(dossier, libre)
    t0 = time.time()
    _demarrer(moteur, dossier, ctx)
    return time.time() - t0


def _demarrer(moteur, dossier, ctx):
    arreter_tout_sauf(moteur)
    env = dict(os.environ)
    if moteur == "acvram":
        arreter("acvram")
        env["CTX"] = str(ctx)
        # ET en argument : une variable d'environnement peut etre ecrasee par
        # le lanceur, un argument non. Le 9/09/2026 `CTX="${2:-}"` ecrasait
        # l'environnement par un argument absent et retombait sur 32768.
        subprocess.run([os.path.join(BIN, "acvram-serveur"), dossier, str(ctx)], env=env,
                       capture_output=True, text=True, timeout=900)
        attendre("acvram", 900); return
    if moteur == "llamacpp":
        arreter("llamacpp")
        subprocess.run([os.path.join(BIN, "llamacpp-serveur"), dossier, str(ctx)], env=env,
                       capture_output=True, text=True, timeout=900)
        attendre("llamacpp", 900); return
    if moteur == "vllm":
        arreter("vllm")
        subprocess.run([os.path.join(BIN, "vllm-serveur"), dossier, str(ctx)], env=env,
                       capture_output=True, text=True, timeout=1800)
        attendre("vllm", 1800); return
    if moteur == "tabby":
        if not pret("tabby"):
            env.update({"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
            subprocess.Popen(["setsid", "nohup", ".venv/bin/python", "main.py"], cwd=TABBY_DIR,
                             env=env, stdout=open(os.path.join(TABBY_DIR, "tabby.log"), "a"),
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            attendre("tabby", 180)
        r = http(5000, "/v1/model/load", cle_tabby(True),
                 {"model_name": os.path.basename(dossier), "max_seq_len": ctx,
                  "cache_mode": "Q8"}, delai=900, methode="POST")
        r.read()
        return
    raise ValueError(moteur)


# --------------------------------------------------------------------------
# mesure
# --------------------------------------------------------------------------
# L'ancienne classe Watt est retiree : elle lancait un sous-processus
# nvidia-smi toutes les 200 ms — sur la machine qu'elle mesurait — pour
# moyenner des echantillons de puissance sur la seule carte 0. Trois defauts,
# tous corriges dans outils/energie.py : le cout de la mesure, la moyenne qui
# n'est pas une energie, et une carte lue sur deux alors qu'un modele exile
# travaille sur les deux. Mesure du 8 septembre 2026 a l'appui.


def binaire_servant(moteur) -> str:
    """Le binaire REELLEMENT en train de servir, lu dans /proc.

    Le lanceur dit ce qu'il croit lancer ; `/proc/<pid>/exe` dit ce qui tourne.
    Le 9/09/2026 la machine portait DEUX llama.cpp — celui du depot et celui
    empaquete par Jan — et le banc lancait le second pendant qu'on verifiait
    les options du premier. Un comparatif qui ne dit pas a quel concurrent il
    se compare ne compare rien.
    """
    pid = pid_port(PORTS[moteur])
    if not pid:
        return "?"
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return "?"


_CLES_CTX = ("--max-model-len", "--ctx-size", "--ctx")


def ctx_servant(moteur) -> str:
    """Le contexte REELLEMENT demande au serveur, lu dans /proc/<pid>/cmdline.

    Le banc dit ce qu'il veut ; le lanceur le transmet ou non. Le 9/09/2026,
    `acvram-serveur` ecrasait la variable d'environnement CTX par un argument
    absent (`CTX="${2:-}"`) et retombait sur son defaut de 32768 : le banc
    croyait 8192, le serveur tournait a 32768, et la colonne `ctx` publiait
    l'intention du banc. Sur un modele de 27,5 Gio, cela suffit a forcer un
    exil — puis a le mesurer comme un defaut d'architecture.

    La garde qui compare les contextes des COUPLES ne pouvait pas l'attraper :
    elle compare ce que le banc demande, tous deux a 8192. Il faut demander au
    processus, pas a soi-meme.
    """
    pid = pid_port(PORTS[moteur])
    if not pid:
        return "?"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            args = fh.read().decode(errors="replace").split("\x00")
    except OSError:
        return "?"
    for i, arg in enumerate(args):
        if arg in _CLES_CTX and i + 1 < len(args):
            return args[i + 1]
        for cle in _CLES_CTX:
            if arg.startswith(cle + "="):
                return arg.split("=", 1)[1]
    return "?"


def modele_servi(moteur):
    try:
        d = json.load(http(PORTS[moteur], "/v1/models", CLES[moteur], delai=5))
        return d["data"][0]["id"]
    except Exception:
        return None


def generer(moteur):
    """Une génération en streaming : (jetons, ttft_s, duree_decodage_s, Energie, texte)."""
    port = PORTS[moteur]
    corps = {"model": modele_servi(moteur) or "x",
             "messages": [{"role": "user", "content": PROMPT}],
             "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
             "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(corps).encode())
    req.add_header("Content-Type", "application/json")
    if CLES[moteur]:
        req.add_header("Authorization", f"Bearer {CLES[moteur]}")
    with Energie() as w:
        t0 = time.time()
        texte: list[str] = []
        premier = dernier = None
        n = 0
        usage = None
        with urllib.request.urlopen(req, timeout=600) as r:
            # Lecture non bufferisée : ``for ligne in r`` attend un bloc de
            # 8 Kio, et un moteur à 260 t/s y fait tenir 200 jetons d'un coup
            # — tous horodatés pareil, débit infini. read1 rend ce qui est là.
            reste = b""
            def lignes():
                nonlocal reste
                while True:
                    bloc = r.read1(65536)
                    if not bloc:
                        if reste:
                            yield reste
                        return
                    reste += bloc
                    while b"\n" in reste:
                        l, reste = reste.split(b"\n", 1)
                        yield l
            for ligne in lignes():
                if not ligne.startswith(b"data:"):
                    continue
                brut = ligne[5:].strip()
                if brut == b"[DONE]":
                    break
                try:
                    ev = json.loads(brut)
                except Exception:
                    continue
                if ev.get("error"):
                    e = ev["error"]
                    raise RuntimeError("serveur : " + str(e.get("message", e) if isinstance(e, dict) else e)[:160])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        texte.append(delta.get("content") or delta.get("reasoning_content") or "")
                        maintenant = time.time()
                        if premier is None:
                            premier = maintenant
                        dernier = maintenant
                        n += 1
    if premier is None:
        raise RuntimeError("aucun jeton reçu")
    morceaux = n
    # Le banc a compté les morceaux du flux ; le moteur, lui, annonce SON
    # nombre de jetons. On garde le sien pour le débit — c'est la grandeur que
    # l'utilisateur reçoit — mais on garde AUSSI le nôtre, parce que le débit
    # comparé entre deux moteurs suppose qu'ils comptent pareil, et rien ne le
    # garantit : jeton de fin, jetons spéciaux, tokenizers différents sur le
    # même texte. Publier les deux, c'est établir cette égalité au lieu de
    # l'espérer.
    #
    # Réserve d'interprétation : `morceaux` compte les ÉVÉNEMENTS du flux, pas
    # les jetons. Un moteur qui groupe plusieurs jetons par événement creuse
    # l'écart sans qu'aucune tokenisation ne diffère. Un écart non nul
    # appelle donc un examen, il ne prouve rien à lui seul.
    source = "flux"
    if usage and usage.get("completion_tokens"):
        n = usage["completion_tokens"]
        source = "moteur"
    if morceaux < 2 or dernier - premier < 1e-3:
        # Le débit se mesure entre le premier et le dernier jeton ; sans flux
        # jeton par jeton, cette durée n'existe pas. On REFUSE la mesure au
        # lieu de la recalculer sur la durée totale : ce serait une seconde
        # définition du débit dans le même tableau, sans marque dans la ligne.
        raise RuntimeError(f"pas de flux jeton par jeton ({morceaux} morceau(x) "
                           f"pour {n} jetons)")
    return n, premier - t0, dernier - premier, w, "".join(texte), morceaux, source


# Un passage nomme, et non un tuple positionnel. Le 9/09/2026, ajouter deux
# colonnes a `passages.append` a casse `tps, ttft, e, n, txt, dt = ordonnes[...]`
# — « too many values to unpack » — et la campagne a rendu deux lignes a
# `t_s 0.0`. Des lignes qui existent et ne valent rien : pire qu'un fichier
# vide, un lecteur presse y voit une campagne faite. Avec des champs nommes,
# ajouter une colonne ne peut plus casser une lecture.
Passage = namedtuple("Passage",
                     "tps ttft energie jetons texte duree morceaux source")


def _temperature() -> int:
    """Temperature de la carte mesuree, pour prouver le palier au lieu de le
    supposer. Sans elle, « les deux moteurs etaient chauds » est une intention."""
    try:
        o = subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                            "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:                            # noqa: BLE001
        return -1


def _dispersion_rel(v: list[float]) -> float:
    """Etendue rapportee a la mediane, en %, sur les FLOTTANTS.

    Calculer une dispersion sur la chaine publiee mesure le pas d'arrondi et
    non la grandeur : sur 35 t/s a une decimale, ce pas vaut 0,286 %, et deux
    sigma tires de la chaine (0,164 % et 0,000 %) ne decrivaient que
    l'affichage.
    """
    v = [x for x in v if x]
    if len(v) < 2:
        return 0.0
    med = sorted(v)[len(v) // 2]
    return (max(v) - min(v)) / med * 100 if med else 0.0


def mesurer(moteur):
    """Débit MÉDIAN et énergie du MÊME passage, puis la ligne de base après lui.

    L'énergie ne se moyenne pas entre passages : elle vient du passage dont on
    publie le débit, sinon les deux colonnes décrivent deux exécutions
    différentes. Et la ligne de base se prend APRÈS : prise une fois au début,
    elle dérive, et la dérive se retrouve attribuée au moteur mesuré — c'est
    ce qui avait fait croire à une décroissance de coût le 7 septembre 2026.
    """
    # ---- CHAUFFE : amener la carte a son palier thermique AVANT de mesurer
    #
    # Mesure du 9/09 sur 45 passages identiques : la puissance derive de
    # 308,61 W (49 degC) a un palier de 317,39 W (54 degC), soit **8,78 W**.
    # Ce n'est ni lineaire ni polynomial — c'est une chauffe exponentielle
    # amortie, atteinte vers le passage 30, environ trois minutes.
    #
    # POURQUOI CELA COMPTE PLUS QU'UNE CORRECTION DE SECOND ORDRE : une
    # campagne mesure un moteur PUIS l'autre. Le premier demarre froid, le
    # second est deja chaud. Nos campagnes attribuent **14 W** d'ecart aux
    # moteurs ; jusqu'a **8 W** peuvent etre thermiques. Plus de la moitie.
    #
    # Le contrebalancement `A B B A` avait ete envisage : il annule un biais
    # LINEAIRE, pas une exponentielle amortie, et il coute deux chargements de
    # plus. La chauffe traite la CAUSE au lieu de compenser l'effet.
    #
    # Au palier, l'etendue tombe a 0,45 W sur douze passages : un ecart de 1 W
    # entre moteurs deviendrait mesurable.
    #
    # RESERVE : ce palier vaut pour CE modele, CE debit, CETTE temperature
    # ambiante. Un modele plus gourmand chauffera plus haut et plus longtemps.
    chauffe = float(os.environ.get("ACVRAM_CHAUFFE_S", "180"))
    t_avant = _temperature()
    if chauffe > 0:
        log(f"           chauffe {chauffe:.0f} s (carte a {t_avant} degC)")
        fin = time.time() + chauffe
        while time.time() < fin:
            try:
                generer(moteur)
            except Exception:                    # noqa: BLE001
                break
        log(f"           chauffe finie : {_temperature()} degC")

    passages = []
    for _ in range(MESURES):
        n, ttft, dt, e, txt, morceaux, source = generer(moteur)
        tps = (n - 1) / dt if n > 1 else 0.0
        passages.append(Passage(tps, ttft, e, n, txt, dt, morceaux, source))
    debits = [p.tps for p in passages]

    # Le passage PUBLIE est celui de debit median, plus celui de debit
    # maximal. « Le meilleur des trois » est un estimateur biaise : il
    # favorise le moteur le plus bruyant. Mesure du 8 septembre 2026 :
    # sur un couple a 21,3 % de dispersion contre 5,2 en face, le meilleur
    # donnait la victoire au plus instable. Le maximum reste publie a part,
    # il n'est simplement plus ce qu'on compare.
    # UNE SEULE definition du regime, pour TOUTES les colonnes publiees.
    #
    # Avant le 8/09 le banc publiait « le meilleur des trois », ce qui ecartait
    # le passage froid DE FAIT. Le passage au median a supprime cet ecartement
    # sans que personne ne le remarque — et toute colonne suivant le passage
    # median s'est mise a pouvoir publier la passe froide.
    #
    # Le 9/09, le cas s'est produit : llama.cpp a publie TTFT 125 ms
    # (passages 125,45,33,33,33), donc le passage median EN DEBIT etait le
    # premier. Les watts, les joules et les jetons/kJ de cette ligne venaient
    # donc eux aussi de la passe froide. Une conclusion a ete inversee sur le
    # TTFT (« +41,6 % en notre faveur » au lieu de -121 %), et l'energie —
    # l'objectif du projet — etait touchee par le meme defaut sans qu'on le
    # voie, parce que les DEUX bras du comparatif le subissaient egalement.
    #
    # La classe du defaut est « quelle passe publie-t-on ». On la traite en
    # une fois plutot qu'en corrigeant les colonnes une a une.
    regime = passages[1:] or passages
    ordonnes = sorted(regime, key=lambda x: x.tps)
    median = ordonnes[len(ordonnes) // 2]
    tps, e, n, txt, dt = (median.tps, median.energie,
                          median.jetons, median.texte, median.duree)
    # Le TTFT a sa propre mediane : le passage le plus representatif en debit
    # ne l'est pas forcement en latence de premier jeton.
    _ttfts = sorted(p.ttft for p in regime)
    ttft = _ttfts[len(_ttfts) // 2]
    # Le TTFT du passage FROID est publie a part : il n'est pas du bruit, c'est
    # ce que paie la premiere requete d'un utilisateur. Deux chiffres vrais
    # dans deux conditions, comme le x2,7 des graphes et sa borne sur un dense.
    ttft_froid = passages[0].ttft
    # `debits` couvre TOUS les passages, froid compris : la dispersion reste
    # conservatrice, et c'est voulu. Le median, lui, ne porte que le regime.
    etendue = (min(debits), max(debits))

    # Empreinte du texte de CHAQUE passage. A temperature zero, le meme
    # prompt doit rendre le meme texte : si les empreintes different, le
    # chemin n'est pas deterministe, et c'est un defaut a part entiere —
    # plus important que le debit qu'on etait venu mesurer. Si elles sont
    # identiques, une dispersion de debit ne peut pas venir du texte, et il
    # faut la chercher ailleurs (passage froid, cache, ordonnancement).
    empreintes = [hashlib.sha256(p.texte.encode()).hexdigest()[:8] for p in passages]
    # Le controle doit porter sur les MEMES donnees que la mesure qu'il garde.
    # Le debit publie ecarte le premier passage ; le compter ici invalidait la
    # campagne sur une passe dont personne ne se sert. Le 9/09, les deux
    # moteurs ont diverge au premier passage et a lui seul (acvram
    # e0e0c3e9 puis 036bb29d x4 ; llamacpp b1171ed9 puis 7dc6fe3e x4) — la
    # selection d'algorithme cuBLAS au premier appel suffit a faire basculer
    # un argmax serre. Les empreintes restent TOUTES publiees.
    # Meme perimetre que le median publie : `regime`, pas un decoupage a part.
    textes_identiques = len({hashlib.sha256(p.texte.encode()).hexdigest()[:8]
                             for p in regime}) == 1
    watts = e.moyenne
    base = repos(secondes=min(max(dt, 5.0), 30.0))
    joules = e.joules
    joules_net = max(joules - base.moyenne * e.duree, 0.0)
    jkj = tps / watts * 1000 if watts else 0.0
    jkj_net = n / joules_net * 1000 if joules_net else 0.0
    dispersion = (100 * statistics.pstdev(debits) / statistics.mean(debits)
                  if len(debits) > 1 and statistics.mean(debits) else 0.0)
    r = e.resume()
    # La dispersion n'invalide pas par elle-meme : elle invalide un ECART
    # annonce plus petit qu'elle. Le banc ne connait pas l'ecart qu'on lui
    # fera dire, il signale donc, il ne tranche pas.
    def signaler(quoi):
        r["invalidations"] = (quoi if r["invalidations"] == "aucune"
                              else r["invalidations"] + " ; " + quoi)

    if dispersion > 5.0:
        signaler(f"dispersion des {len(debits)} passages : {dispersion:.1f} %")
    if not textes_identiques:
        signaler("temperature zero mais textes differents entre passages : "
                 + ",".join(empreintes))
    energie = {
        "J": round(joules, 1), "J_net": round(joules_net, 1),
        "W_repos": round(base.moyenne, 1), "jkj_net": round(jkj_net, 1),
        "plafond_W": r["plafond_w"], "horloge_min": r["horloge_min"],
        "horloge_max": r["horloge_max"], "temp_max": r["temp_max"],
        "bridages": r["bridages"], "invalidations": r["invalidations"],
        "dispersion_pct": round(dispersion, 1),
        # Dispersion des WATTS, calculee sur les flottants comme celle du
        # debit — arrondie seulement a l'affichage, et a DEUX decimales.
        # C'est elle qui decide si un ecart d'energie est resoluble : le debit
        # ne borne la dispersion de `jkj` que PAR LE BAS (jkj = tps / watts),
        # donc un debit parfaitement stable est compatible avec une puissance
        # tres bruyante. Sans cette colonne la question reste indecidable.
        "temp_avant": t_avant, "temp_apres": _temperature(),
        "dispersion_W_pct": round(_dispersion_rel([
            p.energie.get("watts", 0.0) if isinstance(p.energie, dict) else 0.0
            for p in regime]), 2),
        "t_s_min": round(etendue[0], 1), "t_s_max": round(etendue[1], 1),
        "empreintes": ",".join(empreintes),
        "textes_identiques": "oui" if textes_identiques else "NON",
        # Les debits DANS L'ORDRE d'execution. min/max disent qu'un passage
        # s'ecarte, pas LEQUEL : un premier passage froid et une bimodalite
        # rendent le meme min, le meme max et la meme dispersion. Sans
        # l'ordre, la table de decision de docs/SERIE-DETERMINISME.md ne peut
        # pas etre appliquee (mesure du 8 septembre 2026).
        "t_s_passages": ",".join(f"{d:.1f}" for d in debits),
        # Les watts et l'energie de CHAQUE passage, a deux decimales.
        #
        # Au singulier, une colonne ne peut pas etre mise a l'epreuve : le banc
        # ne publiait qu'un `W`, celui du passage median, et la question « le
        # -4,7 % d'energie est-il resoluble ? » etait donc indecidable sur les
        # donnees publiees.
        #
        # LA PRECISION EST LA RAISON D'ETRE DE CES COLONNES. A l'entier, un
        # jkj de 128 est quantifie a 0,78 % — un tiers du seuil de 2,4 %
        # qu'elles existent pour eprouver. Elles naitraient incapables de
        # repondre a la question qui les motive. Meme piege que `t_s_passages`
        # a une decimale : sur 35 t/s, le pas d'arrondi vaut 0,286 %, et deux
        # sigma calcules dessus (0,164 % et 0,000 %) ne mesuraient que
        # l'affichage.
        "W_passages": ",".join(f"{p.energie.get('watts', 0):.2f}"
                               if isinstance(p.energie, dict) else "0.00"
                               for p in regime),
        "jkj_passages": ",".join(
            f"{(p.jetons / (p.energie.get('J', 0) or 1) * 1000):.2f}"
            if isinstance(p.energie, dict) and p.energie.get("J") else "0.00"
            for p in regime),
        # Le temps au premier jeton DE CHAQUE passage. Le debit dit que le
        # premier passage coute ; le TTFT dit ou il coute. Si le premier TTFT
        # est seul eleve, le prix est paye avant la generation — lecture des
        # caches, allocation de l'arene, initialisation du contexte — et un
        # prechargement le supprimerait au lieu de le subir. Releve du
        # 8 septembre 2026 : ni ~/.nv/ComputeCache ni ~/.triton/cache n'ont
        # ete ecrits pendant les series, donc ce n'est PAS de la compilation.
        "ttft_passages": ",".join(f"{p.ttft * 1000:.0f}" for p in passages),
        # Le TTFT de la premiere requete, publie A COTE du regime : les deux
        # sont vrais, dans deux conditions. Ne publier que le regime commet
        # l'erreur symetrique de celle qu'on vient de reparer.
        "ttft_froid_ms": round(ttft_froid * 1000),
        # Les deux comptages, pour que « jetons par seconde » veuille dire la
        # meme chose d'un moteur a l'autre.
        "jetons_moteur": n,
        "jetons_flux": ",".join(str(p.morceaux) for p in passages),
        # D'où vient le nombre publié. Un moteur qui n'annonce pas d'`usage`
        # laisse le banc compter, et alors « jetons_moteur » porterait un nom
        # menteur : c'est le nôtre. Deux moteurs de provenance différente dans
        # le même tableau ne comparent pas la même grandeur.
        "jetons_source": passages[0].source,
        "binaire": binaire_servant(moteur),
        "ctx_servi": ctx_servant(moteur),
    }
    # Un aperçu du texte à côté du débit : 481 t/s de « de de de » sur quatre
    # jetons se lisaient comme un record tant qu'on ne voyait pas le texte.
    apercu = " ".join(txt.split())[:70]
    return tps, ttft, watts, jkj, n, apercu, energie


# --------------------------------------------------------------------------
# boucle
# --------------------------------------------------------------------------
def main():
    global MESURES
    ap = argparse.ArgumentParser()
    ap.add_argument("--moteurs", default="acvram,llamacpp,vllm,tabby")
    ap.add_argument("--modeles", default="", help="motif (regex) sur le nom du modèle")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tous", action="store_true", help="aussi les modèles servis par un seul moteur")
    ap.add_argument("--sortie", default=time.strftime("comparatif-%Y%m%d.tsv"))
    ap.add_argument("--passages", type=int, default=MESURES,
                    help="passages par couple (3 par defaut ; 5 pour la serie "
                         "determinisme, qui distingue un passage froid d'une "
                         "bimodalite)")
    ap.add_argument("--forcer-exil", action="store_true",
                    help="charge meme si les poids ne tiennent pas en RAM hote "
                         "(le 8 septembre 2026, ce cas a fait redemarrer la "
                         "machine et perdu les mesures en cours)")
    ap.add_argument("--simuler", action="store_true")
    a = ap.parse_args()
    MESURES = a.passages
    global FORCER_EXIL
    FORCER_EXIL = a.forcer_exil
    moteurs = a.moteurs.split(",")

    table = parc()
    couples = []
    for nom in sorted(table):
        if a.modeles and not re.search(a.modeles, nom, re.I):
            continue
        dispo = {m: v for m, v in table[nom].items() if m in moteurs}
        if len(dispo) < (1 if a.tous else 2):
            continue
        for m, (alias, d, ctx) in dispo.items():
            couples.append((nom, m, alias, d, min(ctx, a.ctx)))
    log(f"{len(couples)} couples sur {len({c[0] for c in couples})} modèles ; sortie {a.sortie}")
    if a.simuler:
        for nom, m, alias, d, ctx in couples:
            print(f"  {nom[:48]:48s} {m:9s} {alias[:34]:34s} ctx {ctx}")

    # Le contexte effectif est `min(ctx du parc, --ctx)` : un moteur dont le
    # modele est converti a 4096 sera plafonne la, pendant que l'autre reste a
    # 8192. La colonne `ctx` le publie depuis toujours, mais publier n'est pas
    # avertir — et deux debits pris a des contextes differents ne se comparent
    # pas. 9/09/2026 : le cas a failli passer sur Qwen2.5-Coder-14B.
    par_modele = {}
    for nom, m, _alias, _d, ctx in couples:
        par_modele.setdefault(nom, {})[m] = ctx
    for nom, ctxs in sorted(par_modele.items()):
        if len(set(ctxs.values())) > 1:
            log(f"  ATTENTION {nom[:40]} : contextes differents selon le "
                f"moteur ({', '.join(f'{m}={c}' for m, c in sorted(ctxs.items()))})"
                f" — les debits ne sont PAS comparables")
            # sys.exit et non return : un appelant doit pouvoir distinguer
            # « rien a mesurer » d'un refus. Un return silencieux rendait le
            # code 0 avec un TSV vide — un succes qui ne mesure rien.
            sys.exit(2)

    faits = set()
    reussies = echouees = invalides = 0
    if os.path.exists(a.sortie):
        entete = None
        for l in open(a.sortie):
            c = l.rstrip("\n").split("\t")
            if l.startswith("modele\t"):
                entete = c
                continue
            if len(c) < 2:
                continue
            # Une reprise ne doit sauter que ce qui a ABOUTI. Une ligne en
            # erreur ou invalidee etait comptee comme faite : la mesure ne
            # repartait jamais, et le TSV gardait sa ligne sans valeur.
            def col(nom):
                if not entete or nom not in entete:
                    return ""
                i = entete.index(nom)
                return c[i] if i < len(c) else ""
            if col("etat") != "ok":
                continue
            if col("invalidations") not in ("", "aucune", "?"):
                continue
            faits.add((c[0], c[1]))
    else:
        with open(a.sortie, "w") as f:
            f.write("modele\tmoteur\talias\tctx\tt_s\tttft_ms\tW\tj_kJ\tjetons\tchargement_s\tetat\tapercu\t"
                    "J\tJ_net\tW_repos\tj_kJ_net\tplafond_W\thorloge_min\thorloge_max\ttemp_max\t"
                    "W_passages\tjkj_passages\tdispersion_W_pct\ttemp_avant\ttemp_apres\tttft_froid_ms\tbridages\tdispersion_pct\tt_s_min\tt_s_max\tt_s_passages\tttft_passages\t"
                    "jetons_moteur\tjetons_flux\tjetons_source\tbinaire\tctx_servi\t"
                    "empreintes\ttextes_identiques\t"
                    "invalidations\n")

    # par moteur, pour ne pas relancer un serveur lourd à chaque modèle
    for m in moteurs:
        for nom, mm, alias, d, ctx in couples:
            if mm != m or (nom, m) in faits:
                continue
            log(f"{m:9s} {nom}")
            print(f"           cible : {d}", flush=True)
            etat, tps, ttft, watts, jkj, n, charge, apercu = "ok", 0, 0, 0, 0, 0, 0, ""
            energie = {}
            try:
                charge = demarrer(m, d, ctx)
                tps, ttft, watts, jkj, n, apercu, energie = mesurer(m)
                log(f"           {tps:.1f} t/s (médiane ; {energie['t_s_min']}-{energie['t_s_max']}), "
                    f"TTFT {ttft*1000:.0f} ms, {watts:.0f} W, "
                    f"{jkj:.0f} j/kJ brut, {energie['jkj_net']:.0f} net, chargé en {charge:.0f} s")
                if energie["invalidations"] != "aucune":
                    log(f"           MESURE INVALIDE : {energie['invalidations']}")
            except Exception as exc:                       # noqa: BLE001
                etat = f"erreur: {str(exc)[:120]}"
                log(f"           ÉCHEC {etat}")
            if etat.startswith("erreur"):
                echouees += 1
            elif energie.get("invalidations", "aucune") != "aucune":
                # « valide » comptait « n'a pas leve d'exception ». Une mesure
                # invalidee par le bridage etait annoncee valide, et la garde
                # reussies==0 ne pouvait pas la voir : elle protegeait du cas
                # absent, pas du cas faux.
                invalides += 1
            else:
                reussies += 1
            with open(a.sortie, "a") as f:
                def v(cle, defaut=""):
                    return energie.get(cle, defaut)
                f.write(f"{nom}\t{m}\t{alias}\t{ctx}\t{tps:.1f}\t{ttft*1000:.0f}\t{watts:.0f}\t"
                        f"{jkj:.0f}\t{n}\t{charge:.0f}\t{etat}\t{apercu}\t"
                        f"{v('J', 0)}\t{v('J_net', 0)}\t{v('W_repos', 0)}\t{v('jkj_net', 0)}\t"
                        f"{v('plafond_W', 0)}\t{v('horloge_min', -1)}\t{v('horloge_max', -1)}\t"
                        f"{v('temp_max', -1)}\t{v('W_passages', '?')}\t"
                        f"{v('jkj_passages', '?')}\t{v('dispersion_W_pct', -1)}\t"
                        f"{v('temp_avant', -1)}\t{v('temp_apres', -1)}\t"
                        f"{v('ttft_froid_ms', -1)}\t"
                        f"{v('bridages', '?')}\t{v('dispersion_pct', 0)}\t"
                        f"{v('t_s_min', 0)}\t{v('t_s_max', 0)}\t{v('t_s_passages', '?')}\t"
                        f"{v('ttft_passages', '?')}\t"
                        f"{v('jetons_moteur', '?')}\t{v('jetons_flux', '?')}\t"
                        f"{v('jetons_source', '?')}\t{v('binaire', '?')}\t{v('ctx_servi', '?')}\t"
                        f"{v('empreintes', '?')}\t"
                        f"{v('textes_identiques', '?')}\t{v('invalidations', '?')}\n")
        arreter(m)
    log(f"TERMINÉ — {reussies} mesure(s) valide(s), "
        f"{invalides} invalidée(s), {echouees} échec(s)")
    # Trois campagnes de suite ont fini en code 0 sans une seule mesure, le
    # 9/09/2026 : garde inconditionnelle, TSV vide, puis lignes a t_s 0.0. Le
    # code de sortie ne portait aucune information et il a cesse d'etre lu.
    # Une campagne qui ne mesure rien doit echouer, sinon un enchainement la
    # prend pour un succes — et des lignes qui existent sans rien valoir sont
    # pires qu'un fichier vide.
    if reussies == 0:
        log("  AUCUNE mesure valide : la campagne n'a rien produit")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Protocole 2.5 (qr.md § 2.5, demande Maîtresse 22/09) : alternance stricte
# entre moteurs, J net = ∫(P − P_repos) dt / jetons (P_repos pris UNE FOIS,
# 5 s avant la fenêtre, pas par sous-passage), ≥ 6 fenêtres ≥ 20 s, rejet si
# sd (pstdev/moyenne des débits, MÊME définition que `mesurer()` ci-dessus,
# seuil resserré à 10 %) > 10 %, throttle actif (`Energie.bridages`), ou
# écart de charge étrangère hors ligne — et par PAIRE (une fenêtre par
# moteur, dans l'ordre d'alternance) : écart d'horloge médiane ≤ 3 %, sinon
# la PAIRE entière est invalidée (les deux fenêtres, pas une seule).
#
# La logique d'agrégation/validation est SÉPARÉE de l'E/S GPU (`generer`,
# `repos`) : `_agreger_fenetre`/`_valider_*` sont des fonctions pures, testées
# à sec sur des traces synthétiques (`tests/test_banc_4moteurs_protocole25.py`
# côté dépôt principal, ou directement ici en scratchpad selon où ce fichier
# vit). L'orchestration GPU (`mesurer_fenetre_protocole25`/`comparer_alternee`)
# les appelle mais n'a pas de logique propre à tester hors carte.
#
# « charge étrangère » : substitution CONFIRMÉE (Maîtresse 22/09) — le « 5 %
# P_max » venait d'un avis extérieur (qr.md, hors dépôt, ~/Bureau/Vibe, pas
# une note du groupe) ; le critère qui compte est REGLES §2, déjà instrumenté
# ici : charge CPU par processus (pic load1 > nproc/2) ET puissance < 395 W
# (`Energie.invalidations()`/`plafond`) — `_valider_charge` reste tel quel.

def _sd_relatif_pct(debits: list) -> float:
    """Même définition que `mesurer()` (pstdev/moyenne, en %) — pas
    `_dispersion_rel` (étendue/médiane) : la protocole 2.5 demande un sd, pas
    une étendue."""
    if len(debits) < 2:
        return 0.0
    moy = statistics.mean(debits)
    return 100 * statistics.pstdev(debits) / moy if moy else 0.0


def _agreger_fenetre(sous_passages: list) -> dict:
    """Fusionne N sous-passages d'un même moteur (chacun le retour d'un
    `generer()`, additif : J et durée s'additionnent, horloges/températures se
    concatènent) en UNE fenêtre ≥ 20 s. Fonction pure — pas d'accès carte.

    ``sous_passages`` : liste de dicts {n, duree, joules, horloges,
    temperatures, bridages, debit} — la forme que rend `generer()` + son
    `Energie` (voir `mesurer_fenetre_protocole25`)."""
    n_total = sum(p["n"] for p in sous_passages)
    duree_totale = sum(p["duree"] for p in sous_passages)
    joules_totales = sum(p["joules"] for p in sous_passages)
    debits = [p["debit"] for p in sous_passages]
    horloges = [h for p in sous_passages for h in p.get("horloges", []) if h and h >= 0]
    temperatures = [t for p in sous_passages for t in p.get("temperatures", []) if t and t >= 0]
    throttle = any(p.get("bridages") for p in sous_passages)
    return {
        "n": n_total, "duree": round(duree_totale, 2), "joules": round(joules_totales, 1),
        "t_s": round(n_total / duree_totale, 1) if duree_totale else 0.0,
        "sd_pct": round(_sd_relatif_pct(debits), 2),
        "horloge_med": statistics.median(horloges) if horloges else -1,
        "temp_max": max(temperatures) if temperatures else -1,
        "throttle": throttle,
    }


def _joules_net(agg: dict, watts_repos: float) -> float:
    """J net = joules_mesures − P_repos × durée (P_repos pris UNE fois, AVANT
    la fenêtre entière — pas par sous-passage : qr.md § 2.5)."""
    return round(max(agg["joules"] - watts_repos * agg["duree"], 0.0), 1)


def _valider_fenetre(agg: dict) -> list:
    """sd > 10 % ou throttle actif → la fenêtre seule est rejetée."""
    raisons = []
    if agg["sd_pct"] > 10.0:
        raisons.append(f"sd {agg['sd_pct']:.1f} % > 10 %")
    if agg["throttle"]:
        raisons.append("throttle actif")
    return raisons


def _valider_charge(charge_pct) -> list:
    """Voir le commentaire de tête : critère REGLES §2 (load1 pic / nproc, en
    %), confirmé par la Maîtresse — le « % P_max » d'un avis extérieur n'est
    pas retenu. ``None`` = non mesuré, ne rejette pas (silencieux, pas un
    TENU)."""
    if charge_pct is None:
        return []
    return [] if charge_pct <= 5.0 else [f"charge étrangère {charge_pct:.1f} % > 5 %"]


def _valider_paire(horloge_a: float, horloge_b: float) -> list:
    """Écart d'horloge médiane entre les DEUX fenêtres d'une paire (un
    moteur, puis l'autre, consécutives dans l'alternance) : > 3 % invalide LA
    PAIRE entière, pas une fenêtre seule — l'écart de vitesse ne jugerait
    alors rien (même règle que `chaine-sampler-abba.sh`, REGLES § 3)."""
    if horloge_a is None or horloge_b is None or horloge_a <= 0 or horloge_b <= 0:
        return ["horloge indisponible sur au moins une fenêtre de la paire"]
    ecart = 100 * abs(horloge_a - horloge_b) / horloge_a
    return [] if ecart <= 3.0 else [f"écart horloge {ecart:.1f} % > 3 %"]


def mesurer_fenetre_protocole25(moteur, fenetre_s=20.0, repos_avant_s=5.0, charge_pct=None):
    """Une fenêtre ≥ ``fenetre_s`` pour ``moteur`` (accès carte : `generer`,
    `repos`) : `repos()` UNE fois avant, puis `generer()` en boucle jusqu'à
    la durée demandée (chaque appel garde son propre `Energie` — additif,
    jamais imbriqué), agrégé par `_agreger_fenetre` (pure, testée à part)."""
    base = repos(secondes=repos_avant_s)
    sous_passages = []
    t0 = time.time()
    while time.time() - t0 < fenetre_s:
        n, ttft, dt, e, texte, morceaux, source = generer(moteur)
        tps = (n - 1) / dt if n > 1 else 0.0
        sous_passages.append({
            "n": n, "duree": e.duree, "joules": e.joules,
            "horloges": list(e.horloges), "temperatures": list(e.temperatures),
            "bridages": set(e.bridages), "debit": tps,
        })
    agg = _agreger_fenetre(sous_passages)
    agg["joules_net"] = _joules_net(agg, base.moyenne)
    agg["charge_pct"] = charge_pct
    agg["raisons"] = _valider_fenetre(agg) + _valider_charge(charge_pct)
    agg["moteur"] = moteur
    return agg


def comparer_alternee(moteurs, sha, n_fenetres=6, fenetre_s=20.0, sortie_tsv="/dev/stdout"):
    """Alternance stricte `moteurs[0], moteurs[1], moteurs[0], moteurs[1], …`
    (round-robin sur la liste, 2 moteurs typiquement), N ≥ 6 fenêtres, une
    paire = deux fenêtres consécutives de moteurs différents. Écrit le TSV
    avec l'en-tête EXACT demandé : moteur/sha/horloge/temp/charge/J/t/s/date/
    durée, plus deux colonnes de diagnostic en fin (valide/raisons) — sans
    quoi une fenêtre rejetée disparaîtrait sans laisser de trace."""
    assert n_fenetres >= 6, "protocole 2.5 : au moins 6 fenêtres"
    lignes = []
    fenetres = []
    for i in range(n_fenetres):
        moteur = moteurs[i % len(moteurs)]
        agg = mesurer_fenetre_protocole25(moteur, fenetre_s=fenetre_s)
        fenetres.append(agg)
    # validation par paire (i, i+1) sur l'horloge médiane
    for i in range(0, n_fenetres - 1, 2):
        a, b = fenetres[i], fenetres[i + 1]
        raisons_paire = _valider_paire(a["horloge_med"], b["horloge_med"])
        for f in (a, b):
            f["raisons"] = f["raisons"] + raisons_paire
    with open(sortie_tsv, "w") as fh:
        fh.write("moteur\tsha\thorloge\ttemp\tcharge\tJ\tt_s\tdate\tduree\tvalide\traisons\n")
        for f in fenetres:
            valide = not f["raisons"]
            charge = f["charge_pct"] if f["charge_pct"] is not None else "?"
            fh.write(f"{f['moteur']}\t{sha}\t{f['horloge_med']}\t{f['temp_max']}\t"
                    f"{charge}\t{f['joules_net']}\t{f['t_s']}\t"
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{f['duree']}\t"
                    f"{'oui' if valide else 'non'}\t{' ; '.join(f['raisons']) or 'aucune'}\n")
    return fenetres


if __name__ == "__main__":
    main()
