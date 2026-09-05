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
import argparse, json, os, re, subprocess, sys, threading, time, tomllib, urllib.request

KIMI = os.path.expanduser("~/.kimi-code")
BIN = os.path.expanduser("~/.local/bin")
SECRETS = os.path.expanduser("~/.config/ia-secrets.env")
TABBY_DIR = "/mnt/AI_GENERATOR/TabbyAPI"
VLLM_DIR = "/mnt/AI_GENERATOR/vLLM"
PORTS = {"acvram": 8090, "llamacpp": 8080, "vllm": 8000, "tabby": 5000}
PORT_INTERDIT = 8081
PROMPT = ("Explique en détail, en français et en plusieurs paragraphes, comment "
          "fonctionne la mémoire virtuelle d'un système d'exploitation moderne : "
          "pages, tables de pages, TLB, défauts de page, et ce qui se passe quand "
          "la mémoire physique est pleine.")
MAX_TOKENS = 200
MESURES = 2


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


def attendre(moteur, secondes):
    t0 = time.time()
    while time.time() - t0 < secondes:
        if pret(moteur):
            return time.time() - t0
        time.sleep(2)
    raise RuntimeError(f"{moteur} n'a pas démarré en {secondes} s")


def demarrer(moteur, dossier, ctx):
    """Lance le moteur et rend le temps de chargement, lanceur compris (les
    lanceurs attendent eux-mêmes le port avant de rendre la main)."""
    t0 = time.time()
    _demarrer(moteur, dossier, ctx)
    return time.time() - t0


def _demarrer(moteur, dossier, ctx):
    arreter_tout_sauf(moteur)
    env = dict(os.environ)
    if moteur == "acvram":
        arreter("acvram")
        env["CTX"] = str(ctx)
        subprocess.run([os.path.join(BIN, "acvram-serveur"), dossier], env=env,
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
class Watt:
    """Échantillonne la puissance de la carte 0 pendant la génération."""
    def __init__(self, gpu=0):
        self.gpu, self.v, self.stop = gpu, [], False

    def __enter__(self):
        def boucle():
            while not self.stop:
                try:
                    o = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=power.draw",
                                        "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=2).stdout
                    self.v.append(float(o.strip()))
                except Exception:
                    pass
                time.sleep(0.2)
        self.t = threading.Thread(target=boucle, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join(timeout=3)

    @property
    def moyenne(self):
        return sum(self.v) / len(self.v) if self.v else 0.0


def modele_servi(moteur):
    try:
        d = json.load(http(PORTS[moteur], "/v1/models", CLES[moteur], delai=5))
        return d["data"][0]["id"]
    except Exception:
        return None


def generer(moteur):
    """Une génération en streaming : (jetons, ttft_s, duree_decodage_s, W)."""
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
    with Watt() as w:
        t0 = time.time()
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
                        maintenant = time.time()
                        if premier is None:
                            premier = maintenant
                        dernier = maintenant
                        n += 1
    if premier is None:
        raise RuntimeError("aucun jeton reçu")
    morceaux = n
    if usage and usage.get("completion_tokens"):
        n = usage["completion_tokens"]
    if morceaux < 2 or dernier - premier < 1e-3:
        # tout est arrivé d'un bloc : pas de flux jeton par jeton, le débit
        # ne peut se mesurer que sur la durée totale, premier jeton compris
        raise RuntimeError(f"pas de flux jeton par jeton ({morceaux} morceau(x) "
                           f"pour {n} jetons)")
    return n, premier - t0, dernier - premier, w.moyenne


def mesurer(moteur):
    meilleur = None
    for _ in range(MESURES):
        n, ttft, dt, watts = generer(moteur)
        tps = (n - 1) / dt if n > 1 else 0.0
        if meilleur is None or tps > meilleur[0]:
            meilleur = (tps, ttft, watts, n)
    tps, ttft, watts, n = meilleur
    jkj = tps / watts * 1000 if watts else 0.0
    return tps, ttft, watts, jkj, n


# --------------------------------------------------------------------------
# boucle
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moteurs", default="acvram,llamacpp,vllm,tabby")
    ap.add_argument("--modeles", default="", help="motif (regex) sur le nom du modèle")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tous", action="store_true", help="aussi les modèles servis par un seul moteur")
    ap.add_argument("--sortie", default=time.strftime("comparatif-%Y%m%d.tsv"))
    ap.add_argument("--simuler", action="store_true")
    a = ap.parse_args()
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
        return

    faits = set()
    if os.path.exists(a.sortie):
        for l in open(a.sortie):
            c = l.rstrip("\n").split("\t")
            if len(c) >= 2 and not l.startswith("modele\t"):
                faits.add((c[0], c[1]))
    else:
        with open(a.sortie, "w") as f:
            f.write("modele\tmoteur\talias\tctx\tt_s\tttft_ms\tW\tj_kJ\tjetons\tchargement_s\tetat\n")

    # par moteur, pour ne pas relancer un serveur lourd à chaque modèle
    for m in moteurs:
        for nom, mm, alias, d, ctx in couples:
            if mm != m or (nom, m) in faits:
                continue
            log(f"{m:9s} {nom}")
            print(f"           cible : {d}", flush=True)
            etat, tps, ttft, watts, jkj, n, charge = "ok", 0, 0, 0, 0, 0, 0
            try:
                charge = demarrer(m, d, ctx)
                tps, ttft, watts, jkj, n = mesurer(m)
                log(f"           {tps:.1f} t/s, TTFT {ttft*1000:.0f} ms, {watts:.0f} W, "
                    f"{jkj:.0f} j/kJ, chargé en {charge:.0f} s")
            except Exception as exc:                       # noqa: BLE001
                etat = f"erreur: {str(exc)[:120]}"
                log(f"           ÉCHEC {etat}")
            with open(a.sortie, "a") as f:
                f.write(f"{nom}\t{m}\t{alias}\t{ctx}\t{tps:.1f}\t{ttft*1000:.0f}\t{watts:.0f}\t"
                        f"{jkj:.0f}\t{n}\t{charge:.0f}\t{etat}\n")
        arreter(m)
    log("TERMINÉ")


if __name__ == "__main__":
    main()
