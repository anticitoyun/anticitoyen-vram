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
import argparse, hashlib, json, os, re, statistics, subprocess, sys, threading, time, tomllib, urllib.request

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
# L'ancienne classe Watt est retiree : elle lancait un sous-processus
# nvidia-smi toutes les 200 ms — sur la machine qu'elle mesurait — pour
# moyenner des echantillons de puissance sur la seule carte 0. Trois defauts,
# tous corriges dans outils/energie.py : le cout de la mesure, la moyenne qui
# n'est pas une energie, et une carte lue sur deux alors qu'un modele exile
# travaille sur les deux. Mesure du 8 septembre 2026 a l'appui.


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
    if usage and usage.get("completion_tokens"):
        n = usage["completion_tokens"]
    if morceaux < 2 or dernier - premier < 1e-3:
        # tout est arrivé d'un bloc : pas de flux jeton par jeton, le débit
        # ne peut se mesurer que sur la durée totale, premier jeton compris
        raise RuntimeError(f"pas de flux jeton par jeton ({morceaux} morceau(x) "
                           f"pour {n} jetons)")
    return n, premier - t0, dernier - premier, w, "".join(texte)


def mesurer(moteur):
    """Débit MÉDIAN et énergie du MÊME passage, puis la ligne de base après lui.

    L'énergie ne se moyenne pas entre passages : elle vient du passage dont on
    publie le débit, sinon les deux colonnes décrivent deux exécutions
    différentes. Et la ligne de base se prend APRÈS : prise une fois au début,
    elle dérive, et la dérive se retrouve attribuée au moteur mesuré — c'est
    ce qui avait fait croire à une décroissance de coût le 7 septembre 2026.
    """
    passages = []
    for _ in range(MESURES):
        n, ttft, dt, e, txt = generer(moteur)
        tps = (n - 1) / dt if n > 1 else 0.0
        passages.append((tps, ttft, e, n, txt, dt))
    debits = [p[0] for p in passages]

    # Le passage PUBLIE est celui de debit median, plus celui de debit
    # maximal. « Le meilleur des trois » est un estimateur biaise : il
    # favorise le moteur le plus bruyant. Mesure du 8 septembre 2026 :
    # sur un couple a 21,3 % de dispersion contre 5,2 en face, le meilleur
    # donnait la victoire au plus instable. Le maximum reste publie a part,
    # il n'est simplement plus ce qu'on compare.
    ordonnes = sorted(passages, key=lambda x: x[0])
    tps, ttft, e, n, txt, dt = ordonnes[len(ordonnes) // 2]
    etendue = (min(debits), max(debits))

    # Empreinte du texte de CHAQUE passage. A temperature zero, le meme
    # prompt doit rendre le meme texte : si les empreintes different, le
    # chemin n'est pas deterministe, et c'est un defaut a part entiere —
    # plus important que le debit qu'on etait venu mesurer. Si elles sont
    # identiques, une dispersion de debit ne peut pas venir du texte, et il
    # faut la chercher ailleurs (passage froid, cache, ordonnancement).
    empreintes = [hashlib.sha256(p[4].encode()).hexdigest()[:8] for p in passages]
    textes_identiques = len(set(empreintes)) == 1
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
        "t_s_min": round(etendue[0], 1), "t_s_max": round(etendue[1], 1),
        "empreintes": ",".join(empreintes),
        "textes_identiques": "oui" if textes_identiques else "NON",
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
    ap.add_argument("--simuler", action="store_true")
    a = ap.parse_args()
    MESURES = a.passages
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
            f.write("modele\tmoteur\talias\tctx\tt_s\tttft_ms\tW\tj_kJ\tjetons\tchargement_s\tetat\tapercu\t"
                    "J\tJ_net\tW_repos\tj_kJ_net\tplafond_W\thorloge_min\thorloge_max\ttemp_max\t"
                    "bridages\tdispersion_pct\tt_s_min\tt_s_max\tempreintes\ttextes_identiques\t"
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
            with open(a.sortie, "a") as f:
                def v(cle, defaut=""):
                    return energie.get(cle, defaut)
                f.write(f"{nom}\t{m}\t{alias}\t{ctx}\t{tps:.1f}\t{ttft*1000:.0f}\t{watts:.0f}\t"
                        f"{jkj:.0f}\t{n}\t{charge:.0f}\t{etat}\t{apercu}\t"
                        f"{v('J', 0)}\t{v('J_net', 0)}\t{v('W_repos', 0)}\t{v('jkj_net', 0)}\t"
                        f"{v('plafond_W', 0)}\t{v('horloge_min', -1)}\t{v('horloge_max', -1)}\t"
                        f"{v('temp_max', -1)}\t{v('bridages', '?')}\t{v('dispersion_pct', 0)}\t"
                        f"{v('t_s_min', 0)}\t{v('t_s_max', 0)}\t{v('empreintes', '?')}\t"
                        f"{v('textes_identiques', '?')}\t{v('invalidations', '?')}\n")
        arreter(m)
    log("TERMINÉ")


if __name__ == "__main__":
    main()
