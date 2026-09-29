#!/usr/bin/env python3
"""Test RÉEL des menus kimi-modele et claude-modele sur TOUS les alias de ~/.kimi-code/config.toml (bd edz, 28/09).

Pour chaque alias, dans l'ordre : (1) préchargement par le lanceur de son moteur ; (2) /v1/models sert l'id attendu ;
(3) une vraie complétion courte (/v1/chat/completions, contenu ou raisonnement non vide) ; (4) `kimi-modele <alias> -p`
et (5) `claude-modele <alias> -p` sur une question d'une ligne (rc 0 et réponse non vide) ; (6) arrêt du serveur par
SON PID (retrouvé par son port, jamais par motif — REGLES § 2) et VRAM de ce PID rendue.

Une ligne par alias dans le TSV de résultats (reprise : un alias déjà présent n'est pas rejoué, sauf --refaire) ;
journal par alias dans le dossier de journaux. Drapeau de pause, lu ENTRE deux alias seulement. Sans --pour-de-vrai :
liste ce qui serait fait, rien n'est lancé. Les lanceurs de service prennent eux-mêmes le verrou carte.sh (service).

Usage : test-menus-reels.py [--pour-de-vrai] [--alias MOTIF] [--moteurs acvram,llamacpp,...] [--refaire] [--max N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parc" / "lib"))
from acvram_parc import charger  # noqa: E402

ETAT = Path(os.environ.get("TMR_ETAT", Path.home() / ".cache" / "acvram" / "menus-reels"))
RESULTATS = Path(os.environ.get("TMR_RESULTATS", Path.home() / "TSV" / "menus-reels.tsv"))
PAUSE = ETAT / "pause"
# journal du serveur (acvram-serveur, en ajout) : sa part de CE lancement est copiée dans le journal d alias quand
# le préchargement échoue — le 28/09 la seule cause lisible était un avertissement, et /tmp s est vidé au redémarrage
LOG_SERVEUR = Path(os.environ.get("ACVRAM_SERVEUR_LOG", "/tmp/acvram-serveur.log"))
# --rapide (chef 29/09, 164 alias à passer) : la preuve se réduit au modèle servi — /v1/models conforme à l'alias ET
# réponse non vide ; claude/kimi exigent ≥ 15 k de contexte et ne sont pas joués (passe réelle sur échantillon ensuite).
CTX_RAPIDE, JETONS_RAPIDE, PLAFOND_RAPIDE = 4096, 16, 300
QUESTION = "Réponds en un seul mot : quelle est la capitale de la France ?"
# rc 0 et réponse non vide ne jugent rien : le 28/09, GLM-4.7-Flash rendait « ``| » en boucle (kimi rc 0 compté OK) et
# claude enchaînait les suites jusqu'au « Prompt is too long ». La réponse doit contenir ATTENDU (casse ignorée) ;
# le programme juge, personne ne relit la sortie (REGLES § 6).
ATTENDU = "paris"
COLONNES = ["date", "alias", "moteur", "verdict", "etape", "cause", "prechargement_s", "models", "completion",
            "kimi_rc", "claude_rc", "arret", "detail"]
ORDRE = ["acvram", "llamacpp", "vllm", "rapide", "yals", "tabby", "jan"]
# claude parle l'API Anthropic (/v1/messages) : TabbyAPI, YALS et Jan ne l'exposent pas — refus par construction du
# lanceur (claude-modele:163-165), noté « n/a », pas une panne.
SANS_CLAUDE = {"tabby", "yals", "jan"}
# moteurs qui se chargent eux-mêmes à la requête (pas de lanceur de préchargement scriptable) : kimi seul est joué.
AUTO_CHARGES = {"tabby", "yals", "jan"}


def secrets(p) -> dict[str, str]:
    d = {}
    if p.secrets.exists():
        for l in p.secrets.read_text(encoding="utf-8").splitlines():
            if "=" in l and not l.lstrip().startswith("#"):
                k, v = l.split("=", 1)
                d[k.strip()] = v.strip().strip('"')
    return d


def lire_tsv(chemin: Path) -> dict[str, list[str]]:
    d = {}
    if chemin.exists():
        for l in chemin.read_text(encoding="utf-8").splitlines():
            if l and not l.startswith("#"):
                c = l.split("\t")
                d[c[0]] = c[1:]
    return d


def alias_du_menu(p) -> list[tuple[str, str, dict]]:
    m = tomllib.loads(p.config_kimi.read_text(encoding="utf-8")).get("models", {})
    rang = {k: i for i, k in enumerate(ORDRE)}
    return sorted(((a, v.get("provider", ""), v) for a, v in m.items() if isinstance(v, dict)),
                  key=lambda t: (rang.get(t[1], 99), list(m).index(t[0])))


def lanceur(p, alias: str, moteur: str, rapide: bool = False,
            ctx_client: int = 29096) -> tuple[list[str], dict[str, str], str | None] | str:
    """(argv, env, id attendu sur /v1/models) — ou la cause pour laquelle rien n'est lançable. Même résolution que
    claude-modele/kimi-modele (colonnes des TSV), mêmes lanceurs ; CTX_CLIENT_MIN : celui de claude (29 096,
    claude-modele:151) — kimi-modele pose le sien (34 816) et relance si besoin ; un alias trop court pour kimi garde
    ainsi le résultat de claude au lieu de tomber au préchargement."""
    b = p.bin
    if moteur == "acvram":
        if rapide:   # --rapide : contexte court, sans graphes CUDA (chauffe et capture courtes), pas de CTX_CLIENT_MIN
            return [str(b / "acvram-serveur"), alias, str(CTX_RAPIDE)], {"GRAPHES": "1"}, alias
        return [str(b / "acvram-serveur"), alias], {"CTX_CLIENT_MIN": str(ctx_client)}, alias
    if moteur in ("llamacpp", "vllm"):
        t = lire_tsv(p.tsv("gguf" if moteur == "llamacpp" else "vllm")).get(alias)
        if not t:
            return f"alias absent de {p.tsv('gguf' if moteur == 'llamacpp' else 'vllm').name}"
        dossier, ctx = t[0], (str(CTX_RAPIDE) if rapide else (t[1] if len(t) > 1 else "0"))
        if not Path(dossier).exists():
            return f"chemin absent : {dossier}"
        if moteur == "vllm":
            return [str(b / "vllm-serveur"), dossier, ctx], {}, Path(dossier).name
        env = {"GABARIT": t[2]} if len(t) > 2 and t[2] and t[2] != "ABSENT" else {}
        return [str(b / "llamacpp-serveur"), dossier, ctx], env, None     # id = nom du .gguf servi (voir id_conforme)
    if moteur == "rapide":
        return [str(b / "llamacpp-appoint")], {}, None
    return f"moteur {moteur} sans lanceur de préchargement"


def get_json(url: str, cle: str, corps: dict | None = None, delai: float = 10) -> dict:
    req = urllib.request.Request(url, data=None if corps is None else json.dumps(corps).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {cle}"} if cle else {})})
    with urllib.request.urlopen(req, timeout=delai) as r:
        return json.load(r)


def id_conforme(servi: str | None, attendu: str | None, entree: dict) -> bool:
    if not servi:
        return False
    if attendu is not None:
        return servi == attendu
    # llama.cpp sert le nom du fichier .gguf ; config.toml porte ce nom en `model` (convention du poste)
    return Path(servi).stem == Path(str(entree.get("model", ""))).stem or servi == entree.get("model")


def pid_ecoute(port: int) -> int | None:
    """PID qui écoute PORT (ss), comme les lanceurs le font pour retrouver leur propre serveur ; jamais un motif."""
    try:
        r = subprocess.run(["ss", "-tlnpH"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    for l in r.stdout.splitlines():
        if re.search(rf"[:.]{port}\s", l):
            m = re.search(r"pid=(\d+)", l)
            if m:
                return int(m.group(1))
    return None


def vram_de(pid: int) -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=10)
    except Exception:
        return False
    return str(pid) in r.stdout.split()


def arreter(port: int, pid: int | None) -> str:
    """SIGTERM puis SIGKILL au PID qui écoute le port ; « ok » si le PID est mort ET n'a plus de VRAM."""
    pid = pid or pid_ecoute(port)
    if pid is None:
        return "ok (rien n'écoutait)"
    for sig, attente in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        for _ in range(attente * 2):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.5)
        else:
            continue
        break
    for _ in range(20):
        if not vram_de(pid) and pid_ecoute(port) != pid:
            return "ok"
        time.sleep(0.5)
    return f"PID {pid} toujours là ou VRAM non rendue"


def queue(texte: str, n: int = 160) -> str:
    l = [x for x in (texte or "").strip().splitlines() if x.strip()]
    return (l[-1] if l else "")[:n].replace("\t", " ")


def client(p, nom: str, alias: str, journal: Path, delai: int) -> tuple[str, str]:
    """(rc, cause) de `<nom> <alias> -p QUESTION` ; cause vide si rc 0 et réponse non vide."""
    try:
        r = subprocess.run([str(p.bin / nom), alias, "-p", QUESTION], capture_output=True, text=True, timeout=delai,
                           stdin=subprocess.DEVNULL, env={k: v for k, v in os.environ.items() if k != "CTX_CLIENT_MIN"})
    except subprocess.TimeoutExpired:
        return "délai", f"{nom} : aucune réponse en {delai} s"
    with journal.open("a", encoding="utf-8") as f:
        f.write(f"--- {nom} rc={r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}\n")
    if r.returncode != 0:
        return str(r.returncode), f"{nom} rc {r.returncode} : {queue(r.stderr) or queue(r.stdout)}"
    if not r.stdout.strip():
        return "0", f"{nom} : réponse vide"
    if ATTENDU not in r.stdout.casefold():
        return "0", f"{nom} : réponse sans « Paris » ({len(r.stdout)} o : fausse ou dégénérée)"
    return "0", ""


def tester(p, cles: dict, alias: str, moteur: str, entree: dict, a) -> dict:
    ligne = {c: "" for c in COLONNES}
    ligne.update(date=datetime.now().strftime("%d/%m %H:%M"), alias=alias, moteur=moteur)
    journal = ETAT / "journaux" / f"{alias}.log"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(f"{datetime.now()} {alias} ({moteur})\n", encoding="utf-8")

    def panne(etape, cause):
        ligne.update(verdict="PANNE", etape=etape, cause=cause[:300])
        return ligne
    port = p.port(moteur) if moteur in ("acvram", "llamacpp", "vllm", "rapide", "yals", "tabby", "jan") else None
    cle = cles.get({"acvram": "CLE_ACVRAM", "vllm": "CLE_VLLM", "llamacpp": "CLE_LLAMACPP", "rapide": "CLE_RAPIDE",
                    "tabby": "CLE_TABBY", "yals": "CLE_YALS", "jan": "CLE_JAN"}.get(moteur, ""), "")
    pid = None
    try:
        if moteur not in AUTO_CHARGES:
            if port is None:
                return panne("préchargement", f"moteur {moteur} absent de parc.toml (présent = false)")
            lz = lanceur(p, alias, moteur, a.rapide, a.ctx_client)
            if isinstance(lz, str):
                return panne("préchargement", lz)
            argv, env, attendu = lz
            t0 = time.monotonic()
            debut = LOG_SERVEUR.stat().st_size if LOG_SERVEUR.exists() else 0
            try:
                r = subprocess.run(argv, env={**os.environ, "ACVRAM_ATTENTE": str(a.attente), **env},
                                   capture_output=True, text=True, timeout=a.attente + (PLAFOND_RAPIDE if a.rapide else 600),
                                   stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return panne("préchargement", f"délai {a.attente + (PLAFOND_RAPIDE if a.rapide else 600)} s")
            ligne["prechargement_s"] = f"{time.monotonic() - t0:.0f}"
            with journal.open("a", encoding="utf-8") as f:
                f.write(f"--- lanceur rc={r.returncode}\n{r.stdout[-6000:]}\n{r.stderr[-6000:]}\n")
            pid = pid_ecoute(port)
            if r.returncode != 0:
                if LOG_SERVEUR.exists():
                    with LOG_SERVEUR.open("rb") as f:
                        f.seek(debut); neuf = f.read().decode(errors="replace").splitlines()[-60:]
                    with journal.open("a", encoding="utf-8") as f:
                        f.write(f"--- {LOG_SERVEUR} (ce lancement, 60 dernières lignes)\n" + "\n".join(neuf) + "\n")
                return panne("préchargement", f"rc {r.returncode} : {queue(r.stderr) or queue(r.stdout)}")
            try:
                servi = get_json(f"http://127.0.0.1:{port}/v1/models", cle)["data"][0]["id"]
            except Exception as e:  # noqa: BLE001
                return panne("/v1/models", f"{type(e).__name__} : {e}"[:200])
            if not id_conforme(servi, attendu, entree):
                return panne("/v1/models", f"id servi {servi!r}, attendu {attendu or entree.get('model')!r}")
            ligne["models"] = "ok"
            try:
                rep = get_json(f"http://127.0.0.1:{port}/v1/chat/completions", cle,
                               {"model": servi, "messages": [{"role": "user", "content": QUESTION}], "max_tokens": JETONS_RAPIDE if a.rapide else 64,
                                "temperature": 0}, delai=min(a.delai_client, PLAFOND_RAPIDE) if a.rapide else a.delai_client)
                msg = rep["choices"][0]["message"]
                texte = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
            except Exception as e:  # noqa: BLE001
                return panne("complétion", f"{type(e).__name__} : {e}"[:200])
            if not texte:
                return panne("complétion", "réponse vide (contenu et raisonnement)")
            # sans « Paris » : noté, mais les clients sont joués quand même (64 jetons de raisonnement peuvent finir avant)
            ligne["completion"] = "ok" if ATTENDU in texte.casefold() else "sans Paris"
        if a.rapide:
            # 16 jetons ne suffisent pas toujours à écrire « Paris » (raisonnement d'abord) : noté, pas jugé
            ligne.update(claude_rc="rapide", kimi_rc="rapide", verdict="OK", detail="rapide : modèle servi seul")
            return ligne
        # les deux clients sont joués indépendamment : claude d abord (serveur préchargé à son contexte), puis kimi
        # (kimi-modele relance à 34 816 si besoin) ; une panne de l un ne masque pas l autre.
        pannes = [("complétion", "complétion sans « Paris »")] if ligne["completion"] == "sans Paris" else []
        if moteur in SANS_CLAUDE:
            ligne["claude_rc"] = "n/a"
        else:
            ligne["claude_rc"], cause = client(p, "claude-modele", alias, journal, a.delai_client)
            if cause:
                pannes.append(("claude", cause))
        ligne["kimi_rc"], cause = client(p, "kimi-modele", alias, journal, a.delai_client)
        if cause:
            pannes.append(("kimi", cause))
        if pannes:
            return panne("+".join(e for e, _ in pannes), " | ".join(c for _, c in pannes))
        ligne["verdict"] = "OK"
        return ligne
    finally:
        if port is not None and moteur not in AUTO_CHARGES:
            ligne["arret"] = arreter(port, pid)
            if ligne["arret"] != "ok" and not ligne["arret"].startswith("ok"):
                ligne["detail"] = "arrêt : " + ligne["arret"]


def rejuger() -> int:
    """Lignes OK écrites avant le contrôle « Paris » : chaque section client rc 0 du journal est rejugée ; une réponse
    sans ATTENDU passe la ligne en PANNE. TSV réécrit, l'ancien gardé en .avant-rejuger."""
    if not RESULTATS.exists():
        print(f"{RESULTATS} absent : rien à rejuger."); return 0
    lignes, n = RESULTATS.read_text(encoding="utf-8").splitlines(), 0
    for i, l in enumerate(lignes):
        c = l.split("\t")
        if l.startswith("#") or len(c) < len(COLONNES) or c[3] != "OK":
            continue
        journal = ETAT / "journaux" / f"{c[0]}.log"
        t = journal.read_text(encoding="utf-8", errors="replace") if journal.exists() else ""
        pannes = []
        for nom, etape in (("claude-modele", "claude"), ("kimi-modele", "kimi")):
            m = re.search(rf"--- {nom} rc=0\n(.*?)(?=\n--- |\Z)", t, re.S)
            if m and ATTENDU not in m.group(1).casefold():
                pannes.append((etape, f"{nom} : réponse sans « Paris » (rejugé sur le journal)"))
        if pannes:
            c[3:6] = ["PANNE", "+".join(e for e, _ in pannes), " | ".join(x for _, x in pannes)]
            lignes[i], n = "\t".join(c), n + 1
            print(f"PANNE {c[0]} {c[4]}")
    if n:
        RESULTATS.with_name(RESULTATS.name + ".avant-rejuger").write_text(RESULTATS.read_text(encoding="utf-8"),
                                                                           encoding="utf-8")
        RESULTATS.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"{n} ligne(s) OK passée(s) en PANNE.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pour-de-vrai", action="store_true")
    ap.add_argument("--alias", help="motif (regex) sur l'alias")
    ap.add_argument("--moteurs", default="acvram,llamacpp,vllm,rapide",
                    help="liste séparée par des virgules (défaut : les moteurs à lanceur de préchargement ; yals, tabby "
                         "et jan se chargent à la requête, hors verrou de carte : campagne à part)")
    ap.add_argument("--refaire", action="store_true", help="rejoue aussi les alias déjà dans le TSV")
    ap.add_argument("--max", type=int, default=0, help="au plus N alias dans ce passage (0 : tous)")
    ap.add_argument("--attente", type=int, default=1800, help="ACVRAM_ATTENTE des lanceurs (verrou de carte)")
    ap.add_argument("--delai-client", type=int, default=600, help="délai de chaque client (s)")
    ap.add_argument("--liste", type=Path,
                    help="fichier d'alias (1re colonne, un par ligne) : joués dans CET ordre, même déjà dans le TSV")
    ap.add_argument("--ctx-client", type=int, default=29096,
                    help="CTX_CLIENT_MIN du préchargement acvram (34 816 : celui de kimi, évite son rechargement)")
    ap.add_argument("--rapide", action="store_true",
                    help=f"preuve réduite au modèle servi (/v1/models + réponse non vide), contexte {CTX_RAPIDE}, sans "
                         f"graphes, {JETONS_RAPIDE} jetons, plafond {PLAFOND_RAPIDE} s ; pannes rejouées une fois en fin")
    ap.add_argument("--rejuger", action="store_true",
                    help="rejuge les lignes OK du TSV sur leurs journaux (contrôle « Paris »), sans rien lancer")
    a = ap.parse_args()
    if a.rejuger:
        return rejuger()
    p = charger(os.environ.get("ACVRAM_PARC_CONFIG"))
    cles = secrets(p)
    tous = alias_du_menu(p)
    faits = set(lire_tsv(RESULTATS)) if RESULTATS.exists() else set()
    mot = set(a.moteurs.split(",")) if a.moteurs else None
    choix = [(al, m, e) for al, m, e in tous if (not a.alias or re.search(a.alias, al))
             and (mot is None or m in mot) and (a.refaire or al not in faits)]
    if a.liste:
        rang = {l.split("\t")[0].strip(): i for i, l in enumerate(a.liste.read_text(encoding="utf-8").splitlines())
                if l.strip() and not l.startswith("#")}
        choix = sorted([(al, m, e) for al, m, e in tous if al in rang and (mot is None or m in mot)],
                       key=lambda x: rang[x[0]])
    if a.max:
        choix = choix[:a.max]
    par = {}
    for _, m, _ in choix:
        par[m] = par.get(m, 0) + 1
    print(f"{len(tous)} alias au menu ; {len(faits)} déjà testés ; ce passage : {len(choix)} {par}")
    if not a.pour_de_vrai:
        for al, m, _ in choix:
            lz = lanceur(p, al, m) if m not in AUTO_CHARGES else "auto-chargé : kimi seul"
            print(f"  {m:9s} {al:60s} {' '.join(lz[0]) if isinstance(lz, tuple) else lz}")
        print("--pour-de-vrai absent : rien lancé.")
        return 0
    RESULTATS.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTATS.exists():
        RESULTATS.write_text("# " + "\t".join(COLONNES[1:2] + COLONNES[:1] + COLONNES[2:]) + "\n", encoding="utf-8")
    def passe(liste, marque=""):
        pannes = []
        for i, (al, m, e) in enumerate(liste, 1):
            if PAUSE.exists():
                print(f"PAUSE ({PAUSE}) : en attente, aucun alias lancé.", flush=True)
                while PAUSE.exists():
                    time.sleep(5)
                print("reprise.", flush=True)
            l = tester(p, cles, al, m, e, a)
            if marque:
                l["detail"] = (l["detail"] + " ; " if l["detail"] else "") + marque
            with RESULTATS.open("a", encoding="utf-8") as f:
                f.write("\t".join([l["alias"], l["date"]] + [l[c] for c in COLONNES[2:]]) + "\n")
            print(f"[{marque}{i}/{len(liste)}] {l['verdict']:5s} {m:9s} {al} {l['etape']} {l['cause'][:120]}", flush=True)
            # une cause de configuration (alias ou chemin absent) ne change pas au second essai : pas de rejeu
            if l["verdict"] == "PANNE" and not re.search(r"absent", l["cause"]):
                pannes.append((al, m, e))
        return pannes

    pannes = passe(choix)
    if a.rapide and pannes:
        print(f"rejeu unique de {len(pannes)} panne(s) en fin de liste", flush=True)
        passe(pannes, "rejeu ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
