#!/usr/bin/env python3
"""Pièce 145 (poste6, 24/09) : TTFT et énergie du PRÉFILL en service, b=1, contre un `acvram serve` déjà lancé
(`--no-prefix-cache`, sinon la 2e invite ne préfille rien). Pour chaque longueur L : des invites DISTINCTES de L ids
(`invite(k, L)` du banc, mêmes ids d'un bras à l'autre), une requête à la fois, `max_tokens=1`, flux SSE ; TTFT = temps
jusqu'au premier fragment portant un jeton (texte vide compris, pièce 210) ; fenêtre ≥ FENETRE_S secondes et ≥ N_MIN requêtes sous `Energie` ; ligne de
base `repos(8 s)` APRÈS la fenêtre (serveur chargé, inactif) ; J par préfill = (joules − repos × durée) / n.
  TTFT_URL=http://127.0.0.1:PORT TTFT_MODELE=<served-name> ttft-service-p145.py 512,2048,4096 > sortie.log
Sortie : une ligne `RESULTAT {json}` par longueur (ttft_med_ms, ttft_min/max, n, j_par_prefill_net, watts, jetons_s_prefill)."""
import json
import os
import statistics
import sys
import time

import httpx

# Pièce 211 (garde d'import a86fa1dd, comme la 168) : la racine doit venir de CE script, jamais
# de l'installation editable — sinon la sonde de regime d'energie.py importe l'arbre principal
# depuis un worktree et la garde refuse, nommement, « indisponible » (constat poste5, 25/09).
_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_ICI))))
sys.path.insert(0, _ICI)
from energie import Energie, repos  # noqa: E402

URL = os.environ["TTFT_URL"].rstrip("/")
MODELE = os.environ.get("TTFT_MODELE", "coder")
FENETRE_S = float(os.environ.get("TTFT_FENETRE_S", "10"))
N_MIN = int(os.environ.get("TTFT_N_MIN", "5"))
N_MAX = int(os.environ.get("TTFT_N_MAX", "40"))
VOCAB_APPROX = int(os.environ.get("BANC_VOCAB", "150000"))


def invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


VIDES = [0]          # premiers jetons dont le texte est vide (compte de la fenêtre, rendu dans RESULTAT)


def premier_fragment(d: dict) -> bool:
    """Un fragment SSE porte-t-il le premier jeton ? Pièce 210 : tout fragment à `choices`, texte VIDE compris. Le jeton
    unique (max_tokens=1) d'une invite d'ids synthétiques peut se décoder en "" (octet UTF-8 partiel, jeton spécial) :
    le service l'a généré (`completion_tokens` = 1), et l'exiger non vide levait « aucun jeton reçu » — toutes les
    passes L = 512 du mixte et de l'i8c (201, prise 8) et la 1re passe de la 165 (24/09) rendues nulles."""
    return bool(d.get("choices")) and "error" not in d


def ttft(client: httpx.Client, ids: list) -> tuple[float, int]:
    """Temps (s) jusqu'au premier fragment portant un jeton (`premier_fragment`) ; nombre de fragments d'erreur."""
    t0 = time.perf_counter()
    premier = None
    err = 0
    with client.stream("POST", f"{URL}/v1/completions",
                       json={"model": MODELE, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "ignore_eos": True,
                             "stream": True}, timeout=600.0) as r:
        for ligne in r.iter_lines():
            if not ligne.startswith("data: ") or ligne.strip() == "data: [DONE]":
                continue
            d = json.loads(ligne[6:])
            if "error" in d:
                err += 1
            elif premier is None and premier_fragment(d):
                premier = time.perf_counter() - t0
                VIDES[0] += not d["choices"][0].get("text")
    if premier is None:
        raise RuntimeError(f"aucun jeton reçu (erreurs {err})")
    return premier, err


def main(longueurs: list[int]) -> None:
    client = httpx.Client()
    r = client.get(f"{URL}/v1/models", timeout=10.0).json()
    servi = r.get("data", [{}])[0].get("id", "?")
    for L in longueurs:
        # chauffe : deux invites hors mesure (compilation Triton, allocations)
        for k in (1, 2):
            ttft(client, invite(9000 + k, L))
        temps = []
        VIDES[0] = 0
        with Energie() as e:
            t0 = time.perf_counter()
            k = 0
            while (time.perf_counter() - t0 < FENETRE_S or len(temps) < N_MIN) and len(temps) < N_MAX:
                k += 1
                t, err = ttft(client, invite(1000 * L + k, L))
                if err:
                    raise RuntimeError(f"erreur SSE à L={L} (k={k})")
                temps.append(t)
            duree = time.perf_counter() - t0
        base = repos(secondes=8.0)
        joules_net = max(e.joules - base.moyenne * duree, 0.0)
        temps_ms = sorted(x * 1000 for x in temps)
        res = {"L": L, "n": len(temps), "servi": servi, "ttft_med_ms": round(statistics.median(temps_ms), 2),
               "ttft_min_ms": round(temps_ms[0], 2), "ttft_max_ms": round(temps_ms[-1], 2),
               "jetons_s_prefill": round(len(temps) * L / sum(temps), 1),
               "duree_fenetre_s": round(duree, 2), "fenetre_valide": duree >= FENETRE_S,
               "joules": round(e.joules, 1), "joules_net": round(joules_net, 1),
               "j_par_prefill_net": round(joules_net / len(temps), 3), "j_par_jeton_prefill_net": round(joules_net / (len(temps) * L), 5),
               "jetons_texte_vide": VIDES[0], "watts_repos": round(base.moyenne, 1), **e.resume()}
        print("RESULTAT " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main([int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "512,2048,4096").split(",")])
