#!/usr/bin/env python3
"""Campagne A6 (audit Sage, prédiction scellée dans
revue/prediction-a6-int8-snrfloor0-14-09.md) : reconvertir un modèle à
snr_floor=0 (défaut CLI actuel), comparer PPL et débit de décodage b=12
contre le dossier existant (snr_floor=25, converti avant le 4/09).

Coder-30B en premier, seul tant que le verdict n'est pas rendu (Jérôme,
14/09). Source bf16 : téléchargée avec accord explicite (~60 Gio,
Qwen/Qwen3-Coder-30B-A3B-Instruct) sur le HDD — le SSD est à 41 Gio
libres, la sortie -sf0 va aussi sur le HDD (`--out`, jamais le SSD).

Protocole décodage IDENTIQUE à outils/gpu/mesure/banc-horloge-decodage.py
(Laure) : 12 séquences, ctx 2048, 200 jetons, énergie NVML monotone —
mais réécrit ici (pas importé) pour rester sur LE PAQUET acvram de CE
worktree, pas celui du worktree de Laure que son script pointe en dur.

    python outils/campagne-a6-snrfloor0.py convertir <source_hf> <nom>
    python outils/campagne-a6-snrfloor0.py mesurer <avant_dir> <apres_dir>
"""
import json
import subprocess
import sys
import time
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


REPO = Path(__file__).resolve().parent.parent
# SANS CECI, `import acvram` resout via le finder d'installation editable
# du venv partage — qui pointe sur le depot DE BASE
# (anticitoyen-vram/acvram), PAS ce worktree : `python outils/script.py`
# met le dossier du SCRIPT (outils/) en sys.path[0], pas la racine du
# depot, donc rien ne fait gagner ce worktree sans cette ligne. Piege
# trouve le 14/09 en cherchant pourquoi un correctif de graphs.py
# n'apparaissait jamais dans les journaux — les mesures PPL/decode d'A6
# n'etaient PAS affectees (aucun code moteur n'avait encore change dans
# ce worktree a ce moment-la), mais un correctif ulterieur l'aurait ete.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "gpu" / "mesure"))
from energie import Energie, repos  # noqa: E402

VENV_PY = _os.environ.get("ACVRAM_PY", f"{Path(__file__).resolve().parents[1]}/../../anticitoyen-vram/.venv/bin/python3")
CORPUS = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
SORTIE_HDD = Path(_RACINE)  # SSD a 41 Gio libres
SLOTS = 12
CTX = 2048
N_JETONS = 200
VOCAB_APPROX = 150000


def cmd_convertir(source: str, nom: str) -> int:
    """`acvram convert <source> --name <nom> -o <hdd>/<nom>-sf0` — le
    defaut CLI (snr_floor=0) s'applique sans flag, c'est tout le point."""
    dest = SORTIE_HDD / f"{nom}-sf0"
    print(f"BEAD A6 — conversion snr_floor=0 : {source} -> {dest}")
    cmd = [VENV_PY, "-m", "acvram", "convert", source, "--name", nom,
          "-o", str(dest)]
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO))
    return r.returncode


def _ppl(dossier: str) -> dict:
    # `acvram eval` rend une LISTE (un element par modele passe) — un seul
    # dossier ici, donc data[0].
    cmd = [VENV_PY, "-m", "acvram", "eval", dossier, "--corpus", str(CORPUS),
          "--window", "2048", "--stride", "2048", "--min-context", "256",
          "--json"]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        return {"echec": r.stderr[-3000:]}
    # `acvram eval` imprime le cadrage et la progression AVANT le JSON —
    # extraire a partir du DERNIER "\n[\n" (le json.dumps(indent=2) final
    # commence exactement ainsi : "[" seul sur sa ligne puis l'element
    # indente) ; un simple "\n[" se trouve aussi a l'INTERIEUR de l'objet
    # (des listes vides comme "vus_plusieurs_fois": [] suivies d'une
    # virgule et d'une nouvelle ligne) et coupe le JSON au mauvais
    # endroit — piege trouve au premier essai.
    idx = r.stdout.rfind("\n[\n")
    brut = r.stdout[idx + 1:] if idx >= 0 else r.stdout
    try:
        donnees = json.loads(brut)
    except json.JSONDecodeError:
        return {"echec": f"JSON illisible :\n{r.stdout[-2000:]}"}
    return donnees[0] if donnees else {"echec": "liste vide"}


def _invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


def _decode_b12(dossier: str) -> dict:
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    loaded = load_model(dossier, dtype=torch.bfloat16, max_model_len=CTX)
    engine = Engine(loaded, None, max_batch_size=SLOTS, max_model_len=CTX)
    engine.warm_graphs()
    torch.cuda.reset_peak_memory_stats(0)
    params = SamplingParams(temperature=0.0, max_tokens=N_JETONS)
    for k in range(SLOTS):
        engine.add_request(_invite(1000 + k, min(256, CTX // 4)), params,
                           request_id=f"d{k}")

    base = repos(secondes=8.0)
    n_avant = engine.stats.decode_tokens
    with Energie() as e:
        t0 = time.perf_counter()
        n_pas = 0
        while engine.running or engine.waiting:
            engine.step()
            n_pas += 1
            if n_pas > N_JETONS + 20:
                raise RuntimeError("le lot ne se termine pas")
        torch.cuda.synchronize()
        duree = time.perf_counter() - t0
    n = engine.stats.decode_tokens - n_avant
    joules_net = max(e.joules - base.moyenne * duree, 0.0)
    return {"slots": SLOTS, "ctx": CTX, "n_jetons_decodes": n,
           "duree_mesure_s": round(duree, 4), "jetons_s": n / duree if duree else 0.0,
           "joules_net": round(joules_net, 1),
           "j_par_jeton_net": round(joules_net / n, 4) if n else None,
           "watts_repos": round(base.moyenne, 1), **e.resume()}


def cmd_decoder(dossier: str) -> int:
    """Sous-commande dediee, appelee dans son PROPRE processus par
    cmd_mesurer — deux load_model() dans le meme processus (comme mon
    piege du duel vLLM/acvram la veille) laisseraient le premier modele
    resident en VRAM, forcant le second en exil massif (mesure : 39/128
    experts residents au lieu de la totalite) et publiant un debit ×16
    plus lent qui n'aurait rien a voir avec snr_floor."""
    print("RESULTAT_DECODE " + json.dumps(_decode_b12(dossier), ensure_ascii=False))
    return 0


def _mesurer_decode_isole(dossier: str) -> dict:
    cmd = [VENV_PY, str(Path(__file__).resolve()), "decoder", dossier]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        return {"echec": r.stderr[-3000:]}
    idx = r.stdout.rfind("RESULTAT_DECODE ")
    if idx < 0:
        return {"echec": f"pas de ligne RESULTAT_DECODE :\n{r.stdout[-2000:]}"}
    return json.loads(r.stdout[idx + len("RESULTAT_DECODE "):])


def cmd_mesurer(avant: str, apres: str) -> int:
    print(f"BEAD A6 — mesure avant/apres : {avant} vs {apres}")
    print("  PPL avant...", flush=True)
    ppl_avant = _ppl(avant)
    print("  PPL apres...", flush=True)
    ppl_apres = _ppl(apres)
    print("  decodage b=12 avant (processus isole)...", flush=True)
    dec_avant = _mesurer_decode_isole(avant)
    print("  decodage b=12 apres (processus isole)...", flush=True)
    dec_apres = _mesurer_decode_isole(apres)

    resultat = {"avant": {"dossier": avant, "ppl": ppl_avant, "decode": dec_avant},
               "apres": {"dossier": apres, "ppl": ppl_apres, "decode": dec_apres}}
    print("RESULTAT " + json.dumps(resultat, ensure_ascii=False, indent=2))

    if ("echec" not in ppl_avant and "echec" not in ppl_apres
            and "echec" not in dec_avant and "echec" not in dec_apres):
        p_avant = ppl_avant.get("perplexity")
        p_apres = ppl_apres.get("perplexity")
        d_avant = dec_avant["jetons_s"]
        d_apres = dec_apres["jetons_s"]
        if p_avant and d_avant:
            ppl_pct = 100 * (p_apres / p_avant - 1)
            deb_pct = 100 * (d_apres / d_avant - 1)
            seuil_ok = ppl_pct <= 2.0 and deb_pct >= 5.0
            print(f"\n  PPL {p_avant:.4f} -> {p_apres:.4f} ({ppl_pct:+.2f} %)")
            print(f"  debit {d_avant:.1f} -> {d_apres:.1f} t/s ({deb_pct:+.2f} %)")
            print(f"  seuil scelle (PPL <=+2% ET debit >=+5%) : "
                  f"{'RESPECTE' if seuil_ok else 'DEPASSE'}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: campagne-a6-snrfloor0.py {convertir|mesurer} ...")
        return 2
    if sys.argv[1] == "convertir":
        return cmd_convertir(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "mesurer":
        return cmd_mesurer(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "decoder":
        return cmd_decoder(sys.argv[2])
    print(f"ECHEC / CAUSE: sous-commande inconnue {sys.argv[1]!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
