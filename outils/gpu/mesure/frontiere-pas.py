"""Décomposition de la frontière de pas à b=12 (verdict 2.4 de qr.md, chef
21/09) : entre la fin du graphe n et le début du graphe n+1, QUI prend les
0,43 ms ? Mesuré en processus, régime servi (pipeline, graphes, lot constant
de B séquences, ctx 2048, invite 256, comme certifie-b12 CERT_PUR), sans HTTP.

Côté carte (événements CUDA, µs par pas) :
  graphe        rejeu du graphe (tête de sortie incluse : le graphe rend les logits)
  echantillon   noyaux de `_sample_only` (argmax glouton)
  trou_gpu      carte OISIVE entre la fin de l'échantillon n et le rejeu n+1
                → C'EST la frontière ; elle ne peut venir que de l'hôte.
Côté hôte (perf_counter, µs par pas, dans l'ordre du pas recouvert) :
  suite_prep    `_grow` + `_build_batch_device` + `preparer` (bind/fill lus dans graphs.temps_*)
  lancement     `rejouer_suivant` côté hôte (replay() asynchrone)
  echant_hote   `_sample_only` côté hôte (lancements des noyaux)
  attente_evt   `event.synchronize()` du pas précédent (par différence)
  consommer     `.tolist()` (D2H) + `_emit` python
  reste_step    tout le reste de `step()` (admission, planification)
  hors_step     boucle de l appelant entre deux `step()`
À part : tête seule (`model._tete` sur [B, H]) et argmax seul sur [B, V],
mesurés hors pipeline, pour situer la tête dans le graphe.

Usage : outils/carte.sh python outils/gpu/mesure/frontiere-pas.py SORTIE.json [B=12] [N_PAS=300]
Durée : chargement + chauffe + N_PAS × ~8 ms ≈ 2-3 min. Rien n est modifié :
seules des enveloppes chronométrées autour des méthodes du moteur.
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402

MAIN_REPO = os.environ.get("ACVRAM_ARBRE", subprocess.run(
    ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--show-toplevel"],
    capture_output=True, text=True).stdout.strip())
sys.path.insert(0, MAIN_REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch  # noqa: E402
from acvram.engine.loader import load_model  # noqa: E402
from acvram.engine.runner import Engine  # noqa: E402
from acvram.engine.sampler import SamplingParams  # noqa: E402

SORTIE = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 12
N_PAS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
MODEL = os.environ.get("ACVRAM_MODELE_MESURE", _racine_modeles() + "/Qwen3-Coder-30B-A3B-nvfp4")
# FRONTIERE_CTX / FRONTIERE_INVITE (pièce 104 § 5 : frontière à ctx 8 k) ; défaut 2048 / 256 inchangé
CTX = int(os.environ.get("FRONTIERE_CTX", "2048"))
PROMPT_LEN, CHAUFFE = int(os.environ.get("FRONTIERE_INVITE", "256")), 30

if not torch.cuda.is_available():
    sys.exit("frontiere-pas : carte requise (sous outils/carte.sh)")
commit = subprocess.run(["git", "-C", MAIN_REPO, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=CTX, max_concurrent_seqs=B)
vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
engine = Engine(loaded, None, max_batch_size=B, max_model_len=CTX, enable_cuda_graphs=True)
engine._eos = set()
if not (engine.pipeline_actif and engine.graphs is not None):
    sys.exit(f"frontiere-pas : pipeline inactif ({engine.regime_ligne()}) — la frontière mesurée ici est celle du pas recouvert")
engine.warm_graphs()


def invite(k, n):
    return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(n)]


# ---- enveloppes : chaque pas note ses événements et ses durées hôte ----
pas: list[dict] = []
courant: dict = {}
mesure = False


def _ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def envelopper(obj, nom, cle, gpu=False):
    orig = getattr(obj, nom)

    def enveloppe(*a, **kw):
        if not mesure:
            return orig(*a, **kw)
        if gpu:
            courant[cle + "_g0"] = _ev()
        t0 = time.perf_counter()
        r = orig(*a, **kw)
        courant[cle] = courant.get(cle, 0.0) + (time.perf_counter() - t0) * 1e6
        if gpu:
            courant[cle + "_g1"] = _ev()
        return r
    setattr(obj, nom, enveloppe)


envelopper(engine.graphs, "rejouer_suivant", "lancement", gpu=True)
envelopper(engine, "_sample_only", "echant_hote", gpu=True)
envelopper(engine, "_consommer", "consommer")
envelopper(engine, "_pipeline_suite", "suite_total")
envelopper(engine, "_plain_decode_pipeline", "pipeline_total")
envelopper(engine, "step", "step_total")

params = SamplingParams(temperature=0.0, max_tokens=CTX - PROMPT_LEN - 8)
for k in range(B):
    engine.add_request(invite(1000 + k, PROMPT_LEN), params, request_id=f"p{k}")
while any(not s_.prefilled for s_ in engine.running) or engine.waiting:
    engine.step()
for _ in range(CHAUFFE):
    engine.step()
torch.cuda.synchronize()

mesure = True
n_bind = len(getattr(engine.graphs, "temps_bind", []))
t_prec = None
for i in range(N_PAS):
    courant.clear()
    t_hors = time.perf_counter()
    if t_prec is not None:
        courant["hors_step"] = (t_hors - t_prec) * 1e6
    engine.step()
    t_prec = time.perf_counter()
    courant["lot"] = len([s for s in engine.running if not s.finished])
    pas.append(dict(courant))
mesure = False
torch.cuda.synchronize()
temps_bind = getattr(engine.graphs, "temps_bind", [])[n_bind:]
temps_fill = getattr(engine.graphs, "temps_fill", [])[n_bind:]

# ---- dépouillement ----
lignes = []
for i, p in enumerate(pas):
    if "lancement_g0" not in p or "echant_hote_g1" not in p:
        continue                                      # pas d'amorçage ou eager : hors série
    d = {
        "graphe": p["lancement_g0"].elapsed_time(p["lancement_g1"]) * 1e3,
        "echantillon": p["echant_hote_g0"].elapsed_time(p["echant_hote_g1"]) * 1e3,
        "suite_prep": p.get("suite_total", 0.0) - p.get("lancement", 0.0) - p.get("echant_hote", 0.0),
        "lancement": p.get("lancement", 0.0),
        "echant_hote": p.get("echant_hote", 0.0),
        "consommer": p.get("consommer", 0.0),
        "attente_evt": p.get("pipeline_total", 0.0) - p.get("suite_total", 0.0) - p.get("consommer", 0.0),
        "reste_step": p.get("step_total", 0.0) - p.get("pipeline_total", 0.0),
        "hors_step": p.get("hors_step", 0.0),
        "step_total": p.get("step_total", 0.0),
        "lot": p["lot"],
    }
    if i + 1 < len(pas) and "lancement_g0" in pas[i + 1]:
        d["trou_gpu"] = p["echant_hote_g1"].elapsed_time(pas[i + 1]["lancement_g0"]) * 1e3
        d["pas_gpu"] = p["lancement_g0"].elapsed_time(pas[i + 1]["lancement_g0"]) * 1e3
    lignes.append(d)

if len(temps_bind) == len(pas):
    for d, tb, tf in zip(lignes, temps_bind, temps_fill):
        d["prep_bind"], d["prep_fill"] = tb * 1e3, tf * 1e3

lots = sorted({d["lot"] for d in lignes})
pleins = [d for d in lignes if d["lot"] == B]


def resume(cle):
    v = [d[cle] for d in pleins if cle in d]
    return {"mediane_us": round(statistics.median(v), 1), "moyenne_us": round(statistics.fmean(v), 1),
            "p90_us": round(sorted(v)[int(0.9 * (len(v) - 1))], 1), "n": len(v)} if v else None


cles = ["pas_gpu", "graphe", "echantillon", "trou_gpu", "step_total", "suite_prep", "prep_bind", "prep_fill",
        "lancement", "echant_hote", "attente_evt", "consommer", "reste_step", "hors_step"]
r = {"commit": commit, "modele": MODEL, "B": B, "n_pas": N_PAS, "n_pleins": len(pleins), "lots_vus": lots,
     "regime": engine.regime_ligne(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
     "us": {k: resume(k) for k in cles}}

# ---- tête seule et argmax seul, hors pipeline ----
m = loaded.model
H = int(loaded.spec.hidden_size)
x = (torch.randn(B, H, device="cuda:0") * 0.3).to(torch.bfloat16)


def chrono(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    a, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b_.record()
    torch.cuda.synchronize()
    return round(a.elapsed_time(b_) * 1e3 / n, 1)


with torch.inference_mode():
    tete = m._tete if hasattr(m, "_tete") else m.lm_head
    r["us"]["tete_seule"] = chrono(lambda: tete(x))
    logits = tete(x)
    r["us"]["argmax_seul"] = chrono(lambda: torch.argmax(logits, dim=-1))
    r["logits"] = {"forme": list(logits.shape), "dtype": str(logits.dtype)}

json.dump(r, open(SORTIE, "w"), indent=1)
print(f"[frontiere-pas] commit {commit} B={B} pas pleins {len(pleins)}/{len(lignes)} lots vus {lots} régime {r['regime']}")
for k in cles:
    s = r["us"][k]
    if s:
        print(f"  {k:12s} méd {s['mediane_us']:8.1f} µs  moy {s['moyenne_us']:8.1f}  p90 {s['p90_us']:8.1f}")
print(f"  tête seule {r['us']['tete_seule']} µs · argmax seul {r['us']['argmax_seul']} µs · logits {r['logits']}")
print(f"  contrôle : graphe + échantillon + trou_gpu = "
      f"{sum(r['us'][k]['mediane_us'] for k in ('graphe', 'echantillon', 'trou_gpu') if r['us'][k]):.1f} µs "
      f"contre pas_gpu {r['us']['pas_gpu']['mediane_us'] if r['us']['pas_gpu'] else '?'} µs")
