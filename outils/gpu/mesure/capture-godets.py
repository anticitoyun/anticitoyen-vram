"""Passe de capture (REGLES § 3) : warm_graphs + 20 pas de décodage à chaque godet, sous le régime courant.
    capture-godets-17-09.py <sortie.json> [godets=1,2,8,16]
Rend par godet : capture ok / erreur, graphes=on après le premier pas, ms/pas ; un seul processus,
un moteur par godet (déchargé entre deux). À lancer sous carte.sh avant tout défaut de noyau de décodage."""
import json, os, sys, time, traceback
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)

_REPO = os.environ.get("ACVRAM_ARBRE", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _REPO)
import torch, acvram
# 23/09 (pièce 82 ter) : la racine valait outils/gpu depuis le déplacement dans outils/gpu/mesure/ — `acvram` venait
# alors de l'installation (l'arbre principal), pas de l'arbre mesuré, sans rien dire. Refus si ce n'est pas l'arbre.
if not os.path.realpath(acvram.__file__).startswith(os.path.realpath(_REPO) + os.sep):
    raise SystemExit(f"capture-godets : acvram importé de {acvram.__file__}, pas de l'arbre {_REPO}")
print("arbre", _REPO, flush=True)
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
SORTIE = sys.argv[1]; GODETS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,8,16").split(",")]
MODEL = os.environ.get("ACVRAM_MODELE_MESURE", _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4")
CTX, PROMPT_LEN, N_PAS = 2048, 256, 20
res = {"modele": MODEL, "regime_ligne": acvram.regime_ligne(), "version": getattr(acvram, "__version__", "?"), "godets": {}}
print(res["regime_ligne"], flush=True)
loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=CTX, max_concurrent_seqs=max(GODETS))   # le Plan doit connaître le plus grand godet (kv_planned_seqs, garde du 17/09)
vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
def invite(k, n): return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(n)]
for B in GODETS:
    r = {"b": B}
    try:
        # (19/09) plus de variable de régime posée par l'instrument : le plafond hybride
        # suit max_batch_size=B (graphs.py plafond_hybride) ; la valeur posée ICI après
        # l'import était inerte (lue une fois à l'import) → plafond 1, godets 2/8/16 en eager
        # rapportés « ok » (poste2, G1)
        eng = Engine(loaded, None, max_batch_size=B, max_model_len=CTX, enable_cuda_graphs=True)
        eng._eos = set()
        n_captures = eng.warm_graphs()
        r["warm_graphs"] = n_captures
        for k in range(B): eng.add_request(invite(1000 + k, PROMPT_LEN), SamplingParams(temperature=0.0, max_tokens=N_PAS + 8), request_id=f"g{B}s{k}")
        while any(not s.prefilled for s in eng.running) or eng.waiting: eng.step()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N_PAS): eng.step()
        torch.cuda.synchronize(); r["ms_par_pas"] = round(1000 * (time.perf_counter() - t0) / N_PAS, 3)
        r["lot"] = len(eng.running); r["engine_regime"] = eng.regime_ligne(); r["graphes_on"] = "graphes=on" in r["engine_regime"]
        # un repli eager VU pendant les pas (plafond, forme hors godet…) rend le godet faux,
        # même si « graphes=on » : le contrôle lit ce que le rejeu a fait, pas ce que l'objet dit
        r["replis_eager"] = sorted(getattr(eng.graphs, "_raisons_eager_vues", set())) if eng.graphs is not None else ["graphes absents"]
        r["n_replis_eager"] = int(getattr(eng.graphs, "replis_eager", 0)) if eng.graphs is not None else 0
        r["ok"] = bool(r["graphes_on"]) and r["lot"] == B and not r["replis_eager"] and r["n_replis_eager"] == 0
        # photos VRAM du GraphRunner (chantier-gemma-capture-godet1-20-09) : octets
        # libres / réservés / alloués avant la première capture, et après un échec
        r["memoire_avant_capture"] = eng.regime().get("graphes_memoire_avant_capture")
        r["memoire_apres_echec"] = eng.regime().get("graphes_memoire_apres_echec")
        del eng; torch.cuda.empty_cache()
    except Exception as e:
        # ACVRAM_TRACEBACK=1 (même interrupteur que le CLI) : pile ENTIÈRE, au journal
        # aussi — 800 caractères ne montraient pas quel noyau Triton tombait (Ornith)
        pile = traceback.format_exc()
        r["ok"] = False; r["erreur"] = f"{type(e).__name__}: {str(e)[:200]}"
        r["trace"] = pile if os.environ.get("ACVRAM_TRACEBACK") else pile[-800:]
        print("ERREUR godet", B, r["erreur"], flush=True)
        if os.environ.get("ACVRAM_TRACEBACK"):
            print(pile, flush=True)
        res["godets"][str(B)] = r; break   # erreur CUDA collante : le processus ne vaut plus rien
    res["godets"][str(B)] = r
    print("GODET", json.dumps({k: v for k, v in r.items() if k not in ("trace", "engine_regime")}), flush=True)
    json.dump(res, open(SORTIE, "w"), indent=1)
json.dump(res, open(SORTIE, "w"), indent=1)
print("RESULTAT", json.dumps({b: {"ok": g.get("ok"), "ms": g.get("ms_par_pas"), "err": g.get("erreur")} for b, g in res["godets"].items()}))
