"""Non-régression du parc, par alias (S2, sage-tests-rapides-cloture-20-09 § 2 M2).

Le critère est DUR — chargement, exil inattendu, repli eager, plantage — pas un
débit. Par alias : chargement au régime servi (lot 12, contexte 2 304, graphes),
UN jeton décodé au godet 1, puis `torch.cuda.memory_stats` après les piles :
`inactive_split_bytes` publié comme chiffre, sans seuil (Sage : « chiffre, pas
seuil »). Trois issues, toutes écrites : OK / REPLI (annoncé : graphes absents
ou repli eager, couches ou experts exilés) / ECHEC (exception, avec sa classe).
Aucun texte de modèle sur la sortie standard (REGLES § 6) : l'identifiant du
jeton va dans le JSON.

    python non-regression-parc.py <alias> <sortie.json> [famille]

Racine du parc : ACVRAM_MODELES (outils/racine_modeles.py), jamais un chemin
en dur — le parc a changé de disque le 20/09.
"""
import json, os, sys, time, torch
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))   # APRÈS PYTHONPATH : le paquet mesuré (/usr/share/acvram) passe avant l'arbre
from outils.racine_modeles import MODELES
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.server.chat import load_tokenizer
from acvram.engine.layers import QuantLinear

nom, sortie = sys.argv[1], sys.argv[2]
famille = sys.argv[3] if len(sys.argv) > 3 else "-"
LOT, CTX = 12, 2304                                # le régime des cellules servies (serve --max-batch 12 --max-model-len 2304)
INVITE = "Explique en une phrase ce qu'est la photosynthèse."
GIO = 2.0 ** 30


def memoire(etape):
    s = torch.cuda.memory_stats(); libre, total = torch.cuda.mem_get_info()
    g = lambda k: s.get(k, 0) / GIO
    return {"etape": etape, "alloue_gio": round(g("allocated_bytes.all.current"), 3),
            "reserve_gio": round(g("reserved_bytes.all.current"), 3),
            "inactive_split_gio": round(g("inactive_split_bytes.all.current"), 3),
            "inactive_split_petit_gio": round(g("inactive_split_bytes.small_pool.current"), 3),
            "inactive_split_grand_gio": round(g("inactive_split_bytes.large_pool.current"), 3),
            "segments": int(s.get("segment.all.current", 0)), "libre_carte_gio": round(libre / GIO, 3)}


res = {"alias": nom, "famille": famille, "racine": MODELES, "lot": LOT, "ctx": CTX, "verdict": None}
t0 = time.time()
try:
    chemin = os.path.join(MODELES, nom)
    charge = load_model(chemin, max_model_len=CTX, max_concurrent_seqs=LOT)
    tok = load_tokenizer(chemin)
    res["exiles"] = sum(1 for m in charge.model.modules() if isinstance(m, QuantLinear) and m.streamed is not None)
    res["memoire_chargement"] = memoire("apres chargement")
    mot = Engine(charge, tok, max_batch_size=LOT, max_model_len=CTX)
    g = getattr(mot, "graphs", None)
    res["graphes"] = bool(getattr(g, "enabled", False))
    if tok is not None and hasattr(tok, "apply_chat_template"):
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": INVITE}], True)); res["gabarit"] = "chat"
    elif tok is not None:
        ids = tok.encode(INVITE); res["gabarit"] = "brut"
    else:
        ids = [1, 2, 3, 4, 5, 6, 7, 8]; res["gabarit"] = "aucun"
    mot.add_request(ids, SamplingParams(temperature=0.0, max_tokens=1), request_id="r0")
    produits = []
    for _ in range(8):
        for s in (mot.step() or []):
            produits.extend(list(getattr(s, "token_ids", ()) or []))
        if not mot.running and not mot.waiting: break
    torch.cuda.synchronize()
    res["jetons"] = len(produits); res["jeton_id"] = produits[:1]
    reg = mot.regime()                                  # état VIVANT après le pas (graphes retombe à False si une capture a échoué)
    res["graphes"] = bool(reg.get("graphes")); res["repli_eager"] = int(reg.get("repli_eager", 0))
    res["eager_raisons"] = list(reg.get("replis_eager_raisons", [])); res["regime"] = mot.regime_ligne()
    res["memoire_piles"] = memoire("apres un jeton (piles construites)")
    res["t_s"] = round(time.time() - t0, 1)
    repli = (not res["graphes"]) or res["repli_eager"] > 0 or bool(res["eager_raisons"]) or res["exiles"] > 0 or res["jetons"] != 1
    res["verdict"] = "REPLI" if repli else "OK"
except Exception as exc:                                   # noqa: BLE001
    import traceback; traceback.print_exc()
    res["verdict"] = "ECHEC"; res["exception"] = f"{type(exc).__name__}: {str(exc)[:160]}"; res["t_s"] = round(time.time() - t0, 1)
json.dump(res, open(sortie, "w"), indent=1)
m = res.get("memoire_piles") or {}
print(f"RESULTAT\t{famille}\t{nom}\t{res['verdict']}\texil {res.get('exiles', '?')}\tgraphes {'oui' if res.get('graphes') else 'NON'}\t"
      f"eager {res.get('eager_raisons', '?')}\tjetons {res.get('jetons', '?')}\t"
      f"inactive_split {m.get('inactive_split_gio', '?')} Gio (petit {m.get('inactive_split_petit_gio', '?')}, grand {m.get('inactive_split_grand_gio', '?')})\t"
      f"reserve {m.get('reserve_gio', '?')} alloue {m.get('alloue_gio', '?')}\t{res['t_s']} s" + (f"\t{res['exception']}" if 'exception' in res else ""))
sys.exit(0 if res["verdict"] == "OK" else 2 if res["verdict"] == "REPLI" else 1)
