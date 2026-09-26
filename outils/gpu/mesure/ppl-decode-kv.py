"""PPL de décodage (teacher forcing) après un préfixe en prefill, UN chargement
pour N tranches (chantier 3.2, `poste7-tests-30min-20-09` § 3.2, 20/09).

Même arithmétique que `scratchpad/ppl-decode-kv-17-09.py` (mêmes ids, même
préfixe de séquence `PPL_PREFIXE` en tête et compté dans P, même teacher forcing,
mêmes jetons notés) : la PPL d'une tranche doit être IDENTIQUE à 10⁻⁴ à la prise
isolée — c'est le contrôle. Ce qui change : le modèle et le moteur sont chargés
UNE fois, chaque tranche est une requête neuve, et chaque résultat est écrit dès
que la tranche finit (ligne `RESULTAT {json}` + `<sortie>/ppl-<bras>-t<k>.json`).

    ppl-decode-kv.py --tranches t1.txt t2.txt … [--bras nom] [--sortie DIR]
                     [--prefixe 8192] [--notes 512] [--prefixe-seq '[gMASK]<sop>']
                     [--modele DIR] [--depuis k] [--reprendre]
    ppl-decode-kv.py --dossier DIR --motif 'tranche-*.txt' …

Repli d'environnement (chaînes existantes) : ACVRAM_MODELE_MESURE, PPL_DECODE_CORPUS
(une seule tranche si `--tranches` absent), PPL_PREFIXE. Régime : aucun posé ici
(leçon régime-par-défaut, 19/09) ; l'appelant pose ACVRAM_KV_FORMAT et le reste.

Cache de préfixe : GARDÉ au défaut servi (`enable_prefix_cache=True`, runner.py:383) —
contrôle de poste2 (20/09 07 h 55, tranche0, k = 1) : cache désactivé 14,7888 contre 14,9613
avec l'ancien script (−1,2 %), mêmes ids, même régime ; la cause est `_frontiere_insta`
(runner.py:839) : sur un hybride (GLM : créneaux MLA) le cache de préfixe COUPE le prefill à
un multiple de 256 (8 192 → 7 936 + 256) pour photographier l'état, et un prefill en deux
morceaux n'a pas la numérique d'un prefill d'un morceau. Le serveur livré a le cache ON :
c'est ce régime qu'on mesure ; `--sans-cache-prefixe` = l'autre bras, nommé dans RESULTAT
(`enable_prefix_cache`). Deux tranches différentes ne partagent aucun bloc (clé = hash
chaîné des ids, runner.py:918-933) ; un état ne survit donc pas d'une tranche à l'autre Les états des couches à tampons fixes (GDN/KDA/Mamba, MLA de GLM)
sont retirés du magasin à `_finish` (runner.py:994-995) et le créneau est remis à
zéro au premier forward de la requête suivante (`static_load(st, None)` :
gdn.py:212-214 zéro conv et S ; mla.py:613-614 `len` à 0, les lignes du cache
au-delà de `len` sont masquées mla.py:818 et réécrites à l'index `len` mla.py:634),
parce que l'id de séquence est un compteur jamais réutilisé (runner.py:37) et
que `static_owners[slot]` ne vaut donc jamais le nouvel id (model.py:2619).
Les graphes CUDA capturés survivent — c'est le gain visé — et la tranche 1
paie leur capture dans son `t_decode` ; les tranches suivantes disent le régime
établi. Les temps sont relevés après `torch.cuda.synchronize()` (« durée
relevée après le join »)."""
import argparse, glob, json, math, os, subprocess, sys, time
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


_REPO = os.environ.get("ACVRAM_ARBRE", str(Path(__file__).resolve().parents[3]))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

CLES_RESULTAT = ("bras", "k", "tranche", "kv_format_env", "kv_dtype_effectif", "ppl", "nll_moy", "n_jetons_notes",
                 "pas", "capacity_tokens", "bytes_per_block", "prefixe_sequence", "regime_ligne",
                 "t_charge", "t_prefill", "t_decode")
# Schéma de l'ancien script (ppl-decode-kv-17-09.py:34-45) : toujours présent dans chaque JSON, à None si inconnu.
CLES_SCHEMA = ("kv_format_env", "regime_ligne", "modele", "prefixe_sequence", "prefixe_ids", "corpus", "prefixe", "notes",
               "premiers_ids", "acvram", "engine_regime", "kv_dtype_effectif", "bytes_per_block", "block_size",
               "num_blocks", "capacity_tokens", "kv_total_bytes_par_couche", "n_caches", "t_charge")


def decouper_ids(tokenizer, texte: str, prefixe_seq: str, prefixe: int, notes: int) -> tuple[list[int], list[int]]:
    """Découpe IDENTIQUE à ppl-decode-kv-17-09.py:31-33 : le préfixe de séquence
    (encodé sans jetons spéciaux) en tête, compté dans les `prefixe` premiers ids,
    jamais dans les notés ; le tout tronqué à prefixe + notes, longueur exigée."""
    pfx = tokenizer.encode(prefixe_seq, add_special_tokens=False) if prefixe_seq else []
    ids = (pfx + tokenizer.encode(texte))[:prefixe + notes]
    if len(ids) != prefixe + notes:
        raise ValueError(f"tranche trop courte : {len(ids)} ids pour prefixe+notes = {prefixe + notes}")
    return ids, pfx


def chemin_json(sortie: str, bras: str, k: int) -> str:
    return os.path.join(sortie, f"ppl-{bras}-t{k}.json")


def tranches_a_faire(tranches: list[str], sortie: str, bras: str, depuis: int = 1,
                     reprendre: bool = False) -> list[tuple[int, str]]:
    """k est 1-based dans l'ordre donné. `depuis` saute les k < depuis ;
    `reprendre` saute celles dont le JSON existe déjà (reprise après coupure)."""
    faire = []
    for k, t in enumerate(tranches, 1):
        if k < depuis:
            continue
        if reprendre and os.path.exists(chemin_json(sortie, bras, k)):
            continue
        faire.append((k, t))
    return faire


def faire_forceur(ids: list[int], prefixe: int, etat: dict, torch):
    """Le `_sample_only` de teacher forcing, ppl-decode-kv-17-09.py:53-63 à l'identique :
    à chaque pas, la cible est le jeton suivant du corpus, la NLL est lue sur
    les logits fp32, et la cible est rendue comme jeton « échantillonné »."""
    n = len(ids)

    def sample_force(logits, seqs, **_kw):     # 23/09 : pipeline.py passe depuis_graphe= depuis b3c15a01 (22/09) — l'outil plantait
        p = etat["pos"]; etat["pas"] += 1
        cible = ids[p + 1] if p + 1 < n else ids[p]
        lp = torch.log_softmax(logits[0].to(torch.float32), dim=-1)
        v = float(-lp[cible])
        if p + 1 < n:
            etat["nll"] += v; etat["n"] += 1; etat["nll_par_jeton"].append(round(v, 4))
        etat["pos"] = p + 1
        return torch.tensor([cible], device=logits.device, dtype=torch.long), lp[cible].reshape(1)
    return sample_force


def _sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def noter_tranche(engine, ids: list[int], prefixe: int, notes: int, torch, SamplingParams, request_id: str) -> dict:
    """Une requête neuve : prefill du préfixe, puis `notes` pas forcés. Rend
    ppl/nll/pas et t_prefill (jusqu'à `seq.prefilled`), t_decode (le reste)."""
    etat = {"pos": prefixe - 1, "nll": 0.0, "n": 0, "pas": 0, "nll_par_jeton": []}
    engine._sample_only = faire_forceur(ids, prefixe, etat, torch)
    _sync(torch); t0 = time.perf_counter()
    seq = engine.add_request(ids[:prefixe], SamplingParams(temperature=0.0, max_tokens=notes), request_id=request_id)
    while (engine.running or engine.waiting) and not seq.prefilled:
        engine.step()
    _sync(torch); t1 = time.perf_counter()
    while engine.running or engine.waiting:
        engine.step()
    _sync(torch); t2 = time.perf_counter()
    if etat["n"] == 0:
        raise RuntimeError("aucun jeton noté : le moteur n'a pas appelé _sample_only")
    return {"ppl": round(math.exp(etat["nll"] / etat["n"]), 4), "nll_moy": round(etat["nll"] / etat["n"], 6),
            "n_jetons_notes": etat["n"], "pas": etat["pas"], "nll_par_jeton": etat["nll_par_jeton"],
            "t_prefill": round(t1 - t0, 3), "t_decode": round(t2 - t1, 3)}


def courir(tranches, engine, tokenizer, lire_corpus, torch, SamplingParams, *, bras: str, sortie: str,
           prefixe: int, notes: int, prefixe_seq: str, base: dict, depuis: int = 1, reprendre: bool = False,
           sortie_txt=sys.stdout) -> list[dict]:
    """Boucle sur les tranches ; chaque résultat est écrit (JSON + ligne RESULTAT)
    AVANT de passer à la suivante, pour qu'une coupure ne perde que la tranche en cours."""
    os.makedirs(sortie, exist_ok=True)
    resultats = []
    for k, tranche in tranches_a_faire(tranches, sortie, bras, depuis, reprendre):
        ids, pfx = decouper_ids(tokenizer, lire_corpus(tranche), prefixe_seq, prefixe, notes)
        res = {c: None for c in CLES_SCHEMA}
        res.update(base)
        res.update({"bras": bras, "k": k, "tranche": os.path.basename(tranche), "corpus": os.path.basename(tranche),
                    "prefixe": prefixe, "notes": notes, "prefixe_sequence": prefixe_seq, "prefixe_ids": pfx,
                    "premiers_ids": ids[:4]})
        res.update(noter_tranche(engine, ids, prefixe, notes, torch, SamplingParams, request_id=f"t{k}"))
        res["smi_memory_used_mib_fin"] = _smi("memory.used")
        with open(chemin_json(sortie, bras, k), "w") as fh:
            json.dump(res, fh, indent=1)
        print("RESULTAT", json.dumps({c: res.get(c) for c in CLES_RESULTAT}), file=sortie_txt, flush=True)
        resultats.append(res)
    return resultats


def _smi(champ):
    try:
        out = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={champ}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def lister_tranches(args) -> list[str]:
    if args.tranches:
        return list(args.tranches)
    if args.dossier:
        t = sorted(glob.glob(os.path.join(args.dossier, args.motif)))
        if not t:
            raise SystemExit(f"aucune tranche {args.motif!r} dans {args.dossier}")
        return t
    corpus = os.environ.get("PPL_DECODE_CORPUS", "/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
    return [corpus]


def analyser(argv=None):
    p = argparse.ArgumentParser(description="PPL de décodage en teacher forcing, un chargement pour N tranches")
    p.add_argument("--tranches", nargs="+", help="fichiers texte, un par tranche (k = rang, 1-based)")
    p.add_argument("--dossier", help="dossier de tranches (avec --motif, ordre trié)")
    p.add_argument("--motif", default="*.txt")
    p.add_argument("--bras", default=os.environ.get("ACVRAM_KV_FORMAT") or "bras", help="étiquette du bras")
    p.add_argument("--sortie", default=".", help="dossier des JSON ppl-<bras>-t<k>.json")
    p.add_argument("--prefixe", type=int, default=8192, help="P : jetons en prefill, préfixe de séquence compris")
    p.add_argument("--notes", type=int, default=512, help="N : jetons notés un par pas")
    p.add_argument("--prefixe-seq", default=os.environ.get("PPL_PREFIXE", ""), help="ex. '[gMASK]<sop>' (GLM)")
    p.add_argument("--sans-cache-prefixe", action="store_true",
                   help="bras diagnostic : cache de préfixe éteint (prefill d'un morceau) ; défaut = régime servi (ON, prefill coupé à 256 sur un hybride)")
    p.add_argument("--modele", default=os.environ.get("ACVRAM_MODELE_MESURE",
                   _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"))
    p.add_argument("--depuis", type=int, default=1, help="reprise : sauter les tranches k < depuis")
    p.add_argument("--reprendre", action="store_true", help="sauter les tranches dont le JSON existe déjà")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = analyser(argv)
    tranches = lister_tranches(args)
    import torch
    import acvram
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer
    from acvram.evaluate import _load_corpus
    max_model_len = args.prefixe + args.notes + 64
    t_debut = time.perf_counter()
    tokenizer = load_tokenizer(args.modele)
    _sync(torch); t0 = time.perf_counter()
    loaded = load_model(args.modele, dtype=torch.bfloat16, max_model_len=max_model_len)
    _sync(torch); t1 = time.perf_counter()
    engine = Engine(loaded, tokenizer, max_batch_size=1, max_model_len=max_model_len,
                    enable_cuda_graphs=True, enable_prefix_cache=not args.sans_cache_prefixe)
    engine._eos = set()
    _sync(torch); t2 = time.perf_counter()
    c0 = next(iter(loaded.model.caches.values()), None)   # MLA (GLM) : pas de cache paginé par couche → champs KV à None
    cfg = getattr(c0, "cfg", None)
    base = {"kv_format_env": os.environ.get("ACVRAM_KV_FORMAT", ""), "regime_ligne": acvram.regime_ligne(),
            # relatif : jamais un /home dans un artefact suivi (cliquet 20/09). Le 23/09, ce commentaire en fin de ligne
            # avait avalé la clé engine_regime (KeyError l. 229 : trois bras de la 107 bis sans PPL).
            "modele": os.path.basename(os.path.normpath(args.modele)),
            "acvram": os.path.relpath(acvram.__file__, os.path.dirname(os.path.dirname(acvram.__file__))),
            "engine_regime": engine.regime_ligne(),
            "kv_dtype_effectif": cfg.dtype if cfg else None, "bytes_per_block": cfg.bytes_per_block() if cfg else None,
            "block_size": cfg.block_size if cfg else None, "num_blocks": cfg.num_blocks if cfg else None,
            "capacity_tokens": cfg.capacity_tokens if cfg else None,
            "kv_total_bytes_par_couche": cfg.bytes_per_block() * cfg.num_blocks if cfg else None,
            "n_caches": len(loaded.model.caches), "smi_memory_used_mib_apres_chargement": _smi("memory.used"),
            "t_charge_modele": round(t1 - t0, 3), "t_moteur": round(t2 - t1, 3), "t_charge": round(t2 - t0, 3),
            "n_tranches": len(tranches), "enable_prefix_cache": not args.sans_cache_prefixe}
    print("REGIME", json.dumps({c: base[c] for c in ("regime_ligne", "kv_dtype_effectif", "bytes_per_block", "num_blocks",
                                                      "capacity_tokens", "smi_memory_used_mib_apres_chargement",
                                                      "t_charge_modele", "t_moteur", "t_charge")}), flush=True)
    print(base["engine_regime"], flush=True)
    resultats = courir(tranches, engine, tokenizer, _load_corpus, torch, SamplingParams, bras=args.bras,
                       sortie=args.sortie, prefixe=args.prefixe, notes=args.notes, prefixe_seq=args.prefixe_seq,
                       base=base, depuis=args.depuis, reprendre=args.reprendre)
    _sync(torch)
    print("TOTAL", json.dumps({"bras": args.bras, "n_tranches_notees": len(resultats), "t_charge": base["t_charge"],
                               "t_prefill_somme": round(sum(r["t_prefill"] for r in resultats), 3),
                               "t_decode_somme": round(sum(r["t_decode"] for r in resultats), 3),
                               "t_total": round(time.perf_counter() - t_debut, 3)}), flush=True)


if __name__ == "__main__":
    main()
