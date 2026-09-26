#!/usr/bin/env python3
"""Audit de poste7 (revue/poste7-audit-connaissances-14-09.md, item A1) : TTFT
1,465 s contre 0,642 s llama.cpp (×2,29) sur Qwen3-Coder-30B-A3B-nvfp4, 71 %
hors forward jamais profilé. Protocole recommandé par l'audit lui-même :
« temps par étape (tokenizer, gabarit Jinja, forward, _decode_delta) sur 6
requêtes uniques ; forward > 70 % → clos ».

Chaque étape est chronométrée côté CPU (time.perf_counter) ; la première
requête est en plus capturée sous torch.profiler pour compter les
synchronisations hôte (piste connue : `int(ntiles.sum())`, model.py:764,
une sync par couche MoE en prefill — voir revue/plomberie-du-pas-de-
decodage.md).

Rien ne charge le modèle sans --pour-de-vrai.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

BASE = Path(MODELES)
DOSSIER = BASE / "Qwen3-Coder-30B-A3B-nvfp4"

# Six invites UNIQUES (le cache de prefixe fausserait tout réemploi) —
# même protocole que bench.py:_invite : pas la même invite répétée.
INVITES = [
    "Écris une fonction Python qui trie une liste de tuples par leur second élément.",
    "Explique la différence entre un thread et un processus en trois phrases.",
    "Convertis ce JSON en dataclass Python : {\"nom\": \"a\", \"age\": 3}",
    "Quel est l'algorithme le plus rapide pour trouver le k-ième plus petit élément ?",
    "Écris un test unitaire pytest pour une fonction qui divise deux entiers.",
    "Résume en deux lignes ce que fait un ramasse-miettes générationnel.",
]

# Régime du duel du 13/09 (poste4.md:3739-3750) : invites de ~350 mots de
# texte réel, DIFFÉRENTES par essai — reconstruites depuis le même corpus
# (wiki.test.raw) pour comparer au même TTFT 1,465 s / 12 séquences plutôt
# qu'à des invites-jouets de ~30 jetons qui sous-estiment le prefill réel.
CORPUS_LONG = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki.test.raw")


def _invites_longues(n: int, mots: int = 350) -> list:
    texte = CORPUS_LONG.read_text(errors="ignore")
    tous_mots = texte.split()
    fenetre = mots
    invites = []
    for i in range(n):
        depart = (i * fenetre * 7) % max(1, len(tous_mots) - fenetre)
        invites.append(" ".join(tous_mots[depart:depart + fenetre]) +
                      f"\n\nRésume ce texte en une phrase. (#{i})")
    return invites


def _mesurer_concurrent(engine, tokenizer, a) -> int:
    """Regime du chiffre de l'audit : N sequences NOUVELLES soumises
    ENSEMBLE, un pas de prefill batche les traite toutes — TTFT par
    sequence = premier jeton emis pour ELLE, pas la fin du lot."""
    import json
    import time as _time
    from acvram.engine.sampler import SamplingParams

    N = a.concurrence
    textes = (_invites_longues(N) if a.invites_longues
             else [INVITES[i % len(INVITES)] + f" (#{i})" for i in range(N)])
    prompts_ids = []
    for texte in textes:
        messages = [{"role": "user", "content": texte}]
        rendu = tokenizer.apply_chat_template(messages, True, {})
        prompts_ids.append(tokenizer.encode(rendu))

    p0 = engine.stats.to_dict().get("prefill_seconds", 0.0) or 0.0

    profiler_ctx = None
    if a.profil_nsys:
        import torch
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                       torch.profiler.ProfilerActivity.CUDA])
        profiler_ctx.__enter__()

    t0 = _time.perf_counter()
    seqs = [engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=8))
           for ids in prompts_ids]
    premiers = {s.id: None for s in seqs}
    while not all(v is not None for v in premiers.values()):
        for out in engine.step():
            if premiers.get(out.sequence_id) is None:
                premiers[out.sequence_id] = _time.perf_counter()
        if not engine.running and not engine.waiting:
            break
    t_fin = _time.perf_counter()

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        table = profiler_ctx.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=30)
        chemin = sorties() / "profil-ttft-coder30b-trace-concurrent.txt"
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(table)
        print(f"  [profil] trace ecrite dans {chemin}", flush=True)

    p1 = engine.stats.to_dict().get("prefill_seconds", 0.0) or 0.0
    forward_s = p1 - p0

    ttfts = [premiers[s.id] - t0 for s in seqs if premiers[s.id] is not None]
    manquants = sum(1 for s in seqs if premiers[s.id] is None)
    ttfts.sort()
    med = ttfts[len(ttfts) // 2] if ttfts else float("nan")
    hors_forward = med - forward_s
    pct_hors = 100 * hors_forward / med if med else float("nan")

    print(f"  concurrence {N} : {sum(len(p) for p in prompts_ids)} jetons prompt total, "
          f"TTFT median {med*1000:.1f} ms (min {ttfts[0]*1000:.1f}, max {ttfts[-1]*1000:.1f}), "
          f"forward (tout le lot) {forward_s*1000:.1f} ms, "
          f"hors-forward {hors_forward*1000:.1f} ms ({pct_hors:.1f} %), "
          f"lot fini en {t_fin - t0:.2f} s, {manquants} sequence(s) sans 1er jeton",
          flush=True)
    print(f"\n{'CONFIRME >70 %' if pct_hors > 70 else 'INFIRME <=70 %'} — cf. audit poste7")

    sortie = sorties() / "profil-ttft-coder30b" / "resultats-concurrent.json"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(
        {"concurrence": N, "ttft_median_s": med, "ttft_min_s": ttfts[0] if ttfts else None,
         "ttft_max_s": ttfts[-1] if ttfts else None, "forward_s": forward_s,
         "hors_forward_s": hors_forward, "pct_hors_forward": pct_hors,
         "manquants": manquants}, indent=2, ensure_ascii=False))
    print(f"\nFAIT / TESTE: {sortie} / RESTE: rien")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--profil-nsys", action="store_true",
                    help="capture aussi une trace torch.profiler sur la 1re requête")
    ap.add_argument("--concurrence", type=int, default=1,
                    help="nombre de sequences soumises ENSEMBLE avant le "
                    "premier pas (regime du chiffre de l'audit : 12 "
                    "sequences, ou-nous-sommes-10-09.md:66-70)")
    ap.add_argument("--invites-longues", action="store_true",
                    help="invites de ~350 mots reels (regime exact du duel, "
                    "poste4.md:3739-3750) au lieu des invites-jouets courtes")
    a = ap.parse_args()

    print("AUDIT poste7 A1 — TTFT Qwen3-Coder-30B-A3B-nvfp4, decompose par etape")
    print(f"  dossier {DOSSIER}")
    if not a.pour_de_vrai:
        print("\nPLAN SEULEMENT. Rien n'a tourne.")
        return 0

    if not DOSSIER.exists():
        print(f"ECHEC / CAUSE: dossier absent {DOSSIER}")
        return 2

    import torch

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer

    t0 = time.time()
    loaded = load_model(str(DOSSIER), dtype=torch.bfloat16,
                        device_override=a.device, max_model_len=4096)
    tokenizer = load_tokenizer(str(DOSSIER))
    if tokenizer is None:
        print("ECHEC / CAUSE: pas de tokenizer pour ce dossier")
        return 2
    print(f"  charge en {time.time() - t0:.1f} s", flush=True)

    engine = Engine(loaded, tokenizer, max_batch_size=max(1, a.concurrence),
                    max_model_len=4096)

    if a.concurrence > 1:
        return _mesurer_concurrent(engine, tokenizer, a)

    lignes = []
    for i, texte in enumerate(INVITES):
        messages = [{"role": "user", "content": texte}]

        t_gabarit0 = time.perf_counter()
        rendu = tokenizer.apply_chat_template(messages, True, {})
        t_gabarit = time.perf_counter() - t_gabarit0

        t_tok0 = time.perf_counter()
        ids = tokenizer.encode(rendu)
        t_tok = time.perf_counter() - t_tok0

        avant = engine.stats.to_dict()
        p0 = avant.get("prefill_seconds", 0.0) or 0.0

        profiler_ctx = None
        if a.profil_nsys and i == 0:
            profiler_ctx = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                           torch.profiler.ProfilerActivity.CUDA])
            profiler_ctx.__enter__()

        t_req0 = time.perf_counter()
        premier = None
        for out in engine.generate(ids, SamplingParams(temperature=0.0, max_tokens=8)):
            if premier is None:
                premier = time.perf_counter()
        t_fin = time.perf_counter()

        if profiler_ctx is not None:
            profiler_ctx.__exit__(None, None, None)
            table = profiler_ctx.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20)
            chemin = sorties() / "profil-ttft-coder30b-trace.txt"
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(table)
            n_sync = table.count("cudaStreamSynchronize") + table.count("cudaDeviceSynchronize")
            print(f"  [profil] trace ecrite dans {chemin}", flush=True)
            print(f"  [profil] occurrences synchronize dans le top 20 : {n_sync}", flush=True)

        apres = engine.stats.to_dict()
        p1 = apres.get("prefill_seconds", 0.0) or 0.0
        forward_s = p1 - p0

        ttft = (premier - t_req0) if premier is not None else float("nan")
        total = t_fin - t_req0
        hors_forward = ttft - forward_s
        pct_hors = 100 * hors_forward / ttft if ttft else float("nan")

        lignes.append(dict(gabarit=t_gabarit, tokenize=t_tok, ttft=ttft,
                           forward=forward_s, hors_forward=hors_forward,
                           pct_hors_forward=pct_hors, total=total,
                           n_tokens_prompt=len(ids)))
        print(f"  requete {i} : {len(ids)} jetons prompt, gabarit {t_gabarit*1000:.2f} ms, "
              f"tokenize {t_tok*1000:.2f} ms, TTFT {ttft*1000:.1f} ms, "
              f"forward {forward_s*1000:.1f} ms, hors-forward {hors_forward*1000:.1f} ms "
              f"({pct_hors:.1f} %)", flush=True)

    import json
    med = sorted(l["pct_hors_forward"] for l in lignes)[len(lignes) // 2]
    print(f"\nmediane hors-forward : {med:.1f} %  "
          f"({'CONFIRME >70 %' if med > 70 else 'INFIRME <=70 %'} — cf. audit poste7)")
    sortie = sorties() / "profil-ttft-coder30b" / "resultats.json"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps({"requetes": lignes, "mediane_pct_hors_forward": med},
                                 indent=2, ensure_ascii=False))
    print(f"\nFAIT / TESTE: {sortie} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
