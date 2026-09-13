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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--profil-nsys", action="store_true",
                    help="capture aussi une trace torch.profiler sur la 1re requête")
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

    engine = Engine(loaded, tokenizer, max_batch_size=1, max_model_len=4096)

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
            chemin = Path("~/Bureau/Claude/acvram-memoire"
                         "/corpus/profil-ttft-coder30b-trace.txt")
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
    sortie = Path("~/Bureau/Claude/acvram-memoire"
                 "/corpus/profil-ttft-coder30b/resultats.json")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps({"requetes": lignes, "mediane_pct_hors_forward": med},
                                 indent=2, ensure_ascii=False))
    print(f"\nFAIT / TESTE: {sortie} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
