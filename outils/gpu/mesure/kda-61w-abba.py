"""61w (poste1, 01/10) : état KDA en [K, V] — équivalence de bout en bout et temps par pas, ancien chemin (A) contre nouveau (B).
Pas d'interrupteur dans le code : A et B sont deux arbres figés (A = main avant la fusion de poste5-kda, B = la fusion) ; ce pilote
tourne dans l'arbre que PYTHONPATH désigne, et le dit (acvram.__file__).

  bras     (sous carte.sh, une fois par bras ABBA) : charge Kimi-Linear-35B (ACVRAM_MODELE_MESURE), régime servi (graphes),
           1) ids gloutons : b=1 (une invite de 256) puis b=12 (12 invites), 2 048 jetons chacune, eos coupé ;
           2) temps par pas (mur, perf_counter + synchronize aux bornes) : b=1 et b=12, 3 × 128 pas après 32 de chauffe.
           Écrit SORTIE.json : {"arbre", "ids": {"b1": [[…]], "b12": [[…]×12]}, "mur_ms": {"b1": [3], "b12": [3]}}.
  comparer A1.json B1.json B2.json A2.json : témoin A1 = A2 et B1 = B2 (sinon l'instrument n'est pas déterministe, rc 6, rien
           d'autre), puis ids A contre B (premier jeton divergent par séquence), puis temps (médiane des 6 répétitions par chemin,
           ABBA), jugés contre la prédiction scellée (revue/poste1-p81-cake-kda-01-10.md, « Levier ») :
           b=12 mur 11,07 → 9,3-9,6 ms (−13 à −16 %), FAUX si le gain < 1,0 ms ; b=1 mur −9 à −11 %, noyau KDA ≤ 0,25 ms.
"""
import json
import os
import statistics
import sys
import time

# Le bras importe acvram de SON arbre (ACVRAM_ARBRE, contrôlé par acvram._garde_arbre) ; sans lui, la racine de CE fichier (pièce 211)
sys.path.insert(0, os.environ.get("ACVRAM_ARBRE") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

N_GEN, INVITE, CHAUFFE, PAS, REPS = 2048, 256, 32, 128, 3


def bras(sortie: str) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    import acvram
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    if not torch.cuda.is_available():
        sys.exit("kda-61w-abba : carte requise (sous outils/carte.sh)")
    modele = os.environ["ACVRAM_MODELE_MESURE"]
    ctx = INVITE + N_GEN + 64
    loaded = load_model(modele, dtype=torch.bfloat16, max_model_len=ctx, max_concurrent_seqs=12)
    vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
    eng = Engine(loaded, None, max_batch_size=12, max_model_len=ctx, enable_cuda_graphs=True)
    eng._eos = set()
    eng.warm_graphs()
    print(eng.regime_ligne(), flush=True)
    from acvram import kernels
    pre = kernels.build_info().get("precompile") or ""
    if not pre:                                  # chef 01/10 : aucun JIT pendant l'ABBA (une compilation fausserait le bras)
        sys.exit("kda-61w-abba : extension NON précompilée (JIT) — ACVRAM_KERNELS_PRECOMPILES ne couvre pas ce .cu")
    print(f"PRECOMPILE {pre}", flush=True)

    def invite(k):
        return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(INVITE)]

    def generer(b):
        # add_request rend la Sequence ; GenerationOutput.token_ids est INCRÉMENTAL (un jeton par pas, runner.py:2424)
        seqs = [eng.add_request(invite(s), SamplingParams(temperature=0.0, max_tokens=N_GEN)) for s in range(b)]
        acc = {q.id: [] for q in seqs}
        while eng.running or eng.waiting:
            for o in eng.step():
                if o.sequence_id in acc:
                    acc[o.sequence_id].extend(int(t) for t in o.token_ids)
        return [acc[q.id] for q in seqs]

    def chrono(b):
        reps = []
        for s in range(b):
            eng.add_request(invite(100 + s), SamplingParams(temperature=0.0, max_tokens=CHAUFFE + REPS * PAS + 8))
        for _ in range(CHAUFFE):
            eng.step()
        for _ in range(REPS):
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(PAS):
                eng.step()
            torch.cuda.synchronize(); reps.append((time.perf_counter() - t) * 1e3 / PAS)
        while eng.running or eng.waiting:
            eng.step()
        return reps

    res = {"arbre": os.path.dirname(os.path.dirname(os.path.realpath(acvram.__file__))),
           "regime": eng.regime_ligne(), "precompile": pre, "ids": {"b1": generer(1), "b12": generer(12)},
           "mur_ms": {"b1": chrono(1), "b12": chrono(12)}}
    json.dump(res, open(sortie, "w"))
    print(f"bras {res['arbre']} : b1 {len(res['ids']['b1'][0])} jetons, b12 {[len(x) for x in res['ids']['b12']][:3]}… ; "
          f"mur b1 {res['mur_ms']['b1']} b12 {res['mur_ms']['b12']}", flush=True)
    return 0


def _divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def comparer(a1: str, b1: str, b2: str, a2: str) -> int:
    A1, B1, B2, A2 = (json.load(open(p)) for p in (a1, b1, b2, a2))
    if A1["arbre"] != A2["arbre"] or B1["arbre"] != B2["arbre"] or A1["arbre"] == B1["arbre"]:
        print(f"ARBRES incohérents : A {A1['arbre']}/{A2['arbre']} B {B1['arbre']}/{B2['arbre']}"); return 5
    for nom, x, y in (("A1/A2", A1, A2), ("B1/B2", B1, B2)):
        if x["ids"] != y["ids"]:
            print(f"TÉMOIN {nom} ≠ : l'instrument n'est pas déterministe, aucun verdict d'équivalence ni de temps"); return 6
    print("TÉMOINS A1 = A2 et B1 = B2 au jeton près")
    tout = True
    for b in ("b1", "b12"):
        for s, (x, y) in enumerate(zip(A1["ids"][b], B1["ids"][b])):
            d = _divergence(x, y)
            if d is not None:
                tout = False
            print(f"  {b} séq {s:2d} : {len(x)} jetons, " + ("IDENTIQUES" if d is None else f"PREMIÈRE DIVERGENCE au jeton {d}"))
    print("ÉQUIVALENCE : ids gloutons IDENTIQUES sur 2 048 jetons, b=1 et b=12" if tout else
          "ÉQUIVALENCE : ids DIVERGENTS (voir ci-dessus) — décision du chef")
    for b, (lo, hi), plancher in (("b1", (-11.0, -9.0), None), ("b12", (-16.0, -13.0), 1.0)):
        ma = statistics.median(A1["mur_ms"][b] + A2["mur_ms"][b]); mb = statistics.median(B1["mur_ms"][b] + B2["mur_ms"][b])
        d = (mb / ma - 1) * 100
        ecart = (max(A1["mur_ms"][b] + A2["mur_ms"][b]) - min(A1["mur_ms"][b] + A2["mur_ms"][b])) / ma * 100
        lecture = "dans la bande" if lo <= d <= hi else ("au-delà (mieux)" if d < lo else "en deçà")
        faux = plancher is not None and (ma - mb) < plancher
        print(f"  {b} mur : A {ma:.3f} ms, B {mb:.3f} ms, Δ {d:+.1f} % (prédit {lo:+.0f} à {hi:+.0f} %) — {lecture}"
              f"{' — FAUX (gain < 1,0 ms)' if faux else ''} ; étendue témoin A {ecart:.1f} %")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["bras"] and len(sys.argv) == 3:
        sys.exit(bras(sys.argv[2]))
    if sys.argv[1:2] == ["comparer"] and len(sys.argv) == 6:
        sys.exit(comparer(*sys.argv[2:6]))
    sys.exit(__doc__)
