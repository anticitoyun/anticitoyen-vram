#!/usr/bin/env python
"""Banc du pas d échantillonnage seul, sur carte, forme réelle du b=12 Coder : logits [12, 151 936] bf16 (et fp32).
Tranche la cause du verdict eea064fe (sampler vectorisé −2,2 % à b=12) SANS le service : si le vectorisé est plus lent
ICI, la cause est dans ses noyaux (H1 : argmax bf16 non vectorisé, H3 : zeros_like) ; s il est plus rapide ici, la perte
vient de l interaction avec le pipeline (H2 : frontière liée au CPU, ordre des lancements) et se lit au nsys, pas ici.

    ACVRAM_TYPE=mesure outils/carte.sh env CUDA_VISIBLE_DEVICES=0 python outils/gpu/mesure/banc-sampler.py   # ≤ 1 min

Mesure par événements CUDA, 200 répétitions après 20 de chauffe, médiane en µs ; lancements comptés par torch.profiler.
Imprime `RESULTAT {...}` ; aucune sortie de modèle (pas de texte)."""
import os, sys, json, statistics
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from acvram.engine.sampler import SamplingParams, sample, _sample_lent   # noqa: E402

B, V, N, CHAUFFE = int(os.environ.get("LOT", 12)), int(os.environ.get("VOCAB", 151936)), 200, 20


def chrono(fn) -> float:
    for _ in range(CHAUFFE):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(N):
        d, f = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d.record(); fn(); f.record(); f.synchronize()
        t.append(d.elapsed_time(f) * 1000)
    return round(statistics.median(t), 1)


def lancements(fn) -> int:
    """Nombre de noyaux CUDA lancés par un appel (torch.profiler, événements de type CUDA)."""
    from torch.profiler import profile, ProfilerActivity
    from torch._C._autograd import DeviceType
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return sum(1 for e in p.events() if e.device_type == DeviceType.CUDA)


def main() -> int:
    if not torch.cuda.is_available():
        return sec()
    torch.manual_seed(0)
    lbf = (torch.randn(B, V, device="cuda") * 4).to(torch.bfloat16)
    l32 = lbf.to(torch.float32)
    glouton = [SamplingParams(temperature=0.0)] * B
    hist = [() for _ in glouton]
    os.environ["ACVRAM_SAMPLER_LOT"] = "1"
    r = {"lot": B, "vocab": V, "us": {}, "lancements": {}}
    cas = {
        "lent_glouton_bf16": lambda: _sample_lent(lbf, glouton, hist),
        "lot_glouton_bf16": lambda: sample(lbf, glouton, hist),
        "lent_glouton_fp32": lambda: _sample_lent(l32, glouton, hist),
        "lot_glouton_fp32": lambda: sample(l32, glouton, hist),
        "argmax_bf16_seul": lambda: lbf.argmax(dim=-1),
        "cast_fp32_puis_argmax": lambda: lbf.to(torch.float32).argmax(dim=-1),
        "argmax_fp32_seul": lambda: l32.argmax(dim=-1),
        "zeros_like": lambda: torch.zeros(B, dtype=torch.float32, device="cuda"),
    }
    for nom, fn in cas.items():
        r["us"][nom] = chrono(fn)
        try:
            r["lancements"][nom] = lancements(fn)
        except Exception as e:                       # noqa: BLE001
            r["lancements"][nom] = f"profil indisponible : {type(e).__name__}"
    # échantillonné (température 0,8, top_p 0,9) : le chemin qui motivait la vectorisation, pour mémoire
    echant = [SamplingParams(temperature=0.8, top_p=0.9)] * B
    g1, g2 = torch.Generator(device="cuda").manual_seed(1), torch.Generator(device="cuda").manual_seed(1)
    r["us"]["lent_echantillonne_bf16"] = chrono(lambda: _sample_lent(lbf, echant, hist, generator=g1))
    r["us"]["lot_echantillonne_bf16"] = chrono(lambda: sample(lbf, echant, hist, generator=g2))
    a, b = r["us"]["lent_glouton_bf16"], r["us"]["lot_glouton_bf16"]
    r["delta_glouton_bf16_us"] = round(b - a, 1)
    r["verdict"] = ("H1/H3 : le vectorisé est plus lent DANS SES NOYAUX (%.1f µs de plus par pas)" % (b - a) if b > a + 5 else
                    "H2 : le vectorisé est plus rapide ou égal ici (%.1f µs) — la perte de 2,2 %% vient de l interaction avec le pipeline, à lire au nsys" % (b - a))
    print("RESULTAT " + json.dumps(r, ensure_ascii=False))
    return 0


def sec() -> int:
    """Sans carte : les mêmes appels sur processeur, forme réduite — prouve que le banc tourne, aucun chiffre publié."""
    import time
    lbf = (torch.randn(B, 4096) * 4).to(torch.bfloat16); glouton = [SamplingParams(temperature=0.0)] * B; hist = [() for _ in glouton]
    os.environ["ACVRAM_SAMPLER_LOT"] = "1"
    t0 = time.perf_counter(); a = _sample_lent(lbf, glouton, hist)[0]; b = sample(lbf, glouton, hist)[0]
    assert torch.equal(a, b)
    print("RESULTAT sec : banc exécuté sur processeur (%.1f ms), ids égaux ; les chiffres viennent de la carte" % ((time.perf_counter() - t0) * 1000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
