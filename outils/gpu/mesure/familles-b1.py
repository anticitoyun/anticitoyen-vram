"""Pièce 66 : familles de noyaux d'une trace nsys (cuda_gpu_trace.csv) en µs/pas, hôte compris (mur = fenêtre ÷ pas),
pour acvram OU llama.cpp (--moteur). Fenêtre = dernier train continu de pas (marqueur : 1 lancement par couche),
tranches de 50 ms ; pas = marqueurs ÷ couches. Ce qu'aucun motif ne reconnaît est listé sous « autres », jamais tu.
Usage : familles-b1.py <csv> --moteur acvram|llamacpp [--couches 48] [--top 30] [--json f]"""
import argparse, collections, csv, json, re

MOTEURS = {
    "acvram": dict(marqueur="_route_fusee_kernel", familles=[
        ("experts", r"nvfp4_gemv_marlin|nvfp4_gemv_grouped|marlin_moe|moe_wna16|Marlin"),
        ("experts_glue", r"moe_reduce|moe_act|moe_aligner|moe_align|moe_slots"),
        ("routage", r"_route|moe_route|routeur|topk"),
        ("projections", r"_etroit|int8_gemv|nvfp4_gemv\b|nvfp4_gemv<|_quant_a8|_epilogue_i8c|narrow|dense_nvfp4"),
        ("tete", r"lm_head|tete|logits|sampler|argmax|softmax_|penal|gumbel"),
        ("attention", r"fmha|flash|paged|attn|attention"),
        ("normes", r"rmsnorm|rms_norm|layer_norm"),
        ("rope_kv", r"rope|kv_write|kv_cache|reshape_and_cache"),
        ("copies", r"CUDA memcpy|CUDA memset|Memcpy|Memset"),
        ("glue_torch", r"at::native|elementwise|vectorized|reduce_kernel|index|CatArray|arange|fill"),
    ]),
    "llamacpp": dict(marqueur="rms_norm", familles=[      # rms_norm : 2 par couche (+1 final) → couches × 2 par pas (voir --marqueur-par-couche)
        ("experts", r"mul_mat_vec_q.*ids|mul_mat_id|moe|mmvq.*id"),
        ("experts_glue", r"topk|argsort|k_get_rows|get_rows|expert|scale_f32|k_add_id"),
        ("projections", r"mul_mat_vec_q|dequantize_mul_mat_vec|mul_mat_q|mmvq|mmq|gemv"),
        ("tete", r"argmax|soft_max|softmax|sampl"),
        ("attention", r"flash_attn|fattn|attn|attention"),
        ("normes", r"rms_norm|norm_f32"),
        ("rope_kv", r"rope|cpy_f32|set_rows|k_set_rows"),
        ("copies", r"CUDA memcpy|CUDA memset|Memcpy|Memset"),
        ("glue", r"k_bin_bcast|unary|silu|swiglu|mul_f32|add_f32|cpy|k_compute|scale|clamp|cont|transpose"),
    ]),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--moteur", required=True, choices=MOTEURS)
    ap.add_argument("--couches", type=int, default=48); ap.add_argument("--marqueur"); ap.add_argument("--marqueur-par-couche", type=int, default=1)
    ap.add_argument("--top", type=int, default=30); ap.add_argument("--json"); ap.add_argument("--min-marq", type=int, default=24)
    a = ap.parse_args(); M = MOTEURS[a.moteur]; marq_nom = a.marqueur or M["marqueur"]
    rows = list(csv.DictReader(open(a.csv)))
    k = [c for c in rows[0] if "Name" in c][0]; d = [c for c in rows[0] if c.startswith("Duration")][0]; s = [c for c in rows[0] if c.startswith("Start")][0]
    g = [c for c in rows[0] if c.startswith("Grd")] or []
    rows.sort(key=lambda r: int(r[s]))
    # contexte CUDA du moteur = celui qui lance le marqueur ; les autres contextes (client, sondes) sont comptés à part
    ctx_moteur = collections.Counter(r.get("Ctx", "") for r in rows if marq_nom in r[k]).most_common(1)[0][0]
    marq = [int(r[s]) for r in rows if marq_nom in r[k] and r.get("Ctx", "") == ctx_moteur]
    assert marq, f"aucun marqueur {marq_nom!r}"
    # fenêtre = TOUTES les tranches de 50 ms en décodage établi (≥ min_marq marqueurs) : les pauses entre lots
    # (préremplissage, client) sont exclues, mur = tranches × 50 ms ÷ pas
    bin_ns = 50_000_000; bins = collections.Counter(t // bin_ns for t in marq)
    pleines = {b for b, n in bins.items() if n >= a.min_marq}
    hors = [r for r in rows if int(r[s]) // bin_ns in pleines and r.get("Ctx", "") != ctx_moteur]
    rows = [r for r in rows if int(r[s]) // bin_ns in pleines and r.get("Ctx", "") == ctx_moteur]
    n_marq = sum(1 for r in rows if marq_nom in r[k]); pas = n_marq / (a.couches * a.marqueur_par_couche)
    t0, t1 = 0, len(pleines) * bin_ns
    autres_ctx = sum(float(r[d]) for r in hors) / 1e3 / pas if pas else 0
    def fam(nom):
        for f, m in M["familles"]:
            if re.search(m, nom): return f
        return "autres"
    par = collections.defaultdict(lambda: [0.0, 0]); noms = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        f = fam(r[k]); dur = float(r[d]) / 1e3    # ns → µs
        grille = "x".join(r[c] for c in g[:3]) if g else ""
        par[f][0] += dur; par[f][1] += 1
        cle = (f, re.sub(r"\(.*", "", r[k])[:80], grille); noms[cle][0] += dur; noms[cle][1] += 1
    mur = (t1 - t0) / 1e3 / pas; noy = sum(v[0] for v in par.values()) / pas
    res = {"moteur": a.moteur, "ctx": ctx_moteur, "pas": round(pas, 1), "fenetre_ms": round((t1 - t0) / 1e6, 1), "mur_us_pas": round(mur, 1),
           "autres_contextes_us_pas": round(autres_ctx, 1), "lancements_autres_ctx_pas": round(len(hors) / pas, 1),
           "noyaux_us_pas": round(noy, 1), "hors_noyaux_us_pas": round(mur - noy, 1), "lancements_pas": round(sum(v[1] for v in par.values()) / pas, 1),
           "familles": {f: {"us_pas": round(v[0] / pas, 1), "lancements_pas": round(v[1] / pas, 1)} for f, v in par.items()}}
    print(json.dumps({x: res[x] for x in res if x != "familles"}, ensure_ascii=False))
    for f, v in sorted(par.items(), key=lambda kv: -kv[1][0]):
        print(f"  {f:14s} {v[0] / pas:8.1f} µs/pas {v[1] / pas:7.1f} l/pas")
    print(f"  -- noyaux (top {a.top}) --")
    for (f, nom, gr), (us, n) in sorted(noms.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"    {f:13s} {nom:80s} [{gr:>14s}] {us / pas:8.1f} µs {n / pas:6.1f} × {us / n:6.1f} µs")
    if a.json: json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)
main()
