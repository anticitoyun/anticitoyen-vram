#!/usr/bin/env python3
"""KL sous gabarit (pièce 52, 22/09) : KL(bf16 HF ‖ moteur) par position de la réponse, teacher forcing sur la
réponse gloutonne de HF, invites SOUS GABARIT de conversation — le seul protocole qui juge un Gemma 4 -it
(pièce 37 : sur texte brut, HF lui-même rend PPL 10^4).

  kl-gabarit.py hf SOURCE_BF16 DOSSIER            # processeur, à sec : 5 invites, réponse gloutonne, log-probs aux positions de réponse
  kl-gabarit.py acvram ALIAS DOSSIER              # carte : mêmes ids, logits aux positions de réponse, KL par position → DOSSIER/kl-<alias>.json
  kl-gabarit.py hidden ALIAS DOSSIER              # carte : flux résiduel après chaque couche aux positions de réponse → DOSSIER/hidden-<alias>.pt
  kl-gabarit.py divergence DOSSIER A B            # à sec : erreur relative par couche entre hidden-A et hidden-B (A = quantifié, B = bf16)

Le bras hf ne prend jamais la carte (CUDA_VISIBLE_DEVICES="" recommandé ; HF_THREADS). Les dumps sont la donnée commune :
aucun bras ne retokenise. Seuil (Coder, verdict-kl-qkv-nvfp4-22-09) : kl_max ≤ 1,0 par invite, 5/5.
"""
import json, os, sys, time
import torch

INVITES = [
    "Explique en quatre phrases ce qu'est une passerelle d'inférence pour des modèles de langage, et pourquoi la hiérarchie de mémoire (VRAM, RAM, disque) compte.",
    "Écris une fonction Python `mediane(xs)` qui rend la médiane d'une liste de nombres, avec une docstring en français et un exemple.",
    "Traduis en anglais : « Le convertisseur enregistre le modèle en double pour donner à chaque carte le format que son silicium lit le mieux. »",
    "Un train part à 8 h 15 et roule 2 h 50 à 120 km/h de moyenne. À quelle heure arrive-t-il et quelle distance a-t-il parcourue ? Réponds en deux phrases.",
    "Quelles sont les différences entre la quantification NVFP4 et INT8 pour les poids d'un modèle de langage ? Réponds en cinq points courts.",
]
REPONSE = int(os.environ.get("KL_REPONSE", "32"))


def mode_hf(source: str, dossier: str) -> None:
    torch.set_num_threads(int(os.environ.get("HF_THREADS", "8")))
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    m, info = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="cpu",
                                                          attn_implementation="eager", output_loading_info=True)
    m.eval()
    garde = {"classe": type(m).__name__, "manquantes": len(info.get("missing_keys", [])),
             "meta": sum(1 for p in m.parameters() if p.device.type == "meta")}
    assert not garde["manquantes"] and not garde["meta"], garde
    tk = AutoTokenizer.from_pretrained(source)
    os.makedirs(dossier, exist_ok=True)
    for i, q in enumerate(INVITES):
        t0 = time.time()
        ids = tk.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True)["input_ids"]
        n_inv = ids.shape[1]
        with torch.no_grad():
            g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=REPONSE, do_sample=False)
            out = m(input_ids=g, attention_mask=torch.ones_like(g))
        # log-probs aux positions qui PRÉDISENT chaque jeton de réponse : lignes n_inv-1 .. n_total-2
        lp = torch.log_softmax(out.logits[0, n_inv - 1: g.shape[1] - 1].float(), dim=-1)
        torch.save({"ids": g[0].tolist(), "n_invite": n_inv, "logprobs_hf": lp.to(torch.float16),
                    "reponse_texte": tk.decode(g[0, n_inv:], skip_special_tokens=False), "garde": garde},
                   os.path.join(dossier, f"invite{i}.pt"))
        print(json.dumps({"invite": i, "n_invite": n_inv, "n_reponse": int(g.shape[1] - n_inv), "s": round(time.time() - t0, 1),
                          "reponse": tk.decode(g[0, n_inv:], skip_special_tokens=True)[:120]}, ensure_ascii=False), flush=True)
    print("FINI hf", flush=True)


def _charger(alias: str):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    from racine_modeles import racine_modeles
    from acvram.engine.loader import load_model
    from acvram.server.chat import load_tokenizer
    chemin = alias if os.path.isdir(alias) else os.path.join(racine_modeles(), alias)
    tok = load_tokenizer(chemin)
    loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=1024, max_concurrent_seqs=1)
    return chemin, tok, loaded


def _forward(loaded, tok, ids: list[int], positions: torch.Tensor):
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    eng = Engine(loaded, tok, max_batch_size=1, max_model_len=len(ids) + 32, enable_cuda_graphs=False)
    eng.pipeline_actif = False
    eng.add_request(list(ids), SamplingParams(temperature=0.0, max_tokens=1), request_id="kl")
    eng._admit()
    batch = eng._build_batch([eng.running[0]], prefill=True)
    with torch.inference_mode():
        out = loaded.model(batch, logits_positions=positions).float()
    # UN Engine par invite garde son cache KV tant que le ramasse-miettes ne passe pas (cycles) :
    # sur l alias attention-int8 (+4 Gio) la 3e invite tombait en OOM (23/09). On libère tout de suite.
    del eng, batch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def mode_acvram(alias: str, dossier: str) -> None:
    chemin, tok, loaded = _charger(alias)
    res = {"alias": chemin, "regime": loaded.model.regime_ligne() if hasattr(loaded.model, "regime_ligne") else None, "invites": []}
    for i in range(len(INVITES)):
        d = torch.load(os.path.join(dossier, f"invite{i}.pt"), weights_only=False)
        ids, n_inv = d["ids"], d["n_invite"]
        pos = torch.arange(n_inv - 1, len(ids) - 1)
        lg = _forward(loaded, tok, ids, pos)
        lp = torch.log_softmax(lg, dim=-1).cpu()
        lph = d["logprobs_hf"].float()
        ph = lph.exp()
        kl = (ph * (lph - lp)).sum(-1)                              # KL(hf ‖ acvram) par position
        egal = (lp.argmax(-1) == lph.argmax(-1))
        cibles = torch.tensor(ids[n_inv:])
        nll = -lp.gather(1, cibles.unsqueeze(1)).squeeze(1)
        r = {"invite": i, "n_reponse": len(cibles), "kl_par_pas": [round(float(x), 4) for x in kl], "kl_max": round(float(kl.max()), 4),
             "kl_moy": round(float(kl.mean()), 4), "argmax_egaux": f"{int(egal.sum())}/{len(cibles)}",
             "nll_reponse_moy": round(float(nll.mean()), 4), "pas_kl_max": int(kl.argmax())}
        res["invites"].append(r)
        print(json.dumps({k: r[k] for k in ("invite", "kl_max", "kl_moy", "argmax_egaux", "nll_reponse_moy", "pas_kl_max")}), flush=True)
    res["kl_max_global"] = max(r["kl_max"] for r in res["invites"])
    res["tenu_5_sur_5"] = sum(1 for r in res["invites"] if r["kl_max"] <= 1.0)
    nom = os.path.basename(chemin.rstrip("/"))
    json.dump(res, open(os.path.join(dossier, f"kl-{nom}.json"), "w"), ensure_ascii=False, indent=1)
    print("RESULTAT " + json.dumps({"alias": nom, "kl_max_global": res["kl_max_global"], "tenu": f"{res['tenu_5_sur_5']}/5",
                                   "kl_moy_par_invite": [r["kl_moy"] for r in res["invites"]]}), flush=True)


def mode_hidden(alias: str, dossier: str) -> None:
    """Flux résiduel après chaque couche (sortie du module `layers[i]`), aux positions de réponse, en fp32."""
    chemin, tok, loaded = _charger(alias)
    couches = loaded.model.layers
    capt: dict[int, torch.Tensor] = {}
    pos_courantes = {"pos": None}

    def crochet(i):
        def f(mod, entree, sortie):
            h = sortie[0] if isinstance(sortie, tuple) else sortie
            capt[i] = h.reshape(-1, h.shape[-1])[pos_courantes["pos"]].detach().float().cpu()
        return f
    poignees = [couches[i].register_forward_hook(crochet(i)) for i in range(len(couches))]
    sortie = {}
    for i in range(len(INVITES)):
        d = torch.load(os.path.join(dossier, f"invite{i}.pt"), weights_only=False)
        ids, n_inv = d["ids"], d["n_invite"]
        pos = torch.arange(n_inv - 1, len(ids) - 1)
        pos_courantes["pos"] = pos
        capt.clear()
        _forward(loaded, tok, ids, pos)
        sortie[i] = {k: v.clone() for k, v in capt.items()}
        print(json.dumps({"invite": i, "couches_captees": len(capt), "forme": list(next(iter(capt.values())).shape)}), flush=True)
    for p in poignees:
        p.remove()
    nom = os.path.basename(chemin.rstrip("/"))
    torch.save(sortie, os.path.join(dossier, f"hidden-{nom}.pt"))
    print("FINI hidden " + nom, flush=True)


def mode_divergence(dossier: str, a: str, b: str) -> None:
    A = torch.load(os.path.join(dossier, f"hidden-{a}.pt"), weights_only=False)
    B = torch.load(os.path.join(dossier, f"hidden-{b}.pt"), weights_only=False)
    couches = sorted(A[0].keys())
    lignes = []
    for c in couches:
        num = sum(float((A[i][c] - B[i][c]).norm()) ** 2 for i in A)
        den = sum(float(B[i][c].norm()) ** 2 for i in A)
        lignes.append((c, (num / den) ** 0.5))
    # erreur relative cumulée par couche, et saut (accroissement) par couche : les couches où le saut est le plus grand sont les sensibles
    prev = 0.0
    sauts = []
    for c, err in lignes:
        sauts.append((c, err, err - prev)); prev = err
    print("couche  err_rel  saut")
    for c, err, s in sauts:
        print(f"{c:6d}  {err:7.4f}  {s:+8.4f}")
    pires = sorted(sauts, key=lambda t: -t[2])[:8]
    print("RESULTAT " + json.dumps({"err_rel_finale": round(sauts[-1][1], 4), "pires_sauts": [(c, round(s, 4)) for c, _, s in pires]}))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "hf":
        mode_hf(sys.argv[2], sys.argv[3])
    elif mode == "acvram":
        mode_acvram(sys.argv[2], sys.argv[3])
    elif mode == "hidden":
        mode_hidden(sys.argv[2], sys.argv[3])
    elif mode == "divergence":
        mode_divergence(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
