#!/usr/bin/env python3
"""Pièce 125 (24/09, poste6) : lequel des deux chemins d'acvram est le plus juste — le PRÉFILL (GEMM, attention flash
sur K/V bf16) ou le DÉCODAGE (GEMV, attention paginée sur KV int8) — contre une référence HF bf16 sur processeur.
poste5 (revue/poste5-rust-prefill-kl-24-09.md) a mesuré qu'ils remplissent le KV de l'invite différemment (32-51 % des
codes K, jusqu'à 19 % L2 sur V dès la couche 1) sans dire lequel a raison : ici chaque chemin est jugé contre HF.

  kl-chemins-p125.py hf SOURCE_BF16 DOSSIER          # à sec (CUDA_VISIBLE_DEVICES=""), bf16 eager sur processeur :
                                                     #   5 invites, réponse gloutonne de HF, log-probs aux positions de
                                                     #   réponse, flux résiduel après CHAQUE couche à TOUTES les positions
  kl-chemins-p125.py prefill ALIAS DOSSIER           # carte : mêmes ids, chemin de préfill servi (un forward), logits +
                                                     #   flux résiduel par couche → prefill-<alias>-<i>.pt
  kl-chemins-p125.py force ALIAS DOSSIER [graphes]   # carte : mêmes ids, chemin de DÉCODAGE (1er jeton préfillé, puis
                                                     #   chaque jeton FORCÉ un par un, comme vidage_decode_force.py) ;
                                                     #   eager par défaut (les crochets ne voient pas un graphe rejoué) ;
                                                     #   « graphes » = témoin servi sans crochets, logits seuls
  kl-chemins-p125.py compare DOSSIER ALIAS           # à sec : KL(hf ‖ chemin) par position de réponse, erreur L2 relative
                                                     #   par couche (positions d'invite / de réponse), préfill contre
                                                     #   forcé, témoin graphes contre eager (0 ulp attendu)
Les dumps HF sont la donnée commune : aucun bras ne retokenise. Invites = celles de kl-gabarit.py (pièce 52).
Seuils et lecture : revue/poste6-piece125-scelle-24-09.md (écrit avant la mesure)."""
import gc
import json
import os
import sys
import time

import torch

INVITES = [
    "Explique en quatre phrases ce qu'est une passerelle d'inférence pour des modèles de langage, et pourquoi la hiérarchie de mémoire (VRAM, RAM, disque) compte.",
    "Écris une fonction Python `mediane(xs)` qui rend la médiane d'une liste de nombres, avec une docstring en français et un exemple.",
    "Traduis en anglais : « Le convertisseur enregistre le modèle en double pour donner à chaque carte le format que son silicium lit le mieux. »",
    "Un train part à 8 h 15 et roule 2 h 50 à 120 km/h de moyenne. À quelle heure arrive-t-il et quelle distance a-t-il parcourue ? Réponds en deux phrases.",
    "Quelles sont les différences entre la quantification NVFP4 et INT8 pour les poids d'un modèle de langage ? Réponds en cinq points courts.",
]
REPONSE = int(os.environ.get("KL_REPONSE", "32"))


def _crochets(couches, capt: dict, quoi):
    """Un crochet par couche : `quoi(sortie)` → tenseur [n, D] fp32 sur l'hôte, accumulé dans capt[i] (liste)."""
    def f(i):
        def g(mod, entree, sortie):
            h = sortie[0] if isinstance(sortie, tuple) else sortie
            capt.setdefault(i, []).append(quoi(h).detach().float().cpu())
        return g
    return [couches[i].register_forward_hook(f(i)) for i in range(len(couches))]


def _empiler(capt: dict) -> dict:
    return {i: torch.cat(v, 0) for i, v in capt.items()}


# ───────────────────────────── référence HF, à sec ─────────────────────────────
def mode_hf(source: str, dossier: str) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "bras hf : CUDA_VISIBLE_DEVICES=\"\" obligatoire (REGLES § 1)"
    torch.set_num_threads(int(os.environ.get("HF_THREADS", "8")))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m, info = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16,               # sans device_map : accelerate absent du venv (24/09)
                                                   attn_implementation="eager", output_loading_info=True)
    m.eval()
    garde = {"classe": type(m).__name__, "manquantes": len(info.get("missing_keys", [])),
             "meta": sum(1 for p in m.parameters() if p.device.type == "meta"), "couches": len(m.model.layers)}
    assert not garde["manquantes"] and not garde["meta"], garde
    tk = AutoTokenizer.from_pretrained(source)
    os.makedirs(dossier, exist_ok=True)
    for i, q in enumerate(INVITES):
        t0 = time.time()
        msgs = [{"role": "user", "content": q}]
        try:    # Qwen3 -it : pas de bloc de réflexion, la réponse commence tout de suite (le Coder ignore l'option)
            ids = tk.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                         enable_thinking=False)["input_ids"]
        except TypeError:
            ids = tk.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
        n_inv = ids.shape[1]
        capt: dict = {}
        with torch.no_grad():
            g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=REPONSE, do_sample=False)
            poignees = _crochets(m.model.layers, capt, lambda h: h[0])
            out = m(input_ids=g, attention_mask=torch.ones_like(g))
            for p in poignees:
                p.remove()
        lp = torch.log_softmax(out.logits[0, n_inv - 1: g.shape[1] - 1].float(), dim=-1)
        torch.save({"ids": g[0].tolist(), "n_invite": n_inv, "logprobs_hf": lp, "hidden": _empiler(capt),
                    "reponse_texte": tk.decode(g[0, n_inv:], skip_special_tokens=False), "garde": garde, "source": source},
                   os.path.join(dossier, f"invite{i}.pt"))
        print(json.dumps({"invite": i, "n_invite": n_inv, "n_reponse": int(g.shape[1] - n_inv), "s": round(time.time() - t0, 1),
                          "reponse": tk.decode(g[0, n_inv:], skip_special_tokens=True)[:100]}, ensure_ascii=False), flush=True)
    print("FINI hf " + json.dumps(garde), flush=True)


# ───────────────────────────── acvram, carte ─────────────────────────────
def _charger(alias: str):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    from racine_modeles import racine_modeles
    from acvram.engine.loader import load_model
    from acvram.server.chat import load_tokenizer
    chemin = alias if os.path.isdir(alias) else os.path.join(racine_modeles(), alias)
    tok = load_tokenizer(chemin)
    loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=1024, max_concurrent_seqs=1)
    return chemin, tok, loaded


def _liberer(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def _nom(chemin: str) -> str:
    return os.path.basename(chemin.rstrip("/"))


def mode_prefill(alias: str, dossier: str) -> None:
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    chemin, tok, loaded = _charger(alias)
    regime = loaded.model.regime_ligne() if hasattr(loaded.model, "regime_ligne") else None
    for i in range(len(INVITES)):
        d = torch.load(os.path.join(dossier, f"invite{i}.pt"), weights_only=False)
        ids, n_inv, L = d["ids"], d["n_invite"], len(d["ids"])
        pos = torch.arange(n_inv - 1, L - 1)
        eng = Engine(loaded, tok, max_batch_size=1, max_model_len=L + 32, enable_cuda_graphs=False)
        eng.pipeline_actif = False
        eng.add_request(list(ids), SamplingParams(temperature=0.0, max_tokens=1), request_id="p125")
        eng._admit()
        batch = eng._build_batch([eng.running[0]], prefill=True)
        capt: dict = {}
        poignees = _crochets(loaded.model.layers, capt, lambda h: h.reshape(-1, h.shape[-1])[:L])
        with torch.inference_mode():
            lg = loaded.model(batch, logits_positions=pos).float()
        for p in poignees:
            p.remove()
        lp = torch.log_softmax(lg, dim=-1).cpu()
        torch.save({"logprobs": lp, "hidden": _empiler(capt), "regime": regime, "alias": chemin, "chemin": "prefill"},
                   os.path.join(dossier, f"prefill-{_nom(chemin)}-{i}.pt"))
        print(json.dumps({"invite": i, "chemin": "prefill", "couches": len(capt), "positions": L}), flush=True)
        _liberer(eng, batch, lg)
    print("FINI prefill " + _nom(chemin), flush=True)


def mode_force(alias: str, dossier: str, graphes: bool) -> None:
    """Le moteur SERVI remplit le KV par son chemin de décodage : invite réduite à son 1er jeton, puis chaque jeton
    (invite ET réponse de HF) forcé à la place de l'échantillon (`_sample_only`, comme vidage_decode_force.py)."""
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    chemin, tok, loaded = _charger(alias)
    regime = loaded.model.regime_ligne() if hasattr(loaded.model, "regime_ligne") else None
    etat = {"force": [], "logits": []}
    _orig = Engine._sample_only

    def forcer(self, logits, seqs, depuis_graphe=False):
        jetons, lp = _orig(self, logits, seqs, depuis_graphe)
        etat["logits"].append(logits.reshape(-1, logits.shape[-1])[0].detach().float().cpu())
        if etat["force"]:
            jetons = jetons.clone()
            jetons.view(-1)[0] = etat["force"].pop(0)
        return jetons, lp
    Engine._sample_only = forcer
    try:
        for i in range(len(INVITES)):
            d = torch.load(os.path.join(dossier, f"invite{i}.pt"), weights_only=False)
            ids, n_inv, L = d["ids"], d["n_invite"], len(d["ids"])
            eng = Engine(loaded, tok, max_batch_size=1, max_model_len=L + 32, enable_prefix_cache=False,
                         speculator=None, enable_cuda_graphs=graphes)
            if graphes:
                eng.demarrer_service(strict=False, warm_max_len=L + 32)
            eng.pipeline_actif = False        # sans pipeline, le lot suivant se bâtit sur seq.output_ids (hôte) : forçable
            etat["force"], etat["logits"] = list(ids[1:]), []
            capt: dict = {}
            poignees = [] if graphes else _crochets(loaded.model.layers, capt, lambda h: h.reshape(-1, h.shape[-1])[-1:])
            seq = eng.add_request(list(ids[:1]), SamplingParams(temperature=0.0, max_tokens=L, ignore_eos=True), request_id="p125")
            while not seq.finished and len(seq.output_ids) < L - 1:
                eng.step()
            for p in poignees:
                p.remove()
            obtenu = list(seq.output_ids[:L - 1])
            if obtenu != list(ids[1:]):
                j = next((k for k, (a, b) in enumerate(zip(obtenu, ids[1:])) if a != b), min(len(obtenu), L - 1))
                raise SystemExit(f"forçage non tenu (invite {i}) : {len(obtenu)} jetons, 1re divergence {j}, "
                                 f"obtenu {obtenu[j:j + 3]} attendu {list(ids[1:])[j:j + 3]}")
            lg = torch.stack(etat["logits"][: L - 1])                      # logits après le jeton p, p = 0..L-2
            assert lg.shape[0] == L - 1, (lg.shape, L)
            lp = torch.log_softmax(lg[n_inv - 1:], dim=-1)                # positions de réponse n_inv-1 .. L-2
            hid = _empiler(capt) if capt else {}
            torch.save({"logprobs": lp, "hidden": hid, "regime": regime, "alias": chemin,
                        "chemin": "force-graphes" if graphes else "force"},
                       os.path.join(dossier, f"force{'-graphes' if graphes else ''}-{_nom(chemin)}-{i}.pt"))
            print(json.dumps({"invite": i, "chemin": "force", "graphes": graphes, "couches": len(hid), "positions": L - 1}), flush=True)
            if not seq.finished:
                eng.abort("p125")
            eng.fermer()
            _liberer(eng, lg)
    finally:
        Engine._sample_only = _orig
    print("FINI force " + _nom(chemin) + (" graphes" if graphes else ""), flush=True)


# ───────────────────────────── comparaison, à sec ─────────────────────────────
def _kl(lp_ref: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1)


def _l2rel(A: dict, H: dict, pos: torch.Tensor) -> list:
    """Erreur L2 relative par couche sur les positions `pos` : ‖A−H‖ / ‖H‖ (fp32)."""
    out = []
    for c in sorted(H.keys()):
        if c not in A:
            break
        a, h = A[c][pos], H[c][pos]
        out.append(round(float((a - h).norm() / h.norm().clamp_min(1e-12)), 5))
    return out


def mode_compare(dossier: str, alias: str) -> None:
    nom = _nom(alias)
    res = {"alias": nom, "invites": [], "regime": {}}
    for i in range(len(INVITES)):
        d = torch.load(os.path.join(dossier, f"invite{i}.pt"), weights_only=False)
        P = torch.load(os.path.join(dossier, f"prefill-{nom}-{i}.pt"), weights_only=False)
        F = torch.load(os.path.join(dossier, f"force-{nom}-{i}.pt"), weights_only=False)
        res["regime"] = {"prefill": P.get("regime"), "force": F.get("regime")}
        ids, n_inv, L = d["ids"], d["n_invite"], len(d["ids"])
        lph = d["logprobs_hf"]
        n = min(lph.shape[0], P["logprobs"].shape[0], F["logprobs"].shape[0])
        lph, lpp, lpf = lph[:n], P["logprobs"][:n], F["logprobs"][:n]
        klp, klf, kpf = _kl(lph, lpp), _kl(lph, lpf), _kl(lpp, lpf)
        H = d["hidden"]
        pos_inv, pos_rep = torch.arange(0, n_inv), torch.arange(n_inv, L - 1)
        r = {"invite": i, "n_invite": n_inv, "n_reponse": n,
             "kl_hf_prefill": {"moy": round(float(klp.mean()), 5), "max": round(float(klp.max()), 5)},
             "kl_hf_force": {"moy": round(float(klf.mean()), 5), "max": round(float(klf.max()), 5)},
             "kl_prefill_force": {"moy": round(float(kpf.mean()), 5), "max": round(float(kpf.max()), 5)},
             "argmax_hf_prefill": f"{int((lpp.argmax(-1) == lph.argmax(-1)).sum())}/{n}",
             "argmax_hf_force": f"{int((lpf.argmax(-1) == lph.argmax(-1)).sum())}/{n}",
             "l2_invite_prefill": _l2rel(P["hidden"], H, pos_inv), "l2_invite_force": _l2rel(F["hidden"], H, pos_inv),
             "l2_reponse_prefill": _l2rel(P["hidden"], H, pos_rep), "l2_reponse_force": _l2rel(F["hidden"], H, pos_rep),
             "l2_invite_prefill_vs_force": _l2rel(P["hidden"], F["hidden"], pos_inv)}
        g = os.path.join(dossier, f"force-graphes-{nom}-{i}.pt")
        if os.path.exists(g):
            G = torch.load(g, weights_only=False)["logprobs"][:n]
            r["temoin_graphes_vs_eager_max_abs"] = float((G - lpf).abs().max())
        res["invites"].append(r)
        print(json.dumps({k: r[k] for k in ("invite", "kl_hf_prefill", "kl_hf_force", "kl_prefill_force", "argmax_hf_prefill",
                                            "argmax_hf_force")} | {"l2_inv_prefill_fin": r["l2_invite_prefill"][-1] if r["l2_invite_prefill"] else None,
                                                                   "l2_inv_force_fin": r["l2_invite_force"][-1] if r["l2_invite_force"] else None,
                                                                   "graphes=eager": r.get("temoin_graphes_vs_eager_max_abs")}), flush=True)
    # bilan pour le scellé : sur combien d'invites chaque chemin est le moins juste (KL moyenne et L2 d'invite en fin de pile)
    inv = res["invites"]
    res["bilan"] = {
        "kl_prefill_pire_sur": sum(1 for r in inv if r["kl_hf_prefill"]["moy"] > 1.25 * r["kl_hf_force"]["moy"]),
        "kl_force_pire_sur": sum(1 for r in inv if r["kl_hf_force"]["moy"] > 1.25 * r["kl_hf_prefill"]["moy"]),
        "l2_prefill_pire_sur": sum(1 for r in inv if r["l2_invite_prefill"] and r["l2_invite_prefill"][-1] > 1.25 * r["l2_invite_force"][-1]),
        "l2_force_pire_sur": sum(1 for r in inv if r["l2_invite_force"] and r["l2_invite_force"][-1] > 1.25 * r["l2_invite_prefill"][-1]),
        "kl_moy_prefill": round(sum(r["kl_hf_prefill"]["moy"] for r in inv) / len(inv), 5),
        "kl_moy_force": round(sum(r["kl_hf_force"]["moy"] for r in inv) / len(inv), 5),
    }
    json.dump(res, open(os.path.join(dossier, f"compare-{nom}.json"), "w"), ensure_ascii=False, indent=1)
    print("RESULTAT " + json.dumps({"alias": nom} | res["bilan"]), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "hf":
        mode_hf(sys.argv[2], sys.argv[3])
    elif mode == "prefill":
        mode_prefill(sys.argv[2], sys.argv[3])
    elif mode == "force":
        mode_force(sys.argv[2], sys.argv[3], len(sys.argv) > 4 and sys.argv[4] == "graphes")
    elif mode == "compare":
        mode_compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
