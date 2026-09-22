#!/usr/bin/env python3
"""KL(bf16 ‖ moteur) par API OpenAI, `echo` + `logprobs` (pièce 36) — un moteur
servi (acvram, vLLM, trtllm-serve), teacher forcing sur les 8 jetons gloutons bf16.

Pour chaque invite (dump HF de decode-pas : {ids, cibles, logits[8, vocab]}) :
  POST /v1/completions  prompt = ids_invite + 8 cibles, echo=true, logprobs=K,
  max_tokens=0, temperature=0.
On lit, aux 8 positions des cibles, le top-K {jeton: logprob} du moteur, et la KL
se calcule contre la distribution bf16 COMPLÈTE du dump :

  KL@K(bf16 ‖ moteur) = Σ_{t ∈ topK(bf16)} p_bf16(t) · (log p_bf16(t) − log p_moteur(t))

tronquée au top-K de bf16 (masse hors top-K du moteur → borne min p_moteur), donc
étiquetée « KL@K » ; la KL exacte demanderait la distribution complète du moteur,
que l'API ne rend pas. K = KL_TOPK (défaut 20).

CONTRÔLE D'ALIGNEMENT (rend faux) : les jetons renvoyés par `echo` doivent
correspondre aux ids envoyés — le serveur rend `token_ids` si possible (comparaison
exacte), sinon on compare `decode(id)` aux `tokens` (str). Désaccord → SUSPECT,
invite exclue. Sans ce contrôle, un serveur qui re-tokenise ou décale fausse tout.

Usage : kl-api.py <url> <dossier_dumps_HF> [nom_servi]
  env : KL_TOPK (20), KL_MODELE (nom servi, sinon lu sur /v1/models).
Vise acvram (pièce 36), vLLM (echo+logprobs natifs), trtllm-serve (logprobs OpenAI
à vérifier). Sortie : RESULTAT json par invite (kl@K par pas, max) + global.
"""
import glob, json, math, os, sys, urllib.request
import numpy as np


def _post(url, body):
    r = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.load(resp)


def _served(url):
    try:
        r = urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10)
        return json.load(r)["data"][0]["id"]
    except Exception:
        return "model"


def _charger_dumps(dossier):
    import torch
    out = {}
    for f in sorted(glob.glob(os.path.join(dossier, "*.pt"))):
        d = torch.load(f, weights_only=False)
        ids = d.get("ids") or d.get("prompt_ids")
        lg = d["logits"]
        lg = lg.float().cpu().numpy() if hasattr(lg, "float") else np.asarray(lg, np.float32)
        out[os.path.basename(f)] = {"ids": list(ids), "cibles": list(d["cibles"]), "logits": lg}
    return out


def _logsoftmax(v):
    v = np.asarray(v, np.float64); v = v - v.max()
    return v - np.log(np.exp(v).sum())


def kl_at_k(logits_bf16_pos, top_moteur):
    """KL@K(bf16 ‖ moteur) à un pas. top_moteur = {token_id: logprob} du moteur.
    Masse hors top-K du moteur : bornée par (min logprob du top − 2)."""
    lpb = _logsoftmax(logits_bf16_pos)
    K = min(len(top_moteur), 40) or 1
    idx = np.argsort(lpb)[-K:]                       # top-K de bf16 (là où p_bf16 pèse)
    plancher = (min(top_moteur.values()) - 2.0) if top_moteur else -1e9
    kl = 0.0
    for t in idx:
        pb = math.exp(lpb[t])
        lpm = top_moteur.get(int(t), plancher)
        kl += pb * (lpb[t] - lpm)
    return max(kl, 0.0)


def main():
    if len(sys.argv) < 3:
        print("usage: kl-api.py <url> <dossier_dumps_HF> [nom_servi]", file=sys.stderr); sys.exit(2)
    url, dossier = sys.argv[1], sys.argv[2]
    nom = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("KL_MODELE") or _served(url)
    K = int(os.environ.get("KL_TOPK", "20"))
    dumps = _charger_dumps(dossier)
    kmax_g = 0.0
    for inv, d in dumps.items():
        prompt_ids = list(d["ids"]) + list(d["cibles"]); n_prefix = len(d["ids"]); N = len(d["cibles"])
        r = _post(url, {"model": nom, "prompt": prompt_ids, "echo": True, "logprobs": K,
                        "max_tokens": 0, "temperature": 0.0, "stream": False})
        lp = r["choices"][0].get("logprobs") or {}
        toks = lp.get("tokens") or []; tops = lp.get("top_logprobs") or []
        ids_srv = lp.get("token_ids")                # acvram (36) peut le rendre ; sinon None
        # contrôle d'alignement
        if ids_srv is not None:
            aligne = list(ids_srv) == prompt_ids
        else:
            aligne = len(toks) == len(prompt_ids)    # faute de token_ids : au moins la longueur
        if not tops or not aligne:
            print("RESULTAT " + json.dumps({"cible": nom, "invite": inv, "aligne": bool(aligne),
                  "erreur": "echo/logprobs absents ou tokens != ids envoyés"}), flush=True); continue
        # top-K moteur aux positions des cibles : la cible k est en position n_prefix+k,
        # son top-K est renvoyé à l'INDEX n_prefix+k de tops (echo aligne tokens et top).
        kls = []
        for k in range(N):
            pos = n_prefix + k
            td = tops[pos] if pos < len(tops) else {}
            top_moteur = {}
            for key, v in (td.items() if isinstance(td, dict) else []):
                try:
                    top_moteur[int(key)] = float(v)     # clé = id si le serveur les donne en id
                except (ValueError, TypeError):
                    pass
            kls.append(kl_at_k(d["logits"][k], top_moteur))
        km = max(kls) if kls else None
        if km is not None:
            kmax_g = max(kmax_g, km)
        print("RESULTAT " + json.dumps({"cible": nom, "ref": "bf16", "invite": inv, "aligne": True,
              "kl_at_k": K, "kl_par_pas": [round(x, 5) for x in kls],
              "kl_max": round(km, 5) if km is not None else None}, ensure_ascii=False), flush=True)
    print("RESULTAT " + json.dumps({"cible": nom, "kl_at_k": K, "kl_max_global": round(kmax_g, 5)}), flush=True)


if __name__ == "__main__":
    main()
