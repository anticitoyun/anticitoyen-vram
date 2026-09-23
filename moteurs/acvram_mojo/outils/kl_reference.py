"""Porte KL de l'étape 1 Mojo (contrat §3, repli quand l'identité au bit est impossible) : KL(HF bf16 ‖
moteur) par API OpenAI, teacher forcing sur les 8 jetons gloutons HF (dump_hf_reference.py), pour un moteur
donné (acvram ou MAX). Réutilise la formule de outils/gpu/mesure/kl-api.py (KL@K tronquée au top-K de la
référence) mais résout les clés de `top_logprobs` par ID (acvram, pièce 36) OU par CHAÎNE de jeton (MAX,
format OpenAI standard) — les deux moteurs ne rendent pas le même format.

Usage : PYTHONPATH=<racine> python kl_reference.py <url> <dossier_dumps_hf> <modele_hf_pour_tokeniseur> [nom_servi]
"""
import glob
import json
import math
import os
import sys
import urllib.request

import numpy as np

K = int(os.environ.get("KL_TOPK", "20"))


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
        out[os.path.basename(f)[:-3]] = {"ids": list(d["ids"]), "cibles": list(d["cibles"]),
                                          "logits": d["logits"].float().numpy()}
    return out


def _logsoftmax(v):
    v = np.asarray(v, np.float64); v = v - v.max()
    return v - np.log(np.exp(v).sum())


def kl_at_k(logits_ref_pos, top_moteur):
    lpb = _logsoftmax(logits_ref_pos)
    Kx = min(len(top_moteur), 40) or 1
    idx = np.argsort(lpb)[-Kx:]
    plancher = (min(top_moteur.values()) - 2.0) if top_moteur else -1e9
    kl = 0.0
    for t in idx:
        pb = math.exp(lpb[t])
        lpm = top_moteur.get(int(t), plancher)
        kl += pb * (lpb[t] - lpm)
    return max(kl, 0.0)


def resoudre_cles(td, tok):
    """dict {clé_moteur: logprob} -> dict {id_vocab: logprob}. Trois formats vus dans ce projet :
    id en entier (chaîne) ; texte DÉCODÉ du jeton (acvram, ex ' the', '\\n', 'Okay' — on ré-encode seul,
    l'aller-retour donne un seul id) ; pièce BPE brute (MAX/OpenAI standard, ex 'Ġthe')."""
    top = {}
    ratees = 0
    for key, v in (td.items() if isinstance(td, dict) else []):
        try:
            top[int(key)] = float(v); continue
        except (ValueError, TypeError):
            pass
        try:
            ids_re = tok.encode(key, add_special_tokens=False)
            if len(ids_re) == 1:
                top[int(ids_re[0])] = float(v); continue
        except Exception:
            pass
        try:
            i = tok.convert_tokens_to_ids(key)
            if i is not None and i != tok.unk_token_id:
                top[int(i)] = float(v); continue
        except Exception:
            pass
        ratees += 1
    return top, ratees


def main():
    url, dossier, modele_tok = sys.argv[1:4]
    nom = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("KL_MODELE") or _served(url)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(modele_tok)
    dumps = _charger_dumps(dossier)
    kmax_g, total_ratees = 0.0, 0
    for inv, d in dumps.items():
        prompt_ids = list(d["ids"]) + list(d["cibles"]); n_prefix = len(d["ids"]); N = len(d["cibles"])
        r = _post(url, {"model": nom, "prompt": prompt_ids, "echo": True, "logprobs": K,
                        "max_tokens": 0, "temperature": 0.0, "stream": False})
        lp = r["choices"][0].get("logprobs") or {}
        tops = lp.get("top_logprobs") or []
        if os.environ.get("KL_DEBUG"):
            print("DEBUG", inv, "n_prefix=", n_prefix, "N=", N, "len(prompt_ids)=", len(prompt_ids),
                  "len(tops)=", len(tops), "tokens[:3]=", (lp.get("tokens") or [])[:3],
                  "tokens[-3:]=", (lp.get("tokens") or [])[-3:], file=sys.stderr)
        if not tops:
            print("RESULTAT " + json.dumps({"cible": nom, "invite": inv,
                  "erreur": "logprobs absents (echo/logprobs non supportés ?)"})); continue

        # Décalage d'indexation : selon le moteur, tops[i] rend la distribution qui prédit token[i]
        # (acvram, convention "officielle" des complétions OpenAI) OU token[i+1] (MAX, constaté 23/09 —
        # tops[pos] contient alors le prochain jeton, pas celui-ci). Détecté par comparaison au premier
        # jeton attendu contre les deux candidats, pas codé en dur pour un moteur.
        decalage = 0
        if n_prefix < len(tops):
            attendu0 = tok.decode([int(d["cibles"][0])])
            td0 = tops[n_prefix] if isinstance(tops[n_prefix], dict) else {}
            top1_0 = max(td0.items(), key=lambda kv: kv[1])[0] if td0 else None
            td_m1 = tops[n_prefix - 1] if n_prefix - 1 >= 0 and isinstance(tops[n_prefix - 1], dict) else {}
            top1_m1 = max(td_m1.items(), key=lambda kv: kv[1])[0] if td_m1 else None
            if top1_m1 == attendu0 and top1_0 != attendu0:
                decalage = -1

        kls, ratees_inv = [], 0
        for k in range(N):
            pos = n_prefix + k + decalage
            td = tops[pos] if 0 <= pos < len(tops) else {}
            top_moteur, r_ = resoudre_cles(td, tok)
            ratees_inv += r_
            if os.environ.get("KL_DEBUG") and (not top_moteur or r_ or os.environ.get("KL_DEBUG_ALL")):
                attendu = tok.decode([int(d["cibles"][k])])
                top1 = max(td.items(), key=lambda kv: kv[1])[0] if isinstance(td, dict) and td else None
                print("DEBUG", inv, k, "decalage=", decalage, "pos=", pos, "attendu=", repr(attendu),
                      "top1_moteur=", repr(top1), "td=", json.dumps(td)[:300], file=sys.stderr)
            kls.append(kl_at_k(d["logits"][k], top_moteur))
        total_ratees += ratees_inv
        km = max(kls) if kls else None
        if km is not None:
            kmax_g = max(kmax_g, km)
        print("RESULTAT " + json.dumps({"cible": nom, "ref": "hf_bf16", "invite": inv, "kl_at_k": K,
              "decalage": decalage, "kl_par_pas": [round(x, 5) for x in kls],
              "kl_max": round(km, 5) if km is not None else None,
              "cles_non_resolues": ratees_inv}, ensure_ascii=False), flush=True)
    print("RESULTAT " + json.dumps({"cible": nom, "kl_at_k": K, "kl_max_global": round(kmax_g, 5),
          "cles_non_resolues_total": total_ratees}), flush=True)


if __name__ == "__main__":
    main()
