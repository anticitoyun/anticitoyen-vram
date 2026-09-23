"""Vidage de référence pour la porte de l'étape 1 (option A du chef : décodage à préfill injecté).

Moteur construit comme `acvram serve` (graphes, chauffe `demarrer_service`, pas de spéculation), b=1. Pour chaque
invite de `tests/invites.json` : les lignes KV des positions de l'invite juste après le préfill (le décodage
n'écrit jamais ces positions), le premier jeton, puis les ids des 128 jetons gloutons et leur sha256. Plus :
l'empreinte (sha256) de chaque tenseur chargé, la tête liée int8 en entier (quantifiée au chargement,
loader.py:1087-1089) et la structure par couche. Aucun chiffre de débit : ce n'est pas une mesure.

Usage (sous outils/carte.sh) :
  PYTHONPATH=<racine> python moteurs/acvram_rust/outils/vidage_reference.py <modèle> <invites.json> <dossier-sortie>
"""
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

from acvram.engine.layers import QuantLinear
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.server.chat import load_tokenizer, render_chat

MODELE, INVITES, SORTIE = sys.argv[1:4]
N_JETONS, MAX_LEN = 128, 2304
os.makedirs(SORTIE, exist_ok=True)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def sha_ids(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


tok = load_tokenizer(MODELE)
loaded = load_model(MODELE, dtype=torch.bfloat16, max_model_len=MAX_LEN, max_concurrent_seqs=1)
eng = Engine(loaded, tok, max_batch_size=1, max_model_len=MAX_LEN, enable_prefix_cache=True,
             speculator=None, enable_cuda_graphs=True)
eng.demarrer_service(strict=False, warm_max_len=int(os.environ.get("ACVRAM_WARM_GRAPHS", "2048")))
modele = loaded.model
# Porte au bit (ordre du chef, option A suite 1) : le sha256 des logits fp32 de CHAQUE pas, lus là où le moteur
# servi les échantillonne (`Engine._sample_only`, pas normal et pas recouvert du pipeline). Lecture synchrone :
# elle ralentit le vidage, elle ne change aucune valeur.
_journal: list = []
_complets: list = []                       # VIDAGE_LOGITS=1 : logits fp32 complets (porte KL du préfill Rust)
_LOGITS = os.environ.get("VIDAGE_LOGITS") == "1"
_sample_orig = Engine._sample_only


def _espion(self, logits, seqs, depuis_graphe=False):
    ligne = logits.reshape(-1, logits.shape[-1])[0].detach()
    _journal.append({"dtype": str(ligne.dtype), "n": int(ligne.numel()),
                     "sha256": hashlib.sha256(ligne.contiguous().cpu().numpy().tobytes()).hexdigest()})
    if _LOGITS:
        _complets.append(ligne.to(torch.float32).cpu().clone())
    return _sample_orig(self, logits, seqs, depuis_graphe)


Engine._sample_only = _espion
res = {"modele": os.path.basename(MODELE.rstrip("/")), "regime": eng.regime_ligne(),
       "kv_format": eng.kv_format_servi(), "invites": []}

# --- poids tels que chargés : empreintes, structure, tête int8 en entier ---------------------------------
empreintes, structure = {}, {}
for nom, t in modele.state_dict().items():
    if isinstance(t, torch.Tensor):
        empreintes[nom] = {"dtype": str(t.dtype), "shape": list(t.shape), "sha256": sha(t)}
for nom, m in modele.named_modules():
    if isinstance(m, QuantLinear):
        q = m.qweight
        sd = q.state_dict() if hasattr(q, "state_dict") else {"poids": q}
        structure[nom] = {"format": type(q).__name__, "out": m.out_features, "in": m.in_features,
                          "scaler": m.scaler is not None, "bias": m.bias is not None,
                          "tenseurs": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": sha(v)}
                                       for k, v in sd.items() if isinstance(v, torch.Tensor)},
                          "scalaires": {k: v for k, v in vars(q).items() if isinstance(v, (int, float, str, bool))}}
res["couche0"] = repr(modele.layers[0])
json.dump({"empreintes": empreintes, "quantlinear": structure},
          open(os.path.join(SORTIE, "poids.json"), "w"), indent=1)
tete = modele.lm_head.qweight
tete_sd = {k: v.detach().contiguous().cpu() for k, v in tete.state_dict().items() if isinstance(v, torch.Tensor)}
save_file(tete_sd, os.path.join(SORTIE, "tete.safetensors"))
res["tete"] = {"format": type(tete).__name__, **{k: sha(v) for k, v in tete_sd.items()}}

# --- invites ------------------------------------------------------------------------------------------------
for inv in json.load(open(INVITES, encoding="utf-8")):
    ids = tok.encode(render_chat(tok, inv["messages"], True))
    L = len(ids)
    _journal.clear()
    _complets.clear()
    seq = eng.add_request(list(ids), SamplingParams(temperature=0.0, max_tokens=N_JETONS), request_id=inv["nom"])
    kv, premier = None, None
    while not seq.finished:
        eng.step()
        if kv is None and seq.output_ids:
            premier = int(seq.output_ids[0])
            blocs = list(seq.blocks)
            bs = modele.caches.get(0).cfg.block_size
            idx_b = torch.tensor([blocs[p // bs] for p in range(L)])
            idx_o = torch.tensor([p % bs for p in range(L)])
            kv = {}
            for i in range(len(modele.layers)):
                c = modele.caches.get(i)
                assert not c.canal, "KV par canal : format non porté"
                for nom_t in ("k", "v", "k_scale", "v_scale"):
                    t = getattr(c, nom_t)
                    kv[f"couche{i}.{nom_t}"] = t[idx_b.to(t.device), idx_o.to(t.device)].contiguous().cpu()
            save_file(kv, os.path.join(SORTIE, f"kv-{inv['nom']}.safetensors"))
    sortie = [int(x) for x in seq.output_ids]
    if _LOGITS:
        save_file({"logits": torch.stack(_complets).contiguous()}, os.path.join(SORTIE, f"logits-{inv['nom']}.safetensors"))
    r = {"nom": inv["nom"], "longueur_invite": L, "sha256_invite": sha_ids(ids), "premier_jeton": premier,
         "sortie": sortie, "sha256_sortie": sha_ids(sortie), "logits": list(_journal), "fin": seq.finish_reason if hasattr(seq, "finish_reason") else None,
         "sha256_kv": hashlib.sha256(b"".join(v.reshape(-1).view(torch.uint8).numpy().tobytes() for v in kv.values())).hexdigest()}
    res["invites"].append(r)
    print(r["nom"], L, len(sortie), r["sha256_sortie"][:16], flush=True)

json.dump(res, open(os.path.join(SORTIE, "reference.json"), "w"), ensure_ascii=False, indent=1)
eng.fermer()
print("ok", len(res["invites"]))
