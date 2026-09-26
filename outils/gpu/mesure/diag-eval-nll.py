"""Pièce 37 : pourquoi `acvram eval` rend PPL 27 713-59 316 sur Gemma 4 31B
alors que le service décode juste (scellé E, KL 0,87 sur 280 jetons) ?
Test discriminant (verdict croisé duck.ai) : teacher-forcing court, SANS
fenêtrage, NLL jeton par jeton, et la première position qui diverge.

Trois bras sur les MÊMES ids ([BOS] + texte connu, 300 jetons par défaut) :
  eval   : le chemin de `perplexity` — ForwardBatch d évaluation,
           `model(batch, return_hidden=True)` puis `_pertes_par_tranches`
  serve  : le chemin du serveur au préfill — `Engine.add_request` + un
           `step()` (mêmes noyaux que /v1/completions), logits de la dernière
           position seulement, et `forward(logits_positions=arange)` pour
           toutes (même forward que le service, logits à chaque position)
  decode : le chemin de la GÉNÉRATION — `add_request` sur le seul premier
           jeton, puis un pas de décodage par position (cache KV, sampler
           lent, `enable_cuda_graphs=False`), le jeton posé à chaque pas
           étant le jeton RÉEL du texte (teacher-forcing) : les logits
           « tels que la génération les voit » à chaque position
  hf     : transformers bf16 — avec DEUX contrôles de la référence elle-même
           (22/09 : la référence rend PPL 108 039 sur nos ids, poste2 9b757c1b) :
           PPL ≤ 30 sur un paragraphe propre qu elle réencode avec SON
           tokeniseur, et 20 jetons générés depuis nos ids qui ne soient pas
           dégénérés — sinon verdict « RÉFÉRENCE INVALIDE », le moteur n est
           pas jugé (offload, `HF_PYTHON` + `MAXMEM` comme
           decode-pas), NLL par position — la référence
Sortie : NLL médiane par bras, |Δ| eval−serve et eval−hf par position,
première position où |Δ| > 1 nat ; PPL(ctx ≥ 32) de chaque bras.

Prédictions (écrites avant) :
  P1 eval ≈ serve ≈ hf (|Δ| < 0,1 nat, PPL 8-20) → le forward est sain : la
     pathologie vient du fenêtrage/corpus de `perplexity` (d : fenêtre
     locale 1 024 au-delà de 1 024 jetons, ou corpus) → rejouer avec
     --jetons 1500 : divergence attendue APRÈS la position 1 024.
  P2 eval ≠ serve dès la position 1 → la ForwardBatch d évaluation manque
     un champ que le service pose (à lire dans l écart : seq_ids, images,
     positions) ; c est un défaut d eval, corrigeable à sec.
  P3 eval ≈ serve ≠ hf dès le début → le forward acvram diverge sur ce
     modèle hors gabarit de conversation (le scellé E ne l a vu que sous
     gabarit) — défaut moteur, à nommer par position et par couche.
  P4 decode ≈ eval ≈ 11-12 nats (PPL decode > 30) → ni le préfill ni le
     décodage ne sont en cause : le modèle converti est faux HORS GABARIT de
     conversation, ou le protocole (BOS, corpus) — c est la lecture que je
     tiens pour la plus probable (le 22/09 les NLL du préfill sont du bruit à
     TOUTES les positions, dernière comprise : 11,77 nats médians sur 12,48
     possibles, et le scellé E n a jamais vu ce modèle hors gabarit).
  P5 decode 8-20 nats (PPL decode ≤ 30) alors que eval reste à ~12 → le
     décodage est juste et le PRÉFILL de Gemma 4 est faux hors dernière
     position : masque fenêtré ou positions du préfill, à nommer par couche.
Contrôle qui rend « faux » : PPL(ctx ≥ 32) du bras eval ≤ 30 sur ce texte
connu ; > 30 = eval faux, quel que soit le reste.
Contrôle du bras decode lui-même : sa PREMIÈRE NLL (position 1, cache vide,
un seul jeton en contexte) doit égaler celle du bras eval à 1e-2 près — les
deux calculent le même forward sur un contexte d un jeton ; un écart là
signe une erreur de montage du bras, pas un défaut du modèle.

Softcap et échelle d attention, vérifiés dans le code (22/09) :
  * softcap FINAL : `model.py:2653-2656` — `final_logit_softcapping` du
    config (30,0 pour ce 31B, `config.py:761`) appliqué en fp32 après
    `logits_scaling`. Gemma 4 le porte, il est bien appliqué.
  * softcap d ATTENTION : `attn_logit_softcapping` N EXISTE PAS dans le
    config de Gemma 4 (propre à Gemma 2) et n est appliqué nulle part —
    conforme à la génération du modèle.
  * `query_pre_attn_scalar` : absent du config de Gemma 4 également ;
    l échelle est `head_dim ** -0.5` par couche (`attention.py:122-123`),
    avec `head_dim` = 256 sur les couches locales et `global_head_dim` = 512
    sur les globales (`loader.py:541-542`) — deux échelles différentes,
    correctement séparées.
  * QK-norm : `q_norm` / `k_norm` passés par couche (`attention.py:311-318`,
    `404-411`) ; le config de ce 31B ne porte pas `use_qk_norm`, mais les
    poids `self_attn.q_norm.weight` sont lus s ils existent
    (`loader.py:551-552`) — donc pas de norme inventée ni omise.

Usage : outils/carte.sh python outils/gpu/mesure/diag-eval-nll.py ALIAS [--source HF_DIR]
        [--jetons 300] [--texte FICHIER] [--json SORTIE]         (≤ 10 min avec hf ; ≤ 2 min sans)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import torch  # noqa: E402

SEUIL_PPL_PROPRE = 30.0

SCRIPT_HF = r'''
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
ids = json.load(open(sys.argv[2]))
maxmem = sys.argv[3]
texte_propre = open(sys.argv[5], encoding="utf-8").read() if len(sys.argv) > 5 else None
kw = {"dtype": torch.bfloat16, "attn_implementation": "eager", "output_loading_info": True}
if maxmem == "cpu":                      # pièce 37 : processeur seul, zéro offload, zéro carte (59 Go de RAM, ~100 s / 300 jetons)
    torch.set_num_threads(int(os.environ.get("HF_THREADS", "8")))
    kw.update(device_map="cpu")
elif maxmem != "cuda":
    g, c = maxmem.split(",")
    kw.update(device_map="auto", max_memory={0: g, "cpu": c})
else:
    kw.update(device_map="cuda")
try:
    m, info = AutoModelForImageTextToText.from_pretrained(sys.argv[1], **kw)
except Exception:
    m, info = AutoModelForCausalLM.from_pretrained(sys.argv[1], **kw)
m.eval()
# GARDE DE CHARGEMENT (Q(12), 22/09) : une clé manquante ou un paramètre resté sur `meta` fait un modèle
# à poids vides qui génère du bruit en silence ; on le dit AVANT de juger quoi que ce soit.
chargement = {"classe": type(m).__name__, "manquantes": len(info.get("missing_keys", [])),
              "inattendues": len(info.get("unexpected_keys", [])), "mal_formees": len(info.get("mismatched_keys", [])),
              "meta": sum(1 for p in m.parameters() if p.device.type == "meta")}
x = torch.tensor([ids])
dev = next(m.parameters()).device
with torch.no_grad():
    out = m(input_ids=x.to(dev))
lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
nll = -lp.gather(1, x[0, 1:].to(lp.device).unsqueeze(1)).squeeze(1)
r = {"nll": nll.cpu().tolist(), "chargement": chargement}

# CONTRÔLE DE LA RÉFÉRENCE (22/09) : une référence qui rend PPL 108 039 sur
# nos ids ne juge rien. Deux épreuves qui ne dépendent PAS de notre chaîne :
# (1) le modèle réencode lui-même un paragraphe propre avec SON tokeniseur,
# (2) il génère 20 jetons depuis nos ids — si le texte produit est du bruit,
# ce sont les ids ou le chargement qui sont faux, pas le modèle mesuré.
try:
    tk = AutoTokenizer.from_pretrained(sys.argv[1])
except Exception as exc:
    tk = None
    r["controle_erreur"] = f"tokeniseur illisible : {exc}"
if tk is not None and texte_propre:
    ids_hf = tk(texte_propre, return_tensors="pt").input_ids[:, :300]
    with torch.no_grad():
        o2 = m(input_ids=ids_hf.to(dev))
    lp2 = torch.log_softmax(o2.logits[0, :-1].float(), dim=-1)
    n2 = -lp2.gather(1, ids_hf[0, 1:].to(lp2.device).unsqueeze(1)).squeeze(1)
    v = n2[32:] if n2.numel() > 32 else n2
    r["nll_propre"] = n2.cpu().tolist()
    r["ppl_propre"] = float(torch.exp(v.mean()))
    r["ids_propres_en_tete"] = ids_hf[0, :4].tolist()
    r["ids_nos_en_tete"] = ids[:4]
if tk is not None:
    with torch.no_grad():
        g = m.generate(input_ids=x.to(dev), max_new_tokens=20, do_sample=False)
    suite = g[0, x.shape[1]:]
    r["suite_ids"] = suite.tolist()
    r["suite_texte"] = tk.decode(suite, skip_special_tokens=True)
    r["suite_distincts"] = len(set(suite.tolist()))
json.dump(r, open(sys.argv[4], "w"))
'''


def nll_eval(loaded, ids: list[int]) -> list[float]:
    from acvram.engine.model import ForwardBatch
    from acvram.evaluate import _pertes_par_tranches
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    n = len(ids)
    alloc = BlockAllocator((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1, enable_prefix_cache=False)
    blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)], dtype=torch.long)
    batch = ForwardBatch(tokens=torch.tensor(ids, dtype=torch.long), positions=torch.arange(n, dtype=torch.long),
                         seq_lens=[n], query_lens=[n], block_tables=[torch.tensor(blocks, dtype=torch.long)],
                         slot_mapping=slots, is_prefill=True)
    with torch.inference_mode():
        h = loaded.model(batch, return_hidden=True)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=h.device)
        pertes = _pertes_par_tranches(loaded.model, h, targets, 0)
    return pertes.float().cpu().tolist()


def nll_serve(loaded, tok, ids: list[int]) -> tuple[list[float], dict]:
    """Le forward du service au préfill : `Engine` construit la ForwardBatch
    comme /v1/completions (`_build_batch`), le même `model(batch)` est
    appelé avec `logits_positions` = toutes les positions."""
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    eng = Engine(loaded, tok, max_batch_size=1, max_model_len=len(ids) + 32, enable_cuda_graphs=False)
    eng.pipeline_actif = False
    eng.add_request(list(ids), SamplingParams(temperature=0.0, max_tokens=1), request_id="diag")
    eng._admit()                                    # alloue les blocs, comme au premier pas du service
    seq = eng.running[0]
    batch = eng._build_batch([seq], prefill=True)
    champs = {k: (type(getattr(batch, k)).__name__ if getattr(batch, k) is not None else None)
              for k in ("seq_ids", "gdn_store", "images", "mrope_positions", "deepstack") if hasattr(batch, k)}
    with torch.inference_mode():
        logits = loaded.model(batch, logits_positions=torch.arange(len(ids)))
        lp = torch.log_softmax(logits.float(), dim=-1)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=lp.device)
        nll = -lp[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    return nll.cpu().tolist(), champs


def nll_decode(loaded, tok, ids: list[int]) -> list[float]:
    """Le chemin de la génération : un pas de décodage par position, cache KV,
    le jeton du texte posé à chaque pas (teacher-forcing). Les logits sont
    ceux que le sampler lent voit au pas i, donc la NLL « telle que la
    génération la voit » à la position i + 1."""
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    eng = Engine(loaded, tok, max_batch_size=1, max_model_len=len(ids) + 32, enable_cuda_graphs=False)
    eng.pipeline_actif = False
    eng.add_request([ids[0]], SamplingParams(temperature=0.0, max_tokens=len(ids)), request_id="dec")
    eng._admit()
    seq = eng.running[0]
    nll: list[float] = []
    with torch.inference_mode():
        batch = eng._build_batch([seq], prefill=True)            # le seul premier jeton
        logits = loaded.model(batch)
        seq.prefill_len = len(seq.prompt_ids)
        for i in range(1, len(ids)):
            lp = torch.log_softmax(logits.view(-1, logits.shape[-1])[-1].float(), dim=-1)
            nll.append(float(-lp[ids[i]]))
            seq.output_ids.append(ids[i])                        # teacher-forcing : le jeton RÉEL
            if not eng._grow(seq):
                raise RuntimeError(f"blocs épuisés à la position {i} — augmenter max_model_len")
            if i == len(ids) - 1:
                break
            logits = loaded.model(eng._build_batch([seq], prefill=False))
    return nll


def nll_hf(source: str, ids: list[int], texte: str = "") -> dict | None:
    """NLL du bras HF sur NOS ids, plus les deux contrôles de la référence."""
    py = os.environ.get("HF_PYTHON", "/opt/ia/vLLM/.venv/bin/python")
    maxmem = os.environ.get("MAXMEM", "26GiB,80GiB")
    with tempfile.TemporaryDirectory() as d:
        s, i, o = os.path.join(d, "hf.py"), os.path.join(d, "ids.json"), os.path.join(d, "nll.json")
        open(s, "w").write(SCRIPT_HF)
        json.dump(ids, open(i, "w"))
        argv = [py, s, source, i, maxmem, o]
        if texte:
            t = os.path.join(d, "texte.txt")
            open(t, "w", encoding="utf-8").write(texte[:4000])
            argv.append(t)
        r = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(f"[diag] bras hf en échec : {r.stderr[-800:]}", flush=True)
            return None
        return json.load(open(o))


def juger_reference_gabarit(h: dict, depuis: int, seuil: float = 1.0) -> dict:
    """Épreuve de la référence SOUS GABARIT (pièce 37, 22/09) : un Gemma 4 -it ne
    prédit pas la suite d un texte brut (PPL 10^4 sur HF lui-même, T1-T4), il
    prédit des jetons de tour ; la seule épreuve qui rend « faux » pour la bonne
    raison est la NLL moyenne de SA PROPRE réponse gloutonne (ids[depuis:]),
    ≤ `seuil` nat pour un chargement sain, plus la garde de chargement."""
    nll = h.get("nll") or []
    rep = nll[max(depuis - 1, 0):]
    moy = sum(rep) / len(rep) if rep else None
    ch = h.get("chargement") or {}
    motifs = []
    if moy is None:
        motifs.append("aucune position de réponse (depuis trop grand)")
    elif moy > seuil:
        motifs.append(f"NLL moyenne {moy:.3f} > {seuil} nat sur SA propre réponse gloutonne")
    if ch and (ch.get("manquantes") or ch.get("meta") or ch.get("mal_formees")):
        motifs.append(f"chargement : {ch}")
    return {"nll_reponse_hf_moy": None if moy is None else round(moy, 4), "chargement_hf": ch,
            "reference_valide": not motifs, "reference_motifs": motifs}


def juger_reference(h: dict) -> dict:
    """La référence est-elle en état de juger ? Seuils écrits avant : PPL ≤ 30
    sur un paragraphe propre encodé par SON tokeniseur, et une suite de 20
    jetons non dégénérée (≥ 5 jetons distincts). Sinon le moteur n est PAS
    jugé : ce sont les ids ou le chargement qui sont en cause."""
    ppl_propre = h.get("ppl_propre")
    distincts = h.get("suite_distincts")
    motifs = []
    if ppl_propre is None:
        motifs.append("pas de bras propre (tokeniseur ou texte manquant)")
    elif ppl_propre > SEUIL_PPL_PROPRE:
        motifs.append(f"PPL {ppl_propre:.1f} > {SEUIL_PPL_PROPRE} sur SON propre encodage")
    if distincts is not None and distincts < 5:
        motifs.append(f"suite dégénérée : {distincts} jetons distincts sur 20")
    return {"ppl_propre": ppl_propre, "suite_distincts": distincts,
            "suite_texte": (h.get("suite_texte") or "")[:200],
            "reference_valide": not motifs,
            "reference_motifs": motifs}


def ppl(nll: list[float], depuis: int = 32) -> float:
    v = nll[depuis:] if len(nll) > depuis else nll
    return math.exp(sum(v) / len(v)) if v else float("nan")


def premiere_divergence(a: list[float], b: list[float], seuil: float = 1.0):
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > seuil:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("alias")
    ap.add_argument("--source", help="dossier HF bf16 pour le bras hf (sinon deux bras)")
    ap.add_argument("--jetons", type=int, default=300)
    ap.add_argument("--texte", default=None, help="fichier texte connu (défaut : acvram/data/calibration-anglais.txt)")
    ap.add_argument("--sans-decode", action="store_true", help="sauter le bras decode (n pas de décodage)")
    ap.add_argument("--sans-serve", action="store_true", help="sauter le bras serve (tête à toutes les positions en un seul matmul : 5,25 Gio de fp32 sur un alias bf16 à vocabulaire 262 144 — OOM, 22/09)")
    ap.add_argument("--ids", default=None, help="fichier JSON d ids tels quels (gabarit compris) : remplace --texte, aucun BOS ajouté (pièce 37 : Gemma 4 -it ne se juge que sous gabarit)")
    ap.add_argument("--reponse-depuis", type=int, default=0, help="avec --ids : rang du premier id de la réponse gloutonne HF ; l épreuve de référence devient « NLL hf moyenne sur la réponse ≤ 1 nat » (gabarit), et le moteur publie la même moyenne")
    ap.add_argument("--nll-hf", default=None, help="JSON déjà produit par le bras hf (nll_hf, à sec sur processeur) : évite de le rejouer sous le verrou")
    ap.add_argument("--json")
    a = ap.parse_args()
    from acvram.engine.loader import load_model
    from acvram.server.chat import load_tokenizer
    from racine_modeles import racine_modeles
    chemin = a.alias if os.path.isdir(a.alias) else os.path.join(racine_modeles(), a.alias)
    tok = load_tokenizer(chemin)
    texte = open(a.texte or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../acvram/data/calibration-anglais.txt"),
                 encoding="utf-8").read()
    bos = tok.bos_id()
    if a.ids:
        ids = json.load(open(a.ids))[: a.jetons]
    else:
        ids = tok.encode(texte)[: a.jetons - 1]
        if bos is not None:
            ids = [bos] + ids
    loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=len(ids) + 32, max_concurrent_seqs=1)
    r = {"alias": chemin, "jetons": len(ids), "bos": bos, "bos_en_tete": ids[0] == bos if bos is not None else None}
    e = nll_eval(loaded, ids)
    s, champs = (list(e), {"sans_serve": True}) if a.sans_serve else nll_serve(loaded, tok, ids)
    r.update({"nll_eval": e, "nll_serve": s, "champs_batch_serve": champs,
              "ppl_eval": round(ppl(e), 3), "ppl_serve": round(ppl(s), 3),
              "div_eval_serve": premiere_divergence(e, s),
              "delta_eval_serve_max": round(max(abs(x - y) for x, y in zip(e, s)), 4)})
    if not a.sans_decode:
        d = nll_decode(loaded, tok, ids)
        r.update({"nll_decode": d, "ppl_decode": round(ppl(d), 3),
                  "nll_decode_mediane": round(sorted(d)[len(d) // 2], 4),
                  "div_eval_decode": premiere_divergence(e, d),
                  "delta_eval_decode_max": round(max(abs(x - y) for x, y in zip(e, d)), 4),
                  "montage_decode_ok": abs(d[0] - e[0]) <= 1e-2, "delta_position_1": round(abs(d[0] - e[0]), 5)})
    if a.source:
        # 22/09 (poste2) : le modèle acvram (28 Go) restait chargé pendant le
        # sous-processus HF → OOM. On le libère AVANT, et on vérifie que la
        # carte est bien rendue : un bras hf qui tombe sur l OOM ne dit rien.
        del loaded
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            libre = torch.cuda.mem_get_info()[0] / 2 ** 30
            r["vram_libre_avant_hf_gio"] = round(libre, 2)
            print(f"[diag] modèle acvram libéré, {libre:.1f} Gio libres avant le bras hf", flush=True)
            if libre < 20:
                print(f"[diag] ALERTE : {libre:.1f} Gio seulement — le bras hf en offload peut échouer", flush=True)
        brut = json.load(open(a.nll_hf)) if a.nll_hf else nll_hf(a.source, ids, texte)
        if brut is not None:
            h = brut["nll"]
            r.update({"nll_hf": h, "ppl_hf": round(ppl(h), 3), "div_eval_hf": premiere_divergence(e, h),
                      "div_serve_hf": premiere_divergence(s, h),
                      "delta_eval_hf_med": round(sorted(abs(x - y) for x, y in zip(e, h))[len(h) // 2], 4)})
            r.update(juger_reference_gabarit(brut, a.reponse_depuis) if a.ids and a.reponse_depuis else juger_reference(brut))
            if a.ids and a.reponse_depuis:
                d0 = max(a.reponse_depuis - 1, 0)
                r["nll_reponse_moy"] = {b: round(sum(v[d0:]) / len(v[d0:]), 4) for b, v in (("eval", e), ("serve", s), ("hf", h)) if len(v) > d0}
                r["ecarts_confiants"] = ecarts_confiants(e, h)
            r["ids_propres_en_tete"] = brut.get("ids_propres_en_tete")
            r["ids_nos_en_tete"] = brut.get("ids_nos_en_tete")
    r["controle_eval_ppl_le_30"] = r["ppl_eval"] <= 30
    r["verdict"] = verdict(r)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[diag] {r['jetons']} jetons, bos {bos} en tête {r['bos_en_tete']} ; PPL(ctx≥32) eval {r['ppl_eval']} · serve {r['ppl_serve']}"
          + (f" · hf {r['ppl_hf']}" if "ppl_hf" in r else "") + f" ; champs serve {champs}")
    if "ppl_decode" in r:
        print(f"  bras decode : PPL(ctx≥32) {r['ppl_decode']} · NLL médiane {r['nll_decode_mediane']} nats · "
              f"1re divergence eval/decode {r['div_eval_decode']} (|Δ| max {r['delta_eval_decode_max']}) · "
              f"montage (position 1) {'ok' if r['montage_decode_ok'] else 'FAUX, Δ=' + str(r['delta_position_1'])}")
    print(f"  première divergence eval/serve : {r['div_eval_serve']} (|Δ| max {r['delta_eval_serve_max']})"
          + (f" ; eval/hf : {r['div_eval_hf']} ; serve/hf : {r['div_serve_hf']} ; |Δ| eval−hf médian {r['delta_eval_hf_med']}" if "ppl_hf" in r else ""))
    print(f"  contrôle PPL eval ≤ 30 : {r['controle_eval_ppl_le_30']} ; verdict : {r['verdict']}")
    return 0


SEUIL_CONFIANT_NATS = 2.0     # une position « confiante » : NLL_hf ≤ 2 nats (p ≥ 0,135)


def ecarts_confiants(e: list[float], h: list[float], seuil: float = SEUIL_CONFIANT_NATS) -> dict:
    """|Δ| eval−hf sur les positions où hf est confiant. Dans la queue (NLL 15-25 nats,
    p ≈ 1e-7..1e-11) deux moteurs justes diffèrent de plusieurs nats sans qu aucun soit
    faux — un 1er contrôle « |Δ| ≥ 1 nat à une position quelconque » rendait faux 128/202
    positions d invite d un forward par ailleurs à 0,002 nat de médiane sur la réponse
    (22/09, pièce 37) : le seuil ne portait pas le régime de la position."""
    idx = [i for i in range(min(len(e), len(h))) if h[i] <= seuil]
    dd = sorted(abs(e[i] - h[i]) for i in idx)
    if not dd:
        return {"n": 0}
    return {"n": len(dd), "mediane": round(dd[len(dd) // 2], 4), "moyenne": round(sum(dd) / len(dd), 4),
            "part_ge_1nat": round(sum(1 for x in dd if x >= 1.0) / len(dd), 4), "max": round(dd[-1], 3),
            "premiere_ge_1nat": next((i + 1 for i in idx if abs(e[i] - h[i]) >= 1.0), None)}


def verdict_gabarit(r: dict) -> str:
    """SOUS GABARIT (pièce 37) : la PPL absolue des ids d invite ne juge rien (tour
    utilisateur, queue à 15-25 nats) ; ce qui juge, c est l accord avec hf sur les
    positions CONFIANTES (NLL_hf ≤ 2 nats) et la NLL de la réponse gloutonne."""
    m = r["nll_reponse_moy"]
    c = r.get("ecarts_confiants") or {}
    if r["div_eval_serve"] is not None and r["div_eval_serve"] < 8:
        return f"P2 : eval ≠ serve dès la position {r['div_eval_serve']} — la ForwardBatch d évaluation diffère du service (champs {r['champs_batch_serve']})"
    if c.get("n", 0) < 20:
        return f"INDÉCIDABLE : {c.get('n', 0)} positions confiantes seulement (< 20) — allonger la réponse"
    if c["mediane"] > 0.05 or c["part_ge_1nat"] > 0.10:
        return (f"G3 : sous gabarit, acvram ≠ hf sur les positions confiantes (n={c['n']} : |Δ| médian {c['mediane']}, "
                f"{100 * c['part_ge_1nat']:.1f} % à ≥ 1 nat, 1re à {c['premiere_ge_1nat']}) — défaut du forward acvram, à nommer par couche")
    if m.get("eval", 9) > 1.0:
        return f"G4 : positions confiantes tenues mais NLL de la réponse eval {m['eval']} > 1 nat (hf {m.get('hf')}) — conversion abîmée"
    return (f"G1 : moteur juste sous gabarit — positions confiantes n={c['n']} : |Δ| médian {c['mediane']}, "
            f"{100 * c['part_ge_1nat']:.1f} % à ≥ 1 nat (seuils 0,05 / 10 %) ; NLL réponse eval {m['eval']} · serve {m.get('serve')} · hf {m.get('hf')} "
            f"(coût de la quantification : {round(m['eval'] - m.get('hf', 0), 3)} nat/jeton)")


def verdict(r: dict) -> str:
    if r.get("reference_valide") is False:
        return ("RÉFÉRENCE INVALIDE (ids ou chargement) : " + " ; ".join(r["reference_motifs"])
                + f" — suite produite : {r.get('suite_texte', '')!r}. Le moteur n est PAS jugé : "
                  "corriger la référence (gabarit, BOS, tokeniseur, offload) avant toute conclusion.")
    if r.get("nll_reponse_moy"):
        return verdict_gabarit(r)
    if "ppl_decode" in r and not r["montage_decode_ok"]:
        return (f"INVALIDE (bras decode) : à la position 1, contexte d un seul jeton, decode et eval devraient "
                f"coïncider — Δ = {r['delta_position_1']} > 1e-2. Le bras est mal monté, il ne juge rien.")
    if "ppl_decode" in r and r["ppl_eval"] > 30:
        if r["ppl_decode"] <= 30:
            return (f"P5 : le DÉCODAGE est juste (PPL {r['ppl_decode']} ≤ 30, NLL médiane {r['nll_decode_mediane']}) "
                    f"et le préfill est faux hors dernière position (PPL eval {r['ppl_eval']}) — divergence dès la "
                    f"position {r['div_eval_decode']} : masque fenêtré ou positions du préfill, à nommer par couche")
        return (f"P4 : decode ET préfill à ~{r['nll_decode_mediane']} nats (PPL decode {r['ppl_decode']}, eval "
                f"{r['ppl_eval']}) — ni l un ni l autre n est en cause : le modèle converti est faux HORS GABARIT "
                f"de conversation, ou le protocole (BOS, corpus). Suite : le même texte SOUS gabarit, et le bras hf.")
    if "ppl_hf" in r:
        if r["div_eval_hf"] is None and r["div_serve_hf"] is None and r["ppl_eval"] <= 30:
            return "P1 : forward sain (eval ≈ serve ≈ hf) — la pathologie est dans le fenêtrage/corpus de perplexity : rejouer --jetons 1500 (fenêtre locale 1 024)"
        if r["div_eval_serve"] is not None and r["div_eval_serve"] < 8:
            return f"P2 : eval ≠ serve dès la position {r['div_eval_serve']} — la ForwardBatch d évaluation diffère du service (champs {r['champs_batch_serve']})"
        if r["div_serve_hf"] is not None:
            return f"P3 : eval ≈ serve mais ≠ hf dès la position {r['div_serve_hf']} — défaut du forward acvram hors gabarit, à nommer par couche"
        return "non tranché : divergences tardives, à lire par position"
    if r["div_eval_serve"] is not None:
        return f"P2 (sans hf) : eval ≠ serve dès la position {r['div_eval_serve']}"
    return ("P1 ou P3 (sans hf) : eval = serve ; " + ("PPL ≤ 30 : forward plausible, fenêtrage/corpus en cause" if r["ppl_eval"] <= 30
                                                     else "PPL > 30 sur un texte connu : les deux chemins sont faux — bras hf requis"))


if __name__ == "__main__":
    sys.exit(main())
