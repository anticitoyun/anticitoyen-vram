"""Pièce 25 (c) : contrôle KL sur des invites qui ROUTENT vers les experts
sans statistique AWQ (`experts_sans_stats_liste` du manifeste, pièce 25 (b)).

1. `selectionner` : pour chaque invite candidate d un corpus (une par ligne,
   ≥ 64 jetons), une trace de routage (`ACVRAM_TRACE_ROUTAGE`, prefill b=1,
   graphes off) → score = part des (jeton, couche, k) routés vers un expert
   sans stats ; les `--n` meilleures = CIBLÉES, les `--n` les plus basses =
   TÉMOINS (même longueur, même corpus). Écrit `ciblees/NN.txt`, `temoins/NN.txt`
   et `selection.json` (scores, part attendue = |sans stats| / |experts|).
2. `chaine.sh` (écrit à côté) : `decode-pas.py TEXTE_SEUL=1` (scellé E) sur
   chaque invite, KL max sur 8 pas (HF bf16 vs alias nvfp4) → `kl-<groupe>-NN.json`.
3. `juger` : médiane des KL max ciblées / témoins. SEUIL ÉCRIT AVANT :
   ratio ≤ 1,5 → TENU (les experts sans stats ne dégradent pas plus que le
   reste) ; 1,5-2,0 → MARGINAL ; > 2,0 → RÉFUTÉ, les experts sans stats sont
   la cause nommée (et le repli `mediane_couche` se juge par le même ratio,
   prédit ≤ 1,2). Alarme : score max des ciblées < 2 × part attendue → le
   corpus ne les atteint pas, aucune invite ne cible, verdict « non jugé ».

Usage : carte.sh python invites-experts-sans-stats.py selectionner ALIAS CORPUS.txt SORTIE/ [--n 8] [--max-invites 200]
        python invites-experts-sans-stats.py juger SORTIE/            (à sec, après la chaîne)
        python invites-experts-sans-stats.py score TRACE MANIFEST     (à sec : score d une trace)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))      # racine_modeles.py (outils/)

SEUIL_TENU, SEUIL_REFUTE, PREDIT_REPLI = 1.5, 2.0, 1.2


def sans_stats(manifest: dict) -> tuple[set, int]:
    """{(couche, expert)} sans stats (toutes projections confondues) et E."""
    liste = manifest.get("experts_sans_stats_liste", {})
    s = set()
    for couche, projs in liste.items():
        for _proj, experts in projs.items():
            for e in experts:
                s.add((int(couche), int(e)))
    E = 0
    for nom in manifest.get("tensors", {}):
        if ".mlp.experts." in nom:
            try:
                E = max(E, int(nom.split(".mlp.experts.")[1].split(".")[0]) + 1)
            except ValueError:
                pass
    return s, E


def score_trace(trace: str, cibles: set) -> tuple[float, int]:
    """Part des (jeton, couche, k) routés vers une cible, et nombre de triplets."""
    from acvram.memory.trace_routage import relire
    touches = total = 0
    for _jeton, couche, experts in relire(trace):
        for e in experts:
            total += 1
            touches += (couche, e) in cibles
    return (touches / total if total else 0.0), total


def selectionner(a) -> dict:
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.memory import trace_routage
    from acvram.server.chat import load_tokenizer
    from racine_modeles import racine_modeles
    chemin = a.alias if os.path.isdir(a.alias) else os.path.join(racine_modeles(), a.alias)
    manifest = json.load(open(os.path.join(chemin, "acvram_manifest.json")))
    cibles, E = sans_stats(manifest)
    couches = len({c for c, _ in cibles}) or 1
    part_attendue = len(cibles) / max(1, E * manifest.get("num_layers", couches))
    if not cibles:
        sys.exit("ÉCHEC / CAUSE : manifeste sans experts_sans_stats_liste (convertir avec le convert de la pièce 25) / SUITE : reconvertir")
    os.environ["ACVRAM_DISABLE_CUDA_GRAPHS"] = "1"
    tok = load_tokenizer(chemin)
    loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=2048)
    engine = Engine(loaded, tok, max_batch_size=1, max_model_len=2048)
    params = SamplingParams(temperature=0.0, max_tokens=1)
    lignes = [l.strip() for l in open(a.corpus, encoding="utf-8") if len(l.strip()) > 200][: a.max_invites]
    os.makedirs(a.sortie, exist_ok=True)
    scores = []
    for i, texte in enumerate(lignes):
        ids = tok.encode(texte, add_special_tokens=False)[: a.longueur]
        if len(ids) < 64:
            continue
        trace = os.path.join(a.sortie, f"trace-{i:04d}.txt")
        # une trace par invite : le module n a pas d `ouvrir`, on repose son état (chemin, fichier, actif)
        trace_routage.fermer()
        trace_routage._CHEMIN, trace_routage._ACTIF, trace_routage._FICHIER = trace, True, None
        list(engine.generate(ids, params))
        trace_routage.fermer()
        sc, n = score_trace(trace, cibles)
        scores.append({"i": i, "score": sc, "triplets": n, "jetons": len(ids), "texte": tok.decode(ids)})
        os.remove(trace)
    scores.sort(key=lambda s: -s["score"])
    ciblees, temoins = scores[: a.n], scores[-a.n:]
    for groupe, sel in (("ciblees", ciblees), ("temoins", temoins)):
        os.makedirs(os.path.join(a.sortie, groupe), exist_ok=True)
        for j, s in enumerate(sel):
            open(os.path.join(a.sortie, groupe, f"{j:02d}.txt"), "w", encoding="utf-8").write(s["texte"])
    r = {"alias": chemin, "experts_sans_stats": len(cibles), "E": E, "part_attendue": round(part_attendue, 4),
         "invites_notees": len(scores), "ciblees": [{k: v for k, v in s.items() if k != "texte"} for s in ciblees],
         "temoins": [{k: v for k, v in s.items() if k != "texte"} for s in temoins],
         "alarme": (ciblees[0]["score"] < 2 * part_attendue) if ciblees else True}
    json.dump(r, open(os.path.join(a.sortie, "selection.json"), "w"), indent=1)
    ecrire_chaine(a.sortie, chemin, a.n)
    print(f"[invites] {len(scores)} invites notées ; part attendue {part_attendue:.3f} ; ciblées "
          f"{[round(s['score'], 3) for s in ciblees]} ; témoins {[round(s['score'], 3) for s in temoins]}"
          + (" ; ALARME : le corpus n atteint pas les experts sans stats" if r["alarme"] else ""))
    return r


def ecrire_chaine(sortie: str, alias: str, n: int) -> None:
    """Un decode-pas TEXTE_SEUL par invite (KL max sur 8 pas), source HF à donner."""
    p = os.path.join(sortie, "chaine.sh")
    with open(p, "w") as f:
        f.write(f"""#!/usr/bin/env bash
# Pièce 25 (c) : KL max (HF bf16 vs alias nvfp4) sur les invites ciblées et témoins.
# usage : SRC=<source HF> ACVRAM_ARBRE=<arbre figé> outils/carte.sh bash {p}
set -u
SRC=${{SRC:?source HF}}; ARBRE=${{ACVRAM_ARBRE:-$PWD}}; PY=${{PY_ACVRAM:-$ARBRE/.venv/bin/python}}
export HF_PYTHON=${{HF_PYTHON:-/opt/ia/vLLM/.venv/bin/python}} MAXMEM=${{MAXMEM:-26GiB,80GiB}}
for groupe in ciblees temoins; do
  for i in $(seq -f %02g 0 {n - 1}); do
    TEXTE_SEUL=1 INVITE_TEXTE_FICHIER="{sortie}/$groupe/$i.txt" TEMOIN_KL_SORTIE="{sortie}/kl-$groupe-$i.json" \\
      "$PY" "$ARBRE/scratchpad/mm-diag-20-09/decode-pas.py" "{alias}" "$SRC" texte 0 2 > "{sortie}/log-$groupe-$i.txt" 2>&1
    echo "$groupe $i rc $?"
  done
done
"$PY" {os.path.abspath(__file__)} juger "{sortie}"
""")
    os.chmod(p, 0o755)


def juger(sortie: str) -> dict:
    kls = {"ciblees": [], "temoins": []}
    for groupe in kls:
        for fn in sorted(os.listdir(sortie)):
            if fn.startswith(f"kl-{groupe}-") and fn.endswith(".json"):
                d = json.load(open(os.path.join(sortie, fn)))
                v = d.get("kl_max", d.get("KL_max", d.get("kl")))
                if v is not None:
                    kls[groupe].append(float(v))
    sel = json.load(open(os.path.join(sortie, "selection.json"))) if os.path.exists(os.path.join(sortie, "selection.json")) else {}
    if not kls["ciblees"] or not kls["temoins"]:
        return {"verdict": "NON JUGÉ : KL manquantes", **{k: v for k, v in kls.items()}}
    mc, mt = statistics.median(kls["ciblees"]), statistics.median(kls["temoins"])
    ratio = mc / mt if mt > 0 else float("inf")
    if sel.get("alarme"):
        verdict = "NON JUGÉ : aucune invite ne cible les experts sans stats (alarme de sélection)"
    elif ratio <= SEUIL_TENU:
        verdict = f"TENU : ratio {ratio:.2f} ≤ {SEUIL_TENU} — les experts sans stats ne dégradent pas plus que le reste"
    elif ratio <= SEUIL_REFUTE:
        verdict = f"MARGINAL : ratio {ratio:.2f} entre {SEUIL_TENU} et {SEUIL_REFUTE}"
    else:
        verdict = f"RÉFUTÉ : ratio {ratio:.2f} > {SEUIL_REFUTE} — les experts sans stats sont la cause nommée (repli mediane_couche prédit ≤ {PREDIT_REPLI})"
    r = {"kl_ciblees": kls["ciblees"], "kl_temoins": kls["temoins"], "mediane_ciblees": mc, "mediane_temoins": mt,
         "ratio": ratio, "seuils": {"tenu": SEUIL_TENU, "refute": SEUIL_REFUTE, "predit_repli": PREDIT_REPLI}, "verdict": verdict}
    json.dump(r, open(os.path.join(sortie, "verdict.json"), "w"), indent=1)
    print(f"[invites] KL max médiane ciblées {mc:.4f} / témoins {mt:.4f} = {ratio:.2f} ; {verdict}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selectionner"); s.add_argument("alias"); s.add_argument("corpus"); s.add_argument("sortie")
    s.add_argument("--n", type=int, default=8); s.add_argument("--max-invites", type=int, default=200)
    s.add_argument("--longueur", type=int, default=280)
    j = sub.add_parser("juger"); j.add_argument("sortie")
    sc = sub.add_parser("score"); sc.add_argument("trace"); sc.add_argument("manifest")
    a = ap.parse_args()
    if a.cmd == "selectionner":
        selectionner(a)
    elif a.cmd == "juger":
        juger(a.sortie)
    else:
        cibles, E = sans_stats(json.load(open(a.manifest)))
        s_, n = score_trace(a.trace, cibles)
        print(f"score {s_:.4f} sur {n} triplets ({len(cibles)} experts cibles, E={E})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
