#!/usr/bin/env python3
"""Pièce 65 (23/09) — contrôle À SEC du chemin tensor-core sur tous les alias MoE du parc : pour chaque alias
(`acvram_manifest.json` + `config.json`), applique la règle STATIQUE du moteur (`acvram.engine.moe.forme_tensor_refus`,
la même que `MoEBlock._raison_tensor`) couche par couche, et liste les couches acceptées / refusées avec la raison.
Aucun poids chargé, aucune carte. Sortie : une ligne par alias, `--json` pour le détail.

    python outils/controle-moe-tensor-alias.py [--parc DOSSIER_DU_PARC] [--json sortie.json]

Lecture du manifeste : format de chaque tenseur d expert (`tensors[nom].format`), tables AWQ d activation
(`act_scale` dans `keys`) et `experts_sans_stats` (nombre de tables unité : si tous les experts à table sont
sans statistiques, les tables sont unité et le chemin tensor s applique — Coder officiel ; GLM calibA : non)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acvram.engine.moe import forme_tensor_refus  # noqa: E402

RE_EXPERT = re.compile(r"^(?:language_model\.)?(?:model\.)?layers\.(\d+)\.(?:mlp|feed_forward|block_sparse_moe)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$")


def _config_moe(cfg: dict) -> dict:
    c = cfg.get("text_config", cfg)
    E = c.get("num_experts") or c.get("num_local_experts") or c.get("n_routed_experts")
    return {"E": E, "K": c.get("hidden_size"), "I": c.get("moe_intermediate_size") or c.get("intermediate_size"),
            "top_k": c.get("num_experts_per_tok") or c.get("moe_topk")}


def controler_alias(dossier: Path) -> dict | None:
    man, cfg = dossier / "acvram_manifest.json", dossier / "config.json"
    if not man.is_file() or not cfg.is_file():
        return None
    m = json.load(open(man))
    tensors = m.get("tensors") or {}
    couches: dict[int, dict] = defaultdict(lambda: {"formats": set(), "act": 0, "n": 0, "shapes": set(), "hd_down": 0})
    for nom, v in tensors.items():
        r = RE_EXPERT.match(nom)
        if not r:
            continue
        c = couches[int(r.group(1))]
        c["formats"].add(v.get("format", "?")); c["n"] += 1
        c["act"] += any("act_scale" in k for k in v.get("keys", []))
        if r.group(3) in ("gate_proj", "w1") and v.get("shape"):
            c["shapes"].add(tuple(v["shape"]))
        if r.group(3) in ("down_proj", "w2"):
            c["hd_down"] = max(c["hd_down"], int(v.get("hadamard_block", 0) or 0))
    if not couches:
        return None
    info = _config_moe(json.load(open(cfg)))
    E, K, I = info["E"], info["K"], info["I"]
    n_act = sum(c["act"] for c in couches.values())
    sans_stats = m.get("experts_sans_stats")
    # tables unité si aucune table, ou si toutes les tables sont « sans statistiques » (identité explicite)
    awq_unite = n_act == 0 or (isinstance(sans_stats, int) and sans_stats >= n_act)
    detail = {}
    for i, c in sorted(couches.items()):
        detail[i] = forme_tensor_refus(c["formats"], K or 0, I or 0, K or 0, E or 0, awq_unite, c["hd_down"])
    refus = defaultdict(list)
    for i, r in detail.items():
        if r:
            refus[r].append(i)
    return {"alias": dossier.name, "E": E, "K": K, "I": I, "top_k": info["top_k"], "couches_moe": len(detail),
            "acceptees": sum(1 for r in detail.values() if not r), "tables_awq": n_act, "tables_unite": awq_unite,
            "refus": {r: (len(v), v[:6]) for r, v in refus.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parc", default=os.environ.get("ACVRAM_PARC") or _racine_modeles())
    ap.add_argument("--json")
    a = ap.parse_args()
    parc = Path(a.parc)
    if not parc.is_dir():
        print(f"parc absent : {parc}", file=sys.stderr); return 2
    resultats = [r for d in sorted(parc.iterdir()) if d.is_dir() and (r := controler_alias(d))]
    for r in resultats:
        etat = "ACCEPTÉ " if r["acceptees"] == r["couches_moe"] else ("REFUSÉ  " if r["acceptees"] == 0 else "PARTIEL ")
        raisons = " ; ".join(f"{n} couche(s) : {k}" for k, (n, _) in r["refus"].items())
        print(f"{etat} {r['alias']:<72} E={r['E']:<4} k={r['top_k']:<3} K={r['K']:<5} I={r['I']:<5} "
              f"couches {r['acceptees']}/{r['couches_moe']}  {raisons}")
    n_ok = sum(1 for r in resultats if r["acceptees"] == r["couches_moe"])
    print(f"— {len(resultats)} alias MoE : {n_ok} acceptés en entier, {sum(1 for r in resultats if 0 < r['acceptees'] < r['couches_moe'])} partiels, "
          f"{sum(1 for r in resultats if r['acceptees'] == 0)} refusés (repli GEMV nommé sur la ligne de régime)")
    if a.json:
        json.dump(resultats, open(a.json, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
