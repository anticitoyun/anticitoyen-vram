#!/usr/bin/env python3
"""Inventaire enrichi (palier 0) LU DES FICHIERS, reproductible — une règle
par colonne, aucune valeur qui ne vienne d'un fichier lu ; « N/A » quand la
source manque, jamais une supposition. Relu par tests/test_menus.py (e-h) qui
recalcule chaque ligne depuis le disque et compare.

  model_type, max_position_embeddings, rope_scaling : config.json (N/A sans)
  vision : config.json — vision_config présent ou architecture *Vision*/*VL*/
           *ConditionalGeneration → yes, sinon no ; N/A sans config
  tools  : « tools » dans tokenizer_config.json ou chat_template.jinja → yes,
           no si un gabarit existe sans, N/A sans gabarit
  thinking : ND (non déterminable à partir des fichiers)
  bpw    : acvram : formats des tenseurs du manifeste (nvfp4 → W4A16, sinon
           int4_awq / int8 / bf16 / fp16) ; vLLM : hf_quant_config.json
           quant_algo ; EXL3 et GGUF : le nom du dossier (N/A s'il ne dit rien)

    python outils/enrichir-inventaire-17-09.py > acvram-memoire/revue/inventaire-enrichi-palier0-17-09.tsv
"""
import csv
import json
import os
import re
import sys

INVENTAIRE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "acvram-memoire", "revue", "inventaire-disque-brut.tsv")
COLONNES = ("model", "format", "model_type", "max_position_embeddings", "rope_scaling",
            "vision", "tools", "thinking", "bpw")


def _json(chemin):
    try:
        with open(chemin) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def enrichir(chemin: str, fmt: str, nom: str) -> dict:
    c = _json(os.path.join(chemin, "config.json"))
    if c is None:
        mt = mpe = rs = vision = "N/A"
    else:
        mt = str(c.get("model_type", "N/A"))
        mpe = str(c["max_position_embeddings"]) if c.get("max_position_embeddings") is not None else "N/A"
        rs = json.dumps(c["rope_scaling"], sort_keys=True) if c.get("rope_scaling") else "None"
        archs = " ".join(c.get("architectures", []))
        vision = "yes" if ("vision_config" in c or re.search(r"Vision|VL\b|ConditionalGeneration", archs)) else "no"
    gabarit = ""
    for f in ("tokenizer_config.json", "chat_template.jinja"):
        try:
            with open(os.path.join(chemin, f), errors="ignore") as fh:
                gabarit += fh.read()
        except OSError:
            pass
    tools = "N/A" if not gabarit else ("yes" if "tools" in gabarit else "no")
    bpw = "N/A"
    if fmt == "NVFP4_acvram":
        m = _json(os.path.join(chemin, "acvram_manifest.json"))
        if m and isinstance(m.get("tensors"), dict):
            fmts = {v.get("format", "?") for v in m["tensors"].values()}
            for cle, val in (("nvfp4", "W4A16"), ("int4_awq", "int4_awq"), ("int8", "int8"),
                             ("fp16", "fp16"), ("bf16", "bf16")):
                if cle in fmts:
                    bpw = val
                    break
    elif fmt == "vLLM":
        q = _json(os.path.join(chemin, "hf_quant_config.json"))
        if q:
            bpw = str((q.get("quantization") or q).get("quant_algo", "N/A"))
    else:
        m = re.search(r"(Q\d(?:_K(?:_[SML])?|_0|_1)|IQ\d[_A-Z]*|\d(?:\.\d+)?bpw)", nom)
        if m:
            bpw = m.group(1)
    return {"model": nom, "format": fmt, "model_type": mt, "max_position_embeddings": mpe,
            "rope_scaling": rs, "vision": vision, "tools": tools, "thinking": "ND", "bpw": bpw}


def main():
    with open(INVENTAIRE, newline="") as f:
        lignes = list(csv.DictReader(f, delimiter="\t"))
    w = csv.DictWriter(sys.stdout, fieldnames=COLONNES, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in lignes:
        w.writerow(enrichir(r["Chemin"], r["Format"], r["Modèle"]))


if __name__ == "__main__":
    main()
