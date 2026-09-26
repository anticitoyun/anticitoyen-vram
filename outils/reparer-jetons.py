#!/usr/bin/env python3
"""Réécrit le ``tokenizer.json`` des modèles convertis avant la v0.4.64.

Le chemin BPE de la conversion GGUF ne rendait insécables que les jetons de
type 3 (CONTROL) et laissait de côté le type 4 (USER_DEFINED). Sur GLM-4.7,
« <think> » tombe dans le type 4 : le gabarit de conversation le rendait
littéralement, l'encodeur le coupait en « < », « think », « > », et le modèle
répondait par une cascade de balises vides.

Rien à reconvertir : seul le ``tokenizer.json`` est faux, et le GGUF d'origine
porte les types. Le poids ne bouge pas.

    reparer-jetons.py --simuler   liste les modèles atteints
    reparer-jetons.py             les répare
"""
import json
import os
import shutil
import sys
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acvram.quant.gguf import GGUFFile        # noqa: E402

CONVERTIS = MODELES
SOURCES = "/mnt/4TO_SATACMR_2022/Modeles/models_gguf"
SIMULER = "--simuler" in sys.argv


def gguf_de(nom: str):
    """Le premier .gguf du dossier source de même nom — ou, à défaut, du seul
    dossier source dont le nom contient celui du converti (« Ornith-1.0-35B-kimi »
    a été converti depuis « Ornith-1.0-35B-Heretic-kimi-IQ4_XS »)."""
    d = os.path.join(SOURCES, nom)
    if not os.path.isdir(d):
        base = nom.lower().replace("-", "").replace("_", "").replace(".", "")
        cands = [c for c in os.listdir(SOURCES)
                 if os.path.isdir(os.path.join(SOURCES, c))
                 and all(m in c.lower().replace("-", "").replace("_", "").replace(".", "")
                         for m in base.split("kimi")[:1])
                 and base.replace("kimi", "") in c.lower().replace("-", "").replace("_", "").replace(".", "").replace("heretic", "").replace("iq4xs", "").replace("kimi", "")]
        if len(cands) != 1:
            return None
        d = os.path.join(SOURCES, cands[0])
    for f in sorted(os.listdir(d)):
        if f.endswith(".gguf") and "-of-" not in f or f.endswith("00001-of-00002.gguf"):
            return os.path.join(d, f)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
    return os.path.join(d, fs[0]) if fs else None


def main():
    atteints, reparés, sans_source = [], 0, []
    for nom in sorted(os.listdir(CONVERTIS)):
        tj = os.path.join(CONVERTIS, nom, "tokenizer.json")
        if not os.path.isfile(tj):
            continue
        try:
            t = json.load(open(tj))
        except Exception:                                     # noqa: BLE001
            continue
        if t.get("model", {}).get("type") != "BPE":
            continue
        deja = {a["id"] for a in (t.get("added_tokens") or [])}
        src = gguf_de(nom)
        if src is None:
            continue
        try:
            g = GGUFFile(src)
            toks = g.kv.get("tokenizer.ggml.tokens") or []
            tt = g.kv.get("tokenizer.ggml.token_type") or []
        except Exception as e:                                # noqa: BLE001
            sans_source.append((nom, str(e)[:40]))
            continue
        manquants = [i for i, ty in enumerate(tt) if ty == 4 and i not in deja]
        if not manquants:
            continue
        atteints.append((nom, len(manquants),
                         [toks[i] for i in manquants[:4]]))
        if SIMULER:
            continue
        for i in manquants:
            t.setdefault("added_tokens", []).append(
                {"id": i, "content": toks[i], "special": False,
                 "single_word": False, "lstrip": False,
                 "rstrip": False, "normalized": False})
        t["added_tokens"].sort(key=lambda a: a["id"])
        shutil.copy2(tj, tj + ".avant-jetons")
        with open(tj, "w", encoding="utf-8") as fh:
            json.dump(t, fh, ensure_ascii=False)
        reparés += 1

    print(f"{len(atteints)} modèle(s) au tokenizer incomplet")
    for nom, n, ex in atteints:
        print(f"  {nom[:52]:52s} {n:3d} jetons, ex. {ex}")
    for nom, err in sans_source:
        print(f"  SOURCE ILLISIBLE {nom[:44]} : {err}")
    if not SIMULER:
        print(f"{reparés} réparé(s) (ancien fichier gardé en .avant-jetons)")


if __name__ == "__main__":
    main()
