#!/usr/bin/env python3
"""verifier-contexte — la colonne ctx des menus contre le modèle et contre le plan acvram (poste7-menus-test-reel-contexte-20-09 § 3a).

Pour chaque alias des TSV (acvram-chemins, gguf-chemins, vllm-chemins) :
  ctx colonne  = 3e colonne du TSV (ce que les lanceurs passent tel quel)
  ctx modèle   = max_position_embeddings de config.json (text_config déplié) ou context_length de l'en-tête GGUF
  ctx plan     = acvram seulement : `acvram plan <dossier> --max-model-len <colonne> --json` à sec (CUDA_VISIBLE_DEVICES="") →
                 kv_max_tokens si le plan tient, sinon le refus nommé (avertissement « ne tient pas »)
  verdict      = FAUX (colonne > modèle) · RÉDUIT (colonne > plan : proposer le plus grand ctx qui tient = plan) · OK · REFUS (plan impossible)
Sortie : ~/TSV/contexte-verifie.tsv (alias · ctx colonne · ctx modèle · ctx plan · verdict · détail).
--appliquer réécrit la colonne des TSV (sauvegarde .avant-<date>) : RÉDUIT → plan, FAUX → min(modèle, plan) ; la ligne
réécrite porte « plan » en 6e colonne tant que ctx_tenu (moteur) n'a pas prouvé la valeur.
Charge : un `acvram plan` par alias acvram (~s de CPU) → --lot N et --depuis pour avancer par paquets entre les prises annoncées.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

TSV_DIR = Path(os.environ.get("ACVRAM_TSV_DIR", Path.home() / "TSV"))
CLASSES = {"acvram-chemins.tsv": "acvram", "gguf-chemins.tsv": "llamacpp", "vllm-chemins.tsv": "vllm"}
JOURNAL_SERVICE = Path(os.environ.get("ACVRAM_JOURNAL_SERVICE", "/tmp/acvram-serveur.log"))
DATE = dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def lire_ctx_gguf(chemin: Path) -> int | None:
    T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    try:
        with open(chemin, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            if struct.unpack("<I", f.read(4))[0] < 2:
                return None
            _, n_kv = struct.unpack("<QQ", f.read(16))

            def lire(t):
                if t == 8:
                    n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
                if t == 9:
                    st, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8)); return [lire(st) for _ in range(n)]
                fmt = T[t]; return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
            for _ in range(n_kv):
                k = lire(8); t, = struct.unpack("<I", f.read(4)); v = lire(t)
                if k.endswith(".context_length"):
                    return int(v)
    except Exception:
        return None
    return None


def ctx_modele(chemin: Path) -> tuple[int | None, str]:
    """(ctx, source) — config.json (text_config déplié) ou GGUF ; (None, raison) sinon."""
    if chemin.is_file() and chemin.suffix.lower() == ".gguf":
        c = lire_ctx_gguf(chemin); return c, "gguf:context_length" if c else "gguf sans context_length"
    if chemin.is_dir():
        cj = chemin / "config.json"
        if cj.exists():
            try:
                cfg = json.loads(cj.read_text())
            except Exception as exc:
                return None, f"config.json illisible : {exc}"
            for bloc in (cfg, cfg.get("text_config") or {}, cfg.get("llm_config") or {}):
                if isinstance(bloc, dict) and bloc.get("max_position_embeddings"):
                    return int(bloc["max_position_embeddings"]), "config.json:max_position_embeddings"
            return None, "config.json sans max_position_embeddings"
        # le GGUF du modèle, pas le projecteur multimodal (mmproj-*) : le premier fragment ou, sinon, le plus gros
        ggufs = [g for g in sorted(chemin.glob("*.gguf")) if "mmproj" not in g.name.lower()]
        if ggufs:
            premier = [g for g in ggufs if "-00001-of-" in g.name] or sorted(ggufs, key=lambda g: g.stat().st_size, reverse=True)
            c = lire_ctx_gguf(premier[0]); return c, "gguf:context_length" if c else f"gguf sans context_length ({premier[0].name})"
        return None, "ni config.json ni gguf"
    return None, "chemin absent"


def ctx_plan(chemin: Path, colonne: int, acvram: str, timeout: int = 600, profile: str = "") -> tuple[int | None, str]:
    """(kv_max_tokens, détail) via `acvram plan --json` à sec ; (None, refus nommé) si rien ne tient."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    # à sec, la carte est cachée : sans --profile le plan ne voit aucune carte et rend kv_max_tokens = 0 (poste9 21/09,
    # 161 alias « RÉDUIT ») ; le profil déclaré planifie pour la machine, la carte reste à poste2
    cmd = ["nice", "-n", "19", "ionice", "-c3", acvram, "plan", str(chemin), "--max-model-len", str(colonne), "--json"] \
        + (["--profile", profile] if profile else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"acvram plan : délai {timeout} s dépassé"
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()
        return None, "acvram plan rc " + str(r.returncode) + (" : " + msg[-1][:160] if msg else "")
    try:
        d = json.loads(r.stdout[r.stdout.index("{"):])
    except Exception:
        return None, "acvram plan : sortie non JSON"
    plan = d.get("plan", {})
    kv = plan.get("kv_max_tokens")
    avert = " ; ".join(plan.get("warnings", []) or [])
    if any("ne tient pas" in w or "aucune carte visible" in w for w in plan.get("warnings", []) or []):
        return None, avert[:200]
    if kv is None:
        return None, "plan sans kv_max_tokens"
    return int(kv), avert[:200]


def ctx_servi_par_alias(journal: Path | None = None) -> dict[str, int]:
    """{alias: N} — dernière ctx_tenu=N numérique du dernier bloc de service de l'alias
    (poste7-s2-k48-feu-vert-21-09 § 2b(c)) : une colonne ne portant pas cette valeur est FAUX,
    même si le modèle et le plan la laisseraient passer — la ligne servie fait foi."""
    journal = JOURNAL_SERVICE if journal is None else journal
    try:
        texte = journal.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    bornes = [(m.start(), m.group(1)) for m in re.finditer(r"acvram : service de (\S+)", texte)]
    resultat: dict[str, int] = {}
    for i, (pos, alias) in enumerate(bornes):
        fin = bornes[i + 1][0] if i + 1 < len(bornes) else len(texte)
        vals = re.findall(r"ctx_tenu=(\d+)\b", texte[pos:fin])
        if vals:
            resultat[alias] = int(vals[-1])
    return resultat


def verdict(colonne: int, modele: int | None, plan: int | None, plan_detail: str, classe: str,
           servi: int | None = None) -> tuple[str, str]:
    if servi is not None:
        # une valeur réellement servie fait foi avant le modèle et le plan (§ 2b(c))
        return ("OK", f"servi={servi}") if colonne == servi else ("FAUX", f"colonne {colonne} ≠ servi {servi}")
    if modele is not None and colonne > modele:
        return "FAUX", f"colonne {colonne} > modèle {modele}"
    if classe == "acvram":
        if plan is None:
            return "REFUS", plan_detail
        if colonne > plan:
            return "RÉDUIT", f"colonne {colonne} > plan {plan} : proposer {plan}"
    return "OK", plan_detail if classe == "acvram" else ""


def lire_tsv(p: Path) -> list[list[str]]:
    if not p.exists():
        return []
    return [l.split("\t") for l in p.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv-dir", type=Path, default=TSV_DIR)
    ap.add_argument("--sortie", type=Path, default=None, help="défaut : <tsv-dir>/contexte-verifie.tsv")
    ap.add_argument("--acvram", default=shutil.which("acvram") or "acvram")
    ap.add_argument("--profile", default="rig-14900k-5090-3080ti",
                    help="profil déclaré pour planifier à sec (carte cachée) ; '' = sonder la machine")
    ap.add_argument("--alias", action="append", default=[], help="ne traiter que ces alias (répétable)")
    ap.add_argument("--lot", type=int, default=0, help="nombre d'alias acvram à planifier ce passage (0 = tous)")
    ap.add_argument("--depuis", type=int, default=0, help="indice de départ dans la liste des alias (reprise par lots)")
    ap.add_argument("--sans-plan", action="store_true", help="modèle seulement, aucun acvram plan (aucune charge)")
    ap.add_argument("--appliquer", action="store_true", help="réécrit la colonne ctx des TSV (RÉDUIT → plan, FAUX → min(modèle, plan))")
    a = ap.parse_args()
    sortie = a.sortie or a.tsv_dir / "contexte-verifie.tsv"
    anciens = {l[0]: l for l in lire_tsv(sortie)}
    lignes: dict[str, list[str]] = dict(anciens)
    ctx_servi = ctx_servi_par_alias()  # § 2b(c) : une valeur réellement servie prime sur modèle/plan
    plans_faits = 0
    entrees = []
    for nom, classe in CLASSES.items():
        for l in lire_tsv(a.tsv_dir / nom):
            if len(l) >= 3 and (not a.alias or l[0] in a.alias):
                entrees.append((classe, nom, l))
    bilan = {"OK": 0, "RÉDUIT": 0, "FAUX": 0, "REFUS": 0, "?": 0}
    for i, (classe, nom, l) in enumerate(entrees):
        if i < a.depuis:
            continue
        al, chemin, col = l[0], Path(l[1]), l[2]
        try:
            colonne = int(col)
        except ValueError:
            lignes[al] = [al, col, "?", "?", "?", "colonne non numérique"]; bilan["?"] += 1; continue
        modele, src = ctx_modele(chemin)
        plan = None; detail_plan = "sans plan"
        if classe == "acvram" and not a.sans_plan and chemin.exists():
            if a.lot and plans_faits >= a.lot:
                # au-delà du lot : on garde l'ancienne ligne si elle existe, sinon on note « à planifier »
                if al in anciens:
                    continue
                lignes[al] = [al, str(colonne), str(modele or "?"), "?", "?", f"{src} ; plan à faire (lot)"]; bilan["?"] += 1
                continue
            plan, detail_plan = ctx_plan(chemin, colonne, a.acvram, profile=a.profile); plans_faits += 1
        servi = ctx_servi.get(al)
        v, det = verdict(colonne, modele, plan, detail_plan, classe, servi)
        if servi is None and (classe != "acvram" or a.sans_plan):
            v = "FAUX" if (modele is not None and colonne > modele) else ("OK" if modele is not None else "?")
            det = f"{src}" + ("" if modele is not None else " : ctx modèle inconnu")
        lignes[al] = [al, str(colonne), str(modele if modele is not None else "?"), str(plan if plan is not None else ("-" if classe != "acvram" else "?")), v, det]
        bilan[v if v in bilan else "?"] += 1
        print(f"{al}\t{colonne}\t{modele or '?'}\t{plan if plan is not None else '-'}\t{v}\t{det[:100]}", flush=True)
    texte = "# alias\tctx colonne\tctx modèle\tctx plan\tverdict\tdétail  (verifier-contexte, " + DATE + ")\n" + "".join("\t".join(lignes[k]) + "\n" for k in sorted(lignes))
    sortie.parent.mkdir(parents=True, exist_ok=True)
    if sortie.exists():
        shutil.copy2(sortie, sortie.with_name(sortie.name + f".avant-{DATE}"))
    sortie.write_text(texte, encoding="utf-8")
    print("bilan : " + ", ".join(f"{k} {v}" for k, v in bilan.items()) + f" → {sortie}")
    if a.appliquer:
        corriges = 0
        for nom, classe in CLASSES.items():
            p = a.tsv_dir / nom
            if not p.exists():
                continue
            src_lignes = p.read_text(encoding="utf-8").splitlines()
            nouvelles = []
            for l in src_lignes:
                c = l.split("\t")
                if l and not l.startswith("#") and len(c) >= 3 and c[0] in lignes:
                    _, colonne, modele, plan, v, _ = lignes[c[0]]
                    cible = None
                    if v == "RÉDUIT" and plan.isdigit():
                        cible = int(plan)
                    elif v == "FAUX" and modele.isdigit():
                        cible = min(int(modele), int(plan)) if plan.isdigit() else int(modele)
                    if cible is not None and str(cible) != c[2]:
                        # poste7-3b-lanceur-contexte-20-09 : une colonne posée par le plan, pas par une requête tenue,
                        # est marquée « plan » (6e colonne) tant que ctx_tenu (moteur) n'existe pas
                        c[2] = str(cible); corriges += 1
                        c = c[:5] + ["plan"] if len(c) >= 5 else c + [""] * (5 - len(c)) + ["plan"]
                nouvelles.append("\t".join(c) if l and not l.startswith("#") else l)
            if nouvelles != src_lignes:
                shutil.copy2(p, p.with_name(p.name + f".avant-{DATE}"))
                p.write_text("\n".join(nouvelles) + "\n", encoding="utf-8")
        print(f"appliqué : {corriges} colonne(s) réécrite(s) (sauvegardes .avant-{DATE})")
    return 0 if bilan["FAUX"] == 0 and bilan["REFUS"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
