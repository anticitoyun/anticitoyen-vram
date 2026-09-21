#!/usr/bin/env python3
"""Dérive l'usage (tags fermés) de chaque modèle servi depuis des signaux LOCAUX
— jamais une carte HF en ligne, jamais une sortie de modèle. Écrit la colonne
« usage » de ~/TSV/notes-modeles.tsv et le journal de preuves ~/TSV/usage-sources.tsv
(une ligne par tag : alias · tag · source · extrait). Appelé par modeles-a-jour :
la table n'est jamais figée à la main, elle se régénère.

Vocabulaire fermé (15), séparateur « · » dans la colonne usage. Précédence des
tags de contenu : porno ⊃ explicite ⊃ nsfw ⊃ sans-censure — un tag supérieur
écrit aussi tous les inférieurs ; un modèle aligné n'en porte aucun.
"""
from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

VOCAB = ["code", "agent-outils", "chat", "créatif-récit", "jeu-de-rôle",
         "raisonnement-math", "traduction", "résumé-RAG", "vision",
         "long-contexte", "sans-censure", "nsfw", "explicite", "porno", "vedette"]

CTX_LONG = 65536

# (tag, source par défaut, regex sur le texte « alias + modèle » en minuscules).
# vision, long-contexte et vedette ne sont PAS ici : ils viennent de signaux
# structurés (VISION_TSV, ctx servi, colonne vedette), pas d'un mot du nom.
_MOTS = [
    ("code",              r"coder|(?<![a-z])code(?![a-z])"),
    ("agent-outils",      r"agent|tool|function|fonction"),
    ("chat",              r"instruct|(?<![a-z])it(?![a-z])|chat"),
    ("créatif-récit",     r"creativ|story|writer|récit|recit|prose|novel"),
    ("jeu-de-rôle",       r"roleplay|(?<![a-z])rp(?![a-z])|(?<![a-z])erp(?![a-z])"),
    ("raisonnement-math", r"reason|thinking|(?<![a-z])math|(?<![a-z])r1(?![a-z])|deepseek-r"),
    ("traduction",        r"translat|multilingual|multilingue"),
    ("résumé-RAG",        r"summar|(?<![a-z])rag(?![a-z])|résumé|resume-rag"),
]
# tags de contenu adulte : mot dans le nom/modèle OU dans le README local.
_CENSURE = [
    ("sans-censure", r"abliterated|ablitérated|(?<![a-z])abl(?![a-z])|heretic|uncensored|unfiltered|décensuré|decensure"),
    ("nsfw",         r"nsfw|adult|(?<![a-z])18\+"),
    ("explicite",    r"(?<![a-z])erp(?![a-z])|nsfw-finetune|[eé]roti|lewd"),
    ("porno",        r"porn|xxx|hardcore"),
]
_ORDRE_CENSURE = ["sans-censure", "nsfw", "explicite", "porno"]


def _readme_local(dossier: str, limite: int = 4000) -> str:
    """Grep BORNÉ d'un README/model card dans le dossier du modèle (jamais en
    ligne). On lit au plus quelques fichiers de métadonnées, tronqués."""
    if not dossier:
        return ""
    d = Path(dossier)
    if not d.is_dir():
        return ""
    morceaux = []
    for nom in ("README.md", "README.txt", "model_card.md", "MODEL_CARD.md"):
        f = d / nom
        if f.is_file():
            try:
                morceaux.append(f.read_text(errors="ignore")[:limite])
            except Exception:
                pass
    return " ".join(morceaux).lower()


def deriver_usage(alias: str, model: str, ctx_servi: int, vision_status: str,
                  readme: str = "", vedette: bool = False):
    """Fonction PURE : rend (tags ordonnés selon VOCAB, preuves) où chaque preuve
    est (tag, source, extrait). Aucun accès disque ni réseau — le README est passé
    déjà lu. C'est ce que le test cassant joue sur sa fixture."""
    texte = f"{alias} {model}".lower()
    tags: dict[str, tuple[str, str]] = {}   # tag -> (source, extrait)

    for tag, motif in _MOTS:
        m = re.search(motif, texte)
        if m:
            source = "modele" if re.search(motif, model.lower()) else "nom"
            tags[tag] = (source, m.group(0)[:60])

    # « chat » est le tag de base de TOUT modèle servi : la passerelle répond en
    # chat OpenAI, donc aucun alias n'est « sans usage ». La détection instruct/
    # it/chat ci-dessus, si elle a mordu, prime (source « nom »/« modele ») ;
    # sinon on le pose par défaut.
    tags.setdefault("chat", ("défaut", "servi"))

    if vision_status == "vision":
        tags["vision"] = ("tsv-vision", "vision")
    if ctx_servi and int(ctx_servi) >= CTX_LONG:
        tags["long-contexte"] = ("ctx", str(ctx_servi))
    if vedette:
        tags["vedette"] = ("vedette", "≈")

    # contenu adulte : nom/modèle d'abord, README local ensuite.
    niveau = -1
    for i, (tag, motif) in enumerate(_CENSURE):
        hay, src = (texte, "modele" if re.search(motif, model.lower()) else "nom")
        if not re.search(motif, hay) and readme:
            if re.search(motif, readme):
                hay, src = readme, "readme-local"
        mm = re.search(motif, hay)
        if mm:
            niveau = max(niveau, i)
            tags[tag] = (src, mm.group(0)[:60])
    # précédence : le niveau atteint implique tous les inférieurs.
    if niveau >= 0:
        for j in range(niveau + 1):
            t = _ORDRE_CENSURE[j]
            tags.setdefault(t, ("implique", _ORDRE_CENSURE[niveau]))

    ordonnes = [t for t in VOCAB if t in tags]
    preuves = [(t, tags[t][0], tags[t][1]) for t in ordonnes]
    return ordonnes, preuves


def _charger():
    racine = Path(__file__).resolve().parent.parent   # outils/ -> racine du dépôt
    for p in (racine / "parc" / "lib", Path("/usr/share/acvram-parc/lib")):
        if (p / "acvram_parc.py").exists():
            sys.path.insert(0, str(p))
            break
    from acvram_parc import charger
    return charger()


def main() -> int:
    P = _charger()
    conf = tomllib.load(P.config_kimi.open("rb"))
    moteurs = set(P.moteurs)
    vision = {}
    vt = P.tsv("vision")
    if vt.exists():
        for l in vt.read_text().splitlines():
            if l and not l.startswith("#"):
                p = l.split("\t")
                if len(p) >= 2:
                    vision[p[0]] = p[1]
    ctx = {}
    for quoi in ("gguf", "vllm", "acvram"):
        t = P.tsv(quoi)
        if t.exists():
            for l in t.read_text().splitlines():
                if l and not l.startswith("#"):
                    p = l.split("\t")
                    if len(p) >= 3:
                        try:
                            ctx[p[0]] = int(p[2])
                        except ValueError:
                            pass
    notes_p = P.tsv("notes")
    lignes_notes = notes_p.read_text().splitlines() if notes_p.exists() else []
    par_alias = {}
    for l in lignes_notes:
        if l and not l.startswith("#"):
            p = l.split("\t")
            if p:
                par_alias[p[0]] = p

    sources = ["# alias\ttag\tsource\textrait — dérivé par usages-modeles.py, jamais à la main"]
    nouvelles = {}
    for alias, m in conf.get("models", {}).items():
        if m.get("provider") not in moteurs:
            continue
        model = m.get("model", "")
        dossier = ""
        # dossier depuis le TSV du bon moteur (déjà lu pour ctx via chemins)
        ancienne = par_alias.get(alias, [alias, "", "", "", ""])
        vedette = len(ancienne) > 4 and ancienne[4].lstrip().startswith("≈")
        readme = _readme_local(dossier)
        tags, preuves = deriver_usage(alias, model, ctx.get(alias, 0),
                                      vision.get(alias, ""), readme, vedette)
        nouvelles[alias] = " · ".join(tags)
        for t, src, ext in preuves:
            sources.append(f"{alias}\t{t}\t{src}\t{ext}")

    P.tsv_dir.joinpath("usage-sources.tsv").write_text("\n".join(sources) + "\n")
    # réécrit la colonne usage (5e) de notes-modeles.tsv en préservant le reste.
    out = []
    for l in lignes_notes:
        if not l or l.startswith("#"):
            out.append(l); continue
        p = l.split("\t")
        if p and p[0] in nouvelles:
            while len(p) < 5:
                p.append("")
            p[4] = nouvelles[p[0]]
            out.append("\t".join(p[:5]))
        else:
            out.append(l)
    notes_p.write_text("\n".join(out) + "\n")
    print(f"usages-modeles : {len(nouvelles)} modèles, {len(sources)-1} preuves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
