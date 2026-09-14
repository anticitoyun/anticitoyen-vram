#!/usr/bin/env python3
"""Équivalence CPU 2 couches, GLM-4.7-Flash — item (2) de poste7
(revue/poste7-lancement-14-09.md §2, scellé le 14/09) : forward CPU bf16 des
couches 0-1 (dense + première MoE), acvram CONTRE HF transformers, même
invite de 16 jetons. Seuil scellé : max |Δlogit| ≤ 5e-2 ET cosinus ≥ 0,999
sur les logits des 16 positions. Réfuté → PAS de conversion ce soir.

EXÉCUTÉ le 14/09 au soir, sur le correctif d'poste1 fusionné (`16bac2f`,
bead anticitoyen-vram-992, les trois listes MLA portent `glm4_moe_lite`).
RÉSULTAT : RÉFUTÉ — pire |Δlogit| 5,0669 (seuil 0,05), pire cosinus
0,952137 (seuil 0,999). Détail complet, tenseurs manquants écartés par
mesure directe, piste non tranchée (régime DÉGRADÉ sur bf16 pur) :
`revue/verdict-equivalence-glm-2couches-14-09.md`. PAS de conversion.

Incident de procédure corrigé après coup (pas re-exécuté) : la première
exécution de l'étape `acvram` a tourné sur `cuda:0` (auto_plan) sans
`carte.sh`, faute d'un `device_override` explicite — voir le commentaire
dans `etape_acvram`.

Méthode, en quatre étapes (mode `tout`) :
  1. extraire  : découpe couches 0-1 + embed/norm/lm_head du bf16 source
                 (~1,5 Gio, comme prédit par poste7) dans un répertoire HF
                 valide à 2 couches (config.json corrigé + un seul
                 fragment safetensors)
  2. acvram    : `python -m acvram convert` (bf16, sans AWQ, CPU) sur ce
                 mini-répertoire, puis un forward Engine à 16 requêtes de
                 préfixe croissant (1..16 jetons), capture du vecteur de
                 logits complet à chaque étape via `Engine._emit` — dans
                 le VENV DU PROJET (acvram installé)
  3. hf        : `AutoModelForCausalLM.from_pretrained` du même mini-
                 répertoire, un seul forward teacher-forcé des 16 jetons,
                 logits par position — dans le VENV VLLM (transformers
                 5.17, glm4_moe_lite natif, vérifié le 14/09)
  4. comparer  : max|Δlogit| et cosinus par position, verdict contre le
                 seuil scellé

    outils/carte.sh python outils/equivalence-glm-2couches.py tout
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SOURCE = "/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16"
SCRATCH = Path("/tmp/glm-equivalence-2couches")
MINI = SCRATCH / "mini-hf"
CONVERTI = SCRATCH / "mini-acvram"
VENV_PROJET = "~/Bureau/Claude/anticitoyen-vram/.venv/bin/python"
VENV_VLLM = "/opt/ia/vLLM/.venv/bin/python"
N_JETONS = 16
VOCAB_SUR = 150000  # marge sous vocab_size=154880, evite les ids speciaux (154820+)
SEUIL_DELTA = 5e-2
SEUIL_COS = 0.999


def invite() -> list:
    return [(1000 + i * 13) % VOCAB_SUR + 10 for i in range(N_JETONS)]


def etape_extraire() -> None:
    """Écrit un mini-répertoire HF valide à 2 couches — config tronquée,
    UN SEUL fragment safetensors (les tenseurs voulus tiennent large sous
    la limite de fragment habituelle, pas besoin d'un index)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    MINI.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(SOURCE, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["num_hidden_layers"] = 2
    with open(MINI / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    with open(os.path.join(SOURCE, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]

    voulus = {k for k in weight_map
             if k.startswith(("model.embed_tokens.", "model.norm.", "lm_head."))
             or k.startswith("model.layers.0.") or k.startswith("model.layers.1.")}
    print(f"  {len(voulus)} tenseurs voulus", flush=True)

    par_fragment: dict[str, list[str]] = {}
    for k in voulus:
        par_fragment.setdefault(weight_map[k], []).append(k)

    tenseurs = {}
    for fragment, cles in par_fragment.items():
        with safe_open(os.path.join(SOURCE, fragment), framework="pt", device="cpu") as f:
            for k in cles:
                tenseurs[k] = f.get_tensor(k)
    save_file(tenseurs, str(MINI / "model.safetensors"))
    octets = sum(t.numel() * t.element_size() for t in tenseurs.values())
    print(f"  {octets / 1e9:.2f} Gio ecrits dans {MINI}", flush=True)


def etape_acvram() -> None:
    """Convertit le mini-repertoire (bf16, sans AWQ, CPU) puis capture les
    logits complets a 16 positions par prefixes croissants."""
    if CONVERTI.exists():
        shutil.rmtree(CONVERTI)
    r = subprocess.run(
        [VENV_PROJET, "-m", "acvram", "convert", str(MINI),
         "--format", "bf16", "--no-awq", "--quant-device", "cpu",
         "--host-exec", "cpu", "--max-model-len", "64", "-o", str(CONVERTI)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    print(r.stdout[-2000:], file=sys.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"conversion mini echouee (code {r.returncode})")

    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    # `--quant-device cpu` (etape de conversion, plus haut) ne force que le
    # device de la recherche AWQ -- SANS `device_override`, `load_model` (le
    # forward, ici) choisit le GPU par lui-meme via `auto_plan`. Le 14/09 au
    # soir, ce forward a tourne sur cuda:0 sans passer par carte.sh et a
    # chevauche une mesure ncu de poste4. "CPU" en tete de ce script n'etait
    # vrai qu'a moitie -- corrige ici, force explicitement.
    loaded = load_model(str(CONVERTI), dtype=torch.bfloat16, max_model_len=64,
                        device_override="cpu")
    toks = invite()
    logits_par_position = []
    for k in range(1, N_JETONS + 1):
        engine = Engine(loaded, None, max_batch_size=1, max_model_len=64,
                        enable_cuda_graphs=False)
        capture = {}
        orig_emit = engine._emit
        def espion(logits, seqs, _c=capture):
            _c["v"] = logits[0].to(torch.float32).tolist()
            return orig_emit(logits, seqs)
        engine._emit = espion
        engine.add_request(toks[:k], SamplingParams(temperature=0.0, max_tokens=1),
                           request_id="s0")
        for _ in range(3):
            if not engine.running and not engine.waiting:
                break
            engine.step()
        if "v" not in capture:
            raise RuntimeError(f"position {k-1} : aucune capture de logits")
        logits_par_position.append(capture["v"])
        del engine

    SCRATCH.mkdir(parents=True, exist_ok=True)
    with open(SCRATCH / "logits-acvram.json", "w") as fh:
        json.dump(logits_par_position, fh)
    print(f"  {len(logits_par_position)} positions capturees (acvram)", flush=True)


def etape_hf() -> None:
    """Un seul forward teacher-force des 16 jetons via HF transformers,
    dans le venv vLLM (glm4_moe_lite natif, transformers 5.17)."""
    import torch
    from transformers import AutoModelForCausalLM

    modele = AutoModelForCausalLM.from_pretrained(str(MINI), dtype=torch.bfloat16)
    modele.eval()
    toks = invite()
    ids = torch.tensor([toks], dtype=torch.long)
    with torch.no_grad():
        out = modele(ids)
    logits = out.logits[0].to(torch.float32)  # [16, vocab]

    SCRATCH.mkdir(parents=True, exist_ok=True)
    with open(SCRATCH / "logits-hf.json", "w") as fh:
        json.dump(logits.tolist(), fh)
    print(f"  {logits.shape[0]} positions capturees (HF)", flush=True)


def etape_comparer() -> int:
    import math

    with open(SCRATCH / "logits-acvram.json") as fh:
        acv = json.load(fh)
    with open(SCRATCH / "logits-hf.json") as fh:
        hf = json.load(fh)
    if len(acv) != len(hf):
        print(f"ECHEC / CAUSE: nombre de positions different ({len(acv)} vs {len(hf)})")
        return 2

    pire_delta, pire_cos = 0.0, 1.0
    for i, (a, h) in enumerate(zip(acv, hf)):
        deltas = [abs(x - y) for x, y in zip(a, h)]
        d = max(deltas)
        num = sum(x * y for x, y in zip(a, h))
        na = math.sqrt(sum(x * x for x in a))
        nh = math.sqrt(sum(y * y for y in h))
        cos = num / (na * nh) if na and nh else 0.0
        pire_delta = max(pire_delta, d)
        pire_cos = min(pire_cos, cos)
        print(f"  position {i}: |delta| max {d:.4f}  cos {cos:.6f}", flush=True)

    verdict = pire_delta <= SEUIL_DELTA and pire_cos >= SEUIL_COS
    print(f"RESULTAT pire_delta={pire_delta:.4f} (seuil {SEUIL_DELTA}) "
         f"pire_cos={pire_cos:.6f} (seuil {SEUIL_COS}) "
         f"VERDICT={'PASSE' if verdict else 'REFUTE'}")
    return 0 if verdict else 1


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tout"
    if mode in ("tout", "extraire"):
        etape_extraire()
        if mode == "extraire":
            return 0
    if mode in ("tout", "acvram"):
        etape_acvram()
        if mode == "acvram":
            return 0
    if mode in ("tout", "hf"):
        if mode == "tout":
            r = subprocess.run([VENV_VLLM, __file__, "hf"])
            if r.returncode != 0:
                return r.returncode
        else:
            etape_hf()
            return 0
    if mode in ("tout", "comparer"):
        return etape_comparer()
    print(f"ECHEC / CAUSE: mode inconnu {mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
