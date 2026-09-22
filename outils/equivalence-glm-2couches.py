#!/usr/bin/env python3
"""Équivalence CPU 2 couches, GLM-4.7-Flash — item (2) de Sage
(revue/sage-lancement-14-09.md §2, scellé le 14/09) : forward CPU bf16 des
couches 0-1 (dense + première MoE), acvram CONTRE HF transformers, même
invite de 16 jetons.

Critère RÉÉCRIT le 15/09 (Sage, revue/sage-glm-equivalence-15-09.md §2
puis §3) : le seuil absolu (max|Δlogit| ≤ 5e-2 global) est RETIRÉ — sur
des logits de 10-30, l'ulp bf16 vaut 0,0625-0,25 ; un seuil sous l'ulp ne
peut pas rendre « vrai », ce n'est pas un contrôle (REGLES §4). Remplacé
par `acvram.quant.equivalence` : par position, cumulatif, top-1 identique
ET delta ≤ 2 ulp bf16 de max_j|logit_ref,j| ET cos ≥ 0,9999 — sauf
ex-aequo PROUVÉ (écart top-1/top-2 de la référence seule ≤ 2 ulp) ; ≤ 2
ex-aequo autorisés sur 16 ; pas de cosinus global (il masque une position
fausse derrière quinze bonnes — 0,999556 passait avec un delta de 1,815).
Le multiplicateur « 2 ulp » est négocié, pas mesuré (§3) : recalage prévu
sur un témoin référence-contre-elle-même, pas encore fait.

EXÉCUTÉ le 14/09 au soir, sur le correctif d'Océane fusionné (`16bac2f`,
bead anticitoyen-vram-992, les trois listes MLA portent `glm4_moe_lite`).
RÉSULTAT initial (ancien seuil global) : RÉFUTÉ — pire |Δlogit| 5,0669,
pire cosinus 0,952137. Trois bogues trouvés et corrigés depuis (routeur
et biais de correction en bf16 au lieu de fp32, tiers hôte ignorant
`--format` sans GPU visible) : rejoué au nouveau critère par position,
**11/16 passent** (revue/equivalence-glm-14-09.md, table complète avec
delta en ulp). PAS encore de conversion sur ce seul chiffre — voir la
recommandation dans cette même revue.

Incident de procédure corrigé après coup (pas re-exécuté) : la première
exécution de l'étape `acvram` a tourné sur `cuda:0` (auto_plan) sans
`carte.sh`, faute d'un `device_override` explicite — voir le commentaire
dans `etape_acvram`.

Méthode, en quatre étapes (mode `tout`) :
  1. extraire  : découpe couches 0-1 + embed/norm/lm_head du bf16 source
                 (~1,5 Gio, comme prédit par Sage) dans un répertoire HF
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
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SOURCE = "/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16"
SCRATCH = Path("/tmp/glm-equivalence-2couches")
MINI = SCRATCH / "mini-hf"
CONVERTI = SCRATCH / "mini-acvram"
VENV_PROJET = os.environ.get("ACVRAM_PY", f"{Path(__file__).resolve().parents[1]}/../../anticitoyen-vram/.venv/bin/python")
VENV_VLLM = "/opt/ia/vLLM/.venv/bin/python"
N_JETONS = 16
VOCAB_SUR = 150000  # marge sous vocab_size=154880, evite les ids speciaux (154820+)
# SEUIL_DELTA/SEUIL_COS (5e-2 global, 0,999) RETIRES le 15/09 : sous l'ulp
# bf16 lui-meme, un seuil absolu ne peut pas rendre "vrai" (Sage §2). Le
# critere vit maintenant dans acvram.quant.equivalence (par position).


def invite() -> list:
    return [(1000 + i * 13) % VOCAB_SUR + 10 for i in range(N_JETONS)]


# --------------------------------------------------------------------------
# Profil par couche (Sage, revue/sage-bissection-w4a16-verdict-17-09.md §2,
# 17/09) : le chantier « 1 % » (acvram W4A16 1,028 contre Marlin 1,016, MEMES
# poids) n'est dans aucun noyau (deja verifie par Laure/Sage) -- ce qui reste
# est la mathematique commune aux deux chemins. Ici, poids bf16 PURS des deux
# cotes (GLM-4.7-Flash-srcbf16-bf16 contre HF sur la source) : la
# quantification est retiree comme variable, seul le CODE peut differer.
#
# Invite REELLE (pas synthetique comme `invite()` ci-dessus) : une invite
# synthetique a des statistiques d'activation plates, qui masqueraient
# precisement le genre de defaut cherche (RoPE, normes, routeur MoE sensibles
# a la distribution reelle des activations). >= 256 jetons, prefixe GLM
# `[gMASK]<sop>` en tete (obligatoire depuis le 16/09, sinon la comparaison
# ne vaut rien -- modele-sans-son-prefixe-de-sequence.md).
PROFIL_SCRATCH = Path("/tmp/glm-profil-couches")
PROFIL_ACVRAM_BF16 = _RACINE + "/GLM-4.7-Flash-srcbf16-bf16"
PROFIL_CORPUS = str(Path(__file__).resolve().parent.parent
                    / "scratchpad/corpus-calib-k48/bras-A-anglais.txt")
N_JETONS_PROFIL = 300           # marge confortable sur le >= 256 demande
SEUIL_RELATIF_COUCHE = 0.005    # 0,5 % par couche (Sage §2)
SEUIL_CROISSANCE = 1.5          # x1,5 d'une couche a la suivante
SEUIL_SITE = 3.0                # un site suspect : x3 au-dela du profil attendu


def etape_profil_invite() -> None:
    """Tokenise l'invite reelle UNE SEULE FOIS (venv acvram) et publie les
    identifiants partages -- evite tout risque de divergence entre les deux
    tokenisations independantes (acvram et HF) d'un meme texte."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from acvram.server.chat import load_tokenizer

    tok = load_tokenizer(SOURCE)
    prefixe = tok.encode_brut("")
    with open(PROFIL_CORPUS, encoding="utf-8", errors="replace") as fh:
        texte = fh.read()
    corps = tok.encode(texte, add_special_tokens=False)
    n = N_JETONS_PROFIL - len(prefixe)
    assert len(corps) >= n, f"corpus trop court : {len(corps)} < {n}"
    ids = prefixe + corps[:n]
    assert len(ids) >= 256, f"invite {len(ids)} < 256 jetons demandes"

    PROFIL_SCRATCH.mkdir(parents=True, exist_ok=True)
    with open(PROFIL_SCRATCH / "ids.json", "w") as fh:
        json.dump(ids, fh)
    print(f"  invite : {len(ids)} jetons ({len(prefixe)} de prefixe "
         f"[gMASK]<sop> + {n} du corpus), premiers ids {ids[:6]}", flush=True)


def _charger_ids() -> list:
    with open(PROFIL_SCRATCH / "ids.json") as fh:
        return json.load(fh)


def _accroche(module, captures: dict, cle: str):
    """Capture la sortie d'un sous-module (tenseur nu ou premiere valeur
    d'un tuple, HF renvoie (sortie, poids_attention) sur self_attn)."""
    import torch as _torch

    def crochet(mod, entree, sortie):
        t = sortie[0] if isinstance(sortie, tuple) else sortie
        captures.setdefault(cle, []).append(t.detach().to(_torch.float32).clone())
    return module.register_forward_hook(crochet)


def etape_profil_acvram() -> None:
    """Charge GLM-4.7-Flash-srcbf16-bf16 (nos poids bf16 purs) sur CPU, à
    sec, et capture la sortie de chaque couche ET de ses quatre sous-blocs
    (norme d'entree, attention MLA, norme post, MoE/MLP) sur UN SEUL passage
    prefill couvrant toute l'invite -- pas de decoupage (runner.py:114,
    "le prefill n'est pas decoupe"), donc un seul forward par couche."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # avant tout import torch/acvram
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    ids = _charger_ids()
    loaded = load_model(PROFIL_ACVRAM_BF16, dtype=torch.bfloat16,
                        max_model_len=len(ids) + 8, device_override="cpu")
    captures: dict = {}
    crochets = []
    for i, layer in enumerate(loaded.model.layers):
        crochets.append(_accroche(layer, captures, f"couche{i}"))
        crochets.append(_accroche(layer.input_layernorm, captures, f"norme_entree{i}"))
        # MLA : `self.linear_attn` (DecoderLayerGDN, loader.py:736-761), pas
        # `self_attn` -- nom herite du chemin GDN que la MLA partage.
        attn = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        crochets.append(_accroche(attn, captures, f"attention{i}"))
        crochets.append(_accroche(layer.post_attention_layernorm, captures, f"norme_post{i}"))
        crochets.append(_accroche(layer.mlp, captures, f"mlp{i}"))

    # `enable_prefix_cache` (defaut True) photographie l'etat GDN/MLA a une
    # frontiere interieure a l'invite (`_frontiere_insta`, runner.py:637-652)
    # en RAM epinglee -- `torch.empty_like(..., pin_memory=True)` exige un
    # contexte CUDA, absent a sec (`CUDA_VISIBLE_DEVICES=""`). Coupe : aucun
    # sens ici (une seule requete, jamais reprise), et evite le crash.
    engine = Engine(loaded, None, max_batch_size=1,
                    max_model_len=len(ids) + 8, enable_cuda_graphs=False,
                    enable_prefix_cache=False)
    engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=1), request_id="s0")
    n_pas = 0
    while engine.running or engine.waiting:
        engine.step()
        n_pas += 1
        if n_pas > 5:
            raise RuntimeError("prefill non termine en 5 pas — decoupage inattendu ?")
    for c in crochets:
        c.remove()

    n_couches = len(loaded.model.layers)
    assert all(f"couche{i}" in captures for i in range(n_couches)), \
        "toutes les couches n'ont pas ete capturees"
    for i in range(n_couches):
        for cle in (f"couche{i}", f"norme_entree{i}", f"attention{i}",
                   f"norme_post{i}", f"mlp{i}"):
            assert len(captures[cle]) == 1, f"{cle} capture {len(captures[cle])} fois, attendu 1"

    PROFIL_SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.save({"n_couches": n_couches,
               **{k: v[0] for k, v in captures.items()}},
              PROFIL_SCRATCH / "acvram.pt")
    print(f"  {n_couches} couches capturees (acvram, bf16 pur, CPU)", flush=True)


def etape_profil_hf() -> None:
    """Meme invite, HF `AutoModelForCausalLM` sur la source, memes quatre
    sous-blocs par couche. A sec : `CUDA_VISIBLE_DEVICES` deja pose a "" par
    l'appelant en mode "profil" ; pose ici aussi pour un lancement direct."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from transformers import AutoModelForCausalLM

    modele = AutoModelForCausalLM.from_pretrained(SOURCE, dtype=torch.bfloat16)
    modele.eval()
    ids = _charger_ids()

    captures: dict = {}
    crochets = []
    for i, layer in enumerate(modele.model.layers):
        crochets.append(_accroche(layer, captures, f"couche{i}"))
        crochets.append(_accroche(layer.input_layernorm, captures, f"norme_entree{i}"))
        crochets.append(_accroche(layer.self_attn, captures, f"attention{i}"))
        crochets.append(_accroche(layer.post_attention_layernorm, captures, f"norme_post{i}"))
        crochets.append(_accroche(layer.mlp, captures, f"mlp{i}"))

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        modele(input_ids)
    for c in crochets:
        c.remove()

    n_couches = len(modele.model.layers)
    PROFIL_SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.save({"n_couches": n_couches,
               **{k: v[0] for k, v in captures.items()}},
              PROFIL_SCRATCH / "hf.pt")
    print(f"  {n_couches} couches capturees (HF, {modele.config.num_hidden_layers} "
         f"couches declarees, rope_interleave={modele.config.rope_interleave})",
         flush=True)


def _relatif(a, h) -> float:
    """||a - h|| / ||h||, sur le tenseur complet [jetons, dimension]."""
    import torch
    d = (a.to(torch.float64) - h.to(torch.float64)).norm()
    n = h.to(torch.float64).norm()
    return float(d / n) if n > 0 else float("inf")


def etape_profil_comparer() -> int:
    """Erreur relative ||Δh||/||h_HF|| par couche (scellé <= 0,5 %, croissance
    <= x1,5) ; si une couche saute d'un facteur >= 3 au-dela de la croissance
    attendue, detail par sous-bloc de CETTE couche (Sage §2)."""
    import torch

    acv = torch.load(PROFIL_SCRATCH / "acvram.pt", weights_only=True)
    hf = torch.load(PROFIL_SCRATCH / "hf.pt", weights_only=True)
    n = acv["n_couches"]
    if n != hf["n_couches"]:
        print(f"ECHEC / CAUSE: nombre de couches different ({n} vs {hf['n_couches']})")
        return 2

    erreurs = []
    for i in range(n):
        e = _relatif(acv[f"couche{i}"], hf[f"couche{i}"])
        erreurs.append(e)
        croissance = e / erreurs[i - 1] if i > 0 and erreurs[i - 1] > 0 else 1.0
        marque = "SITE >=x3" if (i > 0 and croissance >= SEUIL_SITE) else \
                ("CROISSANCE" if croissance > SEUIL_CROISSANCE else "ok")
        print(f"  couche {i:2d}: erreur relative {e * 100:.4f} % "
             f"(croissance x{croissance:.2f}) {marque}", flush=True)

    pire = max(range(n), key=lambda i: erreurs[i])
    hors_seuil = [i for i, e in enumerate(erreurs) if e > SEUIL_RELATIF_COUCHE]
    croissances_hors_seuil = [
        i for i in range(1, n)
        if erreurs[i - 1] > 0 and erreurs[i] / erreurs[i - 1] > SEUIL_CROISSANCE]

    print(f"\n  pire couche : {pire} (erreur relative {erreurs[pire] * 100:.4f} %)")
    print(f"  couches au-dela de {SEUIL_RELATIF_COUCHE * 100:.1f} % : {hors_seuil}")
    print(f"  croissances au-dela de x{SEUIL_CROISSANCE} : {croissances_hors_seuil}")

    site = next((i for i in range(1, n)
                if erreurs[i - 1] > 0 and erreurs[i] / erreurs[i - 1] >= SEUIL_SITE), None)
    if site is not None:
        print(f"\n  -- detail par sous-bloc de la couche {site} (site suspect) --")
        for sous_bloc in ("norme_entree", "attention", "norme_post", "mlp"):
            e = _relatif(acv[f"{sous_bloc}{site}"], hf[f"{sous_bloc}{site}"])
            print(f"    {sous_bloc:14s}: erreur relative {e * 100:.4f} %", flush=True)
        # Le residu (x + sortie d'attention, x + mlp) est une simple addition,
        # pas un module hooke -- meme calcul des deux cotes (DecoderLayerGDN.
        # forward model.py:1652, _mlp model.py:1787-1791), aucun site possible
        # dans l'addition elle-meme.

    ok = not hors_seuil and not croissances_hors_seuil
    print(f"\nRESULTAT VERDICT={'PASSE' if ok else 'REFUTE'}")
    return 0 if ok else 1


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
    logits complets a 16 positions par prefixes croissants.

    A SEC signifie AUCUN CONTEXTE CUDA, pas seulement "aucune donnee sur le
    GPU". Deux incidents le 14/09 au soir : (1) `device_override="cpu"` sans
    `CUDA_VISIBLE_DEVICES` laisse `load_model`/`torch` INITIALISER cuda:0
    (contexte + libs chargees, quelques centaines de Mio, visible sur
    `nvidia-smi --query-compute-apps`) meme si aucun tenseur n'y est place --
    releve par Jerome (PID 255501, chevauchement d'une prise de Laure) ; (2)
    un forward complet avait deja tourne sur cuda:0 sans carte.sh avant ca.
    `CUDA_VISIBLE_DEVICES=""` cache le GPU au process : `torch.cuda` ne peut
    alors PLUS l'initialiser, meme par accident -- verifie ici avec
    `nvidia-smi` avant d'affirmer que ca tourne a sec, pas suppose."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    if CONVERTI.exists():
        shutil.rmtree(CONVERTI)
    r = subprocess.run(
        [VENV_PROJET, "-m", "acvram", "convert", str(MINI),
         "--format", "bf16", "--no-awq", "--quant-device", "cpu",
         "--host-exec", "cpu", "--max-model-len", "64", "-o", str(CONVERTI)],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parent.parent))
    print(r.stdout[-2000:], file=sys.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"conversion mini echouee (code {r.returncode})")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # avant tout import torch/acvram
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

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
    dans le venv vLLM (glm4_moe_lite natif, transformers 5.17). A sec :
    `CUDA_VISIBLE_DEVICES` deja mis a "" par l'appelant (main() en mode
    "tout"), pose ici aussi pour un lancement direct en mode "hf"."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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
    """Critère par position (Sage, revue/sage-glm-equivalence-15-09.md
    §2 puis §3) : top-1 identique ET delta ≤ 2 ulp bf16 de
    max_j|logit_ref,j| ET cos ≥ 0,9999, sauf ex-aequo prouvé (écart
    top-1/top-2 de la référence seule ≤ 2 ulp) ; ≤ 2 ex-aequo sur 16 ;
    pas de cosinus global."""
    import math

    from acvram.quant.equivalence import ulp_bf16, verdict_global, verdict_position

    with open(SCRATCH / "logits-acvram.json") as fh:
        acv = json.load(fh)
    with open(SCRATCH / "logits-hf.json") as fh:
        hf = json.load(fh)
    if len(acv) != len(hf):
        print(f"ECHEC / CAUSE: nombre de positions different ({len(acv)} vs {len(hf)})")
        return 2

    verdicts = []
    for i, (a, h) in enumerate(zip(acv, hf)):
        delta = max(abs(x - y) for x, y in zip(a, h))
        echelle_ref = max(abs(v) for v in h)
        top1_ref = max(range(len(h)), key=lambda k: h[k])
        top1_nous = max(range(len(a)), key=lambda k: a[k])
        ordre = sorted(range(len(h)), key=lambda k: -h[k])
        ecart_top1_top2_ref = h[ordre[0]] - h[ordre[1]]
        na = math.sqrt(sum(x * x for x in a))
        nh = math.sqrt(sum(y * y for y in h))
        cos = sum(x * y for x, y in zip(a, h)) / (na * nh) if na and nh else 0.0

        v = verdict_position(delta, echelle_ref, top1_ref, top1_nous, cos,
                             ecart_top1_top2_ref)
        verdicts.append(v)
        u = ulp_bf16(echelle_ref)
        marque = "EX-AEQUO" if v.ex_aequo else ("ok" if v.ok else "ECHEC")
        print(f"  position {i}: delta={delta:.4f} ({delta / u:.2f} ulp) "
             f"cos={cos:.6f} {marque} — {v.raison}", flush=True)

    ok, raison = verdict_global(verdicts)
    print(f"RESULTAT {sum(1 for v in verdicts if v.ok)}/{len(verdicts)} positions "
         f"passent — {raison} — VERDICT={'PASSE' if ok else 'REFUTE'}")
    return 0 if ok else 1


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
            r = subprocess.run([VENV_VLLM, __file__, "hf"],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            if r.returncode != 0:
                return r.returncode
        else:
            etape_hf()
            return 0
    if mode in ("tout", "comparer"):
        return etape_comparer()

    # -- profil par couche (Sage §2, 17/09) : quatre etapes independantes,
    # comme "tout" ci-dessus mais sur le modele COMPLET et une invite reelle.
    if mode in ("profil", "profil_invite"):
        etape_profil_invite()
        if mode == "profil_invite":
            return 0
    if mode in ("profil", "profil_acvram"):
        etape_profil_acvram()
        if mode == "profil_acvram":
            return 0
    if mode in ("profil", "profil_hf"):
        if mode == "profil":
            r = subprocess.run([VENV_VLLM, __file__, "profil_hf"],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            if r.returncode != 0:
                return r.returncode
        else:
            etape_profil_hf()
            return 0
    if mode in ("profil", "profil_comparer"):
        return etape_profil_comparer()

    print(f"ECHEC / CAUSE: mode inconnu {mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
