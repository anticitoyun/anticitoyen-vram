#!/usr/bin/env python3
"""Duel A2 (audit poste7, item A2) : duel à TROIS moteurs — acvram, llama.cpp,
vLLM. TabbyAPI retiré le 15/09 (décision de poste7 : hors objectif B, ×2,5-8
derrière, aucun chemin MLA/MoE sm_120 ; ses chiffres du 14/09 restent dans
audit-a2 comme historiques ; son banc est dans outils/archives/). Consigne
de chef (14/09) : valider les rivaux sur Qwen3-Coder-30B-A3B, régime
pp2048, même NVML
power.draw.instant (idle soustrait), même fenêtre — contre notre chiffre
du jour (acvram, prefill 16 938 j/s, poste4 14/09).

Dénominateur COMMUN (poste4, 14/09, après ma première mesure buguée —
cache de préfixe non coupé, invite identique d'un essai à l'autre) :
L / durée d'un `generate(max_tokens=1)` complet, synchronisé aux deux
bouts, invite DIFFÉRENTE à chaque répétition, cache de préfixe désactivé,
2 passes de chauffe, 7 répétitions, médian ± σ. Délègue à deux scripts qui
implémentent CE protocole, chacun dans le venv de son moteur :
  outils/banc_prefill_chaud.py   (acvram, branche poste4 ea98f36)
  outils/banc_prefill_vllm.py    (vLLM, même dénominateur, écrit ici)
  outils/banc_llamacpp_reel.py   (llama.cpp, binaire réel sm_120 de poste8,
                                  même dénominateur, poste2 14/09)
Ce fichier ne fait qu'ENVELOPPER l'appel d'un relevé de puissance NVML sur
TOUTE la fenêtre (2 chauffes + 7 répétitions), comme demandé — pas de
fenêtre par répétition (trop courte pour l'échantillonnage, piège trouvé
sur mon premier essai).

Modèles, un par moteur (même modèle de base ; NVFP4 pour acvram et vLLM,
GGUF Q4_K_M pour llama.cpp — son seul format) :
  acvram     $(outils/racine_modeles.py)/Qwen3-Coder-30B-A3B-nvfp4
  vLLM       /mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4
             (NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4, NVIDIA ModelOpt,
             téléchargé le 14/09 avec accord explicite — 18,1 Gio)
  llama.cpp  GGUF et binaire fixés dans outils/banc_llamacpp_reel.py

Un seul moteur à la fois (VRAM insuffisante pour deux copies d'un 30B
simultanées) — libérer explicitement entre deux appels de ce script.

Rien ne charge de modèle sans --pour-de-vrai.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import sys
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.puissance_nvml import mesurer_idle, mesurer_pendant  # noqa: E402
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BASE = Path(MODELES)
DOSSIER_ACVRAM = BASE / "Qwen3-Coder-30B-A3B-nvfp4"
DOSSIER_VLLM = Path("/mnt/4TO_SATACMR_2022/Modeles/models_vllm"
                    "/Qwen3-Coder-30B-A3B-Instruct-FP4")

GPU = 0
PP_LEN = 2048
SORTIE = sorties() / "banc-4moteurs" / "resultats.json"


def _ecrire(moteur: str, donnees: dict) -> None:
    resultats = {}
    if SORTIE.exists():
        try:
            resultats = json.loads(SORTIE.read_text())
        except (json.JSONDecodeError, OSError):
            resultats = {}
    resultats[moteur] = donnees
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(resultats, indent=2, ensure_ascii=False))


def _lancer_et_parser(cmd: list, cwd: Path) -> dict:
    """Lance ``cmd``, rend la ligne `RESULTAT {json}` de stdout. Rend
    {"echec": stderr} si le processus echoue ou ne l'imprime pas."""
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    if r.returncode != 0:
        return {"echec": r.stderr[-4000:] or r.stdout[-4000:]}
    m = re.search(r"RESULTAT (\{.*\})", r.stdout)
    if not m:
        return {"echec": f"pas de ligne RESULTAT dans stdout :\n{r.stdout[-2000:]}"}
    return json.loads(m.group(1))


def _mesurer_acvram() -> dict:
    idle = mesurer_idle(GPU)
    venv_python = Path(os.environ.get("ACVRAM_PY", f"{REPO}/../../anticitoyen-vram/.venv/bin/python3"))
    cmd = [str(venv_python), "outils/banc_prefill_chaud.py", "m64e4", str(PP_LEN)]
    res, w, n, _med = mesurer_pendant(lambda: _lancer_et_parser(cmd, REPO), gpu=GPU)
    if "echec" in res:
        return res
    return {"pp_len": PP_LEN, "pp_js": res["med_jps"], "pp_sigma": res["sigma"],
           "pp_ms": res["ms"], "compteurs": res.get("compteurs"),
           "watts_idle": idle, "watts_fenetre": w, "watts_net": w - idle,
           "n_releves_puissance": n}


def _mesurer_llamacpp() -> dict:
    # Serveur llama-server lance et arrete PAR le banc (comme un client
    # reel) ; puissance et energie mesurees dedans, meme raison que vLLM.
    venv_python = Path(os.environ.get("ACVRAM_PY", f"{REPO}/../../anticitoyen-vram/.venv/bin/python3"))
    cmd = [str(venv_python), "outils/banc_llamacpp_reel.py"]
    res = _lancer_et_parser(cmd, REPO)
    if "echec" in res:
        return res
    return {"pp_len": PP_LEN, "pp_js": res["pp_js"], "pp_sigma": res["pp_sigma"],
            "pp_ms": round(1000.0 * PP_LEN / res["pp_js"], 2) if res["pp_js"] else None,
            "binaire_version": res.get("binaire_version"),
            "decodage_jetons_s": res.get("jetons_s"),
            "decodage_j_par_jeton_net": res.get("j_par_jeton_net")}


def _mesurer_vllm() -> dict:
    # Puissance mesuree DEDANS banc_prefill_vllm.py, autour de la seule
    # boucle de mesure — pas ici, qui envelopperait aussi le chargement
    # du modele (~30-60 s quasi-idle) et diluerait la mediane vers
    # l'idle (3 W nets mesures ainsi au premier essai, implausible).
    cmd = ["/opt/ia/vLLM/.venv/bin/python", "outils/banc_prefill_vllm.py",
          str(DOSSIER_VLLM), str(PP_LEN)]
    res = _lancer_et_parser(cmd, REPO)
    if "echec" in res:
        return res
    return {"pp_len": PP_LEN, "pp_js": res["med_jps"], "pp_sigma": res["sigma"],
           "pp_ms": res["ms"],
           "watts_idle": res["watts_idle"], "watts_fenetre": res["watts_fenetre"],
           "watts_net": res["watts_net"], "n_releves_puissance": res["n_releves_puissance"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true")
    ap.add_argument("--moteur", required=True, choices=["acvram", "llamacpp", "vllm"])
    a = ap.parse_args()

    print(f"BEAD audit poste7 A2 — duel 3 moteurs, moteur={a.moteur}")
    print(f"  pp {PP_LEN}, GPU {GPU}, denominateur commun "
          f"(poste4 14/09) : generate(max_tokens=1), invite differente "
          f"par repetition, cache de prefixe coupe, 2 chauffes + 7 rep, "
          f"mediane, puissance sur TOUTE la fenetre")
    if not a.pour_de_vrai:
        print("\nPLAN SEULEMENT. Rien n'a tourne.")
        return 0

    if a.moteur == "acvram":
        if not DOSSIER_ACVRAM.exists():
            print(f"ECHEC / CAUSE: dossier absent {DOSSIER_ACVRAM}")
            return 2
        donnees = _mesurer_acvram()
    elif a.moteur == "vllm":
        if not DOSSIER_VLLM.exists():
            print(f"ECHEC / CAUSE: dossier absent {DOSSIER_VLLM}")
            return 2
        donnees = _mesurer_vllm()
    else:
        donnees = _mesurer_llamacpp()

    if "echec" in donnees:
        print(f"ECHEC / CAUSE:\n{donnees['echec']}")
        return 2

    print(f"  pp{PP_LEN} : {donnees['pp_js']} j/s (sigma {donnees['pp_sigma']}, "
          f"{donnees['pp_ms']} ms/pas)  "
          f"puissance fenetre {donnees['watts_fenetre']:.0f} W, "
          f"idle {donnees['watts_idle']:.0f} W, net {donnees['watts_net']:.0f} W "
          f"({donnees['n_releves_puissance']} releves)")
    if donnees.get("compteurs"):
        print(f"  compteurs de chemin : {donnees['compteurs']}")

    _ecrire(a.moteur, donnees)
    print(f"\nFAIT / TESTE: {SORTIE} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
