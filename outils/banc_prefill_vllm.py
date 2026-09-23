#!/usr/bin/env python3
"""Banc de prefill vLLM, MÊME dénominateur que
outils/banc_prefill_chaud.py (Laurine, branche laurine, ea98f36) :
generate(max_tokens=1) complet, invite DIFFÉRENTE à chaque répétition,
cache de préfixe désactivé, 2 passes de chauffe, 7 répétitions, médian ± σ.

Sans ce dénominateur commun, `vllm bench latency` mesure autre chose (lot +
génération) — remarque de Laurine, 14/09, duel A2.

    /opt/ia/vLLM/.venv/bin/python outils/banc_prefill_vllm.py <dossier_modele> <L>
"""
import json
import math
import sys
import time
from pathlib import Path
import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parent.parent))
from outils.puissance_nvml import mesurer_idle, mesurer_pendant  # noqa: E402

from vllm import LLM, SamplingParams

chemin_modele = sys.argv[1]
L = int(sys.argv[2])
REP = 7

# FlashInfer (backend par defaut) n'est pas installe dans ce venv
# ("depuis la reinstallation" de Jerome). FLASH_ATTN refuse ensuite : le
# cache KV FP8 (choisi par defaut pour ce checkpoint FP4) exige FA3/SM90
# ou FA4/SM100 — notre 5090 est SM120, aucun des deux. TRITON_ATTN est
# plus permissif (pas de contrainte d'architecture connue sur le cache
# FP8) — pas de piste plus prudente trouvee dans le temps imparti.
llm = LLM(model=chemin_modele, dtype="auto", enforce_eager=False,
         enable_prefix_caching=False, gpu_memory_utilization=0.85,
         attention_config={"backend": "TRITON_ATTN"},
         max_model_len=4096)  # comme acvram (banc_prefill_chaud.py) ;
         # le defaut du modele (262144) reserve plus de cache KV que la
         # carte n'en a de libre pour un seul prefill de 2048 jetons
SP = SamplingParams(temperature=0.0, max_tokens=1)


def un(rep: int) -> float:
    # Meme formule d'invite que banc_prefill_chaud.py, pour deux raisons :
    # comparable d'un moteur a l'autre, et jamais deux fois la MEME invite
    # (le cache de prefixe la servirait sinon en 27 ms quelle que soit L).
    prompt = [(1000 + rep * 7919 + i * 13) % 150000 + 10 for i in range(L)]
    t0 = time.perf_counter()
    llm.generate([prompt], SP, use_tqdm=False)
    return time.perf_counter() - t0


# Puissance echantillonnee ICI, autour de la SEULE boucle de mesure —
# pas autour du chargement du modele (~30-60 s, quasi-idle cote GPU
# pendant la lecture des poids) qui, mesure de l'exterieur (comme mon
# premier essai), diluait la mediane vers l'idle (3 W nets, implausible).
idle = mesurer_idle(0)


def _boucle():
    for r in range(2):
        un(100 + r)
    return [un(r) for r in range(REP)]


durees, watts, n_releves, _watts_median = mesurer_pendant(_boucle, gpu=0)
jps = sorted(L / d for d in durees)
med = jps[REP // 2]
moy = sum(jps) / REP
sig = math.sqrt(sum((x - moy) ** 2 for x in jps) / REP)
print("RESULTAT " + json.dumps({"L": L, "med_jps": round(med), "sigma": round(sig),
                                "ms": round(1000 * L / med, 1),
                                "watts_idle": idle, "watts_fenetre": watts,
                                "watts_net": watts - idle,
                                "n_releves_puissance": n_releves}))
