#!/usr/bin/env python3
"""Chiffre certifie b=12 mode moyen (Sage, sage-laure-15-09 § 3) : un bras
par processus (constante de module), 20 s au compteur en rondes, ctx 2048,
invite 256, comme la campagne modes-energie du 14/09. Rend pas moyen (ms),
t/s, J/jeton brut, W, en-tete REGLES §3, preuve du bras lue dans le module.

Usage : certifie-b12-15-09.py NOM_BRAS SORTIE.json [B=12]
"""
import json, os, subprocess, sys, time
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)

MAIN_REPO = os.environ.get("ACVRAM_ARBRE", subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() + "/../../anticitoyen-vram")
sys.path.insert(0, MAIN_REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch  # noqa: E402
import acvram.engine.model as modele  # noqa: E402
from acvram.engine.loader import load_model  # noqa: E402
from acvram.engine.runner import Engine  # noqa: E402
from acvram.engine.sampler import SamplingParams  # noqa: E402
from energie import Energie, nvml  # type: ignore  # noqa: E402

BRAS, SORTIE = sys.argv[1:3]
B_ARG = int(sys.argv[3]) if len(sys.argv) > 3 else 12
MODEL = os.environ.get("ACVRAM_MODELE_MESURE",
                       _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4")
B, CTX, PROMPT_LEN, CIBLE_S, REPOS_S = B_ARG, 2048, 256, 20.0, 30.0
if os.environ.get("CERT_PUR", "0") == "1":
    # CTX doit etre pose AVANT load_model (max_model_len) : 2560 -> 2300 pas >= 22 s meme a 9 ms/pas
    CTX = int(os.environ.get("CERT_CTX", "2560")); CIBLE_S = float(os.environ.get("CERT_CIBLE_S", "22"))
MAX_TOKENS_RONDE = CTX - PROMPT_LEN - 4
# Bras témoin du protocole ABAB de Laure (protocole-certifie-smi-abab-18-09,
# relayé par Jérôme) : reproduit l'ancien instrument (sous-processus
# `nvidia-smi` dans la fenêtre) pour comparaison, un run sur deux. Le défaut
# (CERT_SMI_TEMOIN absent) est déjà le correctif de sage-profil-verdict-18-09
# §3 — pas de variable pour l'ACTIVER, il l'est déjà.
CERT_SMI_TEMOIN = os.environ.get("CERT_SMI_TEMOIN", "0") == "1"


def _carte0():
    return nvml().cartes[0][1]


def smi(champs):
    """Ancien instrument (bras témoin ABAB) : sous-processus `nvidia-smi`
    par appel, celui-là même dont sage-profil-verdict-18-09 §3 a mesuré le
    biais (+0,12 ms/pas à b=1). N'est plus appelé que si CERT_SMI_TEMOIN=1."""
    return [c.strip() for c in subprocess.run(
        ["nvidia-smi", "-i", "0", f"--query-gpu={champs}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5).stdout.strip().split(",")]


class GardeSmi:
    """Compte les sous-processus `nvidia-smi` pendant la fenêtre mesurée —
    structurel, pas une consigne à se rappeler (REGLES §4 « instrument
    corrélé à la variable étudiée », sage-profil-verdict-18-09 §3 : un
    `nvidia-smi` par 200 pas biaisait b=1 de +0,12 ms/pas, 3,5 % à 287 t/s)."""

    def __enter__(self):
        self.n = 0
        self._orig = subprocess.run

        def _compte(cmd, *a, **kw):
            if isinstance(cmd, (list, tuple)) and cmd and "nvidia-smi" in str(cmd[0]):
                self.n += 1
            return self._orig(cmd, *a, **kw)
        subprocess.run = _compte
        return self

    def __exit__(self, *a):
        subprocess.run = self._orig


preuve = {"regime_ligne": __import__("acvram").regime_ligne(), "engine_regime": None, "bras": BRAS, "eager": os.environ.get("CERT_EAGER", "0") == "1", "ACVRAM_HYBRID_SLOTS": os.environ.get("ACVRAM_HYBRID_SLOTS"), "_MOE_DECODE_MMA_lu": getattr(modele, "_MOE_DECODE_MMA", "absent"),
          "_MOE_DECODE_MMA_BT_lu": getattr(modele, "_MOE_DECODE_MMA_BT", "absent"),
          "_MOE_DECODE_MMA_MIN_T_lu": getattr(modele, "_MOE_DECODE_MMA_MIN_T", "absent"),
          "_MOE_ROUTE_PACK_lu": getattr(modele, "_MOE_ROUTE_PACK", "absent"),
          "ACVRAM_MOE_DECODE_MMA_MIN_T_env": os.environ.get("ACVRAM_MOE_DECODE_MMA_MIN_T"),
          "modele": MODEL,
          "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "power_limit_w": nvml().plafond_w(_carte0()), "acvram": __import__("os").path.relpath(__import__("acvram").__file__, __import__("os").path.dirname(__import__("os").path.dirname(__import__("acvram").__file__)))}  # relatif : jamais un /home dans un artefact suivi (cliquet 20/09)
print(f"[PREUVE] {preuve}", flush=True)
# CERT_PLAN_LEN disparait avec le correctif du 17/09 (loader.py::_replanifier
# ne passait que max_model_len au planificateur, jamais max_concurrent_seqs
# -- budget KV toujours dimensionne pour 8 sequences, tiering.py:447) :
# passer B directement dimensionne le plan pour le VRAI lot, plus besoin de
# gonfler max_model_len pour compenser. `engine_regime` ci-dessous porte
# desormais `kv_budget=<jetons>/<sequences_planifiees>`.
loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=CTX,
                    max_concurrent_seqs=B)
vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000


def invite(k, n):
    return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(n)]


# (19/09, Sage) certifie ne pose plus AUCUNE variable de régime que le serveur ne pose pas :
# le plafond hybride suit le max_batch_size de l'Engine (graphs.py plafond_hybride, 24b6565) —
# le 568 t/s GLM du 17/09 était un régime d'instrument (HYBRID_SLOTS=12 posé ici), le service
# tournait à 155 (eager dès b=5). Un HYBRID_SLOTS posé PAR L'APPELANT reste respecté et se lit
# dans regime_ligne(), comme pour le serveur.
if os.environ.get("ACVRAM_HYBRID_SLOTS"):
    print(f"[certifie] ACVRAM_HYBRID_SLOTS={os.environ['ACVRAM_HYBRID_SLOTS']} posé par l'appelant (régime explicite)", flush=True)
PUR = os.environ.get("CERT_PUR", "0") == "1"   # fenetre de decodage PUR : lot constant, ni prefill ni traine
engine = Engine(loaded, None, max_batch_size=B, max_model_len=CTX,
                enable_cuda_graphs=os.environ.get("CERT_EAGER", "0") != "1")   # CERT_EAGER=1 : sans graphes (Sage 11)
engine._eos = set(); preuve["engine_regime"] = engine.regime_ligne()
if os.environ.get("CERT_EAGER", "0") != "1":
    engine.warm_graphs()
params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS_RONDE)
time.sleep(REPOS_S)
def _echantillon():
    """(horloge SM MHz, température °C) — sous-processus `nvidia-smi` en
    bras témoin (CERT_SMI_TEMOIN=1), lecture NVML en processus par défaut
    (sage-profil-verdict-18-09 §3)."""
    if CERT_SMI_TEMOIN:
        c, t_ = smi("clocks.sm,temperature.gpu")
        return int(c), int(t_)
    h = _carte0()
    return nvml().horloge_sm(h), nvml().temperature(h)


n_avant, n_rondes, n_pas = engine.stats.decode_tokens, 0, 0
sm, temp = [], []
temp_avant = _echantillon()[1]
if PUR:
    for k in range(B):
        engine.add_request(invite(1000 + k, PROMPT_LEN), params, request_id=f"p{k}")
    while any(not s_.prefilled for s_ in engine.running) or engine.waiting:
        engine.step()
    for _ in range(30): engine.step()
    torch.cuda.synchronize()
    n_avant = engine.stats.decode_tokens
    lot = len(engine.running)
    t_hote = time.perf_counter()
    with GardeSmi() as garde_smi, Energie() as e:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < CIBLE_S:
            engine.step(); n_pas += 1
            if len(engine.running) != lot:
                raise RuntimeError(f"lot variable pendant la fenetre pure : {len(engine.running)} != {lot}")
            if n_pas % 200 == 0:
                c, t_ = _echantillon()
                sm.append(c); temp.append(t_)
        torch.cuda.synchronize()
        duree_hote = time.perf_counter() - t_hote
    n_rondes = 0
else:
  t_hote = time.perf_counter()
  with GardeSmi() as garde_smi, Energie() as e:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < CIBLE_S:
        n_rondes += 1
        for k in range(B):
            engine.add_request(invite(1000 + n_rondes * 97 + k, PROMPT_LEN), params,
                               request_id=f"r{n_rondes}s{k}")
        while engine.running or engine.waiting:
            engine.step(); n_pas += 1
            if n_pas % 200 == 0:
                c, t_ = _echantillon()
                sm.append(c); temp.append(t_)
    torch.cuda.synchronize()
    duree_hote = time.perf_counter() - t_hote
# Garde structurelle (sage-profil-verdict-18-09 §3, REGLES §4) : hors bras
# témoin, un seul nvidia-smi lancé PENDANT la fenêtre invalide la cellule --
# ce n'est plus censé arriver (défaut = NVML en processus), donc un compte
# non nul signale une régression, pas un réglage. En bras témoin, le compte
# est attendu et seulement étiqueté (REGLES §4 « étiqueter, pas effacer »).
if garde_smi.n and not CERT_SMI_TEMOIN:
    print(f"[CERT {BRAS}] REFUS : {garde_smi.n} appel(s) nvidia-smi lance(s) "
         f"PENDANT la fenetre mesuree -- instrument corrompu, cellule invalide "
         f"(sage-profil-verdict-18-09 §3).", flush=True)
    json.dump({"invalide": True,
              "cause": "nvidia-smi lance dans la fenetre mesuree (bras B attendu sans)",
              "appels_smi_fenetre": garde_smi.n, "preuve": preuve},
             open(SORTIE, "w"), indent=1)
    raise SystemExit(1)
n = engine.stats.decode_tokens - n_avant
# Le lot reel doit etre B sur 100% des pas, pas seulement demande (REGLES §3,
# sage-poste-d-verdict-17-09 §2) : une sequence tronquee par budget KV epuise
# quitte le lot en silence cote engine (finish_reason="length" comme un
# max_tokens normal) -- seul ce compteur distingue les deux. Refuser plutot
# que publier un debit qui a tourne a un lot plus petit que B pendant la mesure.
n_tronquees = engine.stats.sequences_tronquees_budget
if n_tronquees:
    print(f"[CERT {BRAS}] REFUS : {n_tronquees} sequence(s) tronquee(s) par "
         f"budget KV epuise pendant la mesure -- {preuve['engine_regime']} -- "
         f"b={B} n'a pas tourne a b={B} sur toute la fenetre, cellule invalide.",
         flush=True)
    json.dump({"invalide": True,
              "cause": "budget KV epuise pendant la mesure : lot reel < b demande",
              "b_demande": B, "sequences_tronquees_budget": n_tronquees,
              "preuve": preuve}, open(SORTIE, "w"), indent=1)
    raise SystemExit(1)
# (19/09, Sage) un repli eager pendant la fenetre = la cellule n'a pas tourne au regime annonce
n_replis = int(engine.regime().get("repli_eager", 0)) if not preuve["eager"] else 0
if n_replis:
    print(f"[CERT {BRAS}] REFUS : {n_replis} pas retombe(s) en eager pendant la mesure "
          f"({', '.join(engine.regime().get('replis_eager_raisons', []))}) -- {engine.regime_ligne()} -- cellule invalide.", flush=True)
    json.dump({"invalide": True, "cause": "repli eager pendant la mesure", "repli_eager": n_replis,
               "raisons": engine.regime().get("replis_eager_raisons", []), "preuve": preuve}, open(SORTIE, "w"), indent=1)
    raise SystemExit(1)
# deux instruments pour la duree (biais b11b8a3 : energie.py relevait apres le join)
ecart_duree = abs(e.duree - duree_hote)
res = {"en_tete": {"instrument": "energie.py compteur TotalEnergyConsumption",
                   "cartes": e.resume().get("cartes"), "fenetre_s": round(e.duree, 3), "fenetre_hote_s": round(duree_hote, 3), "ecart_duree_s": round(ecart_duree, 3),
                   "plafond_w": preuve["power_limit_w"],
                   "horloge_sm_mhz": {"min": min(sm), "moy": round(sum(sm) / len(sm)), "max": max(sm)} if sm else None,
                   "temperature_gpu_c": {"avant": temp_avant, "min": min(temp), "max": max(temp)} if temp else {"avant": temp_avant},
                   "mode": "moyen 400 W, horloge libre", "b": B, "ctx": CTX, "invite": PROMPT_LEN,
                   "fenetre": "decodage pur, lot constant" if PUR else "rondes (prefill + traine incluses)",
                   "instrument_horloge_temp": "nvidia-smi temoin (CERT_SMI_TEMOIN=1)" if CERT_SMI_TEMOIN else "NVML en processus",
                   "appels_smi_fenetre": garde_smi.n},
       "preuve": preuve, "n_rondes": n_rondes, "n_pas": n_pas, "n_jetons": n,
       "pas_ms": round(1000 * e.duree / n_pas, 3) if n_pas else None,
       "jetons_s": round(n / e.duree, 2), "j_par_jeton_brut": round(e.joules / n, 4),
       "puissance_moyenne_w": round(e.moyenne, 1),
       "invalidations": e.invalidations + ([f"durees energie/hote divergent de {ecart_duree:.3f} s"] if ecart_duree > 0.05 else [])}
print(f"[CERT {BRAS}] pas {res['pas_ms']} ms, {res['jetons_s']} t/s, {res['j_par_jeton_brut']} J/j, "
      f"{res['puissance_moyenne_w']} W, sm {res['en_tete']['horloge_sm_mhz']}, "
      f"T {res['en_tete']['temperature_gpu_c']}, inval {res['invalidations']}", flush=True)
json.dump(res, open(SORTIE, "w"), indent=1)
