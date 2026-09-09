"""Une manche du duel, version corrigee de TROIS biais que j'avais introduits.

1. INVITES DE TEXTE REEL, extraites du corpus wiki (sha verifie), un extrait
   DIFFERENT par essai. La version precedente tirait 400 mots dans un
   vocabulaire de 24 : ultra-repetitif, donc taillee pour la speculation ngram
   d'acvram. L'invite fabriquait le resultat, pour la quatrieme fois.
2. SPECULATION EGALISEE : acvram specule par defaut (`--speculative ngram`),
   llama.cpp seulement si on lui passe `--draft`. Le duel opposait un moteur
   speculatif a un moteur qui ne l'est pas.
3. N=256 au lieu de 128 : le debit de decodage se deduit de (tN - t1), et avec
   N trop petit cette difference est du meme ordre que le bruit — d'ou 62,8 %
   de dispersion a la premiere manche.

Separation prefill/decodage sans instrumenter les moteurs : deux requetes sur
la meme invite, a max_tokens=1 puis N. Symetrique, donc equitable.
"""
import json, sys, time, urllib.request, subprocess

url, cle, modele, n_essais = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
N = 256
CORPUS = "/mnt/AI_GENERATOR/corpus/wiki.test.raw"
with open(CORPUS, encoding="utf-8", errors="ignore") as fh:
    TEXTE = fh.read()
MOTS = TEXTE.split()

def invite(k):
    # extrait DIFFERENT par essai, ~350 mots de texte reel
    d = (k * 4001) % max(1, len(MOTS) - 400)
    return " ".join(MOTS[d:d + 350]) + "\n\nResume ce passage en une phrase."

def demander(texte, maxtok):
    corps = json.dumps({"model": modele, "temperature": 0, "max_tokens": maxtok,
                        "messages": [{"role": "user", "content": texte}]}).encode()
    req = urllib.request.Request(url, corps, {"Content-Type": "application/json",
                                              "Authorization": f"Bearer {cle}"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.perf_counter() - t0
    u = d.get("usage") or {}
    return dt, u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0

def watts():
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return float(s)
    except Exception:                          # noqa: BLE001
        return None                            # un releve manquant n'est pas zero

demander(invite(999), 16)                       # chauffe
dec, pre, w, ptok = [], [], [], 0
for k in range(n_essais):
    txt = invite(k)
    t1, p1, _ = demander(txt + " ", 1)
    tN, pN, cN = demander(txt, N)
    if cN <= 1 or tN <= t1:
        continue
    dec.append((cN - 1) / (tN - t1))
    pre.append(p1 / t1 if t1 > 0 else 0)
    ptok = p1
    x = watts()
    if x is not None:
        w.append(x)

def med(v):
    v = sorted(v); return v[len(v) // 2] if v else 0.0
def disp(v):
    return (max(v) - min(v)) / med(v) * 100 if v else float("nan")

pj = med(w) / med(dec) if w and med(dec) else None
print(json.dumps({
    "decode_tok_s": round(med(dec), 2), "decode_disp_pct": round(disp(dec), 2),
    "prefill_tok_s": round(med(pre), 1), "prefill_disp_pct": round(disp(pre), 2),
    "prompt_tokens": ptok, "essais": len(dec),
    "watts_median": round(med(w), 1) if w else None,
    "jetons_par_kJ": round(1000 / pj) if pj else None,
}))
