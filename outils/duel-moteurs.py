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
# CONCURRENCE, ajoutee le 10/09 : les gains du jour sont massifs a douze
# sequences et NULS a une (116,4 tok/s avant comme apres, mesure de poste3). Un
# duel a une seule sequence ne verrait rien de ce que nous avons gagne — et
# c'est le regime ou les duels se font d'habitude. Le bras a 1 reste identique
# au duel du 9/09 pour rester comparable a son tableau.
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 1
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

if CONC > 1:
    # A plusieurs sequences, mesurer un prefill par requete pendant que les
    # autres tournent n'aurait pas de sens : on mesure le DEBIT AGREGE, seul
    # chiffre qui a une definition sous concurrence. Deux tours symetriques :
    # max_tokens=1 pour le TTFT, puis N pour le decodage.
    import threading
    def tour(maxtok, res, base):
        fils, dep = [], time.perf_counter()
        def un(k):
            try:
                res.append(demander(invite(base + k), maxtok))
            except Exception:                   # noqa: BLE001
                pass
        for k in range(CONC):
            t = threading.Thread(target=un, args=(k,)); t.start(); fils.append(t)
        for t in fils:
            t.join()
        return time.perf_counter() - dep
    ttfts, debits, w = [], [], []
    for essai in range(n_essais):
        r1 = []
        mur1 = tour(1, r1, essai * 1000)
        ttfts.append(mur1)
        rN = []
        x = watts()
        murN = tour(N, rN, essai * 1000)
        jetons = sum(c for _, _, c in rN)
        if jetons > 1 and murN > mur1:
            debits.append(jetons / murN)
        y = watts()
        for v in (x, y):
            if v is not None:
                w.append(v)
    def med(v):
        v = sorted(v); return v[len(v) // 2] if v else 0.0
    def disp(v):
        return (max(v) - min(v)) / med(v) * 100 if v else float("nan")
    pj = med(w) / med(debits) if w and med(debits) else None
    print(json.dumps({
        "concurrence": CONC,
        "decode_tok_s_agrege": round(med(debits), 2),
        "decode_disp_pct": round(disp(debits), 2),
        "ttft_mur_s": round(med(ttfts), 3),
        "essais": len(debits),
        # LES VALEURS DANS L ORDRE, pas seulement la mediane et l etendue :
        # une dispersion ne dit pas si le debit DERIVE au fil des essais ou
        # s il SAUTE. La premiere signature designe l histoire de la session
        # (cles de graphe retenues a vie), la seconde un bruit.
        "debits_dans_l_ordre": [round(x, 1) for x in debits],
        "ttft_dans_l_ordre": [round(x, 3) for x in ttfts],
        "watts_median": round(med(w), 1) if w else None,
        "jetons_par_kJ": round(1000 / pj) if pj else None,
    }))
    sys.exit(0)

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
