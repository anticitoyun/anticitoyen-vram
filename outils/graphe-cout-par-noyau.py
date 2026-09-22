"""Un noyau grave dans un GRAPHE CUDA coute-t-il encore quelque chose ?

Nous raisonnons depuis ce matin comme si un graphe faisait disparaitre le cout
des noyaux. Il fait disparaitre le cout CPU du LANCEMENT. Pas les noyaux : un
graphe rejoue execute tous ceux qu'il contient.

PREDICTION ECRITE AVANT LA MESURE
---------------------------------
En eager, N noyaux triviaux enchaines sont limites par ce que le CPU met a les
lancer (les lancements sont asynchrones, le GPU va plus vite qu'eux) : on
attend un temps par noyau de l'ordre de 2 a 5 us, PLAT en N.
Dans un graphe, le CPU ne lance qu'une fois ; reste ce que le GPU met a
enchainer les noeuds. J'attends 0,5 a 2 us par noeud, donc un rapport de 2 a 5.

    ~10 us par noeud dans le graphe aussi -> les graphes ne sauvent rien contre
        un compte eleve ; le seul remede est de REDUIRE LE NOMBRE de noyaux.
    ~2 us  -> les graphes divisent par cinq, la priorite est de les faire tenir.
    ~0 us, les noeuds s'enchainent sans trou -> les trous de 10 us observes en
        eager sont une propriete de l'eager seul.

ET CE QUE MA PREDICTION NE PREVOIT PAS — le point ajoute sans hypothese, parce
qu'ecrire deux issues ne garantit pas d'avoir couvert le domaine (leçon du
jour) : on fait varier N sur deux decades. Si le temps par noyau DEPEND de N
dans l'un des deux regimes, aucune des issues ci-dessus ne s'applique et c'est
cette dependance qu'il faudra expliquer.

MON PLANCHER DE 9,65 us N'EST PAS TRANSPORTABLE ICI, et c'est dit d'avance :
il a ete mesure autour d'UN appel isole, evenements CUDA compris, sur un noyau
qui ecrit 128 flottants par bloc. Ni le meme noyau, ni le meme regime.
"""
import os, sys, torch
# LE VERROU EST DANS L OUTIL, PLUS DANS MON HABITUDE. Je lancais ces scripts
# SOUS carte.sh — ce qui protege quand j y pense, et pas quand quelqu un
# d autre les lance directement. Une habitude ne survit ni a la fatigue ni a
# un autre operateur : c est ce que charge-gpu.py a cesse d etre le 10/09, au
# prix de six manches perdues par une autre session.
def _verrou_tenu() -> bool:
    _c = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "0").split(",")[0]
    if not _c.isdigit():
        _c = "0"
    info = os.environ.get("ACVRAM_VERROU", f"/tmp/acvram-carte-{_c}.lock") + ".qui"
    try:
        with open(info) as fh:
            tenant = int(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return False
    p = os.getpid()
    for _ in range(40):
        if p == tenant:
            return True
        try:
            with open(f"/proc/{p}/stat") as fh:
                p = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        if p <= 1:
            return False
    return False

if not _verrou_tenu() and os.environ.get("ACVRAM_SANS_VERROU") != "1":
    raise SystemExit(
        "REFUS : le verrou de carte n'est pas tenu par un de mes ancetres.\n"
        "    outils/carte.sh " + " ".join(sys.argv[:1]) + " ...\n"
        "  Une mesure lancee sans verrou peut tourner pendant celle d'une "
        "autre session, et aucun des deux resultats ne vaudra.")

if torch.cuda.device_count() != 1:
    raise SystemExit("REFUS : une seule carte doit etre visible")
from acvram import kernels
ext = kernels.get_extension()
if ext is None or not hasattr(ext, "banc_fma"):
    raise SystemExit("REFUS : le binaire ne porte pas banc_fma")
print(f"# binaire sha {kernels._SO_HASH}")
print(f"# {torch.cuda.get_device_properties(0).name}")

# NOYAU TRIVIAL : une grille d'un bloc, K minimal. On mesure l'enchainement,
# pas le travail — si le noyau coutait, il masquerait ce qu'on cherche.
BLOCS, THREADS, K = 1, 32, 1

def chrono(fn, n=11):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        t.append(e0.elapsed_time(e1) * 1000.0)
    t.sort()
    return t[len(t) // 2]

print(f"{'N':>7} {'eager us':>10} {'us/noyau':>9} | {'graphe us':>10} {'us/noeud':>9} | rapport")
for N in (100, 1000, 10000):
    def lot():
        for _ in range(N):
            ext.banc_fma(BLOCS, 1, 1, THREADS, K)
    us_e = chrono(lot)

    # CAPTURE. Le flux dedie et la chauffe sont exiges par CUDA ; sans eux la
    # capture echoue ou capture un etat incomplet.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lot()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            lot()
    except Exception as exc:
        print(f"{N:7d} {us_e:10.1f} {us_e/N:9.3f} | CAPTURE IMPOSSIBLE : "
              f"{type(exc).__name__}: {str(exc)[:80]}")
        continue
    us_g = chrono(g.replay)
    print(f"{N:7d} {us_e:10.1f} {us_e/N:9.3f} | {us_g:10.1f} {us_g/N:9.3f} | "
          f"{us_e/us_g:6.2f}x", flush=True)
