"""Charge GPU CONNUE, pour eprouver la fragilite d'un test sous contention.

Deux grandeurs reglees, pas « un truc qui tourne » :
    --gio     memoire occupee, allouee d'un bloc et gardee
    --calcul  fraction de temps passee a calculer (0 = memoire seule)

Sans cela on ne saurait pas ce qu'on accuse : un test peut tomber par manque
de VRAM, par contention de calcul, ou par les deux — et le remede n'est pas le
meme. La charge s'annonce sur stdout et se retire proprement a l'arret.
"""
import argparse, os, signal, sys, time, torch

def _verrou_tenu() -> bool:
    """Un de mes ancetres tient-il le verrou de carte ?

    CET OUTIL EST LE SEUL DU CIRCUIT DONT LE METIER EST D OCCUPER LA CARTE, et
    c est precisement celui qu on ne pense pas a proteger : il ne mesure rien,
    donc il ne ressemble pas a une manche, donc le reflexe du verrou ne se
    declenche pas. Le 10/09 a 19h41 il a tourne pendant la campagne d une autre
    session, qui a perdu six manches a chercher un intrus — et l en-tete de ce
    fichier promettait pourtant une charge « connue ».
    On ne repare pas cela par une consigne : un outil dont le metier est de
    gener ne doit pas POUVOIR etre lance a la main par distraction.
    """
    info = os.environ.get("ACVRAM_VERROU", "/tmp/acvram-carte-0.lock") + ".qui"
    try:
        with open(info) as fh:
            tenant = int(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return False
    p = os.getpid()
    for _ in range(40):                     # remonter mes ancetres
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

p = argparse.ArgumentParser()
p.add_argument("--gio", type=float, default=20.0)
p.add_argument("--calcul", type=float, default=0.5)
p.add_argument("--duree", type=float, default=600.0)
a = p.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("REFUS : pas de carte")
if not _verrou_tenu() and os.environ.get("ACVRAM_CHARGE_SANS_VERROU") != "1":
    raise SystemExit(
        "REFUS : le verrou de carte n'est pas tenu par un de mes ancetres.\n"
        "  Lancez-moi SOUS le verrou, et dites que la charge est voulue :\n"
        "    ACVRAM_NOM=CHARGE-DELIBEREE outils/carte.sh <votre commande>\n"
        "  Sans cela une autre session cherchera un intrus pendant vingt\n"
        "  minutes — c'est arrive le 10/09 a 19h41, six manches perdues.")

n = int(a.gio * (1 << 30) / 4)
bloc = torch.empty(n, dtype=torch.float32, device="cuda")
libre, total = torch.cuda.mem_get_info()
print(f"charge : {a.gio:.1f} Gio pris · {libre/(1<<30):.1f} Gio libres sur "
      f"{total/(1<<30):.1f} · calcul {a.calcul:.0%}", flush=True)

# Le calcul occupe la carte sans allouer davantage : deux matrices fixes.
m = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
fin = time.time() + a.duree
arret = {"v": False}
signal.signal(signal.SIGTERM, lambda *_: arret.__setitem__("v", True))
signal.signal(signal.SIGINT, lambda *_: arret.__setitem__("v", True))
while time.time() < fin and not arret["v"]:
    t0 = time.time()
    if a.calcul > 0:
        for _ in range(20):
            m = torch.nn.functional.relu(m @ m.T)[:4096, :4096].contiguous()
        torch.cuda.synchronize()
    dt = time.time() - t0
    if a.calcul < 1.0 and dt > 0:
        time.sleep(dt * (1 - a.calcul) / max(a.calcul, 1e-3))
print("charge retiree", flush=True)
