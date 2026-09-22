#!/usr/bin/env python3
"""Reconvertit les 38 modèles sans promotions INT8 (acvram >= 0.4.50).

Chaque modèle est converti dans un dossier voisin, vérifié, puis seulement
échangé : un échec laisse l'ancien modèle intact et servi. Reprenable — un
modèle déjà à zéro promotion est sauté.
"""
import json, os, shutil, subprocess, sys, time

S = os.path.dirname(os.path.abspath(__file__))
RACINE = os.environ.get("ACVRAM_ARBRE", subprocess.run(["git", "-C", S, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() + "/../../anticitoyen-vram")
PY = f"{RACINE}/.venv/bin/python"
JOURNAL = f"{S}/reconv-journal.tsv"

def promotions(rep):
    """Nombre de tenseurs promus et nombre total, ou None si illisible."""
    try:
        m = json.load(open(os.path.join(rep, "acvram_manifest.json")))
    except Exception:
        return None
    t = m["tensors"]
    return sum(1 for v in t.values() if v.get("promoted_from")), len(t), set(t)

def gio(rep):
    return sum(os.path.getsize(os.path.join(rep, f))
               for f in os.listdir(rep)) / 1024 ** 3

modeles = json.load(open(f"{S}/reconv.json"))
faits = set()
if os.path.exists(JOURNAL):
    faits = {l.split("\t")[0] for l in open(JOURNAL) if l.startswith("ok\t") is False}
    faits = {l.split("\t")[1] for l in open(JOURNAL) if l.startswith("ok\t")}

print(f"=== reconversion de {len(modeles)} modèles, snr_floor=0 ===", flush=True)
print(f"    journal : {JOURNAL}", flush=True)
t_debut = time.time()
for i, r in enumerate(modeles, 1):
    dest, src = r["dir"], r["src_path"]
    nom = os.path.basename(dest)
    if nom in faits:
        print(f"[{i}/{len(modeles)}] {nom} : déjà fait", flush=True)
        continue
    etat = promotions(dest)
    if etat and etat[0] == 0:
        print(f"[{i}/{len(modeles)}] {nom} : déjà sans promotion", flush=True)
        continue
    neuf = dest + ".neuf"
    shutil.rmtree(neuf, ignore_errors=True)
    avant = gio(dest)
    print(f"[{i}/{len(modeles)}] {nom} : {avant:.2f} Gio, {etat[0]} promus\n"
          f"    source {src}", flush=True)
    t0 = time.time()
    p = subprocess.run([PY, "-m", "acvram", "convert", src, "--out", neuf],
                       cwd=RACINE, capture_output=True, text=True, timeout=10800)
    if p.returncode != 0:
        print(f"    ÉCHEC conversion : {p.stderr.strip()[-400:]}", flush=True)
        with open(JOURNAL, "a") as f:
            f.write(f"echec\t{nom}\tconversion\n")
        shutil.rmtree(neuf, ignore_errors=True)
        continue
    verif = promotions(neuf)
    # Le convertisseur d'aujourd'hui garde des tenseurs que l'ancien jetait
    # (les couches MTP depuis 426a0f6) : le nouveau doit contenir tous les
    # anciens, sans promotion, et peut en avoir davantage.
    if not verif or verif[0] != 0 or not etat[2] <= verif[2]:
        manque = sorted(etat[2] - verif[2])[:5] if verif else "?"
        print(f"    ÉCHEC vérification : {verif[:2] if verif else verif} attendu "
              f"(0, >= {etat[1]}), manquants {manque}", flush=True)
        with open(JOURNAL, "a") as f:
            f.write(f"echec\t{nom}\tverification\t{verif}\n")
        shutil.rmtree(neuf, ignore_errors=True)
        continue
    apres = gio(neuf)
    vieux = dest + ".vieux"
    shutil.rmtree(vieux, ignore_errors=True)
    os.rename(dest, vieux)
    os.rename(neuf, dest)
    shutil.rmtree(vieux, ignore_errors=True)
    dt = time.time() - t0
    print(f"    OK {avant:.2f} → {apres:.2f} Gio ({100*(apres-avant)/avant:+.1f} %), "
          f"{dt/60:.1f} min", flush=True)
    with open(JOURNAL, "a") as f:
        f.write(f"ok\t{nom}\t{avant:.3f}\t{apres:.3f}\t{dt:.0f}\n")
print(f"=== terminé en {(time.time()-t_debut)/3600:.1f} h ===", flush=True)
