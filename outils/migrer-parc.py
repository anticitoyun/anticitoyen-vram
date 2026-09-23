#!/usr/bin/env python3
"""Réorganise le parc : originaux sur le HDD CMR, convertis acvram sur le SSD.

    SSD  ${ACVRAM_MODELES}/..                            ← models_acvram seul
    HDD  /mnt/4TO_SATACMR_2022/Modeles                   ← tout le reste

Aujourd'hui les originaux sont physiquement sur le SSD et le HDD les voit par
des liens symboliques (par catégorie ou par modèle) ; toutes les configurations
passent par les chemins du HDD. Chaque lien est donc remplacé par le vrai
dossier, sans qu'aucun chemin de configuration ne change. Dans l'autre sens,
models_acvram part sur le SSD et le HDD garde un lien de catégorie.

Chaque dossier est copié dans un `.X.partiel`, vérifié (octets et nombre de
fichiers), puis échangé et sa source supprimée. Les deux files sont entrelacées
selon l'espace libre. Reprenable : ce qui est déjà à destination est sauté.

    migrer.py --simuler     affiche le plan et la convergence en espace
    migrer.py               exécute
"""
import os, shutil, subprocess, sys, time
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

SSD = str(_p.Path(MODELES).parent)
HDD = "/mnt/4TO_SATACMR_2022/Modeles"
ACV = "models_acvram"
MARGE = 25 * 1024 ** 3          # espace à laisser sur un disque après chaque copie
S = os.path.dirname(os.path.abspath(__file__))
JOURNAL = f"{S}/migration-journal.tsv"
SIMULER = "--simuler" in sys.argv


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def taille(p):
    """Octets réels d'un dossier, sans suivre les liens."""
    out = subprocess.run(["du", "-sb", "-P", p], capture_output=True, text=True)
    return int(out.stdout.split("\t")[0])


def nb_fichiers(p):
    n = 0
    for _, _, fs in os.walk(p):
        n += len(fs)
    return n


def libre(p):
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def vrai_dossier(p):
    return os.path.isdir(p) and not os.path.islink(p)


def sous(p, racine):
    return os.path.realpath(p).startswith(os.path.realpath(racine) + "/")


def supprimer(p):
    """rm -rf borné aux deux racines, jamais à travers un lien."""
    assert sous(p, SSD) or sous(p, HDD), f"hors périmètre : {p}"
    assert not os.path.islink(p), f"refus de supprimer à travers un lien : {p}"
    shutil.rmtree(p)


# --------------------------------------------------------------------------
# prépasse : une catégorie du HDD qui est un lien vers le SSD devient un vrai
# dossier de liens par modèle, pour qu'un modèle reste joignable pendant que
# son voisin se copie
# --------------------------------------------------------------------------
def eclater_categorie(cat):
    lien = os.path.join(HDD, cat)
    if not os.path.islink(lien):
        return
    cible = os.path.realpath(lien)
    assert cible == os.path.join(SSD, cat), f"{lien} -> {cible} inattendu"
    log(f"catégorie liée {lien} -> {cible} : éclatée en liens par modèle")
    if SIMULER:
        return
    os.unlink(lien)
    os.mkdir(lien)
    for n in sorted(os.listdir(cible)):
        os.symlink(os.path.join(cible, n), os.path.join(lien, n))


# --------------------------------------------------------------------------
# déplacement d'un dossier
# --------------------------------------------------------------------------
def deplacer(src, dst):
    """src (vrai dossier) → dst ; dst peut être un lien vers src, ou absent."""
    assert vrai_dossier(src), f"source absente ou lien : {src}"
    if os.path.lexists(dst):
        # dst est soit un lien vers src, soit src lui-même vu à travers une
        # catégorie encore liée (simulation) ; tout autre cas est une collision
        if os.path.realpath(dst) != os.path.realpath(src):
            raise RuntimeError(f"collision : {dst} existe déjà et n'est pas {src}")
    parent = os.path.dirname(dst)
    assert SIMULER or vrai_dossier(parent), f"parent non éclaté : {parent}"
    partiel = os.path.join(parent, "." + os.path.basename(dst) + ".partiel")
    if os.path.lexists(partiel):
        supprimer(partiel)
    t0 = time.time()
    octets, fichiers = taille(src), nb_fichiers(src)
    log(f"  {src}\n            -> {dst}  ({octets / 1024**3:.1f} Gio, {fichiers} fichiers)")
    if SIMULER:
        return octets
    r = subprocess.run(["rsync", "-aW", "--no-compress", src + "/", partiel + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rsync a échoué ({r.returncode}) : {r.stderr[-300:]}")
    o2, f2 = taille(partiel), nb_fichiers(partiel)
    if (o2, f2) != (octets, fichiers):
        raise RuntimeError(f"copie divergente : {o2}/{f2} au lieu de {octets}/{fichiers}")
    if os.path.islink(dst):
        os.unlink(dst)
    os.rename(partiel, dst)
    supprimer(src)
    dt = time.time() - t0
    with open(JOURNAL, "a") as f:
        f.write(f"ok\t{src}\t{dst}\t{octets}\t{dt:.0f}\n")
    log(f"     fait en {dt/60:.1f} min ({octets / 1024**2 / max(dt, 1):.0f} Mio/s)")
    return octets


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def file_originaux():
    """(src SSD, dst HDD, octets) pour chaque vrai dossier du SSD hors acvram."""
    out = []
    for cat in sorted(os.listdir(SSD)):
        pc = os.path.join(SSD, cat)
        if cat == ACV or not vrai_dossier(pc):
            continue
        for n in sorted(os.listdir(pc)):
            p = os.path.join(pc, n)
            if vrai_dossier(p):
                out.append((p, os.path.join(HDD, cat, n), taille(p)))
            elif os.path.isfile(p):
                out.append((p, os.path.join(HDD, cat, n), os.path.getsize(p)))
    return out


def file_convertis():
    src = os.path.join(HDD, ACV)
    if not vrai_dossier(src):
        return []
    return [(os.path.join(src, n), os.path.join(SSD, ACV, n),
             taille(os.path.join(src, n)))
            for n in sorted(os.listdir(src)) if vrai_dossier(os.path.join(src, n))]


def deplacer_fichier(src, dst):
    """Un fichier isolé (scripts de models_train) suit son dossier."""
    if SIMULER:
        return
    shutil.copy2(src, dst + ".partiel")
    assert os.path.getsize(dst + ".partiel") == os.path.getsize(src)
    if os.path.islink(dst):
        os.unlink(dst)
    os.rename(dst + ".partiel", dst)
    os.unlink(src)


def main():
    for d in (SSD, HDD):
        assert os.path.ismount(os.path.dirname(d)) or os.path.isdir(d), d
    log(f"SSD {SSD} : {libre(SSD)/1024**3:.0f} Gio libres")
    log(f"HDD {HDD} : {libre(HDD)/1024**3:.0f} Gio libres")
    if SIMULER:
        log("MODE SIMULATION : rien n'est déplacé")

    # catégories du HDD à éclater
    for cat in sorted(os.listdir(SSD)):
        if cat != ACV and vrai_dossier(os.path.join(SSD, cat)):
            eclater_categorie(cat)
    os.makedirs(os.path.join(SSD, ACV), exist_ok=True)

    orig = file_originaux()
    conv = file_convertis()
    log(f"{len(orig)} originaux SSD→HDD ({sum(o for _, _, o in orig)/1024**3:.0f} Gio), "
        f"{len(conv)} convertis HDD→SSD ({sum(o for _, _, o in conv)/1024**3:.0f} Gio)")
    ssd_libre, hdd_libre = libre(SSD), libre(HDD)
    total = 0
    palier = 0
    t_debut = time.time()
    while orig or conv:
        if conv and ssd_libre >= conv[0][2] + MARGE:
            src, dst, o = conv.pop(0)
            sens = "→SSD"
        elif orig and hdd_libre >= orig[0][2] + MARGE:
            src, dst, o = orig.pop(0)
            sens = "→HDD"
        else:
            raise RuntimeError(f"bloqué : SSD {ssd_libre/1024**3:.0f} Gio, HDD "
                               f"{hdd_libre/1024**3:.0f} Gio libres, prochain converti "
                               f"{conv[0][2]/1024**3 if conv else 0:.0f}, prochain original "
                               f"{orig[0][2]/1024**3 if orig else 0:.0f}")
        if os.path.isfile(src):
            deplacer_fichier(src, dst)
        else:
            deplacer(src, dst)
        if sens == "→SSD":
            ssd_libre -= o; hdd_libre += o
        else:
            hdd_libre -= o; ssd_libre += o
        if not SIMULER:
            ssd_libre, hdd_libre = libre(SSD), libre(HDD)
        total += o
        if total // (150 * 1024**3) > palier:
            palier = total // (150 * 1024**3)
            log(f"ÉTAPE {total/1024**3:.0f} Gio déplacés, reste {len(orig)} originaux "
                f"et {len(conv)} convertis, {(time.time()-t_debut)/3600:.1f} h écoulées")

    # finitions
    src_acv = os.path.join(HDD, ACV)
    if vrai_dossier(src_acv) and not os.listdir(src_acv) and not SIMULER:
        os.rmdir(src_acv)
        os.symlink(os.path.join(SSD, ACV), src_acv)
        log(f"{src_acv} -> {SSD}/{ACV} (lien de compatibilité)")
    for cat in sorted(os.listdir(SSD)):
        pc = os.path.join(SSD, cat)
        if cat != ACV and vrai_dossier(pc) and not os.listdir(pc) and not SIMULER:
            os.rmdir(pc)
    # liens cassés du HDD
    for cat in os.listdir(HDD):
        pc = os.path.join(HDD, cat)
        if not vrai_dossier(pc):
            continue
        for n in os.listdir(pc):
            p = os.path.join(pc, n)
            if os.path.islink(p) and not os.path.exists(p):
                log(f"lien cassé supprimé : {p} -> {os.readlink(p)}")
                if not SIMULER:
                    os.unlink(p)
    log(f"TERMINÉ : {total/1024**3:.0f} Gio en {(time.time()-t_debut)/3600:.1f} h ; "
        f"SSD {libre(SSD)/1024**3:.0f} Gio libres, HDD {libre(HDD)/1024**3:.0f} Gio libres")


if __name__ == "__main__":
    main()
