#!/usr/bin/env python3
"""Le sac a dos ordonne-t-il par le bon critere ? Deux bras, un signe d'ecart.

Il ordonne par GAIN DE SNR PAR OCTET (`convert.py`, cle `-gain_db / cout`). La
courbe du quota du 10/09 montre que le rendement en perplexite par tenseur
promu croit d'un facteur SEIZE le long de sa course :

    segment            tenseurs   milli-PPL par tenseur
      0 ->   6              6            0,183
    172 -> 198             26            0,796
    198 -> 225             27            2,867

donc les tenseurs promus EN PREMIER rendent le MOINS de perplexite. Si c'est
vrai, renverser l'ordre a budget egal doit AMELIORER la perplexite.

VALEUR PREVUE, pour que ce controle puisse rendre « faux » :

    bras A (ordre actuel)     PPL 5,4918   mesure le 10/09 a 6,00 Gio
    plafond (tout promu)      PPL 5,4144   la meilleure atteignable
    bras B (ordre inverse)    PREDICTION : 5,4144 <= PPL(B) < 5,4918
                              et si l'ordre est SANS EFFET, PPL(B) ~ 5,4918

Un B au-dessus de 5,4918 refute l'inversion : l'ordre serait bien oriente et le
facteur seize viendrait d'autre chose. Un B egal a 5,4918 a la dispersion pres
dit que l'ordre ne decide de rien — ce qui serait aussi un resultat, et
retirerait le sujet.

LES DEUX BRAS SONT RECONVERTIS PAR LE BINAIRE COURANT. Reutiliser le dossier
quota-6g00 produit plus tot serait plus rapide, mais son manifeste ne porte pas
`budget.ordre_glouton` — il a ete produit avant que ce champ existe — donc rien
n'y attesterait l'ordre employe. C'est la lecon des temoins : un bras produit
par un autre binaire n'est pas un bras.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
CARTE = Path(__file__).resolve().parent / "carte.sh"
BASE = Path(MODELES)
SOURCE = BASE / "Llama-2-7b-hf"
CORPUS = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
CORPUS_SHA = "e52922746ad09bac73b0dba32b2987c0d7924da14337dcd43c1d9113a9f6d0ae"
BUDGET = 6.00
PPL_REFERENCE = 5.4141
PPL_PLAFOND = 5.4144          # tout promu : la meilleure atteignable
PPL_BRAS_A_ATTENDU = 5.4918   # mesure le 10/09 au meme budget

EVAL = ["--corpus", str(CORPUS), "--window", "2048", "--stride", "2048",
        "--max-tokens", "344064", "--min-context", "0", "--json"]


def mon_unite() -> str:
    """L'unite systemd qui contient CE processus, ou une chaine vide.

    Le garde des orphelins comptait le PILOTE lui-meme parmi les services
    vivants et refusait de demarrer — un garde qui accuse son operateur, et qui
    rendait impossible le detachement du pilote, c'est-a-dire la protection
    meme qu'il devait servir. Il faut donc s'exclure, et le seul moyen fiable
    est de lire son propre cgroup : `systemd-run --unit=X` place le processus
    dans `X.service`, et rien dans l'environnement ne le dit.
    """
    try:
        for ligne in open("/proc/self/cgroup", encoding="utf-8"):
            # LE DERNIER segment, pas le premier. Le chemin vaut
            #   /user.slice/user-1000.slice/user@1000.service/app.slice/X.service
            # et le premier segment en « .service » est `user@1000.service` —
            # jamais l'unite. Ma premiere version renvoyait donc toujours le
            # gestionnaire de session, le garde ne s'excluait jamais, et il
            # refusait le lancement une seconde fois apres avoir ete corrige.
            morceaux = [m for m in ligne.strip().split("/")
                        if m.endswith(".service")]
            if morceaux:
                return morceaux[-1]
    except OSError:
        pass
    return ""


def rendre_le_cache(d: Path) -> None:
    for f in sorted(d.glob("*.safetensors")):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def service(nom: str, argv: list[str], journal: Path, inverse: bool,
            minutes: int, python: str, mode_actif: str = "") -> int:
    sent = journal.with_suffix(".fini")
    sent.unlink(missing_ok=True)
    env = ["--setenv=PYTHONPATH=" + str(RACINE),
           "--setenv=ACVRAM_CUDA_HOME=/usr/local/cuda-13.2",
           "--setenv=CUDA_VISIBLE_DEVICES=0",
           f"--setenv=ACVRAM_KERNEL_CACHE={os.environ.get('ACVRAM_KERNEL_CACHE', '/tmp/poste1-noyaux')}"]
    if inverse:
        env.append("--setenv=ACVRAM_ORDRE_SAC_INVERSE=1")
    if mode_actif:
        env.append(f"--setenv=ACVRAM_ORDRE_SAC={mode_actif}")
    if os.environ.get("ACVRAM_MAX_PROMUS"):
        env.append(f"--setenv=ACVRAM_MAX_PROMUS={os.environ['ACVRAM_MAX_PROMUS']}")
    cmd = (["systemd-run", "--user", "--unit", nom, "--collect",
            f"--property=WorkingDirectory={RACINE}",
            "--property=MemoryHigh=40G", "--property=MemoryMax=44G",
            f"--property=RuntimeMaxSec={minutes * 60}"] + env
           + ["/bin/bash", "-c",
              # chien de garde sur le pilote : un bras orphelin tiendrait la
              # carte pour rien, et carte-libre.sh le croirait legitime
              f"( while kill -0 {os.getpid()} 2>/dev/null; do sleep 5; done; "
              f"  kill -TERM 0 ) & "
              f"{CARTE} {' '.join(argv)} > {journal} 2>&1; echo $? > {sent}"])
    subprocess.run(cmd, check=True)
    import time
    t0 = time.time()
    while not sent.exists():
        if time.time() - t0 > minutes * 60 + 120:
            subprocess.run(["systemctl", "--user", "stop", nom], check=False)
            return 124
        time.sleep(10)
    return int(sent.read_text().strip() or "1")


def releve(journal: Path) -> dict | None:
    t = journal.read_text(encoding="utf-8", errors="replace")
    dec, i, dernier = json.JSONDecoder(), 0, None
    while True:
        d = t.find("{", i)
        if d < 0:
            break
        try:
            o, i = dec.raw_decode(t, d)
        except json.JSONDecodeError:
            i = d + 1
            continue
        for c in (o.get("models", [o]) if isinstance(o, dict) else []):
            if isinstance(c, dict) and "perplexity" in c:
                dernier = c
    return dernier


MODES = {"snr": "snr_par_octet_decroissant",
         "inverse": "snr_par_octet_croissant",
         "erreur": "erreur_evitee_par_octet_decroissante",
         "absolu": "erreur_absolue_evitee_par_octet_decroissante",
         "base_croissant": "snr_de_base_croissant_sans_cout"}


def un_bras(etiquette: str, inverse: bool, sortie: Path, python: str,
            mode: str = "", suffixe: str = "") -> dict:
    # `suffixe` distingue une manche a compte impose (n149) des dossiers
    # budget-fill du meme ordre : sans lui, `if not dossier.exists()`
    # reutiliserait le dossier a 173/196 promus au lieu de convertir a 149.
    etiquette = etiquette + suffixe
    dossier = BASE / f"Llama-2-7b-ordre-{etiquette}"
    print(f"\n=== bras {etiquette} : ordre "
          f"{'INVERSE (+gain/cout)' if inverse else 'actuel (-gain/cout)'}",
          flush=True)
    if not dossier.exists():
        code = service(f"acvram-ordre-conv-{etiquette}",
                       [python, "-m", "acvram", "convert", str(SOURCE),
                        "--out", str(dossier), "--bits-budget", str(BUDGET),
                        "--grille-erreurs"],
                       sortie / f"conv-{etiquette}.log", inverse, 120, python,
                       mode_actif=mode)
        if code:
            print(f"  ECHEC conversion, code {code}")
            return {"bras": etiquette, "echec": True}
    else:
        print("  dossier deja present")
    m = json.loads((dossier / "acvram_manifest.json").read_text())
    bud = m.get("budget") or {}
    # TEMOIN DU BRAS : l'ordre employe est ecrit au manifeste. Sans lui, rien
    # ne distinguerait les deux dossiers apres coup.
    attendu = MODES[mode] if mode else (
        "snr_par_octet_croissant" if inverse else "snr_par_octet_decroissant")
    if bud.get("ordre_glouton") != attendu:
        print(f"  MANCHE SANS OBJET : le manifeste porte "
              f"{bud.get('ordre_glouton')!r} au lieu de {attendu!r}. "
              f"L'echappement n'a pas pris, ou le dossier vient d'un autre bras.")
        return {"bras": etiquette, "echec": True, "ordre": bud.get("ordre_glouton")}
    # LE CHAMP EST ECRIT CONDITIONNEL (convert.py:1069) : un tenseur dont la
    # calibration ne produit pas out_ref_norm passerait SANS, en silence. On le
    # COMPTE au lieu de le supposer — remarque de f2.
    avec_ech = sum(1 for v in m.get("tensors", {}).values()
                   if "out_ref_norm" in v)
    tot_t = len(m.get("tensors", {}))
    print(f"  out_ref_norm present sur {avec_ech}/{tot_t} tenseurs", flush=True)
    if avec_ech == 0:
        print("  ATTENTION : aucun tenseur ne porte l'echelle. Le code de "
              "cette conversion precede l'ajout du champ (2082279, 17:51), "
              "ou la calibration ne la produit pas.", flush=True)
    elif avec_ech < tot_t:
        print(f"  ATTENTION : {tot_t - avec_ech} tenseurs sans echelle — la "
              f"simulation `absolu` les ecartera, et il faut savoir lesquels "
              f"avant d'en tirer un classement.", flush=True)
    oct_r = sum(f.stat().st_size for f in dossier.glob("*.safetensors"))
    print(f"  octets {oct_r:,}  promus {bud.get('promus')}/{bud.get('candidats')}  "
          f"depense {bud.get('depense_gib')} Gio".replace(",", " "), flush=True)
    rendre_le_cache(dossier)
    code = service(f"acvram-ordre-eval-{etiquette}",
                   [python, "-m", "acvram", "eval", str(dossier)] + EVAL,
                   sortie / f"eval-{etiquette}.log", inverse, 60, python)
    if code:
        print(f"  ECHEC eval, code {code}")
        return {"bras": etiquette, "echec": True}
    r = releve(sortie / f"eval-{etiquette}.log") or {}
    if r.get("corpus_sha256") != CORPUS_SHA[:24]:
        print(f"  REFUS : corpus {r.get('corpus_sha256')} au lieu de "
              f"{CORPUS_SHA[:24]}")
        return {"bras": etiquette, "echec": True}
    ppl = r["perplexity"]
    print(f"  PPL {ppl}  (etalon {PPL_REFERENCE}, "
          f"ecart {100 * (ppl / PPL_REFERENCE - 1):+.3f} %)", flush=True)
    rendre_le_cache(dossier)
    return {"bras": etiquette, "inverse": inverse, "ppl": ppl,
            "octets": oct_r, "budget": bud}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--modes", default="",
                    help="modes a produire, separes par des virgules "
                         "(snr, inverse, erreur, absolu, base_croissant). "
                         "Sans cet argument, les deux bras historiques.")
    ap.add_argument("--sortie", default=str(sorties() / "ordre-sac"))
    ap.add_argument("--max-promus", type=int, default=0,
                    help="plafond de COMPTE : promeut les N premiers de "
                         "l'ordre, budget ignore. Dossiers suffixes -nN.")
    a = ap.parse_args()
    suffixe = ""
    if a.max_promus > 0:
        os.environ["ACVRAM_MAX_PROMUS"] = str(a.max_promus)
        suffixe = f"-n{a.max_promus}"
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    import hashlib
    if hashlib.sha256(CORPUS.read_bytes()).hexdigest() != CORPUS_SHA:
        print("ECHEC / CAUSE: corpus de sha inattendu")
        return 2
    restes = [m for m in subprocess.run(
        ["systemctl", "--user", "list-units", "acvram*", "--no-legend",
         "--plain"], capture_output=True, text=True).stdout.split()
        if m.endswith(".service") and m != mon_unite()]
    if restes:
        print(f"ECHEC / CAUSE: services acvram vivants : {restes[:3]} / "
              f"SUITE: systemctl --user stop <unite>")
        return 2
    print(f"A/B ORDRE DU SAC A DOS — budget {BUDGET} Gio, un signe d'ecart")
    print(f"  prediction : {PPL_PLAFOND} <= PPL(B) < {PPL_BRAS_A_ATTENDU}")
    if a.modes:
        voulus = [m.strip() for m in a.modes.split(",") if m.strip()]
        inconnus = [m for m in voulus if m not in MODES]
        if inconnus:
            print(f"ECHEC / CAUSE: mode(s) inconnu(s) {inconnus} ; connus "
                  f"{sorted(MODES)} / SUITE: un mode inconnu ferait mesurer "
                  f"le defaut en croyant mesurer autre chose")
            return 2
        res = [un_bras(m, m == "inverse", sortie, a.python, mode=m,
                       suffixe=suffixe) for m in voulus]
    else:
        res = [un_bras("normal", False, sortie, a.python),
               un_bras("inverse", True, sortie, a.python)]
    if any(r.get("echec") for r in res):
        print("\nECHEC / CAUSE: un bras n'a pas rendu / SUITE: voir ci-dessus")
        return 1
    # UN SEUL MODE (ex. base_croissant) NE FAIT PAS UNE PAIRE A/B. La version
    # precedente faisait `res[1]` en aveugle et levait IndexError APRES avoir
    # calcule et sauve la PPL — le resultat etait bon, le tool plantait quand
    # meme. On imprime chaque bras, et le verdict A/B seulement s'il y a deux
    # bras a comparer.
    for r in res:
        print(f"  bras {r['bras']:16s} PPL {r['ppl']}")
    if len(res) < 2:
        (sortie / "ab-ordre.json").write_text(json.dumps(
            {"bras": res, "etalon": PPL_REFERENCE}, indent=1))
        return 0
    A, B = res[0]["ppl"], res[1]["ppl"]
    print(f"  ecart B - A            {B - A:+.4f}")
    if B < A - 0.005:
        verdict = ("L'ORDRE EST MAL ORIENTE : le renverser AMELIORE la "
                   "perplexite a budget egal.")
    elif B > A + 0.005:
        verdict = ("L'ordre est BIEN oriente : le renverser degrade. "
                   "Le facteur seize vient d'autre chose.")
    else:
        verdict = ("L'ORDRE NE DECIDE DE RIEN a la dispersion pres — ce qui "
                   "retire le sujet, et c'est aussi un resultat.")
    print(f"\n  {verdict}")
    (sortie / "ab-ordre.json").write_text(json.dumps(
        {"bras": res, "ecart": B - A, "verdict": verdict,
         "prediction": [PPL_PLAFOND, PPL_BRAS_A_ATTENDU]}, indent=2,
        ensure_ascii=False))
    print(f"\nFAIT / TESTE: 2 bras, ecart {B - A:+.4f} PPL / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
