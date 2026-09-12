#!/usr/bin/env python3
"""Renomme les convertis acvram en <nom>-src<origine>-<format reel>.

Le nom d un converti portait le format de la SOURCE, pas le sien : un dossier
dit Q6_K ou exl3-5.0bpw alors que ses poids sont en nvfp4. La source reste
dans le nom parce que les empreintes de contenu prouvent que deux conversions
de sources differentes ont des poids differents.

Ecrit un journal de retour AVANT d agir. --appliquer pour executer.
"""
import json, os, re, sys, collections
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

A = MODELES
TSV = os.path.expanduser("~/.kimi-code/acvram-chemins.tsv")
JOURNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renommages.tsv")
APPLIQUER = "--appliquer" in sys.argv

MOTIFS = [
    r"[-.]i1[-.]Q\d[_-]K[_-][MSL]", r"[-.]UD[-.]Q\d_K_[MSXL]+",
    r"[-.]Q\d_K_X?[MSL]", r"[-.]Q\d_K", r"[-.]Q\d_\d", r"[-.]IQ\d[MSL]?",
    r"[-.]EXL3[-.]?\d+\.?\d*bpw", r"[-.]exl3[-.]?\d+\.?\d*bpw",
    r"[-.]\d+\.?\d*bpw[-.]?EXL3", r"[-.]\d+\.?\d*bpw",
    r"[-.]EXL3", r"[-.]exl3", r"[-.]AWQ[-.]?\d*bit", r"[-.]awq", r"[-.]AWQ",
    r"[-.]GGUF", r"[-.]gguf", r"[-.]bf16", r"[-.]iq3m", r"[-.]i1",
]

def format_reel(d):
    """format dominant EN OCTETS, jamais en nombre de tenseurs"""
    mf = os.path.join(d, "acvram_manifest.json")
    if not os.path.isfile(mf):
        return None
    try:
        m = json.load(open(mf))
    except Exception:
        return None
    par = collections.Counter()
    for t in (m.get("tensors") or {}).values():
        if not isinstance(t, dict):
            continue
        f = str(t.get("format") or t.get("quant") or t.get("dtype") or "?")
        n = t.get("nbytes") or t.get("bytes") or t.get("size")
        if n is None:
            n = 1
            for x in (t.get("shape") or []):
                n *= x
            n *= {"bf16": 2, "fp16": 2, "int8": 1, "nvfp4": .5, "int4_awq": .5}.get(f, 2)
        par[f] += int(n)
    return par.most_common(1)[0][0] if par else None

def decoupe(nom):
    """rend (base sans etiquette, etiquette d origine trouvee ou '')"""
    src = ""
    base = nom
    for mo in MOTIFS:
        m = re.search(mo, base, flags=re.IGNORECASE)
        if m:
            src = m.group(0).lstrip("-.")
            base = base[:m.start()] + base[m.end():]
            break
    return re.sub(r"[-.]+$", "", base), src

plan = []
for nom in sorted(os.listdir(A)):
    d = os.path.join(A, nom)
    if not os.path.isdir(d):
        continue
    fmt = format_reel(d)
    if fmt is None:
        print(f"IGNORE (manifeste absent ou illisible) : {nom}")
        continue
    base, src = decoupe(nom)
    # l etiquette source ne prend ni point ni tiret : sinon "srci1-Q4_K_M"
    # se lit comme deux champs. Et quand la source vaut deja le format reel,
    # la repeter n apprend rien : "Agents-4B-kimi-srcbf16-bf16".
    plat = re.sub(r"[.-]", "_", src)
    tag = "" if (not src or plat.lower() == fmt.lower()) else "-src" + plat
    cible = f"{base}{tag}-{fmt}"
    if cible != nom:
        plan.append((nom, cible))

doubles = [k for k, v in collections.Counter(b for _, b in plan).items() if v > 1]
if doubles:
    print("COLLISIONS, rien n a ete fait :", *doubles, sep="\n  ")
    sys.exit(2)
occupe = set(os.listdir(A)) - {a for a, _ in plan}
heurt = [b for _, b in plan if b in occupe]
if heurt:
    print("CIBLE DEJA PRISE, rien n a ete fait :", *heurt, sep="\n  ")
    sys.exit(2)

with open(JOURNAL, "w") as f:
    f.write("ancien\tnouveau\n")
    for a, b in plan:
        f.write(f"{a}\t{b}\n")
print(f"{len(plan)} renommages ; journal de retour : {JOURNAL}")
if not APPLIQUER:
    print("(essai a blanc — relancer avec --appliquer)")
    sys.exit(0)

for a, b in plan:
    os.rename(os.path.join(A, a), os.path.join(A, b))
print(f"{len(plan)} dossiers renommes")

# TSV d alias : les chemins suivent, les alias existants restent valides
if os.path.isfile(TSV):
    corr = {a: b for a, b in plan}
    lignes, ajouts = [], []
    for l in open(TSV):
        c = l.rstrip("\n").split("\t")
        if len(c) >= 2 and c[1].startswith(A + os.sep):
            anc = c[1][len(A) + 1:].split(os.sep)[0]
            if anc in corr:
                c[1] = c[1].replace(A + os.sep + anc, A + os.sep + corr[anc], 1)
                # alias supplementaire portant le nom corrige ; l ancien reste
                na = "acvram-" + re.sub(r"[^a-z0-9]+", "-", corr[anc].lower()).strip("-")
                if na != c[0]:
                    ajouts.append("\t".join([na] + c[1:]))
        lignes.append("\t".join(c))
    vus = {l.split("\t")[0] for l in lignes}
    with open(TSV, "w") as f:
        f.write("\n".join(lignes) + "\n")
        for a in ajouts:
            if a.split("\t")[0] not in vus:
                f.write(a + "\n")
    print(f"TSV d alias : {len(lignes)} lignes mises a jour, {len(ajouts)} alias ajoutes")
