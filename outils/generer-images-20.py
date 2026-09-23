"""20 images 896×896 déterministes (graine fixe) pour le chantier multimodal (sage-acvram-multimodal-20-09 § 3) :
scènes synthétiques — formes colorées comptables, texte, dégradés, grilles, graphiques — sans donnée personnelle
(les photos du poste sont privées). Scellées par sha256 (images-20.sha256) ; le harnais est indifférent à la source :
Sage peut y substituer 20 photos libres, le sha256 change et se date.  Usage : python outils/generer-images-20.py <dossier>
Les png ne sont PAS dans le dépôt (perdus avec un worktree le 21/09) : ce script les régénère au bit près
(graine 20260920, DejaVuSans-Bold 72, PIL `optimize=True`) — contrôle : `sha256sum -c scratchpad/corpus-prive/images-20/images-20.sha256`
dans le dossier généré (20/20 OK le 21/09, 3 régénérations). Il écrit aussi descriptions.tsv et images-20.sha256 : générer dans un
dossier neuf, jamais directement dans corpus-prive/images-20 (les fichiers scellés y sont suivis par git).
Le bit près dépend de la bibliothèque, pas seulement du script (Manon 21/09 : 3/20 avec un autre venv) : les octets d un png
dépendent du rendu du texte (FreeType) ET de la compression (zlib, `optimize=True`). Environnement qui reproduit le scellé :
python 3.12.14 (.venv acvram), Pillow 12.3.0, FreeType 2.14.3, zlib 1.3.1, DejaVuSans-Bold.ttf sha256 e1d733afbbfce842…
Tout écart est un REFUS nommé (rc 3) avant d écrire ; ACVRAM_IMAGES_FORCER=1 génère quand même (sha à re-sceller, daté)."""
import os, sys, random, hashlib, math
from PIL import Image, ImageDraw, ImageFont
ATTENDU = {"Pillow": "12.3.0", "freetype2": "2.14.3", "zlib": "1.3.1", "police": "e1d733afbbfce842"}
POLICE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def environnement() -> dict:
    import PIL
    from PIL import features
    return {"Pillow": PIL.__version__, "freetype2": features.version("freetype2"), "zlib": features.version("zlib"),
            "police": hashlib.sha256(open(POLICE, "rb").read()).hexdigest()[:16] if os.path.exists(POLICE) else "absente"}


def ecarts() -> dict:
    """{clé: (attendu, trouvé)} pour tout ce qui diffère de l environnement scellé ; vide = reproductible au bit."""
    env = environnement()
    return {k: (v, env[k]) for k, v in ATTENDU.items() if env[k] != v}


if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "--environnement":
    print(environnement()); sys.exit(0)
if __name__ == "__main__":
    diff = ecarts()
    if diff and os.environ.get("ACVRAM_IMAGES_FORCER") != "1":
        print("REFUS : environnement ≠ scellé, les png ne seraient pas au bit — " +
              ", ".join(f"{k} attendu {a} trouvé {t}" for k, (a, t) in diff.items()) + " (ACVRAM_IMAGES_FORCER=1 pour générer un v2 à re-sceller)")
        sys.exit(3)
D = sys.argv[1]; os.makedirs(D, exist_ok=True); rnd = random.Random(20260920); S = 896
COULEURS = {"rouge": (220, 40, 40), "vert": (40, 170, 60), "bleu": (40, 80, 220), "jaune": (240, 210, 30), "orange": (245, 130, 20),
            "violet": (140, 60, 200), "noir": (20, 20, 20), "blanc": (245, 245, 245), "cyan": (30, 200, 210), "rose": (240, 100, 170)}
try: F = ImageFont.truetype(POLICE, 72)
except Exception: F = ImageFont.load_default()
def fond(): return Image.new("RGB", (S, S), tuple(rnd.randint(200, 255) for _ in range(3)))
def formes(im, n):
    d = ImageDraw.Draw(im); noms = []
    for _ in range(n):
        c = rnd.choice(list(COULEURS)); k = rnd.choice(["cercle", "carré", "triangle"]); r = rnd.randint(50, 110)
        x, y = rnd.randint(r, S - r), rnd.randint(r, S - r)
        if k == "cercle": d.ellipse([x - r, y - r, x + r, y + r], fill=COULEURS[c])
        elif k == "carré": d.rectangle([x - r, y - r, x + r, y + r], fill=COULEURS[c])
        else: d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=COULEURS[c])
        noms.append(f"{k} {c}")
    return noms
def texte(im, mots):
    d = ImageDraw.Draw(im); y = 120
    for m in mots: d.text((80, y), m, fill=(10, 10, 10), font=F); y += 130
def degrade(im, a, b):
    px = im.load()
    for x in range(S):
        t = x / (S - 1); col = tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))
        for y in range(S): px[x, y] = col
def grille(im, n):
    d = ImageDraw.Draw(im); p = S // n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0: d.rectangle([i * p, j * p, (i + 1) * p - 1, (j + 1) * p - 1], fill=(30, 30, 30))
def barres(im, vals):
    d = ImageDraw.Draw(im); w = S // (len(vals) * 2)
    for i, v in enumerate(vals):
        x = 60 + i * 2 * w; d.rectangle([x, S - 80 - v * 6, x + w, S - 80], fill=COULEURS["bleu"]); d.text((x, S - 70), str(v), fill=(0, 0, 0), font=F)
def horloge(im, h, m):
    d = ImageDraw.Draw(im); c = S // 2; R = 380; d.ellipse([c - R, c - R, c + R, c + R], outline=(0, 0, 0), width=8)
    for k in range(12):
        a = math.radians(k * 30 - 90); d.ellipse([c + (R - 40) * math.cos(a) - 12, c + (R - 40) * math.sin(a) - 12, c + (R - 40) * math.cos(a) + 12, c + (R - 40) * math.sin(a) + 12], fill=(0, 0, 0))
    ah = math.radians((h % 12) * 30 + m / 2 - 90); am = math.radians(m * 6 - 90)
    d.line([c, c, c + 200 * math.cos(ah), c + 200 * math.sin(ah)], fill=(0, 0, 0), width=18); d.line([c, c, c + 300 * math.cos(am), c + 300 * math.sin(am)], fill=(200, 0, 0), width=10)
desc = []
for i in range(20):
    im = fond(); k = i % 6
    if k == 0: n = rnd.randint(2, 6); noms = formes(im, n); desc.append(f"{n} formes : " + ", ".join(noms))
    elif k == 1: mots = rnd.sample(["POMME", "TABLE", "SOLEIL", "RIVIERE", "MONTAGNE", "CHAT", "PORTE", "LIVRE", "NUAGE", "TRAIN", "JARDIN", "MUSIQUE"], 3); texte(im, mots); desc.append("texte : " + " / ".join(mots))
    elif k == 2: a, b = rnd.sample(list(COULEURS), 2); degrade(im, COULEURS[a], COULEURS[b]); desc.append(f"dégradé {a} → {b}")
    elif k == 3: n = rnd.choice([4, 6, 8]); grille(im, n); desc.append(f"damier {n}×{n}")
    elif k == 4: vals = [rnd.randint(10, 120) for _ in range(rnd.randint(3, 6))]; barres(im, vals); desc.append("barres " + " ".join(map(str, vals)))
    else: h, m = rnd.randint(1, 12), rnd.choice([0, 15, 30, 45]); horloge(im, h, m); desc.append(f"horloge {h}h{m:02d}")
    im.save(os.path.join(D, f"img{i:02d}.png"), optimize=True)
with open(os.path.join(D, "descriptions.tsv"), "w") as f:
    for i, t in enumerate(desc): f.write(f"img{i:02d}.png\t{t}\n")
with open(os.path.join(D, "images-20.sha256"), "w") as f:
    for i in range(20):
        p = os.path.join(D, f"img{i:02d}.png"); f.write(hashlib.sha256(open(p, "rb").read()).hexdigest() + f"  img{i:02d}.png\n")
print("20 images écrites dans", D)
