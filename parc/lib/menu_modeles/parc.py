"""Commun claude-modeles / kimi-modeles — données du parc.

Une ligne du parc (Modele), la lecture des TSV, la reconstruction de la liste
complète (charger_parc) et la réécriture d'une fiche (ecrire_note), plus les
barèmes de tri (rang_qualite, rang_refus). Les constantes de chemins et de
moteurs viennent de `config` ; aucune circularité neuve (config n'importe pas
parc).
"""
import os
import re
import sys
import tomllib

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import GObject  # noqa: E402

from .config import (  # noqa: E402
    MOTEURS, ORDRE_MOTEUR, FICHE_ABSENTE, NOTE_MOTS, REFUS_RANG,
    CONFIG, NOTES, GGUF_TSV, VLLM_TSV, ACVRAM_TSV, VISION_TSV, MAISON,
)


def rang_qualite(txt):
    t = (txt or "").strip().lower()
    if t in NOTE_MOTS:
        return NOTE_MOTS[t]
    etoiles = txt.count("★") if txt else 0
    return etoiles or -1


def rang_refus(txt):
    t = (txt or "").strip().lower()
    if t in REFUS_RANG:
        return REFUS_RANG[t]
    m = re.match(r"(\d+)\s*/\s*(\d+)", t)          # « 0/5 refus », écrit par banc-refus
    if m and int(m.group(2)):
        return round(int(m.group(1)) * 4 / int(m.group(2)))
    return 9                                        # inconnu : en queue de tri


class Modele(GObject.Object):
    """Une ligne du parc. Les champs de fiche sont modifiables depuis le panneau."""

    __gtype_name__ = "ClaudeModele"

    def __init__(self, alias, provider, nom, ctx, refus, tps, qual, usage,
                 dossier="", ctx_service=0, gabarit="", capacites=(), vision_status=""):
        super().__init__()
        self.alias = alias
        self.provider = provider
        self.nom = nom
        self.ctx = int(ctx or 0)
        self.refus = refus
        self.tps = tps
        self.qual = qual
        self.usage = usage
        self.dossier = dossier
        self.ctx_service = int(ctx_service or 0)
        self.gabarit = gabarit
        self.capacites = tuple(capacites or ())
        self.vision_status = vision_status          # "vision", "texte-seul" ou ""
        self.charge = False
        self.taille = None                          # calculée à la demande

    @property
    def moteur(self):
        return MOTEURS.get(self.provider)

    @property
    def lancable(self):
        """Un alias sans chemin connu ne peut pas être préchargé d'ici. TabbyAPI
        charge le sien par son API et l'appoint n'en sert qu'un : ni l'un ni
        l'autre n'a besoin d'un dossier."""
        return self.provider in ("tabby", "rapide", "yals") or bool(self.dossier)

    @property
    def debit(self):
        # Le premier nombre de la cellule, pas ses chiffres concaténés : « 134,9* » (débit
        # mesuré avant le 20/09, marqué d'une étoile) vaut 134,9, jamais 1349.
        m = re.match(r"\s*(\d+(?:[.,]\d+)?)", self.tps or "")
        return float(m.group(1).replace(",", ".")) if m else -1

    @property
    def outils_etat(self):
        """« ok », « non » ou « ? ». Deux écritures cohabitent dans les fiches :
        le mot-clé agent-ok/agent-non des premiers modèles, et le score mesuré
        par banc-outils (« outils 12/12 », « 0/12 aux outils ») des suivants."""
        u = (self.usage or "").lower()
        if "agent-non" in u:
            return "non"
        if "agent-ok" in u:
            return "ok"
        m = re.search(r"outils\s+(\d+)\s*/\s*(\d+)", u) or re.search(r"(\d+)\s*/\s*(\d+)\s+aux outils", u)
        if m and int(m.group(2)):
            return "ok" if int(m.group(1)) / int(m.group(2)) >= 0.75 else "non"
        return "?"

    @property
    def outils_ok(self):
        return self.outils_etat == "ok"

    def texte_recherche(self):
        # synonymes lisibles des capacités techniques. « vision » et « voit images »
        # NE sont PAS ici : ce sont des mots-clés sémantiques, traités dans
        # _passe_filtre par vision_status. Un VL converti « texte-seul » porte
        # « vision » dans son alias ou son nom mais ne sert pas d'images — il ne
        # doit pas sortir sur « vision ». « vl » et le reste restent substring libre.
        lisibles = []
        if "video_in" in self.capacites:
            lisibles += ["voit vidéos"]
        if "thinking" in self.capacites:
            lisibles += ["réflexion"]
        if self.outils_etat == "ok":
            lisibles += ["génère images", "génère vidéos", "médias"]
        return " ".join([self.alias, self.usage or "", self.nom or "", self.qual or "",
                         " ".join(self.capacites), " ".join(lisibles)]).lower()


def lire_tsv(chemin, mini=2):
    d = {}
    try:
        for ligne in chemin.read_text().splitlines():
            if not ligne.strip() or ligne.startswith("#"):
                continue
            c = ligne.rstrip("\n").split("\t")
            if len(c) >= mini:
                d[c[0]] = c[1:]
    except OSError:
        pass
    return d


def charger_parc():
    """Reconstruit la liste complète depuis les fichiers, dans l'ordre du menu texte.
    `CONFIG` absent (premier lancement, aucun modèle balayé) rend un parc vide, pas une
    erreur : seul un fichier PRÉSENT mais illisible (TOML cassé) est une faute (pièce 84)."""
    if not CONFIG.exists():
        return []
    try:
        conf = tomllib.load(CONFIG.open("rb"))
    except Exception as e:
        raise RuntimeError(f"config.toml illisible : {e}") from e
    notes = lire_tsv(NOTES, 5)
    gguf = lire_tsv(GGUF_TSV, 3)
    vllm = lire_tsv(VLLM_TSV, 3)
    acvr = lire_tsv(ACVRAM_TSV, 3)
    vision = lire_tsv(VISION_TSV, 2)

    parc = []
    sans_fiche = []
    hors_moteur = []
    for alias, m in conf.get("models", {}).items():
        p = m.get("provider")
        if p not in MOTEURS:
            hors_moteur.append(alias)
            continue
        n = notes.get(alias)
        if n is None:
            sans_fiche.append(alias)
            n = list(FICHE_ABSENTE)
        # un champ vide ou « ? » dans la fiche vaut une fiche absente pour ce champ
        n = [v if v.strip() and v.strip() != "?" else FICHE_ABSENTE[i] for i, v in enumerate(n[:4])]
        # multimodal : « 👁 » en tête de l'usage si le moteur servi prend des images
        # (llama.cpp + mmproj, vLLM VL) ; un converti acvram d'un modèle VL reste texte seul
        vs = (vision.get(alias) or [""])[0]
        if vs == "vision":
            n[3] = "👁 " + n[3] + " · vision (images)"
        elif vs == "texte-seul":
            n[3] = n[3] + " · VL converti, texte seul"
        dossier = ctx_service = gab = ""
        if p == "llamacpp" and alias in gguf:
            ligne = gguf[alias]
            dossier = ligne[0]
            ctx_service = ligne[1] if len(ligne) > 1 else 0
            gab = ligne[2] if len(ligne) > 2 else ""
            if gab.startswith("~/"):
                gab = str(MAISON / gab[2:])
        elif p == "vllm" and alias in vllm:
            ligne = vllm[alias]
            dossier = ligne[0]
            ctx_service = ligne[1] if len(ligne) > 1 else 0
        elif p == "acvram" and alias in acvr:
            ligne = acvr[alias]
            dossier = ligne[0]
            ctx_service = ligne[1] if len(ligne) > 1 else 0
        elif p == "rapide":
            ctx_service = 65536          # défaut de llamacpp-appoint, jamais surchargé
        parc.append(Modele(alias, p, m.get("model", "?"), m.get("max_context_size", 0),
                           n[0], n[1], n[2], n[3], dossier, ctx_service, gab,
                           m.get("capabilities", []), vs))
    parc.sort(key=lambda x: (ORDRE_MOTEUR.get(x.provider, 9), x.alias))
    # S1/S3 (poste7-menus-cloture-19-09) : le compte est le contrôle, pas la lecture
    print(f"parc : {len(parc)} modèles, {len(sans_fiche)} alias sans fiche, "
          f"{len(hors_moteur)} hors moteur connu", file=sys.stderr, flush=True)
    for a in sans_fiche:
        print(f"  sans fiche : {a}", file=sys.stderr)
    for a in hors_moteur:
        print(f"  hors moteur : {a} ({conf['models'][a].get('provider')!r})", file=sys.stderr)
    return parc


def ecrire_note(alias, refus, tps, qual, usage):
    """Réécrit la fiche d'un alias en conservant l'ordre du fichier et ses commentaires.
    Écriture atomique : une coupure de courant ne laisse pas un fichier tronqué."""
    lignes = []
    trouve = False
    neuve = "\t".join([alias, refus, tps, qual, usage])
    try:
        lignes = NOTES.read_text().splitlines()
    except OSError:
        pass
    for i, ligne in enumerate(lignes):
        if ligne.startswith(alias + "\t"):
            lignes[i] = neuve
            trouve = True
            break
    if not trouve:
        lignes.append(neuve)
    tmp = NOTES.with_suffix(".tsv.tmp")
    tmp.write_text("\n".join(lignes) + "\n")
    os.replace(tmp, NOTES)
