"""Pièce 283 (relecture chef) : `cmd_serve` (acvram/cli.py) est le SEUL point d'entrée qui
construit un propositeur — `Engine.__init__` prend `speculator=None` par défaut (jamais ngram
lui-même), aucune commande `chat`/`generate` n'existe dans le CLI (grep sur `add_parser`), le
serveur OpenAI n'expose aucun paramètre de requête `speculative` (grep sur `acvram/server/`),
et aucun GUI/HTML/JS du dépôt ne mentionne ngram. Ce test fige ces quatre faits : si l'un
d'eux change (nouvelle commande, nouveau paramètre serveur, nouveau défaut Engine), il doit
casser ici plutôt que de laisser un second chemin par défaut à ngram passer inaperçu."""
import inspect
import pathlib
import re

RACINE = pathlib.Path(__file__).resolve().parent.parent


def test_engine_ne_defaut_jamais_a_un_speculator():
    from acvram.engine.runner import Engine
    sig = inspect.signature(Engine.__init__)
    assert sig.parameters["speculator"].default is None, (
        "Engine.__init__ ne doit jamais construire de propositeur par défaut lui-même — "
        "seul cmd_serve (acvram/cli.py) décide, et c'est là que le défaut 283 vit."
    )


def test_aucune_commande_chat_ou_generate_dans_le_cli():
    texte = (RACINE / "acvram" / "cli.py").read_text(encoding="utf-8")
    noms = set(re.findall(r'add_parser\("([a-z]+)"', texte))
    assert not ({"chat", "generate"} & noms), (
        f"une commande chat/generate est apparue dans le CLI ({noms & {'chat', 'generate'}}) "
        "— vérifier qu'elle ne construit pas son propre propositeur par défaut."
    )


def test_le_serveur_openai_n_expose_aucun_parametre_speculative():
    serveur = RACINE / "acvram" / "server"
    for f in serveur.glob("*.py"):
        assert "speculative" not in f.read_text(encoding="utf-8").lower(), (
            f"{f} mentionne 'speculative' — vérifier qu'aucune requête ne peut choisir ngram "
            "par défaut au niveau de l'API (283 ne couvre que cmd_serve)."
        )


def test_aucun_gui_ne_mentionne_ngram_ou_speculative():
    motifs = [p for ext in ("*.html", "*.js") for p in RACINE.rglob(ext)
             if ".git" not in p.parts and "node_modules" not in p.parts]
    fautifs = [str(p) for p in motifs if "speculative" in p.read_text(encoding="utf-8", errors="ignore").lower()]
    assert not fautifs, f"GUI mentionnant 'speculative' — vérifier son propre défaut : {fautifs}"
