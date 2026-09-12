"""Non-regression du parc : ca charge, et ca produit du texte coherent.

Le critere est DUR — chargement, exil inattendu, charabia, plantage — pas un
debit. A ce stade nous cherchons des regressions visibles, pas des dixiemes de
pourcent. Et un test de texte AFFICHE le texte : un detecteur de charabia qui
juge a notre place a deja menti une fois.
"""
import os, sys, torch
sys.path.insert(0, "~/Bureau/Claude/anticitoyen-vram")
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.server.chat import load_tokenizer
from acvram.engine.layers import QuantLinear
A = "/mnt/2TO_2023_980PRO/Modeles/models_acvram"
nom, famille = sys.argv[1], sys.argv[2]
INVITE = "Explique en une phrase ce qu'est la photosynthese."
try:
    charge = load_model(os.path.join(A, nom), max_model_len=2048)
    tok = load_tokenizer(os.path.join(A, nom))
    exiles = sum(1 for m in charge.model.modules()
                 if isinstance(m, QuantLinear) and m.streamed is not None)
    mot = Engine(charge, tok, max_batch_size=2, max_model_len=2048)
    g = getattr(mot, "graphs", None)
    graphes = bool(getattr(g, "enabled", False))
    # LE GABARIT DE CHAT, sans quoi on mesure « pas de gabarit » et non une
    # regression : une invite brute a temperature 0 fait boucler ou deriver
    # n importe quel modele d instruction. Premier essai sans lui : deux
    # modeles sur quatre rendaient du charabia.
    gabarit = "brut"
    # Le garde testait `chat_template`, que notre enveloppe `Tokenizer`
    # n a pas — alors qu elle a bien `apply_chat_template`. Attribut
    # voisin de celui qui decide : troisieme fois aujourd hui.
    if tok is not None and hasattr(tok, "apply_chat_template"):
        # `apply_chat_template` de notre enveloppe rend du TEXTE, pas des
        # identifiants — il faut encoder derriere.
        ids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": INVITE}], True))
        gabarit = "chat"
    elif tok is not None:
        ids = tok.encode(INVITE)
    else:
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
    mot.add_request(ids, SamplingParams(temperature=0.0, max_tokens=40),
                    request_id="r0")
    produits = []
    for _ in range(64):
        for s in (mot.step() or []):
            produits.extend(list(getattr(s, "token_ids", ()) or []))
        if not mot.running and not mot.waiting: break
    texte = tok.decode(produits) if tok and produits else "(pas de tokenizer)"
    print(f"{famille}\t{nom}\texil {exiles}\tgraphes {'oui' if graphes else 'NON'}\t"
          f"{len(produits)} jetons\tgabarit {gabarit}\tOK")
    print(f"    >>> {texte[:220]!r}")
except Exception as exc:                                   # noqa: BLE001
    import traceback; traceback.print_exc()
    print(f"{famille}\t{nom}\tECHEC\t{type(exc).__name__}: {str(exc)[:120]}")
