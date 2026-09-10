#!/usr/bin/env python3
"""PPL de reference : transformers sur les poids d'origine, protocole GPTQ.

POURQUOI CE POINT AVANT TOUT AUTRE. Si nous convertissons d'abord, un ecart au
chiffre publie ne dira pas s'il vient du MOTEUR, de la CONVERSION ou du
PROTOCOLE. Cette mesure-ci n'emploie ni notre conversion ni notre moteur :
memes poids, meme corpus, meme fenetre, implementation de reference. Elle
separe les trois causes que la conversion melange.

    (1) s'ecarte de la valeur publiee   -> c'est notre PROTOCOLE
    (1) colle et (2) s'ecarte           -> c'est NOTRE CHAINE
    les deux collent                    -> chaine validee de bout en bout

PROTOCOLE, repris de la litterature de quantification (GPTQ et suivants) :
concatener le split test, decouper en segments DISJOINTS de 2048 jetons, et
moyenner. Deux details de leur code que je reproduis a dessein, parce qu'ils
deplacent le chiffre au troisieme decimale :

  - la NLL d'un segment est la loss MOYENNE (sur 2047 positions) multipliee par
    2048, puis la somme est divisee par nsamples * 2048. C'est donc la moyenne
    des losses PAR SEGMENT, pas la moyenne ponderee exacte : un facteur
    2047/2048 dans l'exposant. Negligeable, mais reproduit plutot que corrige —
    un etalon se compare a ce qui est publie, pas a ce qui serait juste ;
  - les segments incomplets sont IGNORES, pas completes.

DIFFERENCE DE CORPUS A DECLARER, et je ne la corrige pas faute de savoir
laquelle est employee : les articles font `"\\n\\n".join(dataset['text'])` sur
le dataset HuggingFace, ou chaque ligne est un element. Nous lisons
`wiki.test.raw`, le fichier brut d'origine, ou les lignes sont separees par un
seul `\\n`. Ce n'est PAS la meme chaine de caracteres, et l'ecart peut atteindre
quelques centiemes de perplexite. Le banc mesure les DEUX variantes pour que
la difference soit chiffree au lieu d'etre supposee.
"""
import argparse
import json
import math
import sys
import time


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fenetre", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--sortie", default="/tmp/etalon-ppl-transformers.json")
    a = ap.parse_args()

    dt = getattr(torch, a.dtype)
    # use_fast=False : c'est le tokenizer que GPTQ emploie (datautils.py:16),
    # et le lent ne segmente pas toujours comme le rapide. Un etalon se compare
    # a ce qui est publie, donc jusqu'au choix du tokenizer.
    tok = AutoTokenizer.from_pretrained(a.modele, use_fast=False)
    t0 = time.time()
    # PAS de device_map : il exige le paquet `accelerate`, absent du venv, et
    # echoue avec un message qui parle de tp_plan et de set_default_device —
    # trois pistes pour une dependance manquante. Le .to() fait la meme chose
    # sans dependance, au prix d'un passage en RAM hote au chargement.
    mod = AutoModelForCausalLM.from_pretrained(a.modele, dtype=dt).to("cuda:0")
    mod.eval()
    print(f"charge en {time.time() - t0:.0f} s, dtype {next(mod.parameters()).dtype}",
          flush=True)

    brut = open(a.corpus, encoding="utf-8").read()

    # BRAS A — LE TEXTE DE GPTQ, RECONSTRUIT ET VERIFIE. Leur datautils.py fait
    # `"\n\n".join(load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    # ['text'])`. Le paquet `datasets` n'est pas dans le venv, donc je
    # reconstruis le texte depuis le fichier brut — mais je ne le SUPPOSE pas :
    # les elements du dataset wikitext sont les lignes du fichier AVEC leur
    # saut final, et le split test en compte 4 358. Si notre fichier n'en donne
    # pas autant, la reconstruction est fausse et le bras A est refuse.
    lignes = brut.split("\n")
    elements = [l + "\n" for l in lignes[:-1]] if lignes[-1] == "" else \
               [l + "\n" for l in lignes]
    if len(elements) != 4358:
        print(f"BRAS A REFUSE : {len(elements)} elements reconstruits pour "
              "4 358 attendus dans le split test de wikitext-2-raw-v1. La "
              "reconstruction ne reproduit pas le dataset, et le comparer a "
              "une valeur GPTQ n'aurait aucun sens.", file=sys.stderr)
        variantes = {"llamacpp-fichier-brut": brut}
    else:
        variantes = {
            "gptq-elements-joints": "\n\n".join(elements),
            "llamacpp-fichier-brut": brut,
        }

    releves = []
    for nom, texte in variantes.items():
        ids = tok(texte, return_tensors="pt").input_ids
        n = ids.numel() // a.fenetre
        somme, t1 = 0.0, time.time()
        with torch.no_grad():
            for i in range(n):
                seg = ids[:, i * a.fenetre:(i + 1) * a.fenetre].to("cuda:0")
                lg = mod(seg).logits
                perte = torch.nn.functional.cross_entropy(
                    lg[:, :-1, :].reshape(-1, lg.shape[-1]).float(),
                    seg[:, 1:].reshape(-1))
                somme += perte.item() * a.fenetre
                if (i + 1) % 40 == 0:
                    print(f"  {nom} {i + 1}/{n} segments, "
                          f"PPL partielle {math.exp(somme / ((i + 1) * a.fenetre)):.4f}",
                          flush=True)
        ppl = math.exp(somme / (n * a.fenetre))
        d = {"variante": nom, "jetons": int(ids.numel()), "segments": n,
             "fenetre": a.fenetre, "dtype": a.dtype, "ppl": ppl,
             "secondes": round(time.time() - t1, 1)}
        releves.append(d)
        print(f"{nom:>20} : PPL {ppl:.4f} sur {n} segments de {a.fenetre} "
              f"({ids.numel()} jetons, {d['secondes']} s)", flush=True)

    if len(releves) == 2:
        e = 100 * (releves[0]["ppl"] / releves[1]["ppl"] - 1)
        print(f"\necart entre les deux variantes de corpus : {e:+.3f} % — "
              "c'est l'incertitude que porte le seul choix de la mise en forme "
              "du texte, avant tout moteur et toute conversion.")
    json.dump(releves, open(a.sortie, "w"), indent=1)
    print(f"TERMINE — releve dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
