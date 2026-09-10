"""La perplexité, pour que les choix de format reposent sur une mesure.

Le rapport signal/bruit d'une matrice de poids et la similarité cosinus entre
vecteurs de logits sont des approximations. Elles corrèlent avec la qualité,
elles sont bon marché, et c'est ce que le convertisseur rapporte par tenseur —
mais elles ne savent pas répondre à « le NVFP4 avec un plancher de promotion à
25 dB vaut-il mieux que de l'INT4 simple sur ce modèle ». La perplexité, si.

L'évaluation est une fenêtre glissante sans détour : on présente une fenêtre de
jetons, on note la prédiction du modèle pour chaque jeton connaissant tout ce
qui précède, on glisse de ``stride`` et on ne compte que les positions
nouvellement exposées, afin qu'aucun jeton ne soit noté deux fois avec des
quantités de contexte différentes.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

__all__ = ["EvalResult", "perplexity", "compare_models", "DEFAULT_CORPUS"]

DEFAULT_CORPUS = """The transformer architecture replaced recurrence with attention,
which made it possible to train on far longer sequences without the gradient
path growing with the sequence length. Quantization reduces the number of bits
each weight occupies; the difficulty is not the storage but the distribution,
because a handful of channels carry activations orders of magnitude larger
than the rest and a uniform grid spends most of its resolution on the wrong
values.

def sliding_window(tokens, size, stride):
    for start in range(0, len(tokens), stride):
        window = tokens[start:start + size]
        if len(window) < 2:
            return
        yield start, window

La memoire d'un ordinateur n'est pas un mur mais une hierarchie: registres,
caches, memoire vive, disque. Un modele de langage qui ne tient pas dans la
memoire video n'est pas pour autant hors de portee, a condition d'accepter que
certaines couches soient lues plus lentement que d'autres et de placer au bon
endroit celles qui comptent.

SELECT model, AVG(tokens_per_second) AS throughput
FROM benchmarks WHERE quantization IN ('nvfp4', 'int4') GROUP BY model;
"""


@dataclass
class EvalResult:
    model: str
    perplexity: float = 0.0
    nll: float = 0.0
    tokens: int = 0
    windows: int = 0
    # Perplexite par tranche de contexte : {jetons de contexte disponibles au
    # minimum: (somme des nll, positions)}. Un modele sain coute une dizaine de
    # nats sur son premier jeton et moins de deux au millieme ; melanger les
    # deux dans une moyenne rend un chiffre qui ne decrit aucun regime.
    par_contexte: dict[int, tuple[float, int]] = field(default_factory=dict)
    seconds: float = 0.0
    weights_bytes: int = 0
    bits_per_weight: float = 0.0
    formats: dict[str, int] = field(default_factory=dict)
    avertissement: str = ""
    # Perplexite CUMULATIVE apres n fenetres, aux jalons de `_JALONS`. Le
    # chiffre final ne dit pas comment il s'est forme : le wikitext oscille de
    # 6,8 a 9,2 avant de se stabiliser, et un biais d'instrument peut dependre
    # de la longueur de contexte — croissant sur une famille d'architecture,
    # decroissant sur une autre. Comparer deux moteurs au meme nombre de
    # fenetres exige de connaitre ce cumul ; llama.cpp le rend, acvram non.
    cumul: dict[int, float] = field(default_factory=dict)
    # Le cadrage voyage avec le chiffre : sans lui, « 7,23 » et « 137 » ont
    # l'air de decrire le meme objet.
    min_context: int = 0
    window: int = 0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "perplexity": round(self.perplexity, 4),
            "nll": round(self.nll, 6),
            "tokens": self.tokens,
            "windows": self.windows,
            "seconds": round(self.seconds, 2),
            "weights_bytes": self.weights_bytes,
            "bits_per_weight": round(self.bits_per_weight, 3),
            "formats": self.formats,
            "avertissement": self.avertissement,
            "cumul": {str(k): round(v, 4) for k, v in sorted(self.cumul.items())},
            "min_context": self.min_context,
            "window": self.window,
            "par_contexte": {str(k): {"ppl": round(math.exp(min(v[0] / v[1], 60.0)), 4),
                                      "jetons": v[1]}
                             for k, v in sorted(self.par_contexte.items()) if v[1]},
        }


# Jalons du cumul : puissances de deux jusqu'au corpus entier. Choisis pour
# qu'une mesure courte et une mesure longue partagent des points de comparaison.
_JALONS = (16, 32, 64, 128, 256, 512)

# Sous ce nombre de positions notees, le resultat porte un avertissement.
# Choisi comme l'ordre de grandeur en dessous duquel l'ecart-type de la
# moyenne des log-vraisemblances depasse l'ecart typique entre deux formats.
_POSITIONS_MINIMALES = 512

# Bornes des tranches de contexte, en jetons vus par la position notee.
_TRANCHES = ((0, 8), (8, 32), (32, 128), (128, 512), (512, 0))


def _load_corpus(path: Optional[str]) -> str:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return DEFAULT_CORPUS


def perplexity(model_dir: str, corpus_path: Optional[str] = None,
               window: int = 512, stride: int = 256,
               max_tokens: int = 8192, device: Optional[str] = None,
               dtype: torch.dtype = torch.bfloat16,
               progress: Optional[Callable[[int, int], None]] = None,
               min_context: int = 0) -> EvalResult:
    """Perplexité par fenêtre glissante sur un modèle converti.

    ``min_context`` écarte du décompte les positions qui ont moins de tant de
    jetons devant elles. Un jeton prédit sans contexte coûte une dizaine de
    nats quel que soit le modèle : sur un corpus court, ces quelques positions
    portent l'essentiel de la moyenne et la perplexité obtenue ne mesure plus
    le modèle mais la longueur du corpus. llama.cpp ne note pour cette raison
    que la seconde moitié de chaque fenêtre. La valeur par défaut reste zéro
    pour que les mesures déjà publiées restent comparables ; toute comparaison
    de formats devrait passer au moins 64.
    """
    from .engine.loader import load_model
    from .engine.model import ForwardBatch
    from .memory.kvcache import BLOCK_SIZE, BlockAllocator
    from .server.chat import load_tokenizer

    t0 = time.time()
    loaded = load_model(model_dir, dtype=dtype, device_override=device)
    tokenizer = load_tokenizer(model_dir)
    if tokenizer is None:
        from .server.chat import pourquoi_pas_de_tokenizer
        raison = pourquoi_pas_de_tokenizer(model_dir) or "cause inconnue"
        raise ValueError(f"la perplexité a besoin d'un tokenizer et n'en a "
                         f"pas : {raison}")

    ids = tokenizer.encode(_load_corpus(corpus_path))[:max_tokens]
    if len(ids) < 16:
        raise ValueError("corpus trop court pour être évalué")

    model = loaded.model
    result = EvalResult(model=os.path.basename(os.path.abspath(model_dir)),
                        min_context=min_context, window=window)
    result.weights_bytes = model.nbytes
    n_params = loaded.spec.total_params
    result.bits_per_weight = (result.weights_bytes * 8 / n_params) if n_params else 0.0
    for entry in loaded.manifest.get("tensors", {}).values():
        f = entry.get("format", "?")
        result.formats[f] = result.formats.get(f, 0) + 1

    blocks_per_window = (window + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    total_nll = 0.0
    counted = 0
    n_windows = max(1, (len(ids) - 1 + stride - 1) // stride)

    for w, start in enumerate(range(0, len(ids) - 1, stride)):
        chunk = ids[start:start + window]
        if len(chunk) < 2:
            break
        alloc = BlockAllocator(blocks_per_window, enable_prefix_cache=False)
        blocks = alloc.allocate(blocks_per_window)
        n = len(chunk)
        slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
                              for i in range(n)], dtype=torch.long)
        batch = ForwardBatch(
            tokens=torch.tensor(chunk, dtype=torch.long),
            positions=torch.arange(n, dtype=torch.long),
            seq_lens=[n], query_lens=[n],
            block_tables=[torch.tensor(blocks, dtype=torch.long)],
            slot_mapping=slots, is_prefill=True)

        logits = model(batch, logits_positions=batch.all_token_indices())
        logits = logits[:-1].to(torch.float32)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=logits.device)

        # On ne note que les positions que cette fenêtre expose pour la
        # première fois, afin qu'un jeton ne soit jamais compté deux fois avec
        # des quantités de contexte différentes.
        first_new = 0 if start == 0 else max(0, (window - stride) - 1)
        # Le logit d'indice i predit le jeton i+1 en ayant vu i+1 jetons.
        first_new = max(first_new, min_context)
        if first_new >= logits.shape[0]:
            break
        nll = torch.nn.functional.cross_entropy(
            logits[first_new:], targets[first_new:], reduction="sum")
        total_nll += float(nll)
        counted += int(targets[first_new:].numel())
        # Le meme cout, reparti par quantite de contexte disponible : c'est ce
        # qui dit si un chiffre eleve vient du modele ou des premieres
        # positions. Sans ce detail, un corpus de 283 jetons et un corpus de
        # 100 000 rendent deux nombres qu'on croit comparables.
        pertes = torch.nn.functional.cross_entropy(
            logits[first_new:], targets[first_new:], reduction="none")
        for k, (bas, haut) in enumerate(_TRANCHES):
            i0 = max(0, bas - first_new)
            i1 = min(pertes.shape[0], haut - first_new) if haut else pertes.shape[0]
            if i1 > i0:
                som, n_pos = result.par_contexte.get(bas, (0.0, 0))
                result.par_contexte[bas] = (som + float(pertes[i0:i1].sum()),
                                            n_pos + i1 - i0)
        result.windows += 1
        if result.windows in _JALONS and counted:
            result.cumul[result.windows] = math.exp(
                min(total_nll / counted, 60.0))
        if progress:
            progress(w + 1, n_windows)
        if start + window >= len(ids):
            break

    # LE PROTOCOLE DIT « SEGMENTS DISJOINTS » QUAND stride == window, et il
    # n'a pas besoin d'un autre mode : first_new vaut alors max(0, -1) = 0,
    # donc chaque segment est contigu au precedent, sans recouvrement, et
    # toutes ses positions sont notees une fois. C'est le protocole employe
    # par la litterature de quantification pour la perplexite WikiText-2 —
    # concatener le split, decouper en segments de 2048, moyenner la NLL.
    # A appeler ainsi : perplexity(m, corpus_path=..., window=2048,
    # stride=2048, min_context=0).
    #
    # ET LE COMPTE SE VERIFIE, sans quoi rien ne garantit qu'on note ce qu'on
    # croit. En mode disjoint chaque jeton sauf le premier est predit
    # exactement une fois, donc `counted` doit valoir len(ids) - 1 aux
    # segments tronques pres. Un ecart signale un recouvrement ou un oubli, et
    # se lirait sinon comme une perplexite legerement differente — la pire
    # forme de defaut, celle qui ne se voit pas.
    if stride == window and min_context == 0:
        # L'INVARIANT N'EST PAS len(ids) - 1, ET MON PREMIER JET L'A ECRIT.
        # En segments disjoints chaque forward est independant : le premier
        # jeton d'un segment n'a aucun predecesseur, donc il n'est PAS predit.
        # `logits[:-1]` contre `chunk[1:]` note WINDOW - 1 positions par
        # segment, pas window. L'invariant est donc nsamples x (window - 1) —
        # 168 x 2047 = 343 896 sur wikitext-2 — et non len(ids) - 1 = 344 063.
        # Ecrit autrement, il aurait averti A TORT sur un protocole correct,
        # et l'avertissement aurait envoye chercher un defaut inexistant.
        #
        # C'est aussi ce qui montre que notre cadrage est celui de GPTQ : leur
        # nll_i vaut loss_i x 2048 avec loss_i moyennee sur 2047, puis divise
        # par nsamples x 2048 — le facteur 2048/2047 s'annule et il reste la
        # moyenne exacte sur nsamples x 2047. Aucune convention a corriger.
        # LA FORMULE SUIT LA BOUCLE, ET DEUX VERSIONS S'Y SONT TROMPEES.
        # La boucle parcourt `range(0, len(ids) - 1, stride)` : le nombre de
        # segments est donc le PLAFOND de (len(ids) - 1) / window, pas son
        # plancher — sur 344 064 jetons cela fait 168 segments et non 167, et
        # 343 896 positions notables et non 341 849. Chacune de mes deux
        # premieres versions a donc averti A TORT sur un protocole correct,
        # et un garde-fou qui crie toujours est un garde-fou qu'on desactive.
        # Verifie contre la mesure : 168 x 2047 = 343 896, exactement ce que
        # le compteur rend sur wikitext-2 tronque a 344 064 jetons.
        debuts = range(0, max(0, len(ids) - 1), window)
        attendu = sum(min(window, len(ids) - d) - 1 for d in debuts
                      if min(window, len(ids) - d) >= 2)
        if counted != attendu:
            # `avertissement` et non une liste : c'est le champ que porte
            # EvalResult. Un garde-fou qui leve AttributeError au moment
            # d'alerter ne garde rien — mon premier jet ecrivait dans
            # result.warnings, qui n'existe pas.
            manque = (f"mode disjoint : {counted} positions notees pour "
                      f"{attendu} attendues ({len(debuts)} segments, "
                      f"{window - 1} positions notables chacun sur {len(ids)} "
                      "jetons) — un jeton est compte deux fois ou pas du "
                      "tout, et la perplexite ne porte pas sur le corpus "
                      "annonce")
            result.avertissement = (result.avertissement + " | " + manque
                                    if result.avertissement else manque)
    result.tokens = counted
    if counted == 0:
        raise ValueError(
            f"aucune position notee : le corpus fait {len(ids)} jetons et "
            f"min_context={min_context} les ecarte toutes. Allonger le corpus "
            f"ou baisser min_context.")
    # Un chiffre tire de trop peu de positions n'est pas un chiffre : sa
    # variance depasse l'ecart qu'on veut mesurer. Le corpus interne fait 283
    # jetons ; note comme llama.cpp (fenetre 512, min_context 256) il n'en
    # laisserait que vingt-six. Trois sessions ont interprete deux jours durant
    # un 137 obtenu sur 282 positions notees des le premier jeton — l'avertir
    # est le minimum, et il voyage avec le resultat, pas seulement a l'ecran.
    if counted < _POSITIONS_MINIMALES:
        result.avertissement = (
            f"{counted} positions notees seulement (moins de "
            f"{_POSITIONS_MINIMALES}) : la variance de ce chiffre depasse "
            f"probablement les ecarts entre formats. Corpus plus long requis.")
    if min_context == 0 and len(ids) < window:
        note = (f"corpus de {len(ids)} jetons plus court que la fenetre de "
                f"{window}, note des la premiere position ({counted} positions "
                f"notees) : les jetons sans contexte dominent la moyenne. "
                f"Comparer a llama.cpp demande --min-context {window // 2}, "
                f"qui ne laisserait ici que "
                f"{max(0, len(ids) - 1 - window // 2)} positions.")
        result.avertissement = (result.avertissement + " " + note
                                if result.avertissement else note)
    result.nll = total_nll / max(1, counted)
    result.perplexity = math.exp(min(result.nll, 60.0))
    result.seconds = time.time() - t0
    return result


def compare_models(model_dirs: list[str], **kwargs: Any) -> list[EvalResult]:
    """Évalue plusieurs modèles convertis sur le même corpus et les classe."""
    out = [perplexity(d, **kwargs) for d in model_dirs]
    return sorted(out, key=lambda r: r.perplexity)


def render(results: list[EvalResult]) -> str:
    if not results:
        return "aucun resultat"
    width = max(len(r.model) for r in results)
    cadres = {(r.window, r.min_context) for r in results}
    lines = []
    if len(cadres) == 1:
        w, mc = next(iter(cadres))
        lines.append(f"  cadrage : fenetre {w}, contexte minimal {mc} jeton(s)")
        lines.append("")
    lines.append(f"  {'modele':<{width}}  {'ppl':>9}  {'bpp':>6}  {'taille':>10}  "
                 f"{'jetons':>8}")
    best = min(r.perplexity for r in results)
    for r in results:
        delta = "" if r.perplexity == best else f"  (+{100*(r.perplexity/best-1):.1f}%)"
        lines.append(f"  {r.model:<{width}}  {r.perplexity:9.3f}  "
                     f"{r.bits_per_weight:6.2f}  {r.weights_bytes/2**20:8.1f}Mio  "
                     f"{r.tokens:8d}{delta}")
    # Les formats REELS du modele evalue, a cote du chiffre. Le 8/09/2026 un
    # dossier nomme « temoin-int8 » avait ses 72 projections de perceptron en
    # q3n a 3,25 bits : la garde anti-grossissement les avait basculees, en
    # l'annoncant dans un journal detache que personne n'a lu. La perplexite
    # qui en est sortie a fait chercher un biais d'instrument une demi-journee.
    # Ce qui n'est pas imprime a cote du chiffre finit par etre suppose.
    for r in results:
        if r.formats:
            detail = ", ".join(f"{f} {n}" for f, n in
                               sorted(r.formats.items(), key=lambda x: -x[1]))
            lines.append(f"  {r.model} : {detail}")
    if any(r.formats for r in results):
        lines.append("")
    avertis = {r.avertissement for r in results if r.avertissement}
    for a in sorted(avertis):
        lines.append("")
        lines.append(f"  ATTENTION : {a}")
    for r in results:
        if r.cumul:
            lines.append("")
            lines.append(f"  {r.model}, perplexite cumulative :")
            for n, v in sorted(r.cumul.items()):
                lines.append(f"    apres {n:4d} fenetres  {v:9.4f}")
    for r in results:
        if r.par_contexte:
            lines.append("")
            lines.append(f"  {r.model} par contexte disponible :")
            for bas, (som, n) in sorted(r.par_contexte.items()):
                if n:
                    # La borne affichee est celle de la tranche ET du
                    # min_context : une tranche 128-512 filtree a 256 ne
                    # contient que des positions a 256 jetons ou plus, et
                    # l'annoncer « a partir de 128 » decrirait un objet plus
                    # facile que celui qu'on a mesure.
                    reel = max(bas, r.min_context)
                    lines.append(f"    a partir de {reel:>4} jetons  "
                                 f"ppl {math.exp(min(som / n, 60.0)):9.3f}  "
                                 f"sur {n:5d} positions")
    return "\n".join(lines)
