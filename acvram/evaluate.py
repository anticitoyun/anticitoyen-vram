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

import hashlib
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
    bos: Optional[int] = None            # BOS posé en tête de chaque fenêtre (None : le tokeniseur n en a pas)
    # Perplexite par tranche de contexte : {jetons de contexte disponibles au
    # minimum: (somme des nll, positions)}. Un modele sain coute une dizaine de
    # nats sur son premier jeton et moins de deux au millieme ; melanger les
    # deux dans une moyenne rend un chiffre qui ne decrit aucun regime.
    par_contexte: dict[int, tuple[float, int]] = field(default_factory=dict)
    seconds: float = 0.0
    weights_bytes: int = 0
    # Nom explicite : ce chiffre est nbytes/params APRES chargement, donc il
    # depend du dtype de chargement (26,99 en fp32, 20,04 en bf16 pour un meme
    # dossier fp16 a 16,00 bits sur disque) et il inclut caches et tampons.
    # Indicatif seulement. La densite qui DECIDE est analytique :
    # quant/formats.py:bits_per_weight(fmt, group_size).
    bits_par_poids_en_memoire: float = 0.0
    nbytes_detail: dict = field(default_factory=dict)
    # Un releve de perplexite qui ne dit pas SUR QUOI il porte ne peut plus
    # etre compare ensuite. Le 10/09, trois releves archives ont oblige a une
    # reconstitution arithmetique (168 x 2047 = 343 896 positions notables ne
    # sont possibles qu'avec >= 344 064 jetons, donc avec wiki-gptq.txt et pas
    # wiki.test.raw, qui n'en rend que 335 688) pour savoir si notre chaine et
    # l'etalon exterieur parlaient du meme texte. Ils en parlaient, mais rien
    # dans les fichiers ne le disait.
    corpus_chemin: str = ""
    corpus_octets: int = 0
    corpus_sha256: str = ""
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
            "bits_par_poids_en_memoire": round(self.bits_par_poids_en_memoire, 3),
            "nbytes_detail": self.nbytes_detail,
            "corpus_chemin": self.corpus_chemin,
            "corpus_octets": self.corpus_octets,
            "corpus_sha256": self.corpus_sha256,
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
# positions par tranche de tête dans `perplexity` (ACVRAM_PPL_TRANCHE) : 256
# → logits fp32 256 × 151 936 = 156 Mio par tranche au lieu de 1,2 Gio
_PPL_TRANCHE = int(os.environ.get("ACVRAM_PPL_TRANCHE", "256"))


def h_device_cuda(model) -> bool:
    try:
        return next(model.parameters()).device.type == "cuda"
    except Exception:                                        # noqa: BLE001
        return False


def _pertes_par_tranches(model, h: torch.Tensor, targets: torch.Tensor,
                         first_new: int, tranche: int = 0) -> torch.Tensor:
    """NLL par position, positions `first_new` … n−2 (le logit i prédit le
    jeton i+1), la tête appliquée par tranches de `tranche` positions sur les
    états cachés normalisés `h` [n, hidden] — arithmétique de `forward`
    (`_tete` puis `_logits_finaux`, logits fp32), fenêtre entière ou non."""
    tranche = tranche or _PPL_TRANCHE
    n = h.shape[0]
    morceaux = []
    for a in range(first_new, n - 1, tranche):
        b = min(a + tranche, n - 1)
        logits = model._logits_finaux(model._tete(h[a:b])).to(torch.float32)
        morceaux.append(torch.nn.functional.cross_entropy(
            logits, targets[a:b], reduction="none"))
        del logits
    return torch.cat(morceaux) if morceaux else torch.zeros(0, device=h.device)


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
               min_context: int = 0,
               apres_chargement: Optional[Callable[[Any], None]] = None) -> EvalResult:
    """Perplexité par fenêtre glissante sur un modèle converti.

    ``min_context`` écarte du décompte les positions qui ont moins de tant de
    jetons devant elles. Un jeton prédit sans contexte coûte une dizaine de
    nats quel que soit le modèle : sur un corpus court, ces quelques positions
    portent l'essentiel de la moyenne et la perplexité obtenue ne mesure plus
    le modèle mais la longueur du corpus. llama.cpp ne note pour cette raison
    que la seconde moitié de chaque fenêtre. La valeur par défaut reste zéro
    pour que les mesures déjà publiées restent comparables ; toute comparaison
    de formats devrait passer au moins 64.

    ``apres_chargement``, appelé avec le modèle chargé juste après
    `load_model` et avant la première fenêtre : point d'extension pour des
    sondes de recherche (hooks de fake-quant d'activation, par exemple —
    voir outils/hooks_activations_a4.py) sans dupliquer la boucle de
    fenêtre glissante ci-dessous dans chaque script qui en a besoin.
    """
    from .engine.loader import load_model
    from .engine.model import ForwardBatch
    from .memory.kvcache import BLOCK_SIZE, BlockAllocator
    from .server.chat import load_tokenizer

    t0 = time.time()
    # 22/09 (`verdict-ppl-31b-ab-22-09`) : sans `max_model_len` ni
    # `max_concurrent_seqs`, `_replanifier` (loader.py) dimensionne le cache KV
    # comme un SERVEUR — `kv_planned_seqs` (8) × `kv_max_tokens` (9 791) du
    # manifeste, 4,5 Gio — et la réserve de préfill sur 9 791 jetons (2,9 Gio) :
    # 23,8 Gio de poids + 7,4 Gio de budget > 29,5 Gio libres → 11 MLP exilés,
    # 619 % de PCIe, PPL aberrante (5 446) sur un 31B qui tient seul. L évaluation
    # ne traite qu UNE fenêtre à la fois : le budget est 1 séquence de
    # `window` (+ un bloc pour l allocateur, `blocks_per_window`), 1,7 Gio.
    loaded = load_model(model_dir, dtype=dtype, device_override=device,
                        max_model_len=window + BLOCK_SIZE, max_concurrent_seqs=1)
    tokenizer = load_tokenizer(model_dir)
    if tokenizer is None:
        from .server.chat import pourquoi_pas_de_tokenizer
        raison = pourquoi_pas_de_tokenizer(model_dir) or "cause inconnue"
        raise ValueError(f"la perplexité a besoin d'un tokenizer et n'en a "
                         f"pas : {raison}")

    texte = _load_corpus(corpus_path)
    ids = tokenizer.encode(texte)[:max_tokens]
    if len(ids) < 16:
        raise ValueError("corpus trop court pour être évalué")

    model = loaded.model
    if apres_chargement is not None:
        apres_chargement(model)
    result = EvalResult(model=os.path.basename(os.path.abspath(model_dir)),
                        min_context=min_context, window=window)
    if corpus_path and os.path.isfile(corpus_path):
        result.corpus_chemin = os.path.abspath(corpus_path)
        result.corpus_octets = os.path.getsize(corpus_path)
        result.corpus_sha256 = hashlib.sha256(
            open(corpus_path, "rb").read()).hexdigest()[:24]
    else:
        result.corpus_chemin = "(corpus par defaut, integre)"
        result.corpus_octets = len(texte.encode("utf-8"))
        result.corpus_sha256 = hashlib.sha256(
            texte.encode("utf-8")).hexdigest()[:24]
    result.weights_bytes = model.nbytes
    # INDICATIF, jamais diviseur : voir la mise en garde de bench.py. Les
    # octets reellement alloues sont dans nbytes_detail.octets_stockage_uniques.
    # Le champ ci-dessus a une valeur PREVUE, tiree du manifeste : embedding au
    # dtype de chargement + somme des tenseurs quantifies. Elle a rendu « faux »
    # des son premier usage (15,9994 prevu contre 26,987 releve sur
    # Llama-2-7b-fp16pur). Le detail est joint au releve pour que le prochain
    # chargement nomme l'ecart au lieu de le reconstater.
    try:
        result.nbytes_detail = model.nbytes_detail()
    except Exception as e:                    # noqa: BLE001
        result.nbytes_detail = {"erreur": f"{type(e).__name__}: {e}"}
    n_params = loaded.spec.total_params
    result.bits_par_poids_en_memoire = ((result.weights_bytes * 8 / n_params)
                                       if n_params else 0.0)
    for entry in loaded.manifest.get("tensors", {}).values():
        f = entry.get("format", "?")
        result.formats[f] = result.formats.get(f, 0) + 1

    blocks_per_window = (window + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    total_nll = 0.0
    counted = 0
    # 22/09 (`verdict-ppl-31b-ab-v2`) : Gemma 4 31B rendait PPL 936-5 446 au
    # lieu de 10-20, identique avant et après le correctif d exil — aucune
    # fenêtre ne commençait par le BOS. Le gabarit de conversation le pose
    # (`{{ bos_token }}`), pas le post-traitement de tokenizer.json, et un
    # modèle entraîné avec BOS n a sans lui aucun puits d attention (même
    # effondrement que GLM sans `[gMASK]<sop>`, Sage § 10). Protocole : chaque
    # fenêtre = [BOS] + (window − 1) jetons du corpus quand le tokeniseur en
    # a un (llama-perplexity fait de même, add_bos par bloc) ; le BOS n est
    # jamais une cible, il est compté dans le contexte. Sans BOS (Qwen) : rien
    # ne change, au bit.
    bos = getattr(tokenizer, "bos_id", lambda: None)()
    result.bos = bos
    corps = window - 1 if bos is not None else window
    n_windows = max(1, (len(ids) - 1 + stride - 1) // stride)

    for w, start in enumerate(range(0, len(ids) - 1, stride)):
        chunk = ids[start:start + corps]
        if bos is not None:
            chunk = [bos] + chunk
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

        # On ne note que les positions que cette fenêtre expose pour la
        # première fois, afin qu'un jeton ne soit jamais compté deux fois avec
        # des quantités de contexte différentes.
        first_new = 0 if start == 0 else max(0, (window - stride) - 1)
        # Le logit d'indice i predit le jeton i+1 en ayant vu i+1 jetons.
        first_new = max(first_new, min_context)
        if first_new >= n - 1:
            break
        # La tête et la log-softmax par TRANCHES de positions (sage-p2-ppl-
        # instrument-file-7h-19-09) : les logits fp32 d'une fenêtre entière
        # (2 047 × 151 936 = 1,2 Gio) plus la déquant de la tête plus la
        # log-softmax faisaient un pic de 4 Gio que le service ne connaît
        # jamais (tête à n ≤ 12) — OOM sur le converti i8c à 30 Gio pris.
        # Les états cachés normalisés sont rendus une fois ; chaque tranche
        # passe par `_tete` + `_logits_finaux`, les MÊMES fonctions que
        # `forward`, donc les mêmes valeurs (tests/test_ppl_tranches.py).
        h = model(batch, return_hidden=True)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=h.device)
        pertes = _pertes_par_tranches(model, h, targets, first_new)
        nll = pertes.sum()
        total_nll += float(nll)
        counted += int(pertes.numel())
        del h
        if torch.cuda.is_available() and h_device_cuda(model):
            torch.cuda.empty_cache()
        # Le meme cout, reparti par quantite de contexte disponible : c'est ce
        # qui dit si un chiffre eleve vient du modele ou des premieres
        # positions. Sans ce detail, un corpus de 283 jetons et un corpus de
        # 100 000 rendent deux nombres qu'on croit comparables.
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
        bos = {r.bos for r in results}
        lines.append("  bos     : " + (", ".join(f"{r.model}={'aucun' if r.bos is None else r.bos}" for r in results)
                                       if len(bos) > 1 else
                                       ("aucun (le tokeniseur n en pose pas)" if bos == {None}
                                        else f"{next(iter(bos))} en tête de chaque fenêtre")))
        lines.append("")
    lines.append(f"  {'modele':<{width}}  {'ppl':>9}  {'bpp':>6}  {'taille':>10}  "
                 f"{'jetons':>8}")
    best = min(r.perplexity for r in results)
    for r in results:
        delta = "" if r.perplexity == best else f"  (+{100*(r.perplexity/best-1):.1f}%)"
        lines.append(f"  {r.model:<{width}}  {r.perplexity:9.3f}  "
                     f"{r.bits_par_poids_en_memoire:6.2f}  {r.weights_bytes/2**20:8.1f}Mio  "
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
