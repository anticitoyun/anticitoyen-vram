"""Hooks de fake-quant sur les 7 projections lineaires — reutilises entre le
test a sec (modele jouet, CPU, tests/test_hooks_activations_a4.py) et la
campagne reelle (Llama-2-7B, carte, outils/fake-quant-a4-llama2-7b.py).

Bead anticitoyen-vram-brd, etape 1. Nom en underscore (pas de tiret) pour
rester IMPORTABLE : outils/campagne-quota.py etc. sont des scripts qu'on ne
peut pas `import`, celui-ci doit l'etre par le test ET par le script de
campagne, comme outils/racine_modeles.py.
"""
import sys as _s
import pathlib as _p

_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))

import torch

from acvram.quant.fakequant_activation import (fake_quantize_e4m3_activation,
                                               fake_quantize_nvfp4_activation)
from acvram.quant.smoothquant import replier_smoothquant

# Les 7 projections que la MMA mxf4nvf4 exige en E2M1 pour ses DEUX
# operandes (bead anticitoyen-vram-brd) — les memes que celles nommees dans
# le manifeste (model.layers.N.self_attn.*_proj, model.layers.N.mlp.*_proj).
GENRES_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
GENRES_MLP = ("gate_proj", "up_proj", "down_proj")

REGIMES = {
    "a16": None,                              # temoin : aucun fake-quant
    "a4": fake_quantize_nvfp4_activation,     # E2M1 bloc 16, echelle UE4M3
    "a8": fake_quantize_e4m3_activation,      # E4M3 bloc 16 (repli mxf8f6f4)
}


def modules_a_hooker(model) -> dict:
    """Rend {chemin: module} pour chaque genre PRESENT sur chaque couche.

    Un genre absent n'est pas une erreur : `self_attn` manque sur les
    couches recurrentes (DecoderLayerGDN, sans attention standard),
    `k_proj`/`v_proj` peuvent etre partages (k_eq_v) et `gate_proj` manque
    sur MLP2 (architectures sans porte SwiGLU classique). Le dry-run compte
    ce qui a ete TROUVE, pour dire honnetement combien de modules seront
    reellement fake-quantifies plutot que de supposer une couverture de 7
    par couche partout.
    """
    trouves = {}
    for i, layer in enumerate(model.layers):
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        for genre in GENRES_ATTN:
            mod = getattr(attn, genre, None) if attn is not None else None
            if mod is not None:
                trouves[f"layers.{i}.self_attn.{genre}"] = mod
        for genre in GENRES_MLP:
            mod = getattr(mlp, genre, None) if mlp is not None else None
            if mod is not None:
                trouves[f"layers.{i}.mlp.{genre}"] = mod
    return trouves


def installer_hooks(model, regime: str):
    """Installe le fake-quant `regime` ('a16', 'a4' ou 'a8') sur toutes les
    projections trouvees. Rend (handles, modules) — `handle.remove()` pour
    chaque handle retire le hook ; `modules` est le dict de
    `modules_a_hooker`, pour verifier la couverture sans le relire."""
    if regime not in REGIMES:
        raise ValueError(f"regime {regime!r} inconnu ; attendu {sorted(REGIMES)}")
    fq = REGIMES[regime]
    trouves = modules_a_hooker(model)
    handles = []
    if fq is None:
        return handles, trouves

    def _hook(module, args):
        # args[0] est l'activation d'entree ; le reste (masques, positions,
        # KV cache...) traverse intact.
        return (fq(args[0]),) + tuple(args[1:])

    for mod in trouves.values():
        handles.append(mod.register_forward_pre_hook(_hook))
    return handles, trouves


def retirer_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


def installer_hooks_genres(model, genres: set, regime: str):
    """Comme `installer_hooks`, mais restreint aux GENRES nommes (par ex.
    {'down_proj'} pour isoler la contribution d'un seul genre — piste de
    chef 13/09 : si la degradation A4 vient surtout de down_proj, on
    peut le laisser en A8 et garder le reste en A4/A16."""
    if regime not in REGIMES:
        raise ValueError(f"regime {regime!r} inconnu ; attendu {sorted(REGIMES)}")
    fq = REGIMES[regime]
    trouves = {k: v for k, v in modules_a_hooker(model).items()
              if k.rsplit(".", 1)[-1] in genres}
    handles = []
    if fq is None:
        return handles, trouves

    def _hook(module, args):
        return (fq(args[0]),) + tuple(args[1:])

    for mod in trouves.values():
        handles.append(mod.register_forward_pre_hook(_hook))
    return handles, trouves


def _cle_hf(chemin_module: str) -> str:
    """'layers.3.mlp.down_proj' -> 'model.layers.3.mlp.down_proj.weight',
    la convention de nommage de `collect_activation_stats` (memes cles que
    le manifeste de conversion, verifie sur test_calibration.py)."""
    return f"model.{chemin_module}.weight"


class _PoigneeFonction:
    """Imite l'interface `.remove()` des handles PyTorch
    (`register_forward_pre_hook`), pour que `retirer_hooks` reste uniforme
    quel que soit le mecanisme d'installation en dessous."""

    def __init__(self, fn):
        self._fn = fn

    def remove(self):
        self._fn()


class _ExtensionSansAttribut:
    """Delegue tout SAUF un attribut nomme, cache pour la duree du hook.

    Sert a forcer le chemin de repli (gate et up separes, chacun passant
    par `_grouped`) sur un MoEBlock qui prendrait sinon le noyau fusionne
    `nvfp4_gemv_grouped_gateup` — lequel bypasse `_grouped` pour gate ET up
    et ne serait donc JAMAIS fake-quantifie."""

    def __init__(self, ext, nom_cache: str):
        object.__setattr__(self, "_ext", ext)
        object.__setattr__(self, "_nom_cache", nom_cache)

    def __getattr__(self, nom):
        if nom == self._nom_cache:
            raise AttributeError(nom)
        return getattr(self._ext, nom)


def installer_hooks_moe_experts(model, regime: str = "a4"):
    """Fake-quant sur l'ENTREE des GEMV groupes gate/up/down des MoEBlock.

    Les experts MoE ne passent PAS par `QuantLinear.forward()` (jamais
    appele : `_forward_grouped` appelle directement `self._grouped(x32,
    pile, ...)`, qui lance le noyau CUDA sur les poids EMPILES). Les hooks
    `torch.nn.Module` de ce fichier n'y voient donc rien — ce mecanisme
    monkeypatch `_grouped` sur chaque instance de MoEBlock trouvee, et
    masque temporairement le noyau fusionne gate+up (`_ExtensionSansAttribut`)
    pour garantir que gate ET up passent, eux aussi, par `_grouped` pendant
    la mesure — sinon "A4 sur gate+up+down" ne le serait qu'a moitie.

    Rend (handles, blocs) — `handles[0].remove()` restaure tout (l'extension
    CUDA globale ET chaque `_grouped` d'origine), comme les autres
    installateurs de ce fichier."""
    if regime not in REGIMES:
        raise ValueError(f"regime {regime!r} inconnu ; attendu {sorted(REGIMES)}")
    fq = REGIMES[regime]
    blocs = [m for m in model.modules() if type(m).__name__.startswith("MoEBlock")]
    if fq is None or not blocs:
        return [], blocs

    from acvram import kernels as _kernels

    ext_original = _kernels.get_extension

    def _get_extension_sans_fusion(*a, **kw):
        ext = ext_original(*a, **kw)
        return (_ExtensionSansAttribut(ext, "nvfp4_gemv_grouped_gateup")
                if ext is not None else ext)

    _kernels.get_extension = _get_extension_sans_fusion

    restaurations = [(bloc, bloc._grouped) for bloc in blocs]
    for bloc, original in restaurations:
        def _grouped_fq(x32, pile, expert_ids, token_ids, _orig=original):
            return _orig(fq(x32), pile, expert_ids, token_ids)
        bloc._grouped = _grouped_fq

    def _restaurer():
        _kernels.get_extension = ext_original
        for bloc, original in restaurations:
            bloc._grouped = original

    return [_PoigneeFonction(_restaurer)], blocs


def installer_hooks_smoothquant(model, act_stats: dict, alpha: float):
    """Replie l'echelle SmoothQuant dans chaque projection trouvee (poids
    NVFP4 REMPLACE en memoire, jamais sur disque) puis hooke l'activation :
    divise par la meme echelle avant le fake-quant E2M1.

    `act_stats` : {cle HF: ActStats}, typiquement le retour de
    `acvram.quant.collect.collect_activation_stats` sur le point de
    controle HF d'origine (PAS le dossier converti — les statistiques
    d'activation se relevent sur un forward en pleine precision).

    Un genre sans statistiques correspondantes est SAUTE, pas remplace par
    une echelle inventee — compte dans `manques`, publie par l'appelant."""
    trouves = modules_a_hooker(model)
    handles = []
    manques = []
    for chemin, mod in trouves.items():
        cle = _cle_hf(chemin)
        st = act_stats.get(cle)
        if st is None or st.max_abs is None:
            manques.append(chemin)
            continue
        x_absmax = st.max_abs.to(mod.qweight.qweight.device).to(torch.float32)
        qw2, s = replier_smoothquant(mod.qweight, x_absmax, alpha)
        mod.qweight = qw2.to(mod.qweight.qweight.device)
        s_dev = s.to(mod.qweight.qweight.device)

        def _hook(module, args, _s=s_dev):
            x = args[0] / _s
            return (fake_quantize_nvfp4_activation(x),) + tuple(args[1:])

        handles.append(mod.register_forward_pre_hook(_hook))
    return handles, trouves, manques
