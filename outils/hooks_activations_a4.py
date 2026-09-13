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

from acvram.quant.fakequant_activation import (fake_quantize_e4m3_activation,
                                               fake_quantize_nvfp4_activation)

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
