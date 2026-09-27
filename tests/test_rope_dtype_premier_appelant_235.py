"""Pièce 235 (poste2, à sec) : `quant/collect.py:257` construisait son `RotaryEmbedding` sans passer
`dtype` -- défaut `torch.float32` (`engine/layers.py:657`), silencieusement différent du dtype réel des
activations de calibration. `self._dtype` est lu par `tables32()` (`layers.py:777`, noyau RoPE fusionné,
tenté avant le repli PyTorch dans `attention.py`).

Cause racine partagée avec la 213 (service, `_ensure`, layers.py:721) : poste5 porte le correctif unique
(origin/poste5-213b) qui fixe `self._dtype` pour de bon et fait convertir au dtype demandé à la lecture,
plutôt que de reconstruire le cache à chaque appel alternant (mon 1er essai reconstruisait sur tout
changement de dtype -- rejeté par chef/poste5 : lève sous capture de graphe, layers.py:725, si
`tables32` et `forward` alternent par couche). La 235 se limite donc à ce que 213b exige de tout
appelant : passer `dtype` (et `device`) à la construction, comme `loader.py` le fait déjà pour gemma4/MLA.
Ce test est repris par poste5 pour couvrir aussi le côté service (213b)."""
import ast
from pathlib import Path

_COLLECT_PY = Path(__file__).resolve().parents[1] / "acvram" / "quant" / "collect.py"


def _appels_rotary_embedding() -> list[ast.Call]:
    arbre = ast.parse(_COLLECT_PY.read_text())
    return [n for n in ast.walk(arbre)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "RotaryEmbedding"]


def test_collect_construit_rope_avec_dtype_explicite_235():
    """Falsificateur : chaque appel `RotaryEmbedding(...)` de `collect.py` doit porter au moins 6
    arguments positionnels (head_dim, max_position, base, scaling, device, dtype) -- pas seulement les
    4 premiers, qui laissaient `dtype` au défaut fp32 de la classe. Rouge avant la 235."""
    appels = _appels_rotary_embedding()
    assert appels, "aucun appel RotaryEmbedding trouvé dans collect.py -- gabarit du test obsolète"
    for appel in appels:
        assert len(appel.args) >= 6, (
            f"collect.py:{appel.lineno} — RotaryEmbedding construit avec {len(appel.args)} arguments "
            "positionnels seulement, dtype n'est pas passé explicitement (défaut fp32 de la classe, "
            "motif de la 235)")
