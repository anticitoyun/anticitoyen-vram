"""Pièce 235 (poste2, à sec), repris dans le correctif unique 213b (arbitrage chef 26/09) : la table RoPE ne dépend
plus du premier appelant. Côté calibration, `quant/collect.py` construisait son `RotaryEmbedding` sans dtype (fp32) ;
`rope_fusee` (attention.py, tenté AVANT le repli `forward`) gelait la table en fp32 sur carte, le repli la gelait en
bf16 sur processeur : deux calibrations différentes selon le chemin. Désormais la table est au dtype FIXE du module,
et tout constructeur du dépôt passe le dtype du service.
Chaque test casse si la faute revient : table au dtype du 1er appelant (1, 3), dtype de retour gelé (2), un
constructeur sans dtype (4)."""
import ast
import pathlib

import torch

import acvram
from acvram.engine.layers import RotaryEmbedding

CPU = torch.device("cpu")
POS = torch.arange(8, dtype=torch.long)


def _rope(dtype=torch.float32):
    return RotaryEmbedding(64, 128, 10000.0, None, dtype=dtype)


def test_tables32_apres_forward_bf16_rend_un_calcul_fp32_frais_235():
    """Module fp32 : un 1er appel bf16 (repli forward) ne doit plus arrondir la table des appelants suivants."""
    rope = _rope()
    cos_bf16, _ = rope(POS, CPU, torch.bfloat16, max_pos=64)
    assert cos_bf16.dtype == torch.bfloat16
    cos32, sin32 = rope.tables32(64, CPU)
    ref = _rope()
    ref._ensure(64, CPU, torch.float32)
    assert torch.equal(cos32, ref._cos) and torch.equal(sin32, ref._sin), \
        "tables32() a rendu le cache bf16 du 1er appelant remonté en fp32 (motif de la 213)"


def test_forward_apres_tables32_rend_bien_le_dtype_demande_235():
    rope = _rope()
    rope.tables32(64, CPU)
    cos_bf16, sin_bf16 = rope(POS, CPU, torch.bfloat16, max_pos=64)
    assert cos_bf16.dtype == sin_bf16.dtype == torch.bfloat16, \
        f"forward(dtype=bf16) a rendu {cos_bf16.dtype} : le cache est resté au dtype du 1er appelant"


def test_calibration_chemin_processeur_egal_chemin_carte_simule_235():
    """Le RoPE tel que collect.py le construit (dtype de la calibration, bf16) : chemin processeur (repli forward
    d'abord) et chemin carte simulé (tables32 de rope_fusee d'abord) rendent les mêmes tables et les mêmes cos/sin."""
    proc, carte = _rope(torch.bfloat16), _rope(torch.bfloat16)
    cp, sp = proc(POS, CPU, torch.bfloat16, max_pos=64)
    proc.tables32(64, CPU)
    carte.tables32(64, CPU)
    cc, sc = carte(POS, CPU, torch.bfloat16, max_pos=64)
    assert torch.equal(proc._cos32, carte._cos32) and torch.equal(proc._sin32, carte._sin32)
    assert torch.equal(cp, cc) and torch.equal(sp, sc)


def test_tout_constructeur_du_depot_passe_un_dtype_235():
    """loader.py (RoPE principal du service) et quant/collect.py construisaient sans dtype → fp32 par défaut."""
    racine = pathlib.Path(acvram.__file__).resolve().parent
    sans = []
    for f in racine.rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "RotaryEmbedding":
                if len(n.args) < 6 and not any(k.arg == "dtype" for k in n.keywords):
                    sans.append(f"{f.relative_to(racine)}:{n.lineno}")
    assert not sans, f"RotaryEmbedding construit sans dtype (fp32 par défaut) : {sans}"
