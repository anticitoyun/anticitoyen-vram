"""Pièce 213 : les tables RoPE ne dépendent plus du PREMIER appelant.

Avant : `RotaryEmbedding._ensure` bâtissait la table au dtype de l'appel qui la créait et ne la reconstruisait que sur
la longueur. Le RoPE principal (loader.py, construit sans dtype → fp32) était rempli d'abord par `tables32` (préfill
fusionné) en fp32 NON arrondi, puis reconstruit en bf16 par `reserver` : le 1er lot d'un processus tournait sur
d'autres tables que tous les suivants (contam2 de la 213, mixte b=8 : Y seul ≠ Y après n'importe quel lot, 700-2 j15 /
700-5 j3 ; défaut présent en 0.6.38).
Processeur : (1) table identique quel que soit l ordre des appelants (module fp32 appelé en bf16, la configuration
fautive), (2) dtype de retour = celui de l appelant, (3) reconstruction sur changement de dtype, (4) chemin processeur
= chemin carte simulé, (5) tout constructeur du moteur passe un dtype. Carte + modèle : (6) le même lot joué deux fois
dans UN moteur rend les mêmes jetons au bit. Bras cassant (prise de la 213b) : ce fichier sur main → rouges."""
import ast
import os
import pathlib
import sys

import pytest
import torch

import acvram
from acvram.engine.layers import RotaryEmbedding

POS = torch.tensor([0, 1, 7, 300, 1152])
POS8 = torch.arange(8, dtype=torch.long)


def _rope(dtype=torch.bfloat16):
    return RotaryEmbedding(128, 32768, 1e6, None, None, dtype)


def test_table_ne_depend_pas_du_premier_appelant():
    # module sans dtype (fp32 par défaut) : la configuration fautive du RoPE principal avant la 213b, appelé en bf16
    a, b = RotaryEmbedding(128, 32768, 1e6), RotaryEmbedding(128, 32768, 1e6)
    a.tables32(1153, torch.device("cpu"))                         # préfill fusionné d'abord (l'ordre du 1er lot)
    ca, sa = a(POS, torch.device("cpu"), torch.bfloat16, max_pos=1153)
    cb, sb = b(POS, torch.device("cpu"), torch.bfloat16, max_pos=1153)   # chemin eager d'abord
    b.tables32(1153, torch.device("cpu"))
    assert torch.equal(a._cos32, b._cos32) and torch.equal(a._sin32, b._sin32)
    assert torch.equal(ca, cb) and torch.equal(sa, sb)
    # les tables du noyau fusionné sont celles du chemin eager, remontées en fp32 (arrondi bf16 compris)
    assert torch.equal(a._cos32, a._cos.float())


def test_dtype_de_retour_est_celui_de_l_appelant():
    r = RotaryEmbedding(64, 4096)                                  # fp32 par défaut
    r.tables32(1153, torch.device("cpu"))
    c, s = r(POS, torch.device("cpu"), torch.bfloat16, max_pos=1153)
    assert c.dtype == s.dtype == torch.bfloat16
    ref = RotaryEmbedding(64, 4096)
    c32, _ = ref(POS, torch.device("cpu"), torch.float32, max_pos=1153)
    assert torch.equal(c, c32.to(torch.bfloat16))


def test_reconstruction_sur_changement_de_dtype():
    r = _rope(torch.float32)
    r.tables32(1153, torch.device("cpu"))
    avant = r._cos32.clone()
    r._dtype = torch.bfloat16
    r.tables32(1153, torch.device("cpu"))
    assert r._cos.dtype == torch.bfloat16
    assert torch.equal(r._cos32, r._cos.float()) and not torch.equal(r._cos32, avant)


def test_chemin_processeur_egal_chemin_carte_simule():
    """RoPE bf16 (le service ; collect.py après la 235) : chemin processeur (repli forward d'abord) et chemin carte
    simulé (tables32 de rope_fusee d'abord) rendent les mêmes tables et les mêmes cos/sin."""
    proc, carte = RotaryEmbedding(64, 128, 10000.0, None, dtype=torch.bfloat16), RotaryEmbedding(64, 128, 10000.0, None, dtype=torch.bfloat16)
    cp, sp = proc(POS8, torch.device("cpu"), torch.bfloat16, max_pos=64)
    proc.tables32(64, torch.device("cpu"))
    carte.tables32(64, torch.device("cpu"))
    cc, sc = carte(POS8, torch.device("cpu"), torch.bfloat16, max_pos=64)
    assert torch.equal(proc._cos32, carte._cos32) and torch.equal(proc._sin32, carte._sin32)
    assert torch.equal(cp, cc) and torch.equal(sp, sc)


def test_tout_constructeur_du_moteur_passe_un_dtype():
    """loader.py:313 (RoPE principal du service) construisait sans dtype → fp32 par défaut. Portée : engine/ (le
    service) ; quant/collect.py relève de la 235 (poste2)."""
    racine = pathlib.Path(acvram.__file__).resolve().parent / "engine"
    sans = []
    for f in racine.rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "RotaryEmbedding":
                if len(n.args) < 6 and not any(k.arg == "dtype" for k in n.keywords):
                    sans.append(f"{f.relative_to(racine)}:{n.lineno}")
    assert not sans, f"RotaryEmbedding construit sans dtype (fp32 par défaut) : {sans}"


# ---- (6) le même lot deux fois dans un moteur : au bit (cellule contam de la 213) ----
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils"))


def _modele():
    from racine_modeles import alias
    return os.environ.get("ACVRAM_MODELE_213") or alias("Qwen3.8-27B-unsloth-mixte-i8c")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
def test_meme_lot_deux_fois_au_bit():
    m = _modele()
    if not os.path.exists(os.path.join(m, "acvram_manifest.json")):
        pytest.skip(f"modèle absent : {m}")
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    B, CTX, N = 8, 1024, 120                  # la cellule contam de la 213, à l identique

    def invite(k, n=128):
        return [(1000 + k * 7919 + i * 13) % 150000 + 10 for i in range(n)]

    def lot(e):
        e._eos = set()
        rid, j = {}, {}
        for k in range(B):
            s = e.add_request(invite(700 + k), SamplingParams(temperature=0.0, max_tokens=N - 11 * k),
                              request_id=f"y{k}")
            rid[s.id] = f"y{k}"; j[f"y{k}"] = []
        for _ in range(N + 40):
            for o in e.step():
                j[rid[o.sequence_id]].extend(o.token_ids)
            if not e.running and not e.waiting:
                break
        return j

    loaded = load_model(m, dtype=torch.bfloat16, max_model_len=CTX, max_concurrent_seqs=B)
    e = Engine(loaded, None, max_batch_size=B, max_model_len=CTX)
    premier, second = lot(e), lot(e)
    assert premier == second, {k: next(i for i, (x, y) in enumerate(zip(premier[k], second[k])) if x != y)
                               for k in premier if premier[k] != second[k]}
