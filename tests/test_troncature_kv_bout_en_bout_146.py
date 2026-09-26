"""Pièce 146 (1), de bout en bout : une requête qui épuise le budget KV en décodage, servie par un vrai serveur sur le
chemin PAR DÉFAUT (graphes CUDA + pipeline), se clôt avec finish_reason "length". Avant 146, le pipeline jetait la
sortie de fin : la requête HTTP ne se fermait jamais (délai du sous-processus dépassé → rouge)."""
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="graphes CUDA et pipeline : carte requise")


def _tokenizer(d):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    vocab = {f"tok{i}": i for i in range(1024)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(d, "tokenizer.json"))
    json.dump({"eos_token": "tok1", "bos_token": "tok2"}, open(os.path.join(d, "tokenizer_config.json"), "w"))


def test_requete_close_quand_le_kv_s_epuise_en_decodage(converted, tmp_path):
    d = str(tmp_path / "jouet")
    shutil.copytree(converted, d)
    _tokenizer(d)
    env = {**os.environ, "ACVRAM_PIPELINE": "1", "ACVRAM_KV_PLAN_OVERRIDE": "1"}
    script = os.path.join(os.path.dirname(__file__), "serveur_troncature_146.py")
    try:
        p = subprocess.run([sys.executable, script, d], capture_output=True, text=True, timeout=240, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("requête jamais close : la fin par budget KV épuisé n'a pas été livrée (délai 240 s)")
    lignes = [l for l in p.stdout.splitlines() if l.startswith("RESULTAT ")]
    assert lignes, f"rc={p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-3000:]}"
    r = json.loads(lignes[0][len("RESULTAT "):])
    assert r["graphes"] and r["pipeline"], f"chemin par défaut non servi : {r}"
    assert r["finish_reason"] == "length" and r["tronquees"] == 1, r
    assert r["completion_tokens"] < 200, r
