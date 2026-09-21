"""`outils/dequantiser-awq-bf16.py --test` : ma déquantification int4 g32 == compressed-tensors au bit (synthétique,
tenseur réel si l AWQ est sur disque, bout en bout sur une source factice). compressed-tensors n est pas dans le venv
acvram : le test se joue par le venv vLLM, sauté s il est absent."""
import os, subprocess, sys, pytest

HFPY = os.environ.get("HF_PYTHON", "/opt/ia/vLLM/.venv/bin/python")
AWQ = "/mnt/4TO_SATACMR_2022/Modeles/models_awq/Qwen3-VL-30B-A3B-abliterated-AWQ"
ICI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outils", "dequantiser-awq-bf16.py")


@pytest.mark.skipif(not os.path.exists(HFPY), reason="venv vLLM (compressed-tensors) absent")
def test_dequantisation_au_bit_contre_compressed_tensors():
    args = [HFPY, ICI, "--test"] + ([AWQ] if os.path.isdir(AWQ) else [])
    r = subprocess.run(args, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=600)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "test synthétique" in r.stdout and "test bout en bout" in r.stdout
