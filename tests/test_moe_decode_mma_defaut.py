"""Cliquet : le décodage MoE par la MMA groupée est le DÉFAUT depuis le 15/09
(Sage, revue/sage-moe-mma-qualite-15-09.md § 2 : PPL 0,9995, −13,2 % de
J/jeton en mode moyen). Ce test casse si le défaut revient à 0 ; le témoin
reste ACVRAM_MOE_DECODE_MMA=0 posé explicitement (bancs, A/B)."""
import os
import subprocess
import sys

CODE = "import acvram.engine.model as m; print(int(m._MOE_DECODE_MMA))"


def _valeur(env_sup: dict) -> str:
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_MOE_DECODE_MMA"}
    env.update(env_sup)
    return subprocess.run([sys.executable, "-c", CODE], env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_le_defaut_est_la_mma():
    assert _valeur({}) == "1"


def test_le_temoin_zero_est_respecte():
    assert _valeur({"ACVRAM_MOE_DECODE_MMA": "0"}) == "0"
