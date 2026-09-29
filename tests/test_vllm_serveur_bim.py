"""bim (29/09) : les 9 alias vLLM du parc mouraient au démarrage (22/22 depuis le 21/09, serveur.log).
(a) moteur laissé au choix de vLLM → FLASHINFER, dont le décodage xqa sm_120 exige nvcc (absent du service) ;
(b) MLA (GLM-4.7-Flash) : TRITON_MLA avec KV fp8 = 102 400 o de mémoire partagée > 101 376 sur la 5090.
Le lanceur choisit donc TRITON_ATTN par défaut et un KV bf16 pour les MLA. À sec : `vllm-serveur --afficher`."""
import json
import os
import pathlib
import subprocess

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent
LANCEUR = RACINE / "parc" / "bin" / "vllm-serveur"


def _afficher(tmp_path, config, **env):
    m = tmp_path / "modele"
    m.mkdir()
    (m / "config.json").write_text(json.dumps(config))
    e = {k: v for k, v in os.environ.items() if k not in ("VLLM_ATTENTION_BACKEND", "KVDT")}
    e.update(env, HOME=str(tmp_path))                     # pas de ~/.config/ia-secrets.env réel
    r = subprocess.run(["bash", str(LANCEUR), "--afficher", str(m)], capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[-1]


DENSE = {"hidden_size": 5120, "num_attention_heads": 32, "head_dim": 128}
MLA = {"hidden_size": 2048, "num_attention_heads": 20, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
VL = {"text_config": {"hidden_size": 2048, "num_attention_heads": 32, "head_dim": 128},
      "vision_config": {"hidden_size": 1152, "num_heads": 16}}              # têtes de 72


def test_dense_triton_attn_par_defaut(tmp_path):
    # Casse si le défaut redevient « laisser vLLM choisir » (→ FLASHINFER, mort au profilage sans nvcc)
    assert _afficher(tmp_path, DENSE).startswith("attention_backend=TRITON_ATTN kv_cache_dtype=fp8")


def test_vision_tetes_72_ne_retombe_pas_sur_flashinfer(tmp_path):
    assert _afficher(tmp_path, VL).startswith("attention_backend=TRITON_ATTN")


def test_mla_kv_bf16_et_moteur_mla(tmp_path):
    # Casse si le KV fp8 revient sur un MLA (102 400 o de mémoire partagée) ou si TRITON_ATTN est imposé à un MLA
    assert _afficher(tmp_path, MLA).startswith("attention_backend=auto kv_cache_dtype=auto")


@pytest.mark.parametrize("env,attendu", [({"KVDT": "fp8"}, "kv_cache_dtype=fp8"),
                                         ({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}, "attention_backend=FLASH_ATTN")])
def test_le_choix_de_l_appelant_est_respecte(tmp_path, env, attendu):
    assert attendu in _afficher(tmp_path, MLA if "KVDT" in env else DENSE, **env)
