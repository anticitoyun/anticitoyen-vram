import json
import os

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Un point de contrôle de 4 couches, de forme llama, assez petit pour être
    converti en une seconde.

    Avec une graine fixe. Sans elle, toute la suite est non déterministe, et les
    tests qui comparent des formats de quantification sur ce modèle sont assez
    proches du plancher de bruit pour qu'un tirage différent les fasse basculer
    — c'est exactement ainsi qu'une exécution verte dans un répertoire de travail
    est devenue rouge dans une extraction propre.
    """
    from safetensors.torch import save_file

    torch.manual_seed(20260830)
    H, I, L, NH, NKV, V = 256, 688, 4, 8, 2, 1024
    d = tmp_path_factory.mktemp("hf")
    json.dump({
        "architectures": ["LlamaForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
    }, open(d / "config.json", "w"))

    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * 0.02}
    for i in range(L):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(NH * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.k_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.v_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * hd, dtype=torch.bfloat16) * .02
        sd[p + "mlp.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16) * .02
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture(scope="session")
def target_rig():
    from acvram.hardware.profiles import load_profile
    return load_profile("rig-14900k-5090-3080ti")


@pytest.fixture(scope="session")
def converted(tiny_checkpoint, target_rig, tmp_path_factory):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("acvram"))
    convert_checkpoint(tiny_checkpoint, plan,
                       ConversionOptions(out_dir=out), spec=spec)
    return out


@pytest.fixture(scope="session")
def tiny_checkpoint_qknorm(tmp_path_factory):
    """Le même modèle minuscule, mais de forme Qwen3 : une RMSNorm par tête sur
    Q et sur K, appliquée avant la RoPE.

    Les poids de ces normalisations sont volontairement tirés au hasard autour
    de 1 : à un, la normalisation reste visible mais un test qui les ignorerait
    resterait proche, ce qui masquerait l'erreur.
    """
    from safetensors.torch import save_file

    torch.manual_seed(20260831)
    H, I, L, NH, NKV, V = 256, 688, 4, 8, 2, 1024
    hd = H // NH
    d = tmp_path_factory.mktemp("hf_qknorm")
    json.dump({
        "architectures": ["Qwen3ForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 2048, "head_dim": hd,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "torch_dtype": "bfloat16",
        "model_type": "qwen3",
    }, open(d / "config.json", "w"))

    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * 0.02}
    for i in range(L):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(NH * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.k_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.v_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * hd, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.q_norm.weight"] = (1 + torch.randn(hd) * .1).to(torch.bfloat16)
        sd[p + "self_attn.k_norm.weight"] = (1 + torch.randn(hd) * .1).to(torch.bfloat16)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16) * .02
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture(scope="session")
def converted_qknorm(tiny_checkpoint_qknorm, target_rig, tmp_path_factory):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint_qknorm, "tiny-qknorm")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("acvram_qknorm"))
    convert_checkpoint(tiny_checkpoint_qknorm, plan,
                       ConversionOptions(out_dir=out), spec=spec)
    return out


@pytest.fixture(autouse=True)
def _carte_disponible(request):
    """Une carte saturée par une mesure en cours rend « ignoré », pas « échec ».

    Le 8/09, sept tests graphes/MoE ont échoué pendant qu'une perplexité
    occupait la 5090 : graphes non capturés (OOM silencieux), pile d'experts
    refusée faute de place. Trois sessions se partagent la machine ; une carte
    occupée est l'état normal. Un test qui exige de la VRAM la vérifie avant
    de courir — le seuil couvre le modèle-jouet, ses graphes et la marge de
    capture.
    """
    if not any(m.name == "gpu_requis" for m in request.node.iter_markers()):
        return
    if not torch.cuda.is_available():
        pytest.skip("pas de carte CUDA")
    libre, _ = torch.cuda.mem_get_info()
    if libre < 4 << 30:
        pytest.skip(f"carte occupée : {libre / (1 << 30):.1f} Gio libres, "
                    "4 requis — mesure en cours ailleurs ?")


# Au-dela de cette occupation, un debit mesure n'est plus celui du code teste.
OCCUPATION_MAX = 50


def occupation_gpu(index: "int | None" = None) -> "int | None":
    """Occupation de la carte courante en pour cent, ou None si indisponible.

    Separee de la fixture pour etre eprouvee : un garde-fou qu'on ne peut pas
    faire mordre a la demande n'est pas un garde-fou.
    """
    import subprocess
    if index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    try:
        lignes = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")
        return int(lignes[index].strip())
    except Exception:
        return None                 # pas de nvidia-smi : on laisse courir


@pytest.fixture(autouse=True)
def _carte_au_repos(request):
    """Un test de DEBIT exige une carte au repos, pas seulement de la place.

    La garde ci-dessus regarde la memoire libre. Un debit ne s'effondre pas
    par manque de place mais par partage des multiprocesseurs : le 8/09,
    `test_le_gemv_nvfp4_ne_part_pas_en_emulation` a rendu 57 Go/s au lieu de
    300 avec huit gigaoctets libres et une conversion voisine a cent pour cent
    d'occupation. Le test annoncait « reparti en emulation logicielle » — un
    diagnostic faux tire d'une mesure vraie, exactement ce que cette journee a
    produit trois fois ailleurs.

    On mesure donc l'instrument avant la mesure : si la carte travaille deja
    pour quelqu'un d'autre, le chiffre ne dira rien du code teste.
    """
    if not any(m.name == "debit_requis" for m in request.node.iter_markers()):
        return
    if not torch.cuda.is_available():
        pytest.skip("pas de carte CUDA")
    occupation = occupation_gpu()
    if occupation is not None and occupation > OCCUPATION_MAX:
        pytest.skip(f"carte a {occupation} % d'occupation — un debit mesure "
                    f"pendant le travail d'autrui ne dit rien du code teste")


# Seuil de KL entre deux jeux de logits : nos propres mesures du 10/09
# donnent 1,9e-4 a 7,0e-3 nats pour un changement d'ordre d'accumulation
# legitime (noyau fusionne vs torch, decodage batche vs par sequence), et
# 0,8 a 5,0 nats pour un vrai defaut de traitement par lot. `torch.equal`
# ne discriminait plus rien depuis que graphe et eager peuvent emprunter
# des chemins numeriquement differents mais tous deux corrects (fusion de
# noyau, batching) : le KL, qui pese chaque composante par sa probabilite
# plutot que de comparer bit a bit, reste discriminant entre les deux
# regimes. Le seuil est pose a 1e-2, un ordre de grandeur au-dessus du
# plus grand ecart legitime mesure et deux ordres en dessous du plus petit
# defaut reel observe.
KL_LOGITS_MAX = 1e-2


def assert_logits_proches(a: torch.Tensor, b: torch.Tensor, msg: str = "",
                          seuil: float = KL_LOGITS_MAX) -> None:
    """Remplace un ``torch.equal`` devenu aveugle : compare deux jeux de
    logits par KL plutot que bit a bit. Voir ``KL_LOGITS_MAX`` pour l'origine
    du seuil."""
    pa = F.log_softmax(a.double().reshape(-1, a.shape[-1]), dim=-1)
    pb = F.softmax(b.double().reshape(-1, b.shape[-1]), dim=-1)
    kl = F.kl_div(pa, pb, reduction="batchmean").item()
    assert kl < seuil, f"{msg} (KL={kl:.4e}, seuil={seuil:.4e})"
