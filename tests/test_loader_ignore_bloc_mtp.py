"""Le loader ne doit pas exécuter le bloc MTP (`model.layers.{num_layers}.*`)
comme une couche de décodage supplémentaire — poste2 a trouvé que le
convertisseur écrivait ce bloc sous la même clé qu'une couche MoE
normale (revue/prediction-mtp-loader-15-09.md, chef/poste7 15/09) ; ce
test vérifie le LOADER indépendamment de ce correctif."""

import json

import torch
from safetensors.torch import save_file

from acvram.engine.config import load_model_spec
from acvram.engine.loader import load_model
from acvram.memory.tiering import PlannerOptions, auto_plan
from acvram.quant.convert import ConversionOptions, convert_checkpoint


def _checkpoint_avec_bloc_fantome(tmp_path):
    """Point de contrôle minuscule, forme llama, DEUX couches déclarées
    (`num_hidden_layers=2`) mais avec un troisième jeu de tenseurs
    `model.layers.2.*` présent sur le disque — simule ce que produit le
    convertisseur actuel de GLM-4.7-Flash avant le correctif de poste2
    (le bloc MTP écrit comme une couche normale)."""
    torch.manual_seed(20260915)
    H, I, NH, NKV, V = 64, 172, 4, 2, 256
    d = tmp_path
    json.dump({
        "architectures": ["LlamaForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": 2,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
    }, open(d / "config.json", "w"))

    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * .02}
    for i in range(3):     # 0, 1 : couches reelles ; 2 : bloc fantome (MTP mal etiquete)
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
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * .02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


def test_le_bloc_fantome_layers_2_est_ignore(tmp_path, target_rig):
    src = _checkpoint_avec_bloc_fantome(tmp_path)
    spec = load_model_spec(src, "fantome")
    assert spec.num_layers == 2, "le config.json ne declare que 2 couches"

    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512))
    out = str(tmp_path / "acvram")
    # awq=False, quant_device="cpu" : ce test verifie le compte de couches
    # du loader, pas la qualite de quantification -- la recherche AWQ par
    # defaut (device="cuda:0", 20 valeurs de grille) rendait ce test lent
    # (mesure : suite complete 60s -> 380s pour deux conversions).
    convert_checkpoint(src, plan, ConversionOptions(
        out_dir=out, awq=False, quant_device="cpu"), spec=spec)

    loaded = load_model(out, dtype=torch.float32, device_override="cpu")
    assert loaded.spec.num_layers == 2
    assert len(loaded.model.layers) == 2, (
        f"le loader a construit {len(loaded.model.layers)} couches -- "
        f"le bloc fantome model.layers.2.* a ete execute comme une 3e couche")


def test_le_manifeste_converti_ne_declare_pas_le_bloc_fantome(tmp_path, target_rig):
    """Meme sans aller jusqu'au chargement : le manifeste lui-meme doit
    dire 2 couches, pas 3 -- c'est CE nombre (spec.to_dict()['num_layers'])
    que load_model relit, pas un recomptage des tenseurs presents."""
    src = _checkpoint_avec_bloc_fantome(tmp_path)
    spec = load_model_spec(src, "fantome")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512))
    out = str(tmp_path / "acvram2")
    convert_checkpoint(src, plan, ConversionOptions(
        out_dir=out, awq=False, quant_device="cpu"), spec=spec)

    with open(f"{out}/acvram_manifest.json") as fh:
        manifest = json.load(fh)
    assert manifest["model"]["num_layers"] == 2

    # Le bogue de conversion (poste2, 15/09) est REPRODUIT ici : le
    # convertisseur ecrit bien les tenseurs de la 3e couche fantome sur
    # le disque (orphelins, jamais relus) -- ce test n'est pas vide.
    cles_fantome = [k for k in manifest["tensors"] if k.startswith("model.layers.2.")]
    assert len(cles_fantome) > 0, "le convertisseur ne reproduit plus le bogue -- ce test ne prouve plus rien"
