#!/usr/bin/env python
"""Déquantifie UNE FOIS sur disque un checkpoint AWQ compressed-tensors (pack-quantized int4, groupes, symétrique) en
safetensors bf16 lisibles par `transformers` sans compressed-tensors ni offload — la référence P3 (3) « AWQ déquantifié ».

Pourquoi sur disque : `CompressedTensorsConfig(run_compressed=False)` avec `device_map=auto` déballe sur un tenseur `meta`
dès qu'une couche est exilée en RAM (verdict-p3-3-30b-21-09) ; on déquantifie donc tenseur par tenseur, à sec, sans
instancier le modèle, et la référence standard (`references-transformers.py`, chemin prouvé sur le 2B) charge le bf16.

    python outils/dequantiser-awq-bf16.py <source AWQ> <destination>      # venv avec safetensors + torch ; ≈ 60 Go écrits
    python outils/dequantiser-awq-bf16.py --test [<source AWQ>]           # test cassant : ma déquantification == compressed-tensors

Noms de tenseurs conservés 1:1 (`X.weight_packed` + `X.weight_scale` + `X.weight_shape` → `X.weight`), `config.json` sans
`quantization_config`, fichiers du tokenizer et du processeur copiés, `acvram_source.json` = provenance (source, sha256 des
shards, méthode). PAS d'`acvram_manifest.json` : un dossier qui en porte un est un alias acvram attendu au menu
(tests/test_menus.py:211-222) et lu par le chargeur — celui-ci est une source brute bf16, témoin hors menu.
"""
import os, sys, json, glob, shutil, hashlib, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

COPIES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json",
          "preprocessor_config.json", "video_preprocessor_config.json")
TAILLE_SHARD = int(os.environ.get("TAILLE_SHARD", 5 * 1024 ** 3))


def dequantiser(packed: torch.Tensor, scale: torch.Tensor, shape) -> torch.Tensor:
    """int32 [R, ceil(C/8)] (8 nibbles par mot, le premier dans les bits de poids faible), non signé décalé de 8 ;
    scale [R, C/g] ; produit dans le dtype de scale (bf16), comme `_dequantize` de compressed-tensors."""
    R, C = int(shape[0]), int(shape[1])
    assert packed.dtype == torch.int32 and packed.shape[0] == R, (packed.dtype, tuple(packed.shape), (R, C))
    q = torch.empty((R, packed.shape[1] * 8), dtype=torch.int32)
    for i in range(8):
        q[:, i::8] = (packed >> (4 * i)) & 0xF
    q = q[:, :C] - 8
    g = C // scale.shape[1]
    assert g * scale.shape[1] == C, (C, tuple(scale.shape))
    w = q.to(scale.dtype).view(R, scale.shape[1], g) * scale.unsqueeze(-1)
    return w.reshape(R, C).contiguous()


def reference(packed, scale, shape, num_bits=4, group_size=32):
    """La déquantification de compressed-tensors elle-même (celle que transformers applique avec run_compressed=False)."""
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
        num_bits=num_bits, type="int", symmetric=True, strategy="group", group_size=group_size))
    return PackedQuantizationCompressor.decompress(
        {"weight_packed": packed, "weight_scale": scale, "weight_shape": torch.tensor(list(shape))}, scheme)["weight"]


def sha256_fichier(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 24), b""):
            h.update(bloc)
    return h.hexdigest()


def test(source=None):
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (64, 96), dtype=torch.int8)                       # 3 groupes de 32, 12 mots int32 par ligne
    scale = (torch.rand(64, 3) * 0.05 + 0.001).to(torch.bfloat16)
    packed = pack_to_int32(q, 4)
    a, b = dequantiser(packed, scale, (64, 96)), reference(packed, scale, (64, 96))
    assert a.dtype == b.dtype == torch.bfloat16 and torch.equal(a, b), "synthétique : divergence avec compressed-tensors"
    print("test synthétique : 64×96 au bit == compressed-tensors")
    if source:
        idx = json.load(open(os.path.join(source, "model.safetensors.index.json")))["weight_map"]
        k = sorted(k for k in idx if k.endswith(".weight_packed"))[0]; p = k[: -len("weight_packed")]
        lire = lambda n: safe_open(os.path.join(source, idx[n]), "pt").get_tensor(n)
        packed, scale, shape = lire(k), lire(p + "weight_scale"), lire(p + "weight_shape")
        a, b = dequantiser(packed, scale, shape), reference(packed, scale, shape)
        assert torch.equal(a, b), "%s : divergence avec compressed-tensors" % k
        print("test réel : %s %s au bit == compressed-tensors" % (k, tuple(a.shape)))
    # bout en bout sur une source factice à deux shards (scale dans l autre shard que packed, un tenseur clair, config)
    import tempfile
    d = tempfile.mkdtemp(); src, dst = os.path.join(d, "src"), os.path.join(d, "dst"); os.makedirs(src)
    scale2 = (torch.rand(64, 3) * 0.05 + 0.001).to(torch.bfloat16); clair = torch.randn(8, 8).to(torch.bfloat16)
    save_file({"m.a.weight_packed": packed, "m.a.weight_shape": torch.tensor([64, 96]), "n.weight": clair}, os.path.join(src, "s1.safetensors"))
    save_file({"m.a.weight_scale": scale2}, os.path.join(src, "s2.safetensors"))
    json.dump({"weight_map": {"m.a.weight_packed": "s1.safetensors", "m.a.weight_shape": "s1.safetensors", "n.weight": "s1.safetensors",
                              "m.a.weight_scale": "s2.safetensors"}}, open(os.path.join(src, "model.safetensors.index.json"), "w"))
    json.dump({"x": 1, "quantization_config": {"format": "pack-quantized", "config_groups": {"g": {"weights": {
        "type": "int", "num_bits": 4, "symmetric": True, "strategy": "group", "group_size": 32}}}}}, open(os.path.join(src, "config.json"), "w"))
    os.environ["MIN_LIBRE_GO"] = "0"; assert convertir(src, dst) == 0
    with safe_open(os.path.join(dst, "model-00001.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["m.a.weight", "n.weight"], list(f.keys())
        assert torch.equal(f.get_tensor("m.a.weight"), reference(packed, scale2, (64, 96))) and torch.equal(f.get_tensor("n.weight"), clair)
    cfg = json.load(open(os.path.join(dst, "config.json"))); assert "quantization_config" not in cfg and cfg["x"] == 1
    assert not os.path.exists(os.path.join(dst, "acvram_manifest.json")) and os.path.exists(os.path.join(dst, "acvram_source.json"))
    shutil.rmtree(d); print("test bout en bout : source factice 2 shards → bf16 au bit, config sans quantization_config, pas de manifeste")
    return 0


def convertir(source, dest):
    cfg = json.load(open(os.path.join(source, "config.json")))
    qc = cfg.pop("quantization_config")
    grp = list(qc["config_groups"].values())[0]["weights"]
    assert qc["format"] == "pack-quantized" and grp["type"] == "int" and grp["num_bits"] == 4 and grp["symmetric"] \
        and grp["strategy"] == "group", grp                                    # seul schéma couvert par `dequantiser`
    idx = json.load(open(os.path.join(source, "model.safetensors.index.json")))["weight_map"]
    fichiers = sorted(set(idx.values()))
    libre = shutil.disk_usage(os.path.dirname(os.path.abspath(dest)) if not os.path.isdir(dest) else dest).free
    min_go = float(os.environ.get("MIN_LIBRE_GO", 70))                              # ≈ 60 Go écrits ; 0 pour le test
    if libre < min_go * 1024 ** 3:
        print("ECHEC : %.0f Go libres < %g Go sous %s" % (libre / 1e9, min_go, dest)); return 3
    os.makedirs(dest, exist_ok=True)
    t0 = time.time(); shard, taille, n_shard, carte, n_deq, n_copie = {}, 0, 0, {}, 0, 0
    sortie = []

    def flush():
        nonlocal shard, taille, n_shard
        if not shard: return
        n_shard += 1; nom = "model-%05d.safetensors" % n_shard
        save_file(shard, os.path.join(dest, nom), metadata={"format": "pt"})
        for k in shard: carte[k] = nom
        sortie.append(nom); print("[shard] %s %d tenseurs %.2f Go %.0f s" % (nom, len(shard), taille / 1e9, time.time() - t0), flush=True)
        shard, taille = {}, 0

    for fichier in fichiers:
        with safe_open(os.path.join(source, fichier), "pt") as f:
            for k in sorted(f.keys()):
                if k.endswith(".weight_scale") or k.endswith(".weight_shape"):
                    continue                                                   # consommés avec leur weight_packed
                if k.endswith(".weight_packed"):
                    p = k[: -len("weight_packed")]
                    lire = lambda n: (f if idx[n] == fichier else safe_open(os.path.join(source, idx[n]), "pt")).get_tensor(n)
                    t = dequantiser(f.get_tensor(k), lire(p + "weight_scale"), lire(p + "weight_shape")); k = p + "weight"; n_deq += 1
                else:
                    t = f.get_tensor(k); n_copie += 1
                shard[k] = t; taille += t.numel() * t.element_size()
                if taille >= TAILLE_SHARD: flush()
    flush()
    total = sum(os.path.getsize(os.path.join(dest, n)) for n in sortie)
    json.dump({"metadata": {"total_size": total}, "weight_map": carte}, open(os.path.join(dest, "model.safetensors.index.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(dest, "config.json"), "w"), indent=2)
    for n in COPIES:
        if n != "config.json" and os.path.exists(os.path.join(source, n)): shutil.copy2(os.path.join(source, n), dest)
    prov = {"role": "témoin P3 (3), hors menu : source brute bf16 = l AWQ déquantifié, pas un alias acvram",
            "source": {"chemin": os.path.abspath(source), "quantization_config": qc,
                       "sha256": {n: sha256_fichier(os.path.join(source, n)) for n in fichiers}},
            "methode": "outils/dequantiser-awq-bf16.py : nibbles int4 → (q − 8) × scale par groupe, produit bf16 (== compressed-tensors PackedQuantizationCompressor.decompress au bit)",
            "sortie": {"shards": {n: sha256_fichier(os.path.join(dest, n)) for n in sortie}, "octets": total,
                       "tenseurs_dequantises": n_deq, "tenseurs_copies": n_copie, "duree_s": round(time.time() - t0)}}
    json.dump(prov, open(os.path.join(dest, "acvram_source.json"), "w"), indent=1, ensure_ascii=False)
    print("RESULTAT " + json.dumps({"dest": dest, "shards": len(sortie), "octets": total, "dequantises": n_deq, "copies": n_copie,
                                    "duree_s": prov["sortie"]["duree_s"], "sha256_sortie": {n: h[:12] for n, h in prov["sortie"]["shards"].items()}}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        sys.exit(test(sys.argv[2] if len(sys.argv) > 2 else None))
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    sys.exit(convertir(sys.argv[1], sys.argv[2]))
