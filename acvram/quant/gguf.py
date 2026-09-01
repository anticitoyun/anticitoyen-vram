"""Lecture des points de contrôle GGUF, comme source de conversion.

GGUF est le conteneur de llama.cpp : un en-tête de métadonnées typées, puis les
tenseurs, la plupart déjà quantifiés par blocs (Q4_K, Q6_K, ...). acvram ne
l'exécute pas tel quel — ses noyaux ont leurs propres formats — mais sait le
*lire* : chaque tenseur est déquantifié vers float32 puis passe par la chaîne
de conversion habituelle. Re-quantifier des poids déjà quantifiés additionne
deux erreurs ; c'est le prix à payer quand le point de contrôle d'origine n'est
plus disponible, et le rapport de conversion chiffre ce qu'il en coûte.

La déquantification suit ggml littéralement, bloc par bloc, vectorisée en
numpy ; les dispositions (entrelacement des quartets, échelles 6 bits pliées
sur deux octets) sortent de `ggml-quants.c`, pas d'une spécification — il n'y
en a pas d'autre.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Iterator, Optional

import numpy as np
import torch

__all__ = ["GGUFFile", "is_gguf", "find_gguf"]

_KV_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}

# ggml_type -> (octets par bloc, poids par bloc)
_BLOCK = {0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32),
          7: (24, 32), 8: (34, 32), 12: (144, 256), 13: (176, 256),
          14: (210, 256), 23: (136, 256), 30: (2, 1)}

_KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                           3, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def is_gguf(path: str) -> bool:
    if os.path.isfile(path) and path.endswith(".gguf"):
        return True
    return os.path.isdir(path) and bool(find_gguf(path))


def find_gguf(path: str) -> Optional[str]:
    """Le fichier .gguf d'un répertoire.

    Les points de contrôle éclatés (``-00001-of-00002.gguf``) ne sont pas
    encore lus : mieux vaut le dire que de charger un tiers du modèle. Les
    projecteurs multimodaux (``mmproj-*``) sont ignorés — ils accompagnent le
    modèle de langage, ils ne le contiennent pas.
    """
    if os.path.isfile(path):
        return path
    cands = sorted(f for f in os.listdir(path)
                   if f.endswith(".gguf") and not f.startswith("mmproj"))
    if not cands:
        return None
    import re
    if any(re.search(r"-\d{5}-of-\d{5}\.gguf$", c) for c in cands):
        raise ValueError(
            f"{path} : point de contrôle GGUF en plusieurs fragments "
            f"(-NNNNN-of-NNNNN) — non géré pour l'instant. Recollez-le avec "
            f"llama-gguf-split --merge, ou convertissez depuis la source.")
    return os.path.join(path, cands[0])


class GGUFFile:
    def __init__(self, path: str) -> None:
        self.path = find_gguf(path)
        if self.path is None:
            raise FileNotFoundError(f"aucun .gguf sous {path}")
        self.kv: dict[str, Any] = {}
        self.tensors: dict[str, tuple[tuple[int, ...], int, int]] = {}
        self._parse()

    # -- en-tete ---------------------------------------------------------
    def _parse(self) -> None:
        with open(self.path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                raise ValueError(f"{self.path} : pas un fichier GGUF")
            version, = struct.unpack("<I", fh.read(4))
            if version < 2:
                raise ValueError(f"GGUF version {version} : trop ancien")
            n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))
            for _ in range(n_kv):
                key = self._string(fh)
                self.kv[key] = self._value(fh, struct.unpack("<I", fh.read(4))[0])
            infos = []
            for _ in range(n_tensors):
                name = self._string(fh)
                nd, = struct.unpack("<I", fh.read(4))
                dims = struct.unpack(f"<{nd}Q", fh.read(8 * nd))
                ttype, offset = struct.unpack("<IQ", fh.read(12))
                infos.append((name, dims, ttype, offset))
            align = int(self.kv.get("general.alignment", 32))
            base = fh.tell()
            base = (base + align - 1) // align * align
            for name, dims, ttype, offset in infos:
                # ggml range la dimension contiguë en premier ; torch en dernier.
                self.tensors[name] = (tuple(reversed(dims)), ttype, base + offset)

    @staticmethod
    def _string(fh) -> str:
        n, = struct.unpack("<Q", fh.read(8))
        return fh.read(n).decode("utf-8", errors="replace")

    def _value(self, fh, vtype: int) -> Any:
        if vtype in _KV_FMT:
            fmt = _KV_FMT[vtype]
            return struct.unpack("<" + fmt, fh.read(struct.calcsize(fmt)))[0]
        if vtype == 8:
            return self._string(fh)
        if vtype == 9:
            etype, n = struct.unpack("<IQ", fh.read(12))
            if etype in _KV_FMT:
                fmt = _KV_FMT[etype]
                size = struct.calcsize(fmt)
                raw = fh.read(size * n)
                return list(struct.unpack(f"<{n}{fmt}", raw))
            return [self._value(fh, etype) for _ in range(n)]
        raise ValueError(f"type de metadonnee GGUF inconnu : {vtype}")

    # -- dequantification ------------------------------------------------
    def load(self, name: str) -> torch.Tensor:
        shape, ttype, offset = self.tensors[name]
        n = 1
        for d in shape:
            n *= d
        if ttype not in _BLOCK:
            raise ValueError(f"{name} : type ggml {ttype} non gere "
                             f"(F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0, "
                             f"Q4_K/Q5_K/Q6_K, IQ4_XS)")
        bbytes, bweights = _BLOCK[ttype]
        nblocks = n // bweights
        raw = np.memmap(self.path, dtype=np.uint8, mode="r",
                        offset=offset, shape=(nblocks * bbytes,))
        out = _DEQUANT[ttype](raw.reshape(nblocks, bbytes))
        return torch.from_numpy(np.ascontiguousarray(out)).reshape(shape)

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Tenseurs sous leurs noms Hugging Face, experts MoE éclatés."""
        gdn = self.arch() in ("qwen35", "qwen35moe", "qwen3next")
        detile = self.arch() in ("qwen35", "qwen35moe")

        def _detile(t: torch.Tensor, dim: int, nk: int, nvpk: int,
                    hd: int) -> torch.Tensor:
            """Ordre « tiled » de ggml -> ordre groupé de la référence.

            Le convertisseur Qwen3.5 réordonne les têtes V pour le broadcast
            de ggml : [G0v0, G1v0, ..., G0v1, ...]. La référence transformers
            — et notre moteur — attendent l'ordre groupé par tête K.
            """
            forme = list(t.shape)
            neuf = forme[:dim] + [nvpk, nk, hd] + forme[dim + 1:]
            t = t.reshape(*neuf)
            perm = list(range(len(neuf)))
            perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
            return t.permute(*perm).contiguous().reshape(*forme)

        if detile:
            a = self.arch()
            nk = int(self.kv.get(f"{a}.ssm.group_count", 16))
            nv = int(self.kv.get(f"{a}.ssm.time_step_rank", 32))
            dv = int(self.kv.get(f"{a}.ssm.inner_size", 4096)) // nv
            dk = int(self.kv.get(f"{a}.ssm.state_size", 128))
            nvpk = nv // nk

        for gname in self.tensors:
            hname = _map_name(gname, gdn)
            if hname is None:
                continue
            t = self.load(gname)
            if detile and ".linear_attn." in hname:
                if hname.endswith("qkv.weight"):
                    kd = nk * dk
                    v = _detile(t[2 * kd:], 0, nk, nvpk, dv)
                    t = torch.cat([t[:2 * kd], v])
                elif hname.endswith("gate.weight"):
                    t = _detile(t, 0, nk, nvpk, dv)
                elif hname.endswith(("alpha.weight", "beta.weight",
                                     "dt_bias.weight", "a_log.weight")):
                    t = _detile(t, 0, nk, nvpk, 1)
                elif hname.endswith("conv1d.weight"):
                    qk = 2 * nk * dk
                    v = _detile(t[qk:], 0, nk, nvpk, dv)
                    t = torch.cat([t[:qk], v])
                elif hname.endswith("out.weight"):
                    t = _detile(t, 1, nk, nvpk, dv)
            if hname.endswith("__exps__"):
                stem = hname[: -len("__exps__")]
                for e in range(t.shape[0]):
                    yield stem.replace("{e}", str(e)), t[e]
            else:
                yield hname, t

    # -- configuration ---------------------------------------------------
    def arch(self) -> str:
        return str(self.kv.get("general.architecture", "llama"))

    # Architectures que le moteur ne sait PAS exécuter : récurrences linéaires
    # (SSM, Gated DeltaNet, KDA). Les convertir quand même produirait un modèle
    # mutilé qui répond du charabia — le pire des échecs, le silencieux.
    UNSUPPORTED = ("kimi-linear", "qwen35", "qwen35moe", "qwen3next",
                   "nemotron_h", "nemotron_h_moe", "falcon-h1", "falcon_h1",
                   "lfm2", "lfm2moe", "mamba", "jamba", "granitehybrid")
    # Architectures transformeurs mais aux blocs différents des nôtres
    # (softcap, laurel, attention partagée...) : à mapper avant de convertir.
    UNTRANSLATED = ("glm4", "glm4moe", "glm4_moe", "gemma4", "gemma3",
                    "gemma3n", "chatglm", "granite", "internlm2", "phi3",
                    "starcoder2", "kat", "deci", "olmoe")

    def check_executable(self) -> None:
        a = self.arch()
        if a == "qwen35" and os.environ.get("ACVRAM_GDN"):
            return                        # chemin Gated DeltaNet (expérimental)
        if a in self.UNSUPPORTED or any(".ssm_" in n for n in self.tensors):
            raise ValueError(
                f"architecture « {a} » : hybride à récurrence linéaire "
                f"(couches SSM/DeltaNet) — le moteur acvram est un "
                f"transformeur pur et ne peut pas l'exécuter. La convertir "
                f"produirait un modèle mutilé. Non converti.")
        if a in self.UNTRANSLATED:
            raise ValueError(
                f"architecture GGUF « {a} » : pas encore traduite vers le "
                f"moteur acvram (blocs différents du transformeur llama). "
                f"Non convertie plutôt que mutilée.")

    def hf_config(self) -> dict:
        a = self.arch()

        def g(suffix: str, default=None):
            return self.kv.get(f"{a}.{suffix}", default)

        archmap = {"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM",
                   "qwen3": "Qwen3ForCausalLM",
                   "qwen2moe": "Qwen2MoeForCausalLM",
                   "qwen3moe": "Qwen3MoeForCausalLM",
                   "mistral": "MistralForCausalLM",
                   "gemma2": "Gemma2ForCausalLM"}
        heads = int(g("attention.head_count", 32))
        tokens = self.kv.get("tokenizer.ggml.tokens") or []
        cfg = {
            "architectures": [archmap.get(a, "LlamaForCausalLM")],
            "model_type": a,
            "hidden_size": int(g("embedding_length", 4096)),
            "intermediate_size": int(g("feed_forward_length", 11008)),
            "num_hidden_layers": int(g("block_count", 32)),
            "num_attention_heads": heads,
            "num_key_value_heads": int(g("attention.head_count_kv", heads)),
            "max_position_embeddings": int(g("context_length", 4096)),
            "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon", 1e-5)),
            "rope_theta": float(g("rope.freq_base", 10000.0)),
            "vocab_size": int(g("vocab_size", len(tokens) or 32000)),
            "torch_dtype": "bfloat16",
            "tie_word_embeddings": "output.weight" not in self.tensors,
        }
        if g("attention.key_length"):
            cfg["head_dim"] = int(g("attention.key_length"))
        if g("expert_count"):
            cfg["num_experts"] = int(g("expert_count"))
            cfg["num_experts_per_tok"] = int(g("expert_used_count", 2))
            cfg["moe_intermediate_size"] = int(
                g("expert_feed_forward_length", cfg["intermediate_size"]))
            if g("expert_shared_feed_forward_length"):
                cfg["shared_expert_intermediate_size"] = int(
                    g("expert_shared_feed_forward_length"))
        if a == "qwen35":
            interval = int(g("full_attention_interval", 4))
            nl = cfg["num_hidden_layers"]
            cfg["model_type"] = "qwen3_next"
            cfg["layer_types"] = [
                "full_attention" if (i + 1) % interval == 0
                else "linear_attention" for i in range(nl)]
            cfg["linear_num_value_heads"] = int(g("ssm.time_step_rank", 32))
            cfg["linear_num_key_heads"] = int(g("ssm.group_count", 16))
            cfg["linear_key_head_dim"] = int(g("ssm.state_size", 128))
            cfg["linear_value_head_dim"] = (
                int(g("ssm.inner_size", 4096))
                // int(g("ssm.time_step_rank", 32)))
            cfg["linear_conv_kernel_dim"] = int(g("ssm.conv_kernel", 4))
            cfg["rotary_dim"] = int(g("rope.dimension_count", 0)) or None
            cfg["attn_output_gate"] = True

        bos = self.kv.get("tokenizer.ggml.bos_token_id")
        eos = self.kv.get("tokenizer.ggml.eos_token_id")
        if bos is not None:
            cfg["bos_token_id"] = int(bos)
        if eos is not None:
            cfg["eos_token_id"] = int(eos)
        return cfg

    # -- fichiers annexes ------------------------------------------------
    def export_sidecars(self, out_dir: str) -> list[str]:
        """Écrit config.json, tokenizer et generation_config reconstruits.

        Le GGUF embarque le vocabulaire ; un tokenizer BPE de style GPT-2 (ce
        qu'emploient Llama 3, Qwen et leurs dérivés) se reconstruit fidèlement.
        Le style SentencePiece n'est pas reconstruit : passer --tokenizer.
        """
        os.makedirs(out_dir, exist_ok=True)
        written = ["config.json"]
        with open(os.path.join(out_dir, "config.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(self.hf_config(), fh, indent=1)

        gen: dict[str, Any] = {}
        for k, ck in (("bos_token_id", "tokenizer.ggml.bos_token_id"),
                      ("eos_token_id", "tokenizer.ggml.eos_token_id"),
                      ("pad_token_id", "tokenizer.ggml.padding_token_id")):
            if ck in self.kv:
                gen[k] = int(self.kv[ck])
        if gen:
            with open(os.path.join(out_dir, "generation_config.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(gen, fh, indent=1)
            written.append("generation_config.json")

        model = self.kv.get("tokenizer.ggml.model")
        tokens = self.kv.get("tokenizer.ggml.tokens")
        merges = self.kv.get("tokenizer.ggml.merges")
        ttypes = self.kv.get("tokenizer.ggml.token_type") or []
        if model == "gpt2" and tokens and merges:
            vocab = {t: i for i, t in enumerate(tokens)}
            added = [{"id": i, "content": t, "special": True,
                      "single_word": False, "lstrip": False,
                      "rstrip": False, "normalized": False}
                     for i, t in enumerate(tokens)
                     if i < len(ttypes) and ttypes[i] == 3]
            tok = {
                "version": "1.0",
                "added_tokens": added,
                "pre_tokenizer": {"type": "ByteLevel",
                                  "add_prefix_space": False,
                                  "trim_offsets": True, "use_regex": True},
                "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                            "trim_offsets": True, "use_regex": True},
                "model": {"type": "BPE", "dropout": None, "unk_token": None,
                          "continuing_subword_prefix": None,
                          "end_of_word_suffix": None, "fuse_unk": False,
                          "byte_fallback": False, "ignore_merges": True,
                          "vocab": vocab, "merges": merges},
            }
            with open(os.path.join(out_dir, "tokenizer.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(tok, fh, ensure_ascii=False)
            written.append("tokenizer.json")

            tcfg: dict[str, Any] = {"tokenizer_class": "PreTrainedTokenizerFast"}
            if "tokenizer.chat_template" in self.kv:
                tcfg["chat_template"] = self.kv["tokenizer.chat_template"]
            for k, ck in (("bos_token", "tokenizer.ggml.bos_token_id"),
                          ("eos_token", "tokenizer.ggml.eos_token_id")):
                if ck in self.kv and int(self.kv[ck]) < len(tokens):
                    tcfg[k] = tokens[int(self.kv[ck])]
            with open(os.path.join(out_dir, "tokenizer_config.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(tcfg, fh, ensure_ascii=False, indent=1)
            written.append("tokenizer_config.json")
        return written


# --------------------------------------------------------------------------
# noms de tenseurs
# --------------------------------------------------------------------------

_DIRECT = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

# Couches à récurrence linéaire (qwen35 / qwen3-next) : la projection qkv de
# la partie linéaire reste fusionnée (le module la découpe en plat), le gate
# et les projections a/b sont séparés, les petits tenseurs d'état passent en
# clair. `attn_gate` sert aussi aux couches d'attention pleines (porte de
# sortie) — même nom GGUF, rôle choisi par le type de couche au chargement.
_LAYER_GDN = {
    "attn_qkv": "linear_attn.qkv",
    "attn_gate": "linear_attn.gate",
    "ssm_alpha": "linear_attn.alpha",
    "ssm_beta": "linear_attn.beta",
    "ssm_out": "linear_attn.out",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_norm": "linear_attn.norm",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
}

_LAYER = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate_inp": "mlp.gate",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
}

_EXPS = {"ffn_gate_exps": "mlp.experts.{e}.gate_proj",
         "ffn_up_exps": "mlp.experts.{e}.up_proj",
         "ffn_down_exps": "mlp.experts.{e}.down_proj"}


def _map_name(g: str, gdn: bool = False) -> Optional[str]:
    if g in _DIRECT:
        return _DIRECT[g]
    if not g.startswith("blk."):
        return None                        # rope_freqs et autres auxiliaires
    _, idx, rest = g.split(".", 2)
    stem, _, kind = rest.rpartition(".")   # "attn_q", "weight"|"bias"
    if gdn:
        if rest == "ssm_a":                # A_log — seul tenseur SANS suffixe
            return f"model.layers.{idx}.linear_attn.a_log.weight"
        if stem == "ssm_dt":               # dt_bias, range en .bias
            return f"model.layers.{idx}.linear_attn.dt_bias.weight"
        if stem in _LAYER_GDN:
            return f"model.layers.{idx}.{_LAYER_GDN[stem]}.{kind}"
        # les couches d'attention pleines retombent sur le mapping classique
    if stem in _LAYER:
        return f"model.layers.{idx}.{_LAYER[stem]}.{kind}"
    if stem in _EXPS:
        return f"model.layers.{idx}.{_EXPS[stem]}.{kind}__exps__"
    return None


# --------------------------------------------------------------------------
# dequantification par type (dispositions de ggml-quants.c)
# --------------------------------------------------------------------------

def _f16(b: np.ndarray, off: int) -> np.ndarray:
    return b[:, off:off + 2].copy().view(np.float16).astype(np.float32)


def _dq_f32(b): return b.reshape(-1, 4).copy().view(np.float32).reshape(-1)
def _dq_f16(b): return b.reshape(-1, 2).copy().view(np.float16).astype(np.float32).reshape(-1)


def _dq_bf16(b):
    u = b.reshape(-1, 2).copy().view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(-1)


def _dq_q8_0(b):
    d = _f16(b, 0)
    q = b[:, 2:34].view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)


def _nibbles(qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (qs & 0xF).astype(np.float32), (qs >> 4).astype(np.float32)


def _dq_q4_0(b):
    d = _f16(b, 0)
    lo, hi = _nibbles(b[:, 2:18])
    return (d * (np.concatenate([lo, hi], 1) - 8.0)).reshape(-1)


def _dq_q4_1(b):
    d, m = _f16(b, 0), _f16(b, 2)
    lo, hi = _nibbles(b[:, 4:20])
    return (d * np.concatenate([lo, hi], 1) + m).reshape(-1)


def _qh_bits(b, off):
    qh = b[:, off:off + 4].copy().view(np.uint32)
    return ((qh >> np.arange(32, dtype=np.uint32)) & 1).astype(np.float32)


def _dq_q5_0(b):
    d = _f16(b, 0)
    h = _qh_bits(b, 2)
    lo, hi = _nibbles(b[:, 6:22])
    q = np.concatenate([lo, hi], 1) + 16.0 * h
    return (d * (q - 16.0)).reshape(-1)


def _dq_q5_1(b):
    d, m = _f16(b, 0), _f16(b, 2)
    h = _qh_bits(b, 4)
    lo, hi = _nibbles(b[:, 8:24])
    return (d * (np.concatenate([lo, hi], 1) + 16.0 * h) + m).reshape(-1)


def _kscales(b, off):
    """Les 8 paires (échelle, minimum) sur 6 bits, pliées dans 12 octets."""
    s = b[:, off:off + 12].astype(np.uint16)
    sc = np.empty((b.shape[0], 8), np.float32)
    mn = np.empty_like(sc)
    for j in range(8):
        if j < 4:
            sc[:, j] = (s[:, j] & 63).astype(np.float32)
            mn[:, j] = (s[:, j + 4] & 63).astype(np.float32)
        else:
            sc[:, j] = ((s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
                        ).astype(np.float32)
            mn[:, j] = ((s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
                        ).astype(np.float32)
    return sc, mn


def _dq_q4_k(b):
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, mn = _kscales(b, 4)
    qs = b[:, 16:144]
    out = np.empty((b.shape[0], 256), np.float32)
    for j in range(4):                     # 64 valeurs par tour, 2 sous-blocs
        chunk = qs[:, 32 * j:32 * (j + 1)]
        lo, hi = _nibbles(chunk)
        out[:, 64 * j:64 * j + 32] = d * sc[:, [2 * j]] * lo \
            - dmin * mn[:, [2 * j]]
        out[:, 64 * j + 32:64 * j + 64] = d * sc[:, [2 * j + 1]] * hi \
            - dmin * mn[:, [2 * j + 1]]
    return out.reshape(-1)


def _dq_q5_k(b):
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, mn = _kscales(b, 4)
    qh = b[:, 16:48]
    qs = b[:, 48:176]
    out = np.empty((b.shape[0], 256), np.float32)
    for j in range(4):
        chunk = qs[:, 32 * j:32 * (j + 1)]
        lo, hi = _nibbles(chunk)
        b1 = ((qh >> (2 * j)) & 1).astype(np.float32)
        b2 = ((qh >> (2 * j + 1)) & 1).astype(np.float32)
        out[:, 64 * j:64 * j + 32] = d * sc[:, [2 * j]] * (lo + 16.0 * b1) \
            - dmin * mn[:, [2 * j]]
        out[:, 64 * j + 32:64 * j + 64] = d * sc[:, [2 * j + 1]] \
            * (hi + 16.0 * b2) - dmin * mn[:, [2 * j + 1]]
    return out.reshape(-1)


def _dq_q6_k(b):
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    scales = b[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(b, 208)
    out = np.empty((b.shape[0], 256), np.float32)
    for n in range(2):                     # deux moitiés de 128
        lq = ql[:, 64 * n:64 * n + 64]
        lh = qh[:, 32 * n:32 * n + 32]
        sc = scales[:, 8 * n:8 * n + 8]
        q1 = (lq[:, :32] & 0xF) | (((lh >> 0) & 3) << 4)
        q2 = (lq[:, 32:] & 0xF) | (((lh >> 2) & 3) << 4)
        q3 = (lq[:, :32] >> 4) | (((lh >> 4) & 3) << 4)
        q4 = (lq[:, 32:] >> 4) | (((lh >> 6) & 3) << 4)
        base = 128 * n
        for i, q in enumerate((q1, q2, q3, q4)):
            qf = q.astype(np.float32) - 32.0
            s = np.repeat(sc[:, 2 * i:2 * i + 2], 16, axis=1)
            out[:, base + 32 * i:base + 32 * (i + 1)] = d * s * qf
    return out.reshape(-1)


def _dq_iq4_xs(b):
    d = _f16(b, 0)
    sh = b[:, 2:4].copy().view(np.uint16).astype(np.uint32)
    sl = b[:, 4:8]
    qs = b[:, 8:136]
    out = np.empty((b.shape[0], 256), np.float32)
    for sb in range(8):                    # 8 sous-blocs de 32
        low = ((sl[:, sb // 2] >> (4 * (sb % 2))) & 0xF).astype(np.uint32)
        high = (sh[:, 0] >> (2 * sb)) & 3
        ls = ((low | (high << 4)).astype(np.float32) - 32.0)
        chunk = qs[:, 16 * sb:16 * (sb + 1)]
        lo = _KVALUES_IQ4NL[(chunk & 0xF).astype(np.int64)]
        hi = _KVALUES_IQ4NL[(chunk >> 4).astype(np.int64)]
        vals = np.concatenate([lo, hi], 1)
        out[:, 32 * sb:32 * (sb + 1)] = d * ls[:, None] * vals
    return out.reshape(-1)


_DEQUANT = {0: _dq_f32, 1: _dq_f16, 2: _dq_q4_0, 3: _dq_q4_1, 6: _dq_q5_0,
            7: _dq_q5_1, 8: _dq_q8_0, 12: _dq_q4_k, 13: _dq_q5_k,
            14: _dq_q6_k, 23: _dq_iq4_xs, 30: _dq_bf16}
