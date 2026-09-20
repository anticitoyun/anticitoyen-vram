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

# Architectures dont les sections de RoPE sont CONTIGUES : en texte pur les
# trois axes portent la meme position, les frequences ne sont pas permutees, et
# le resultat est celui d'un RoPE ordinaire. Le raccourci y est demontre.
#
# Il ne l'est PAS pour l'entrelacement (IMROPE, `LLAMA_ROPE_TYPE_IMROPE` chez
# llama.cpp) : la, les frequences sont permutees entre paires de dimensions, et
# meme a positions egales le resultat differe d'un NEOX standard. L'ecart croit
# avec la distance entre positions, ce qui donne une sortie correcte au debut
# puis de plus en plus fausse — mesure le 8/09/2026 sur un qwen35, dont les
# sections valent [11, 11, 10, 0] : perplexite qui REMONTE avec le contexte
# (85 a 128 jetons, 173 a 512) la ou un modele dense descend proprement.
#
# La liste est BLANCHE a dessein : une architecture inconnue portant des
# sections non triviales est refusee jusqu'a ce que quelqu'un ait verifie
# laquelle des deux dispositions elle emploie. Une demi-journee a ete perdue
# sur un modele servi faux en silence ; un dossier qu'on ne sait pas servir se
# conserve, il ne se sert pas.
_ROPE_SECTIONS_CONTIGUES = {
    "qwen2vl", "qwen3vl", "qwen3vlmoe",
    # qwen35 : sections [11, 11, 10, 0] et RoPE ENTRELACE chez llama.cpp
    # (`LLAMA_ROPE_TYPE_IMROPE`). Verifie le 8/09/2026 dans les deux sources,
    # et le raccourci tient quand meme, pour deux raisons qui doivent aller
    # ensemble :
    #  - `ggml-cpu/ops.cpp` : l'entrelacement ne change ni l'ordre ni la
    #    valeur des frequences ; le selecteur ne choisit que QUEL axe de
    #    position fournit l'angle ;
    #  - `llama-batch.cpp:712-719` : pour un lot de JETONS, la meme position
    #    est diffusee sur les quatre axes (`src_off = batch.token ? 0 : ...`).
    # Les quatre axes portant la meme position et les frequences etant
    # inchangees, IMROPE se reduit exactement a NEOX. **En texte pur
    # seulement** : le jour ou l'on servira des images a ce modele, les axes
    # porteront des positions differentes et il faudra l'implementer.
    "qwen35", "qwen35moe",
}


def _garde_rope_sections(arch: str, sections) -> None:
    if not sections or arch in _ROPE_SECTIONS_CONTIGUES:
        return
    utiles = [int(x) for x in sections if int(x) > 0]
    if len(utiles) <= 1:
        return                     # une seule section : RoPE ordinaire
    raise ValueError(
        f"architecture {arch!r} : rope.dimension_sections = {list(sections)}, "
        f"soit un RoPE multi-axes dont acvram ne connait pas la disposition. "
        f"Si les sections sont contigues, l'ajouter a "
        f"_ROPE_SECTIONS_CONTIGUES apres verification ; si elles sont "
        f"entrelacees (IMROPE), il faut l'implementer. Servir ce modele avec "
        f"un NEOX standard rend une sortie qui se degrade avec la longueur du "
        f"contexte, sans erreur.")



def mmproj_a_cote(path: str) -> Optional[str]:
    """Le projecteur multimodal (``mmproj-*.gguf``) rangé à côté d'un GGUF,
    ou None. La voie GGUF-mmproj est hors périmètre : `convert_checkpoint`
    le nomme dans un refus et convertit le modèle de langue seul."""
    d = path if os.path.isdir(path) else os.path.dirname(path) or "."
    try:
        cands = sorted(f for f in os.listdir(d)
                       if f.startswith("mmproj") and f.endswith(".gguf"))
    except OSError:
        return None
    return cands[0] if cands else None


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
            return self._load_gguf_py(name, shape, ttype, offset, n)
        bbytes, bweights = _BLOCK[ttype]
        nblocks = n // bweights
        raw = np.memmap(self.path, dtype=np.uint8, mode="r",
                        offset=offset, shape=(nblocks * bbytes,))
        out = _DEQUANT[ttype](raw.reshape(nblocks, bbytes))
        return torch.from_numpy(np.ascontiguousarray(out)).reshape(shape)

    def _load_gguf_py(self, name: str, shape: list[int], ttype: int,
                      offset: int, n: int) -> torch.Tensor:
        """Types à grille (IQ1/IQ2/IQ3, TQ…) : déquantification par le
        gguf-py de llama.cpp — la référence elle-même, tables comprises."""
        try:
            from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
            from gguf.quants import dequantize
        except ImportError as e:
            raise ValueError(f"{name} : type ggml {ttype} non gere sans gguf-py "
                             f"(pip install llama.cpp/gguf-py)") from e
        qtype = GGMLQuantizationType(ttype)
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        nbytes = n // block_size * type_size
        raw = np.memmap(self.path, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,))
        out = dequantize(np.asarray(raw).reshape(tuple(shape[:-1]) + (shape[-1] // block_size * type_size,)), qtype)
        return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32)).reshape(shape)

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Tenseurs sous leurs noms Hugging Face, experts MoE éclatés."""
        gdn = self.arch() in ("qwen35", "qwen35moe", "qwen3next")
        detile = self.arch() in ("qwen35", "qwen35moe")
        kimi_rec = None
        if self.arch() == "kimi-linear":
            hkv = self.kv.get("kimi-linear.attention.head_count_kv") or []
            kimi_rec = {i for i, n in enumerate(hkv) if int(n) == 0}
        elif self.arch() == "deepseek2":
            kimi_rec = set()                  # toutes les couches sont MLA
        # couches MTP (nextn) en fin de pile : ignorées entièrement
        nextn = int(self.kv.get(f"{self.arch()}.nextn_predict_layers", 0) or 0)
        premiere_mtp = (int(self.kv.get(f"{self.arch()}.block_count", 0))
                        - nextn) if nextn else -1

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

        # Le convertisseur llama.cpp de la famille llama (LlamaModel, dont
        # Mistral et Granite héritent) permute q et k vers la disposition
        # RoPE entrelacée de Meta ; la référence HF (et nous) travaillent en
        # demi-rotation : on défait la permutation à la lecture.
        permute = self.arch() in ("llama", "granite")
        if permute:
            a = self.arch()
            p_nh = int(self.kv.get(f"{a}.attention.head_count"))
            p_nkv = int(self.kv.get(f"{a}.attention.head_count_kv", p_nh))

        def _depermute(t: torch.Tensor, n_head: int) -> torch.Tensor:
            forme = t.shape
            hd = forme[0] // n_head
            return (t.reshape(n_head, hd // 2, 2, *forme[1:])
                    .transpose(1, 2).reshape(*forme).contiguous())

        phi3 = self.arch() == "phi3"
        # Gemma 4 : Gemma4RMSNorm multiplie par w tel quel (q_norm ≈ 1) ; les
        # GGUF issus d'un convertisseur qui décalait encore les normes de +1
        # (héritage Gemma 3) se reconnaissent à un attn_q_norm proche de 2
        gemma_shift = False
        if self.arch() == "gemma4" and "blk.0.attn_q_norm.weight" in self.tensors:
            gemma_shift = float(self.load("blk.0.attn_q_norm.weight").float().mean()) > 1.5
            if gemma_shift:
                print("  gemma4 : normes décalées de +1 dans ce GGUF, ramenées à w")
        if phi3:
            a = self.arch()
            nh = int(self.kv.get(f"{a}.attention.head_count"))
            nkv = int(self.kv.get(f"{a}.attention.head_count_kv", nh))
            hd = int(self.kv.get(f"{a}.attention.key_length",
                                 int(self.kv.get(f"{a}.embedding_length")) // nh))

        for gname in self.tensors:
            if premiere_mtp >= 0 and gname.startswith("blk."):
                idx = int(gname.split(".", 2)[1])
                if idx >= premiere_mtp:
                    # Couche de prédiction multi-jetons. Elle porte un bloc de
                    # transformeur complet plus quatre tenseurs propres
                    # (eh_proj, enorm, hnorm, shared_head_norm) ; on la range
                    # sous ``model.mtp.<n>`` au lieu de la jeter, pour servir
                    # de brouillon spéculatif — un dixième du coût d'un modèle
                    # brouillon séparé.
                    mtp_i = idx - premiere_mtp
                    reste = gname.split(".", 2)[2]
                    if reste.startswith("nextn."):
                        yield f"model.mtp.{mtp_i}.{reste[6:]}", self.load(gname)
                        continue
                    sous = _map_name(f"blk.0.{reste}", gdn, kimi_rec, phi3)
                    if sous is None:
                        continue
                    sous = sous.replace("model.layers.0.", "")
                    yield f"model.mtp.{mtp_i}.{sous}", self.load(gname)
                    continue
            hname = _map_name(gname, gdn, kimi_rec, phi3)
            if hname is None:
                continue
            if self.arch() == "gemma4" and gname.endswith("ffn_norm.weight"):
                # chez Gemma, ffn_norm précède le MLP (post_attention_norm existe à part)
                hname = hname.replace("post_attention_layernorm", "pre_feedforward_layernorm")
            t = self.load(gname)
            if gemma_shift and gname.endswith("norm.weight"):
                t = t.to(torch.float32) - 1.0
            if hname.endswith("shared_expert_gate.weight") and t.dim() == 1:
                t = t.reshape(1, -1)      # vecteur GGUF -> Linear(d, 1)
            if hname.endswith("linear_attn.ba.weight"):
                # qwen3next : b et a fusionnés (in_proj_ba, b d'abord)
                nv_ = t.shape[0] // 2
                base = hname[: -len("ba.weight")]
                yield base + "beta.weight", t[:nv_].contiguous()
                yield base + "alpha.weight", t[nv_:].contiguous()
                continue
            if hname.endswith(("mamba.norm.weight", "mamba.A.weight", "mamba.D.weight")):
                t = t.reshape(-1)
            if kimi_rec is not None:
                if ".conv1d_" in hname:   # [d_inner, 1, k] -> [d_inner, k]
                    t = t.reshape(t.shape[0], t.shape[-1])
                elif hname.endswith("linear_attn.a.weight"):
                    t = t.reshape(-1)     # [nh, 1] -> [nh]
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
            if permute and hname.endswith(("self_attn.q_proj.weight", "self_attn.q_proj.bias")):
                t = _depermute(t, p_nh)
            elif permute and hname.endswith(("self_attn.k_proj.weight", "self_attn.k_proj.bias")):
                t = _depermute(t, p_nkv)
            if phi3 and hname.endswith("self_attn.qkv_proj.weight"):
                # Phi-3/4 : q, k, v fusionnés [q (nh·hd) ; k (nkv·hd) ; v (nkv·hd)]
                q, k, v = torch.split(t, [nh * hd, nkv * hd, nkv * hd], dim=0)
                base = hname[: -len("qkv_proj.weight")]
                yield base + "q_proj.weight", q
                yield base + "k_proj.weight", k
                yield base + "v_proj.weight", v
                continue
            if phi3 and hname.endswith("mlp.gate_up_proj.weight"):
                g_, u_ = torch.chunk(t, 2, dim=0)      # gate d'abord (Phi3MLP)
                base = hname[: -len("gate_up_proj.weight")]
                yield base + "gate_proj.weight", g_
                yield base + "up_proj.weight", u_
                continue
            if hname.endswith("__exps__"):
                stem = hname[: -len("__exps__")]
                for e in range(t.shape[0]):
                    nom_e = stem.replace("{e}", str(e))
                    if nom_e.endswith("gate_up_proj.weight"):     # gemma4 : gate puis up
                        g_, u_ = torch.chunk(t[e], 2, dim=0)
                        yield nom_e.replace("gate_up_proj", "gate_proj"), g_
                        yield nom_e.replace("gate_up_proj", "up_proj"), u_
                    else:
                        yield nom_e, t[e]
            else:
                yield hname, t

    # -- configuration ---------------------------------------------------
    def arch(self) -> str:
        return str(self.kv.get("general.architecture", "llama"))

    # Architectures que le moteur ne sait PAS exécuter : récurrences linéaires
    # (SSM, Gated DeltaNet, KDA). Les convertir quand même produirait un modèle
    # mutilé qui répond du charabia — le pire des échecs, le silencieux.
    UNSUPPORTED = ("kimi-linear", "qwen35", "qwen35moe", "qwen3next",
                   "mamba", "jamba", "granitehybrid")
    # Architectures transformeurs mais aux blocs différents des nôtres
    # (softcap, laurel, attention partagée...) : à mapper avant de convertir.
    UNTRANSLATED = ("glm4", "glm4moe", "glm4_moe", "gemma3",
                    "gemma3n", "chatglm", "internlm2",
                    "starcoder2", "kat", "deci", "olmoe")

    def check_executable(self) -> None:
        a = self.arch()
        if a in ("qwen35", "qwen35moe", "qwen3next", "kimi-linear",
                 "nemotron_h", "nemotron_h_moe", "falcon-h1") \
                and os.environ.get("ACVRAM_GDN", "1") != "0":
            return                        # récurrences linéaires (expérimental)
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
                   "gemma2": "Gemma2ForCausalLM",
                   # vision-langage servis en texte seul : la partie texte est
                   # un qwen2/qwen3 ; en texte pur les trois axes du mrope
                   # portent la même position, soit un RoPE ordinaire
                   "phi3": "Phi3ForCausalLM",
                   "granite": "GraniteForCausalLM",
                   "gemma4": "Gemma4ForCausalLM",
                   "qwen2vl": "Qwen2ForCausalLM",
                   "qwen3vl": "Qwen3ForCausalLM",
                   "qwen3vlmoe": "Qwen3MoeForCausalLM"}
        vl = {"qwen2vl": "qwen2", "qwen3vl": "qwen3", "qwen3vlmoe": "qwen3_moe"}
        _garde_rope_sections(a, g("rope.dimension_sections"))
        heads = int(g("attention.head_count", 32))
        tokens = self.kv.get("tokenizer.ggml.tokens") or []
        cfg = {
            "architectures": [archmap.get(a, "LlamaForCausalLM")],
            "model_type": vl.get(a, a),
            "hidden_size": int(g("embedding_length", 4096)),
            "intermediate_size": (lambda v: int(max(v)) if isinstance(v, list) else int(v))(
                g("feed_forward_length", 11008)),
            "num_hidden_layers": int(g("block_count", 32)),
            "num_attention_heads": heads,
            # head_count_kv peut être une liste (hybrides) : prendre le max
            "num_key_value_heads": (lambda v: int(max(v) or heads)
                                    if isinstance(v, list) else int(v))(
                g("attention.head_count_kv", heads)),
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
        if a in ("qwen35", "qwen35moe", "qwen3next"):
            interval = int(g("full_attention_interval", 4))
            # les couches MTP (nextn) sont stockées en fin de pile : le modèle
            # principal s'arrête avant elles
            nextn = int(g("nextn_predict_layers", 0))
            cfg["num_hidden_layers"] -= nextn
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
            cfg["gdn_a_log_negexp"] = True       # convention du convertisseur llama.cpp

        if a == "falcon-h1":
            cfg["model_type"] = "falcon_h1"
            cfg["architectures"] = ["FalconH1ForCausalLM"]
            cfg["layer_types"] = ["parallel"] * cfg["num_hidden_layers"]
            cfg["head_dim"] = int(g("attention.key_length", cfg["hidden_size"] // heads))
            inner = int(g("ssm.inner_size")); H = int(g("ssm.time_step_rank"))
            cfg["mamba_num_heads"] = H; cfg["mamba_head_dim"] = inner // H
            cfg["n_groups"] = int(g("ssm.group_count", 1))
            cfg["ssm_state_size"] = int(g("ssm.state_size", 128))
            cfg["conv_kernel"] = int(g("ssm.conv_kernel", 4))
        if a in ("nemotron_h", "nemotron_h_moe"):
            cfg["model_type"] = "nemotron_h"
            cfg["architectures"] = ["NemotronHForCausalLM"]
            hkv = self.kv.get(f"{a}.attention.head_count_kv") or []
            kinds = []
            for i in range(cfg["num_hidden_layers"]):
                st = {n.split(".", 2)[2] for n in self.tensors if n.startswith(f"blk.{i}.")}
                kinds.append("mamba" if "ssm_in.weight" in st else
                             ("full_attention" if "attn_q.weight" in st else
                              ("moe" if "ffn_gate_inp.weight" in st else "mlp")))
            cfg["layer_types"] = kinds
            cfg["num_key_value_heads"] = int(max(hkv)) if hkv else heads
            cfg["head_dim"] = int(g("attention.key_length", cfg["hidden_size"] // heads))
            cfg["attention_rope"] = False
            inner = int(g("ssm.inner_size")); H = int(g("ssm.time_step_rank"))
            cfg["mamba_num_heads"] = H; cfg["mamba_head_dim"] = inner // H
            cfg["n_groups"] = int(g("ssm.group_count", 1))
            cfg["ssm_state_size"] = int(g("ssm.state_size", 128))
            cfg["conv_kernel"] = int(g("ssm.conv_kernel", 4))
            cfg["hidden_act"] = "relu2"
            cfg["first_k_dense_replace"] = 0
            if g("expert_count"):
                cfg["router_scoring"] = "sigmoid"
                cfg["norm_topk_prob"] = bool(g("expert_weights_norm", True))
                cfg["routed_scaling_factor"] = float(g("expert_weights_scale", 1.0) or 1.0)
        if a in ("lfm2", "lfm2moe"):
            cfg["model_type"] = "lfm2_moe" if a == "lfm2moe" else "lfm2"
            cfg["architectures"] = ["Lfm2MoeForCausalLM" if a == "lfm2moe" else "Lfm2ForCausalLM"]
            hkv = self.kv.get(f"{a}.attention.head_count_kv") or []
            cfg["layer_types"] = ["conv" if int(n) == 0 else "full_attention" for n in hkv]
            cfg["num_key_value_heads"] = int(max(hkv)) if hkv else heads
            cfg["conv_L_cache"] = int(g("shortconv.l_cache", 3))
            cfg["first_k_dense_replace"] = int(g("leading_dense_block_count", 0))
            fn = int(g("expert_gating_func", 0) or 0)
            cfg["router_scoring"] = "sigmoid" if fn == 2 else "softmax"
            cfg["norm_topk_prob"] = bool(g("expert_weights_norm", False))
            cfg["routed_scaling_factor"] = float(g("expert_weights_scale", 1.0) or 1.0)
        if a == "deepseek2":
            cfg["model_type"] = "deepseek_v2"
            cfg["architectures"] = ["DeepseekV2ForCausalLM"]
            cfg.pop("head_dim", None)
            nl = cfg["num_hidden_layers"]
            cfg["layer_types"] = ["full_attention"] * nl
            cfg["kv_lora_rank"] = int(g("attention.kv_lora_rank", 512))
            cfg["q_lora_rank"] = int(g("attention.q_lora_rank", 0) or 0)
            cfg["qk_rope_head_dim"] = int(g("rope.dimension_count", 64))
            cfg["qk_nope_head_dim"] = int(g("attention.key_length_mla", 192)) - cfg["qk_rope_head_dim"]
            cfg["v_head_dim"] = int(g("attention.value_length_mla", 128))
            cfg["mla_rope"] = True
            cfg["first_k_dense_replace"] = int(g("leading_dense_block_count", 0))
            cfg["routed_scaling_factor"] = float(g("expert_weights_scale", 1.0))
            cfg["norm_topk_prob"] = bool(g("expert_weights_norm", False))
            fn = int(g("expert_gating_func", 0) or 0)
            if fn == 0:                       # heuristique llama.cpp : GLM 4.7
                fn = 2 if (nl in (47, 48) and cfg["vocab_size"] == 154880) else 1
            cfg["router_scoring"] = "sigmoid" if fn == 2 else "softmax"
            cfg["shared_expert_intermediate_size"] = int(
                g("expert_shared_feed_forward_length", cfg.get("moe_intermediate_size", 0)) or 0)
        if a == "gemma4":
            cfg["model_type"] = "gemma4_text"
            hkv = self.kv.get(f"{a}.attention.head_count_kv") or []
            swa = self.kv.get(f"{a}.attention.sliding_window_pattern") or []
            cfg["layer_types"] = ["sliding_attention" if bool(x) else "full_attention"
                                  for x in swa]
            cfg["head_dim"] = int(g("attention.key_length_swa", 256))
            cfg["global_head_dim"] = int(g("attention.key_length", 512))
            cfg["num_key_value_heads"] = int(max(hkv)) if hkv else heads
            cfg["num_global_key_value_heads"] = int(min(hkv)) if hkv else heads
            cfg["sliding_window"] = int(g("attention.sliding_window", 1024))
            cfg["rope_theta"] = float(g("rope.freq_base", 1e6))
            cfg["rope_theta_swa"] = float(g("rope.freq_base_swa", 1e4))
            # rope_freqs : facteurs 1 sur les paires qui tournent, ~1e30 sinon
            try:
                rf = self.load("rope_freqs.weight")
                cfg["partial_rotary_factor_full"] = int((rf < 1e6).sum()) / float(rf.numel())
            except Exception:                    # noqa: BLE001
                cfg["partial_rotary_factor_full"] = 0.25
            cfg["final_logit_softcapping"] = float(g("final_logit_softcapping", 0.0) or 0.0)
            cfg["hidden_activation"] = "gelu_pytorch_tanh"
            globales = {i for i, x in enumerate(swa) if not x}
            cfg["attention_k_eq_v"] = not any(
                n.endswith("attn_v.weight") and int(n.split(".")[1]) in globales
                for n in self.tensors)
            cfg["embedding_multiplier"] = float(cfg["hidden_size"]) ** 0.5
            if int(g("expert_count", 0) or 0):      # 26B-A4B : MoE en parallèle du MLP dense
                cfg["num_experts"] = int(g("expert_count"))
                cfg["num_experts_per_tok"] = int(g("expert_used_count", 8))
                cfg["moe_intermediate_size"] = int(g("expert_feed_forward_length", 0))
                cfg["gemma_moe"] = True
            cfg["attention_multiplier"] = 1.0
        if a == "granite":
            # multiplicateurs Granite (llama sinon)
            if g("attention.scale"): cfg["attention_multiplier"] = float(g("attention.scale"))
            if g("embedding_scale"): cfg["embedding_multiplier"] = float(g("embedding_scale"))
            if g("residual_scale"): cfg["residual_multiplier"] = float(g("residual_scale"))
            if g("logit_scale"): cfg["logits_scaling"] = float(g("logit_scale"))
        if a == "kimi-linear":
            hkv = self.kv.get("kimi-linear.attention.head_count_kv") or []
            cfg["model_type"] = "kimi_linear"
            cfg["architectures"] = ["KimiLinearForCausalLM"]
            cfg.pop("head_dim", None)      # key_length (576) est la clé MLA
            cfg["layer_types"] = [
                "linear_attention" if int(n) == 0 else "full_attention"
                for n in hkv]
            hd = int(g("kda.head_dim", 128))
            cfg["linear_num_value_heads"] = int(g("attention.head_count", 32))
            cfg["linear_num_key_heads"] = cfg["linear_num_value_heads"]
            cfg["linear_key_head_dim"] = hd
            cfg["linear_value_head_dim"] = hd
            cfg["linear_conv_kernel_dim"] = int(g("ssm.conv_kernel", 4))
            cfg["kv_lora_rank"] = int(g("attention.kv_lora_rank", 512))
            cfg["qk_rope_head_dim"] = int(g("rope.dimension_count", 64))
            cfg["qk_nope_head_dim"] = (int(g("attention.key_length_mla", 192))
                                       - cfg["qk_rope_head_dim"])
            cfg["v_head_dim"] = int(g("attention.value_length_mla", 128))
            cfg["first_k_dense_replace"] = int(g("leading_dense_block_count", 0))
            cfg["router_scoring"] = "sigmoid"
            cfg["routed_scaling_factor"] = float(g("expert_weights_scale", 1.0))
            cfg["norm_topk_prob"] = True

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
        # Fins de tour du gabarit de chat : le GGUF ne déclare qu'un eos, mais
        # les modèles instruits terminent par un marqueur (<|im_end|>...) que
        # llama.cpp reconnaît par heuristique de nom. Sans lui, la génération
        # continue après la réponse.
        toks = self.kv.get("tokenizer.ggml.tokens") or []
        FINS = ("<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end|>",
                "<|eot|>", "<end_of_turn>", "</s>")
        eog = [i for i, t in enumerate(toks) if t in FINS]
        if eog and "eos_token_id" in gen:
            gen["eos_token_id"] = sorted({int(gen["eos_token_id"]), *eog})
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
            # Types GGUF : 3 = CONTROL, 4 = USER_DEFINED. Les deux doivent
            # rester insécables. Ne garder que le 3, comme ici jusqu'à la
            # v0.4.63, découpait « <think> » de GLM-4.7 en « < », « think »,
            # « > » : le modèle recevait un préfixe de conversation corrompu et
            # répondait par une cascade de balises vides. Le chemin
            # SentencePiece prenait déjà les deux types.
            added = [{"id": i, "content": t, "special": ttypes[i] == 3,
                      "single_word": False, "lstrip": False,
                      "rstrip": False, "normalized": False}
                     for i, t in enumerate(tokens)
                     if i < len(ttypes) and ttypes[i] in (3, 4)]
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
        elif model in ("llama", "gemma", "gemma4", "spm", "t5") and tokens:
            # SentencePiece (Unigram) : pièces avec leurs scores, repli
            # octet par octet, préfixe « ▁ » selon add_space_prefix
            scores = self.kv.get("tokenizer.ggml.scores") or [0.0] * len(tokens)
            prefixe = bool(self.kv.get("tokenizer.ggml.add_space_prefix", model == "llama"))
            unk = self.kv.get("tokenizer.ggml.unknown_token_id")
            added = [{"id": i, "content": t, "special": ttypes[i] == 3,
                      "single_word": False, "lstrip": False,
                      "rstrip": False, "normalized": False}
                     for i, t in enumerate(tokens)
                     if i < len(ttypes) and ttypes[i] in (3, 4)]
            normalizers = [{"type": "Replace", "pattern": {"String": " "},
                            "content": "▁"}]
            if prefixe:
                normalizers.insert(0, {"type": "Prepend", "prepend": "▁"})
            decoders = [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                        {"type": "ByteFallback"}, {"type": "Fuse"}]
            if prefixe:
                decoders.append({"type": "Strip", "content": " ", "start": 1, "stop": 0})
            tok = {
                "version": "1.0",
                "added_tokens": added,
                "normalizer": {"type": "Sequence", "normalizers": normalizers},
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": {"type": "Sequence", "decoders": decoders},
                "model": {"type": "Unigram",
                          "unk_id": int(unk) if unk is not None else 0,
                          "vocab": [[t, float(sc)] for t, sc in zip(tokens, scores)],
                          "byte_fallback": True},
            }
            with open(os.path.join(out_dir, "tokenizer.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(tok, fh, ensure_ascii=False)
            written.append("tokenizer.json")
        if "tokenizer.json" in written:
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
    "token_embd_norm.weight": "model.norm.weight",      # lfm2 : norme finale
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
    "ssm_ba": "linear_attn.ba",
    "ssm_beta": "linear_attn.beta",
    "ssm_out": "linear_attn.out",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_norm": "linear_attn.norm",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
}

# kimi-linear : couches KDA (récurrentes) et MLA (attention latente).
_LAYER_KDA = {
    "attn_q": "linear_attn.q_proj", "attn_k": "linear_attn.k_proj",
    "attn_v": "linear_attn.v_proj", "attn_output": "linear_attn.out_proj",
    "ssm_conv1d_q": "linear_attn.conv1d_q",
    "ssm_conv1d_k": "linear_attn.conv1d_k",
    "ssm_conv1d_v": "linear_attn.conv1d_v",
    "ssm_f_a": "linear_attn.f_a", "ssm_f_b": "linear_attn.f_b",
    "ssm_g_a": "linear_attn.g_a", "ssm_g_b": "linear_attn.g_b",
    "ssm_beta": "linear_attn.beta",
    "ssm_norm": "linear_attn.norm",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
}
_LAYER_MLA = {
    "attn_q": "self_attn.q_proj",
    "attn_q_a": "self_attn.q_a_proj", "attn_q_a_norm": "self_attn.q_a_layernorm",
    "attn_q_b": "self_attn.q_b_proj",
    "attn_kv_a_mqa": "self_attn.kv_a_proj_with_mqa",
    "attn_kv_a_norm": "self_attn.kv_a_layernorm",
    "attn_k_b": "self_attn.k_b_proj", "attn_v_b": "self_attn.v_b_proj",
    "attn_output": "self_attn.o_proj",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
}

_LAYER = {
    "post_attention_norm": "post_attention_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "layer_output_scale": "layer_scalar",
    "post_ffw_norm_1": "post_feedforward_layernorm_1",     # gemma4 MoE
    "post_ffw_norm_2": "post_feedforward_layernorm_2",
    "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "ssm_in": "mamba.in_proj", "ssm_out": "mamba.out_proj",
    "ssm_conv1d": "mamba.conv1d", "ssm_norm": "mamba.norm",
    "ssm_d": "mamba.D", "ssm_dt": "mamba.dt_bias",
    "shortconv.in_proj": "conv.in_proj", "shortconv.conv": "conv.conv",
    "shortconv.out_proj": "conv.out_proj",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate_inp": "mlp.gate",
    "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
}

_EXPS = {"ffn_gate_up_exps": "mlp.experts.{e}.gate_up_proj",   # gemma4 : [E, 2·inter, h]
         "ffn_gate_exps": "mlp.experts.{e}.gate_proj",
         "ffn_up_exps": "mlp.experts.{e}.up_proj",
         "ffn_down_exps": "mlp.experts.{e}.down_proj"}


def _map_name(g: str, gdn: bool = False,
              kimi_recurrent: Optional[set] = None,
              phi3: bool = False) -> Optional[str]:
    if g in _DIRECT:
        return _DIRECT[g]
    if not g.startswith("blk."):
        return None                        # rope_freqs et autres auxiliaires
    _, idx, rest = g.split(".", 2)
    stem, _, kind = rest.rpartition(".")   # "attn_q", "weight"|"bias"
    if phi3:
        if stem == "attn_qkv":
            return f"model.layers.{idx}.self_attn.qkv_proj.{kind}"
        if stem == "ffn_up":
            return f"model.layers.{idx}.mlp.gate_up_proj.{kind}"
    if stem == "ffn_gate_inp" and kind == "scale":       # gemma4 : échelle du routeur [h]
        return f"model.layers.{idx}.mlp.router_scale.weight"
    if stem == "ffn_down_exps" and kind == "scale":      # gemma4 : échelle par expert [E]
        return f"model.layers.{idx}.mlp.per_expert_scale.weight"
    if kimi_recurrent is None and rest == "ssm_a" and not gdn:
        return f"model.layers.{idx}.mamba.A.weight"          # nemotron_h, sans suffixe
    if kimi_recurrent is None and rest == "ssm_d" and not gdn:
        return f"model.layers.{idx}.mamba.D.weight"
    if rest == "ffn_norm":                       # falcon-h1 : sans suffixe
        return f"model.layers.{idx}.post_attention_layernorm.weight"
    if kimi_recurrent is None and stem == "ssm_dt" and not gdn:
        return f"model.layers.{idx}.mamba.dt_bias.weight"       # rangé en .bias dans le GGUF
    if kimi_recurrent is not None:
        if rest == "ssm_a":
            return f"model.layers.{idx}.linear_attn.a.weight"
        if stem == "ssm_dt":
            return f"model.layers.{idx}.linear_attn.dt_bias.weight"
        if stem == "exp_probs_b":
            return f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
        table = _LAYER_KDA if int(idx) in kimi_recurrent else _LAYER_MLA
        if stem in table:
            return f"model.layers.{idx}.{table[stem]}.{kind}"
        if stem in _LAYER:                 # ffn_* et normes partagées
            return f"model.layers.{idx}.{_LAYER[stem]}.{kind}"
        if stem in _EXPS:
            return f"model.layers.{idx}.{_EXPS[stem]}.{kind}__exps__"
        return None
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
