"""Construction de l'invite : tokeniseur et gabarits de conversation.

Les gabarits de conversation voyagent dans le point de contrôle sous forme de
source Jinja : l'invite correcte pour un modèle donné est donc celle que produit
son propre gabarit. Quand Jinja est disponible, ce gabarit est utilisé tel quel.
Quand il ne l'est pas — ou que le point de contrôle n'en porte aucun — un repli
au format ChatML garde le serveur utilisable, et le dit, plutôt que de produire
en silence une invite taillée pour un autre modèle.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

__all__ = ["Tokenizer", "load_tokenizer", "render_chat"]


class Tokenizer:
    """Fine enveloppe autour de ``tokenizers.Tokenizer``, avec gabarits."""

    def __init__(self, backend: Any, config: dict, template: Optional[str],
                 template_source: str) -> None:
        self.backend = backend
        self.config = config
        self.template = template
        self.template_source = template_source
        self._env = None

    # -- encode / decode --------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def eos_token_id(self) -> Optional[int]:
        eos = self.config.get("eos_token")
        if isinstance(eos, dict):
            eos = eos.get("content")
        if isinstance(eos, str):
            return self.backend.token_to_id(eos)
        return None

    @property
    def bos_token(self) -> str:
        bos = self.config.get("bos_token")
        if isinstance(bos, dict):
            bos = bos.get("content")
        return bos or ""

    # -- chat -------------------------------------------------------------
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool
                            ) -> str:
        if self.template:
            try:
                return self._render_jinja(messages, add_generation_prompt)
            except Exception:                        # noqa: BLE001
                pass
        return _chatml(messages, add_generation_prompt)

    def _render_jinja(self, messages: list[dict], add_generation_prompt: bool) -> str:
        if self._env is None:
            from jinja2 import Environment
            from jinja2.exceptions import TemplateError

            def raise_exception(msg: str) -> None:
                raise TemplateError(msg)

            self._env = Environment(trim_blocks=True, lstrip_blocks=True)
            self._env.globals["raise_exception"] = raise_exception
            self._env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        tmpl = self._env.from_string(self.template)
        return tmpl.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token,
            eos_token=self.config.get("eos_token", "") or "",
            **{k: v for k, v in self.config.items()
               if isinstance(v, (str, int, float, bool))},
        )


def _chatml(messages: list[dict], add_generation_prompt: bool) -> str:
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m.get('role','user')}\n"
                   f"{m.get('content','')}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def load_tokenizer(path: str) -> Optional[Tokenizer]:
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError:
        return None
    tok_path = os.path.join(path, "tokenizer.json")
    if not os.path.isfile(tok_path):
        return None
    backend = HFTokenizer.from_file(tok_path)

    config: dict = {}
    cfg_path = os.path.join(path, "tokenizer_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

    template = None
    source = "chatml-fallback"
    jinja_path = os.path.join(path, "chat_template.jinja")
    if os.path.isfile(jinja_path):
        with open(jinja_path, "r", encoding="utf-8") as fh:
            template = fh.read()
        source = "chat_template.jinja"
    elif isinstance(config.get("chat_template"), str):
        template = config["chat_template"]
        source = "tokenizer_config.json"
    elif isinstance(config.get("chat_template"), list):
        for entry in config["chat_template"]:
            if entry.get("name") == "default":
                template = entry.get("template")
                source = "tokenizer_config.json[default]"
                break
    return Tokenizer(backend, config, template, source)


def render_chat(tokenizer: Optional[Tokenizer], messages: list[dict],
                add_generation_prompt: bool = True) -> str:
    if tokenizer is None:
        return _chatml(messages, add_generation_prompt)
    return tokenizer.apply_chat_template(messages, add_generation_prompt)
