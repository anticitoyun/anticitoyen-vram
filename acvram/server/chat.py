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
import sys
import re
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
    def prefixe_gabarit(self) -> str:
        """Le texte littéral par lequel le gabarit de conversation commence,
        avant sa première balise Jinja — ce que le modèle voit en tête de
        TOUTE conversation. GLM-4.x : ``[gMASK]<sop>`` (poste7 § 10, cause
        trouvée par poste3 : sans ce préfixe le modèle n'a aucun puits
        d'attention et s'effondre). Vide pour les gabarits qui commencent par
        une balise (Llama, Qwen : leur préfixe est le BOS du tokeniseur)."""
        t = self.template or ""
        i = min([k for k in (t.find("{{"), t.find("{%"), t.find("{#")) if k >= 0] or [len(t)])
        return t[:i].strip()

    def encode_brut(self, text: str) -> list[int]:
        """L'invite d'une complétion BRUTE (/v1/completions, plongements) :
        les jetons spéciaux du tokeniseur (BOS des modèles qui en posent un
        par post-traitement), puis, si rien n'a été posé, le préfixe littéral
        du gabarit — sauf si le texte le porte déjà. Le chemin conversation
        n'en a pas besoin : son gabarit rendu contient déjà ce préfixe."""
        ids = self.encode(text, add_special_tokens=True)
        sans = self.encode(text, add_special_tokens=False)
        prefixe = self.prefixe_gabarit
        if not prefixe or text.lstrip().startswith(prefixe) or len(ids) > len(sans):
            return ids
        return self.encode(prefixe, add_special_tokens=False) + ids

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
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool,
                            extra: Optional[dict] = None) -> str:
        if self.template:
            try:
                return self._render_jinja(messages, add_generation_prompt, extra)
            except Exception:                        # noqa: BLE001
                pass
        return _chatml(messages, add_generation_prompt)

    def _render_jinja(self, messages: list[dict], add_generation_prompt: bool,
                      extra: Optional[dict] = None) -> str:
        if self._env is None:
            from jinja2 import Environment
            from jinja2.exceptions import TemplateError

            def raise_exception(msg: str) -> None:
                raise TemplateError(msg)

            self._env = Environment(trim_blocks=True, lstrip_blocks=True)
            self._templates: dict[str, object] = {}
            self._env.globals["raise_exception"] = raise_exception
            self._env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        # Le Template COMPILE est mis en cache, pas seulement l'Environment.
        # Mesure du 9/09 sur le gabarit de Qwen2.5 (2507 caracteres) :
        #     from_string (compilation)  3,975 ms
        #     render seul                0,007 ms   -> facteur 570
        # Recompiler a chaque requete coutait donc 3,975 ms, soit 3,7 % des
        # 108 ms hors forward du TTFT. llama.cpp compile UNE FOIS au chargement
        # (common_chat_templates_init, common/chat.cpp:591) et ne reparse
        # jamais ensuite.
        #
        # Le cache est indexe par le TEXTE du gabarit : une requete qui fournit
        # le sien (parametre `chat_template`) ne se voit pas servir celui d'une
        # autre, et un modele recharge avec un gabarit different recompile.
        tmpl = self._templates.get(self.template)
        if tmpl is None:
            tmpl = self._env.from_string(self.template)
            self._templates[self.template] = tmpl
        return tmpl.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token,
            eos_token=self.config.get("eos_token", "") or "",
            **{k: v for k, v in self.config.items()
               if isinstance(v, (str, int, float, bool))
               and k not in ("messages", "add_generation_prompt",
                             "bos_token", "eos_token")},
            # les variables de la requête priment sur celles de la config
            **{k: v for k, v in (extra or {}).items()
               if k not in ("messages", "add_generation_prompt")},
        )


def _chatml(messages: list[dict], add_generation_prompt: bool) -> str:
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m.get('role','user')}\n"
                   f"{m.get('content','')}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def pourquoi_pas_de_tokenizer(path: str) -> Optional[str]:
    """La raison pour laquelle ``load_tokenizer`` rendrait ``None``, ou None.

    DEUX CAUSES, ET UN SEUL RETOUR. `load_tokenizer` rend `None` quand le
    paquet `tokenizers` manque ET quand le fichier manque ; ses appelants
    n'annoncaient que la seconde. Le 10/09 cette confusion a coute deux manches
    et une conclusion fausse transmise au circuit — « aucune perplexite n'est
    mesurable sur les convertis issus de GGUF » — alors que les fichiers
    etaient la et que seul l'interpreteur employe n'avait pas le paquet. Un
    message a cause unique fait chercher la ou il n'y a rien.
    """
    try:
        import tokenizers                                  # noqa: F401
    except ImportError:
        return ("le paquet Python `tokenizers` n'est pas installe dans cet "
                f"interpreteur ({sys.executable}) ; le projet utilise "
                "`.venv/bin/python`, qui le porte")
    tok_path = os.path.join(path, "tokenizer.json")
    if not os.path.isfile(tok_path):
        return f"pas de fichier tokenizer.json dans {path}"
    return None


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
                add_generation_prompt: bool = True,
                extra: Optional[dict] = None) -> str:
    if tokenizer is None:
        return _chatml(messages, add_generation_prompt)
    return tokenizer.apply_chat_template(messages, add_generation_prompt, extra)


# -- appels d'outils ---------------------------------------------------------
_APPEL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# Qwen3-Coder : <tool_call><function=nom><parameter=clé>valeur</parameter>…</function></tool_call>
_APPEL_XML = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_XML = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def _valeur(v: str):
    """Une valeur de paramètre XML : nombre, booléen, JSON ou texte tel quel."""
    v = v.strip()
    for essai in (v, v.lower()):
        if essai in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[essai]
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def messages_pour_gabarit(messages: list) -> list[dict]:
    """Les messages tels que les gabarits HF les attendent : rôle et texte,
    plus ``name``, ``tool_calls`` et ``tool_call_id`` quand ils existent —
    sans eux un second tour d'outil ne se rend pas."""
    out = []
    for m in messages:
        d = {"role": m.role, "content": m.text()}
        for k in ("name", "tool_calls", "tool_call_id"):
            v = getattr(m, k, None)
            if v is not None:
                d[k] = v
        out.append(d)
    return out


def extraire_appels(texte: str) -> tuple[str, list[dict]]:
    """Sépare les ``<tool_call>{json}</tool_call>`` du texte (Qwen3, Nemotron,
    Hermes, KAT…). Rend (texte restant, appels au format OpenAI).

    Un bloc dont le JSON ne se lit pas reste dans le texte : mieux vaut un
    appel manqué qu'un appel inventé."""
    trouves = []                                     # (début, fin, nom, args)
    for m in _APPEL.finditer(texte):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj.get("name"), str):
            trouves.append((m.start(), m.end(), obj["name"],
                            obj.get("arguments", obj.get("parameters", {}))))
    for m in _APPEL_XML.finditer(texte):
        params = {k: _valeur(v) for k, v in _PARAM_XML.findall(m.group(2))}
        trouves.append((m.start(), m.end(), m.group(1), params))
    trouves.sort()
    appels, garder, pos = [], [], 0
    for debut, fin, nom, args in trouves:
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        appels.append({"id": f"call_{len(appels)}_{abs(hash(texte[debut:fin])) % 10**8:08d}",
                       "type": "function",
                       "function": {"name": nom, "arguments": args}})
        garder.append(texte[pos:debut]); pos = fin
    garder.append(texte[pos:])
    reste = "".join(garder).strip() if appels else texte
    return reste, appels
