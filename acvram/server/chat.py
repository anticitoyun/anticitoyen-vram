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
import threading
import uuid
from typing import Any, Optional

__all__ = ["Tokenizer", "load_tokenizer", "render_chat",
           "ProcesseurVision", "charger_processeur_vision", "preparer_images"]


_VERROU_GABARIT = threading.Lock()      # 272 : construction paresseuse de l'Environment jinja (voir _render_jinja)


class Tokenizer:
    """Fine enveloppe autour de ``tokenizers.Tokenizer``, avec gabarits."""

    _repli_averti = False        # averti une fois par processus (repli ChatML non muet)

    def __init__(self, backend: Any, config: dict, template: Optional[str],
                 template_source: str) -> None:
        self.backend = backend
        self.config = config
        self.template = template
        self.template_source = template_source
        self._env = None
        # gabarit réellement appliqué au rendu, lisible pour la ligne de régime :
        # "jinja" (gabarit du modèle), "chatml" (pas de gabarit → ChatML natif),
        # "chatml-repli" (jinja présent mais illisible → repli, cf. render_chat).
        self.gabarit_effectif = "jinja" if template else "chatml"

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
        pre = self.encode(prefixe, add_special_tokens=False)
        # Garde (poste7) : n'injecter que si le littéral s'encode ENTIÈREMENT en
        # jetons spéciaux (vocabulaire ajouté) — un gabarit qui commence par
        # du texte ordinaire n'est pas un préfixe de modèle.
        if not self._que_des_speciaux(pre):
            return ids
        return pre + ids

    def _que_des_speciaux(self, ids: list[int]) -> bool:
        getter = getattr(self.backend, "get_added_tokens_decoder", None)
        if getter is None or not ids:
            return False
        try:
            speciaux = set(getter().keys())
        except Exception:                        # noqa: BLE001
            return False
        return all(i in speciaux for i in ids)

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

    def bos_id(self) -> Optional[int]:
        """L identifiant du BOS que le GABARIT pose en tête de toute conversation
        (`{{ bos_token }}` chez Gemma, encodé comme jeton spécial), ou None :
        le post-traitement de tokenizer.json ne le pose PAS toujours
        (Gemma 4 : `TemplateProcessing` sans jeton spécial), `encode(...,
        add_special_tokens=True)` rend alors une suite sans BOS — et un modèle
        entraîné avec BOS s effondre sans lui (22/09, `acvram eval` 31B : PPL
        936 au lieu de 10-20, `verdict-ppl-31b-ab-v2`)."""
        bos = self.bos_token
        if not bos:
            return None
        ident = self.backend.token_to_id(bos)
        return int(ident) if ident is not None else None

    # -- chat -------------------------------------------------------------
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool,
                            extra: Optional[dict] = None) -> str:
        if self.template:
            try:
                r = self._render_jinja(messages, add_generation_prompt, extra)
                self.gabarit_effectif = "jinja"
                return r
            except Exception as exc:                 # noqa: BLE001
                # Repli ChatML : sortie inchangée, mais NON muet (repli silencieux
                # trouvé le 22/09, pièce 38). Averti une fois par processus ; le
                # gabarit effectif reste lisible via `gabarit_effectif`.
                if not Tokenizer._repli_averti:
                    Tokenizer._repli_averti = True
                    print(f"acvram: gabarit jinja illisible ({type(exc).__name__}: {exc}) "
                          "→ repli ChatML (gabarit=chatml-repli)", file=sys.stderr)
                self.gabarit_effectif = "chatml-repli"
        return _chatml(messages, add_generation_prompt)

    def _render_jinja(self, messages: list[dict], add_generation_prompt: bool,
                      extra: Optional[dict] = None) -> str:
        if self._env is None:
            # 272 : depuis la 268 (étape 2), le gabarit est rendu dans des fils — deux requêtes de la première rafale
            # pouvaient voir `_env` publié avant `_templates` et ses globales (AttributeError rattrapé → repli ChatML
            # silencieux, ou JSON en ensure_ascii). Construit en local, publié en dernier, sous verrou.
            with _VERROU_GABARIT:
                if self._env is None:
                    from jinja2 import Environment
                    from jinja2.exceptions import TemplateError

                    def raise_exception(msg: str) -> None:
                        raise TemplateError(msg)

                    env = Environment(trim_blocks=True, lstrip_blocks=True)
                    env.globals["raise_exception"] = raise_exception
                    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}

                    # iqm : le `tojson` de transformers (chat_template_utils), que vLLM sert aussi. Celui de Jinja
                    # refuse `ensure_ascii`/`separators` (GLM-4.7 : TypeError → repli ChatML SANS outils) et
                    # échappe < > & ' en \u003c… : un schéma d'outil ne se rendait pas comme chez HF.
                    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                                          separators=separators, sort_keys=sort_keys)
                    env.filters["tojson"] = tojson
                    self._templates: dict[str, object] = {}
                    self._env = env
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


def messages_pour_gabarit(messages: list, avec_images: bool = False) -> list[dict]:
    """Les messages tels que les gabarits HF les attendent : rôle et texte,
    plus ``name``, ``tool_calls`` et ``tool_call_id`` quand ils existent —
    sans eux un second tour d'outil ne se rend pas.

    ``avec_images`` : le contenu reste une liste de fragments
    (``{"type": "image"}`` / ``{"type": "text"}``), forme que les gabarits
    multimodaux lisent pour poser leur jeton d'image. Faux par défaut : le
    chemin texte rend exactement ce qu'il rendait."""
    out = []
    for m in messages:
        d = {"role": m.role,
             "content": m.fragments() if avec_images else m.text()}
        for k in ("name", "tool_calls", "tool_call_id"):
            v = getattr(m, k, None)
            if v is not None:
                d[k] = arguments_en_objet(v) if k == "tool_calls" else v
        out.append(d)
    return out


def arguments_en_objet(tool_calls: list) -> list:
    """1w1 (poste6 30/09, hypothèse poste1 sf2) : ``function.arguments`` d'un ``tool_call`` OpenAI est une CHAÎNE JSON
    (contrat de l'API) ; les gabarits HF attendent un objet — Qwen3-Coder fait ``tool_call.arguments|items`` (TypeError sur
    une chaîne → repli ChatML sans outils), d'autres ``tojson`` (la chaîne encodée deux fois). vLLM décode de même avant le
    rendu ; ``messages_anthropic`` garde déjà l'objet côté /v1/messages. Copie (la requête n'est pas modifiée) ; JSON
    invalide ou non-objet → ``{"arguments": <valeur d'origine>}``, le repli nommé d'``_entree_outil`` : rien n'est perdu,
    rien ne casse. Déjà un objet, ou forme inconnue : inchangé. Forme HF à plat (``arguments`` au premier niveau) : idem."""
    out = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            out.append(tc); continue
        tc = dict(tc)
        fn = tc.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            fn = dict(fn); fn["arguments"] = _entree_outil(fn["arguments"]); tc["function"] = fn
        elif isinstance(tc.get("arguments"), str):
            tc["arguments"] = _entree_outil(tc["arguments"])
        out.append(tc)
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


# -- API Anthropic (/v1/messages) : outils ---------------------------------------
# Claude Code ne parle que cette API ; sans `tools` rendus au gabarit ni
# `tool_use` rendus au client, il tourne sur un alias local SANS AUCUN OUTIL
# (bd sf2, edz 28/09). Même chemin que /v1/chat/completions : le gabarit HF
# décrit les outils, `extraire_appels` relit les appels générés.
def _texte_blocs(contenu: Any) -> str:
    if isinstance(contenu, str):
        return contenu
    return "".join(b.get("text", "") for b in contenu or []
                   if isinstance(b, dict) and b.get("type") == "text")


def outils_anthropic(tools: Any) -> list[dict]:
    """``tools`` Anthropic ({name, description, input_schema}) au format que
    les gabarits HF lisent (celui d'OpenAI). Les outils serveur d'Anthropic
    (``type`` web_search_…, bash_…) n'ont pas de schéma : écartés."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t.get("description", ""),
                          "parameters": t.get("input_schema")
                          or {"type": "object", "properties": {}}}}
            for t in tools or []
            if isinstance(t, dict) and isinstance(t.get("name"), str)
            and t.get("type", "custom") == "custom"]


def messages_anthropic(req: dict) -> list[dict]:
    """Corps /v1/messages -> messages du gabarit. Sans bloc d'outil, rend
    exactement ce que rendait la route avant sf2 (texte des blocs joint).

    ``tool_use`` (assistant) -> ``tool_calls`` dont les ``arguments`` restent
    un dict : les gabarits les écrivent par ``tojson`` ou ``|items``
    (Qwen3-Coder), une chaîne y serait encodée deux fois. ``tool_result``
    (user) -> un message ``tool`` par résultat, AVANT le texte du même tour :
    le gabarit attend la réponse juste après l'appel."""
    out = []
    systeme = req.get("system")
    if systeme:
        out.append({"role": "system", "content": _texte_blocs(systeme)})
    noms = {}                                        # tool_use_id -> nom d'outil
    for m in req.get("messages", []):
        role, contenu = m.get("role", "user"), m.get("content", "")
        blocs = ([b for b in contenu if isinstance(b, dict)]
                 if isinstance(contenu, list) else [])
        appels = [b for b in blocs if b.get("type") == "tool_use"]
        resultats = [b for b in blocs if b.get("type") == "tool_result"]
        for b in resultats:
            d = {"role": "tool", "content": _texte_blocs(b.get("content", "")),
                 "tool_call_id": b.get("tool_use_id")}
            if b.get("tool_use_id") in noms:
                d["name"] = noms[b["tool_use_id"]]
            out.append(d)
        texte = _texte_blocs(contenu)
        if resultats and not texte:
            continue
        d = {"role": role, "content": texte}
        if appels:
            d["tool_calls"] = [{"id": b.get("id"), "type": "function",
                                "function": {"name": b.get("name"),
                                             "arguments": b.get("input") or {}}}
                               for b in appels]
            noms.update((b.get("id"), b.get("name")) for b in appels)
        out.append(d)
    return _joindre_systemes_tardifs(out)


def _joindre_systemes_tardifs(messages: list[dict]) -> list[dict]:
    """Claude Code glisse des messages « system » dans ``messages`` (contexte des
    crochets SessionStart : 17 Ko mesurés, iqm 28/09). La plupart des gabarits
    n'acceptent le système qu'en tête — KAT lève « System message must be at the
    beginning », le rendu se replie sur ChatML et les outils DISPARAISSENT sans
    bruit. Chaque système hors tête est joint au user qui le précède (ou au
    suivant, ou devient un user), AVANT son texte : le contexte d'abord, la
    question de l'utilisateur en dernier. Pièce claude-2 (poste6, 29/09) : joint
    APRÈS le texte (0.7.13), les 17 Ko du crochet suivaient la question et
    Devstral, Nemo-12B-Claude, Qwen3-0.6B répondaient au crochet (versions,
    .deb, tests) ou répétaient la question — edz liste-2, « sans Paris »."""
    out: list[dict] = []
    attente: list[str] = []
    avant: dict[int, list[str]] = {}                 # indice dans out -> systèmes à placer avant le texte
    for i, m in enumerate(messages):
        if m["role"] == "system" and i > 0:
            if out and out[-1]["role"] == "user" and not attente:
                avant.setdefault(len(out) - 1, []).append(m["content"])
            else:
                attente.append(m["content"])
            continue
        if attente and m["role"] == "user":
            avant[len(out)] = attente
            attente = []
        out.append(m)
    if attente:
        out.append({"role": "user", "content": ""})
        avant[len(out) - 1] = attente
    for k, blocs in avant.items():
        texte = out[k]["content"]
        out[k] = {**out[k], "content": "\n\n".join(blocs + ([texte] if texte else []))}
    return out


def _entree_outil(arguments: str) -> dict:
    """``input`` d'un ``tool_use`` : toujours un objet (contrat Anthropic)."""
    try:
        v = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {"arguments": arguments}
    return v if isinstance(v, dict) else {"arguments": v}


def blocs_anthropic(texte: str, outils: bool) -> tuple[list[dict], bool]:
    """Texte généré -> blocs ``content`` Anthropic (texte puis ``tool_use``),
    et vrai s'il y a au moins un appel (stop_reason ``tool_use``). Les id sont
    tirés au hasard : ceux d'`extraire_appels` se répètent d'un tour à
    l'autre pour un appel identique, et Claude Code apparie par id."""
    if outils:
        reste, appels = extraire_appels(texte)
        if appels:
            blocs = [{"type": "text", "text": reste}] if reste else []
            blocs += [{"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}",
                       "name": a["function"]["name"],
                       "input": _entree_outil(a["function"]["arguments"])}
                      for a in appels]
            return blocs, True
    return [{"type": "text", "text": texte}], False


class FiltreAppels:
    """Flux avec outils : le texte passe au fil de l'eau, tout ce qui suit
    « <tool_call> » est retenu (rendu à la fin en blocs structurés) ; un « < »
    isolé attend, le temps de savoir s'il ouvre la balise — règle de
    `_stream_chat`. ``reste()`` rend ce qui n'est pas encore sorti quand
    aucun appel ne se lit : un bloc illisible reste du texte, jamais perdu."""

    def __init__(self) -> None:
        self.total, self.pend, self.retenu, self.emis = "", "", False, 0

    def pousser(self, delta: str) -> str:
        self.total += delta
        if self.retenu:
            return ""
        self.pend += delta
        if "<tool_call>" in self.pend:
            self.retenu = True
            sortie, self.pend = self.pend[:self.pend.index("<tool_call>")], ""
        elif "<" in self.pend and len(self.pend) < 64:
            return ""
        else:
            sortie, self.pend = self.pend, ""
        self.emis += len(sortie)
        return sortie

    def reste(self) -> str:
        sortie = self.total[self.emis:]
        self.emis = len(self.total)
        return sortie


# -- images : AutoProcessor -> requête interne ---------------------------------
class ProcesseurVision:
    """L'``AutoProcessor`` d'un dossier converti, et ce qu'il faut autour :
    ``ouvrir`` (octets -> image telle que le processeur la lit, Pillow) et
    ``image_token_id`` (le jeton que le processeur expanse en N jetons image).
    Le serveur ne le charge qu'au premier ``image_url``, jamais sur le chemin
    texte."""

    def __init__(self, processeur: Any, image_token_id: int,
                 ouvrir: Any = None) -> None:
        self.processeur = processeur
        self.image_token_id = int(image_token_id)
        self.ouvrir = ouvrir or _ouvrir_pillow

    def __call__(self, prompt: str, images: list[Any]) -> dict:
        return self.processeur(text=prompt, images=images, return_tensors="pt",
                               add_special_tokens=False)


def _ouvrir_pillow(octets: bytes) -> Any:
    import io
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("image refusée : le paquet Python Pillow (PIL) n'est pas "
                         f"installé dans cet interpréteur ({sys.executable})") from None
    try:
        img = Image.open(io.BytesIO(octets))
        img.load()
    except Exception as exc:                                    # noqa: BLE001
        raise ValueError(f"image refusée : octets illisibles comme image ({exc})") from None
    return img.convert("RGB")


def _image_token_id(proc: Any, path: str) -> Optional[int]:
    v = getattr(proc, "image_token_id", None)
    if isinstance(v, int):
        return v
    tok = getattr(proc, "tokenizer", None)
    nom = getattr(proc, "image_token", None)
    if tok is not None and isinstance(nom, str):
        try:
            i = tok.convert_tokens_to_ids(nom)
            if isinstance(i, int) and i >= 0:
                return i
        except Exception:                                       # noqa: BLE001
            pass
    cfg_path = os.path.join(path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        for k in ("image_token_id", "image_token_index"):
            if isinstance(cfg.get(k), int):
                return cfg[k]
    return None


def charger_processeur_vision(path: str) -> ProcesseurVision:
    """Charge l'``AutoProcessor`` du dossier (``processor_config.json`` /
    ``preprocessor_config.json`` copiés par la conversion). Chaque manque
    est nommé : paquet absent, fichier absent, jeton d'image introuvable."""
    try:
        from transformers import AutoProcessor
    except ImportError:
        raise ValueError("modèle avec tour de vision mais AutoProcessor indisponible : "
                         "le paquet Python transformers n'est pas installé dans cet "
                         f"interpréteur ({sys.executable})") from None
    if not any(os.path.isfile(os.path.join(path, f))
               for f in ("processor_config.json", "preprocessor_config.json")):
        raise ValueError("modèle avec tour de vision mais ni processor_config.json ni "
                         f"preprocessor_config.json dans {path}")
    try:
        proc = AutoProcessor.from_pretrained(path)
    except Exception as exc:                                    # noqa: BLE001
        raise ValueError(f"AutoProcessor illisible dans {path} : {exc}") from None
    tid = _image_token_id(proc, path)
    if tid is None:
        raise ValueError(f"jeton d'image introuvable (processeur, tokeniseur, config.json) dans {path}")
    return ProcesseurVision(proc, tid)


def _plages(ids: list[int], jeton: int) -> list[tuple[int, int]]:
    """Les plages [debut, fin) des suites contiguës de ``jeton`` dans ``ids``."""
    plages, debut = [], -1
    for i, t in enumerate(ids + [None]):
        if t == jeton and debut < 0:
            debut = i
        elif t != jeton and debut >= 0:
            plages.append((debut, i)); debut = -1
    return plages


# Annexes du processeur qui accompagnent pixel_values IMAGE PAR IMAGE, de même première dimension
# (Gemma 4 : image_position_ids [n, patches, 2] — sans elles get_image_features casse, poste2 14 h 40 ;
# Qwen3-VL : image_grid_thw). Tout ce qui n'est pas une entrée du modèle (mm_token_type_ids, input_ids,
# attention_mask) reste ici. Un nom inconnu de même forme est transmis aussi : c'est la tour qui sait.
_ANNEXES_HORS_TOUR = {"pixel_values", "input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids",
                      "pixel_values_videos", "video_grid_thw"}


def _annexes(sortie: dict, n: int) -> dict:
    """{nom: tenseur [n, …]} des annexes découpables par image (même première dimension que le nombre d'images)."""
    out = {}
    for k, v in sortie.items():
        if k in _ANNEXES_HORS_TOUR or k == "image_grid_thw" or not hasattr(v, "shape") or v.ndim == 0:
            continue
        if v.shape[0] == n:
            out[k] = v
    return out


def _par_image(sortie: dict, n: int) -> list[tuple[Any, dict]]:
    """``pixel_values`` (et annexes) découpés par image : une ligne par image
    (Gemma, forme [n, patches, d] ou [n, 3, H, W]) ou par ``image_grid_thw``
    (Qwen, patches concaténés). Sinon, refus : on ne devine pas une découpe.
    Le ``supplement`` de chaque image porte ses annexes (``image_position_ids``,
    ``image_grid_thw``…), que la tour reçoit en kwargs."""
    pv = sortie.get("pixel_values")
    if pv is None:
        raise ValueError("le processeur n'a rendu aucun pixel_values")
    annexes = _annexes(sortie, n)
    thw = sortie.get("image_grid_thw")
    if thw is not None and len(thw) == n:
        out, pos = [], 0
        for i in range(n):
            k = int(thw[i].prod()) if hasattr(thw[i], "prod") else int(
                thw[i][0] * thw[i][1] * thw[i][2])
            supp = {"image_grid_thw": thw[i:i + 1], **{a: v[i:i + 1] for a, v in annexes.items()}}
            out.append((pv[pos:pos + k], supp))
            pos += k
        if pos != pv.shape[0]:
            raise ValueError(f"image_grid_thw couvre {pos} patches, pixel_values en porte {pv.shape[0]}")
        return out
    if pv.shape[0] == n:
        return [(pv[i:i + 1], {a: v[i:i + 1] for a, v in annexes.items()}) for i in range(n)]
    if n == 1:
        return [(pv, dict(annexes))]
    raise ValueError(f"{n} images mais pixel_values de forme {tuple(pv.shape)} "
                     "sans image_grid_thw : découpe par image inconnue")


def preparer_images(pv: ProcesseurVision, prompt: str, octets: list[bytes]
                    ) -> tuple[list[int], list]:
    """La requête interne d'une invite à images : ``(prompt_ids, fragments)``.

    ``prompt_ids`` vient de l'``AutoProcessor`` (jetons image DÉJÀ expansés
    en N par image) ; ``fragments`` : un ``ImageFragment`` par image, dans
    l'ordre, avec sa plage, ses pixels et leur sha256."""
    from ..engine.images import ImageFragment

    images = [pv.ouvrir(o) for o in octets]
    sortie = pv(prompt, images)
    ids = sortie["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(t) for t in ids]
    plages = _plages(ids, pv.image_token_id)
    if len(plages) != len(images):
        raise ValueError(f"{len(images)} images envoyées, {len(plages)} plages de jetons "
                         f"image (id {pv.image_token_id}) dans l'invite rendue : le gabarit "
                         "de conversation ne pose pas un jeton d'image par image")
    fragments = []
    for (debut, fin), (pix, supp) in zip(plages, _par_image(sortie, len(images))):
        fragments.append(ImageFragment(debut, fin, pix, supplement=supp))
    return ids, fragments
