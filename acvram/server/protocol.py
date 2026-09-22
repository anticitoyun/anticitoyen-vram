"""Schémas de requête et de réponse compatibles OpenAI.

Les noms et les formes des champs suivent l'API HTTP d'OpenAI, pour que les
clients existants — le SDK openai, LangChain, Open WebUI, Continue, de simples
scripts curl — fonctionnent avec ce serveur sans modification. C'est aussi la
raison pour laquelle ces noms restent en anglais : ils constituent un protocole,
pas de la prose. Les paramètres que le moteur ne sait pas honorer sont acceptés
et ignorés plutôt que rejetés, car les clients envoient couramment le jeu
complet quel que soit le serveur.
"""

from __future__ import annotations

import base64
import os
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChatMessage", "ChatCompletionRequest", "ChatCompletionResponse",
    "ChatCompletionChunk", "CompletionRequest", "CompletionResponse",
    "EmbeddingRequest", "EmbeddingResponse", "ModelList", "ModelCard",
    "Usage", "ErrorResponse", "new_id",
    "ImageRefusee", "charger_image", "VAR_DOSSIER_IMAGES",
]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"] = "user"
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def text(self) -> str:
        """Aplatit le contenu, qui peut être une chaîne ou une liste de fragments."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for part in self.content:
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)

    def images(self) -> list[str]:
        """Les URL des fragments ``image_url`` dans l'ordre ; [] pour un
        contenu chaîne. Un fragment ``image_url`` sans ``url`` est refusé,
        pas ignoré : un client qui envoie une image doit savoir ce qu'elle
        devient (trois états, jamais un silence)."""
        if not isinstance(self.content, list):
            return []
        urls = []
        for part in self.content:
            if part.get("type") != "image_url":
                continue
            iu = part.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if not isinstance(url, str) or not url:
                raise ImageRefusee("fragment image_url sans url")
            urls.append(url)
        return urls

    def fragments(self) -> list[dict[str, Any]]:
        """Le contenu sous la forme que les gabarits multimodaux HF lisent :
        ``{"type": "text", "text": …}`` et ``{"type": "image"}`` dans l'ordre."""
        if self.content is None:
            return []
        if isinstance(self.content, str):
            return [{"type": "text", "text": self.content}]
        out = []
        for part in self.content:
            t = part.get("type")
            if t == "text":
                out.append({"type": "text", "text": part.get("text", "")})
            elif t == "image_url":
                out.append({"type": "image"})
        return out


# -- sources d'image ----------------------------------------------------------
# Trois sources, chacune bornée, et un refus qui dit lequel des trois a
# échoué : `data:` (octets dans la requête), `file://` sous le seul dossier
# ACVRAM_IMAGES_DIR (chemin résolu par realpath, `..` refusé avant résolution),
# `http(s)://` vers 127.0.0.1 ou localhost seulement — jamais une origine
# distante depuis le serveur.
VAR_DOSSIER_IMAGES = "ACVRAM_IMAGES_DIR"
_HOTES_LOCAUX = {"127.0.0.1", "localhost", "::1", "[::1]"}
_MAX_OCTETS = 32 * 1024 * 1024


class ImageRefusee(ValueError):
    """Refus nommé d'une image : rendu en HTTP 400 par le serveur."""


def _refus(source: str, pourquoi: str) -> ImageRefusee:
    return ImageRefusee(f"image refusée ({source}) : {pourquoi}")


def _depuis_data(url: str) -> bytes:
    tete, sep, corps = url.partition(",")
    if not sep:
        raise _refus("data", "pas de virgule entre l'en-tête et les données")
    if not tete[5:].startswith("image/"):
        raise _refus("data", f"type MIME « {tete[5:].split(';')[0]} » : image/* attendu")
    if not tete.endswith(";base64"):
        raise _refus("data", "seul l'encodage base64 est lu")
    try:
        octets = base64.b64decode(corps, validate=True)
    except (ValueError, TypeError) as exc:
        raise _refus("data", f"base64 illisible : {exc}") from None
    if not octets:
        raise _refus("data", "image vide")
    if len(octets) > _MAX_OCTETS:
        raise _refus("data", f"{len(octets)} octets, plus de {_MAX_OCTETS}")
    return octets


def _depuis_fichier(url: str) -> bytes:
    dossier = os.environ.get(VAR_DOSSIER_IMAGES, "")
    if not dossier:
        raise _refus("file", f"{VAR_DOSSIER_IMAGES} n'est pas défini : aucun dossier autorisé")
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in ("", "localhost"):
        raise _refus("file", f"hôte « {parsed.netloc} » : seul un chemin local est lu")
    chemin = urllib.parse.unquote(parsed.path)
    if ".." in chemin.split("/"):
        raise _refus("file", "« .. » dans le chemin")
    racine = os.path.realpath(dossier)
    reel = os.path.realpath(chemin)
    if os.path.commonpath([racine, reel]) != racine:
        raise _refus("file", f"{chemin} hors du dossier autorisé {VAR_DOSSIER_IMAGES}={dossier}")
    if not os.path.isfile(reel):
        raise _refus("file", f"{chemin} n'est pas un fichier")
    if os.path.getsize(reel) > _MAX_OCTETS:
        raise _refus("file", f"{chemin} : plus de {_MAX_OCTETS} octets")
    with open(reel, "rb") as fh:
        return fh.read()


def _depuis_http(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname not in _HOTES_LOCAUX:
        raise _refus("http", f"hôte « {parsed.hostname} » : seuls 127.0.0.1 et localhost sont lus")
    try:
        with urllib.request.urlopen(url, timeout=10) as rep:   # noqa: S310 — hôte local vérifié
            octets = rep.read(_MAX_OCTETS + 1)
    except Exception as exc:                                    # noqa: BLE001
        raise _refus("http", f"{url} : {exc}") from None
    if not octets:
        raise _refus("http", f"{url} : réponse vide")
    if len(octets) > _MAX_OCTETS:
        raise _refus("http", f"{url} : plus de {_MAX_OCTETS} octets")
    return octets


def charger_image(url: str) -> bytes:
    """Les octets d'une image d'après son URL, ou ``ImageRefusee`` nommée."""
    if url.startswith("data:"):
        return _depuis_data(url)
    if url.startswith("file://"):
        return _depuis_fichier(url)
    if url.startswith(("http://", "https://")):
        return _depuis_http(url)
    schema = url.split(":", 1)[0] if ":" in url else "(aucun)"
    raise _refus("url", f"schéma « {schema} » : data:, file:// ou http(s)://127.0.0.1 attendu")


class _SamplingFields(BaseModel):
    # Un champ inconnu (faute de frappe « temprature », option d'un autre
    # serveur « reasoning_effort ») était accepté et IGNORÉ en silence : le
    # client croyait régler quelque chose, le moteur servait le défaut, et rien
    # ne le disait (MECANISMES : « champs inconnus acceptés en silence »). On
    # les garde (`extra="allow"`) pour les NOMMER : `champs_inconnus()` les rend,
    # le serveur les journalise une fois par nom. Le contrat OpenAI est
    # inchangé : la requête passe, les champs connus gardent leur effet.
    model_config = ConfigDict(extra="allow")

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    n: int = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    logprobs: Optional[Union[bool, int]] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    # Même nom et sémantique que vLLM/llama-server (sage-harnais-egal-
    # ignore-eos-18-09) : ignore l'EOS, continue jusqu'à `max_tokens`. Sans
    # équivalent HTTP ici, un harnais comparatif qui force `--ignore-eos`
    # côté llama.cpp fait tomber le lot acvram sous b pendant la fenêtre —
    # asymétrie non neutre, pas un réglage de confort.
    ignore_eos: bool = False

    def champs_inconnus(self) -> list[str]:
        """Noms des champs reçus que ce serveur ne lit pas, triés ; [] sinon."""
        return sorted((self.model_extra or {}).keys())

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    def token_budget(self, default: int = 512) -> int:
        return self.max_completion_tokens or self.max_tokens or default


class ChatCompletionRequest(_SamplingFields):
    model: str
    messages: list[ChatMessage]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    response_format: Optional[dict[str, Any]] = None
    stream_options: Optional[dict[str, Any]] = None
    add_generation_prompt: bool = True
    # Variables passées au gabarit Jinja, comme chez vLLM et llama.cpp :
    # {"enable_thinking": false} coupe la réflexion d'un Qwen3.
    chat_template_kwargs: Optional[dict[str, Any]] = None


class CompletionRequest(_SamplingFields):
    model: str
    prompt: Union[str, list[str], list[int], list[list[int]]]
    echo: bool = False
    suffix: Optional[str] = None
    best_of: Optional[int] = None
    # Même option que côté chat : {"include_usage": true} fait porter le
    # décompte de jetons par le dernier morceau du flux.
    stream_options: Optional[dict[str, Any]] = None


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = ""
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatChoice] = []
    usage: Usage = Field(default_factory=Usage)


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChunkChoice] = []
    usage: Optional[Usage] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None
    logprobs: Optional[dict[str, Any]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionChoice] = []
    usage: Usage = Field(default_factory=Usage)


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str], list[int], list[list[int]]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: Union[list[float], str] = []


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData] = []
    model: str = ""
    usage: Usage = Field(default_factory=Usage)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "acvram"
    # hors norme, mais assez utile pour mériter d'être exposé
    acvram: Optional[dict[str, Any]] = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []


class ErrorResponse(BaseModel):
    error: dict[str, Any]

    @staticmethod
    def make(message: str, err_type: str = "invalid_request_error",
             code: Optional[str] = None) -> "ErrorResponse":
        return ErrorResponse(error={"message": message, "type": err_type,
                                    "param": None, "code": code})
