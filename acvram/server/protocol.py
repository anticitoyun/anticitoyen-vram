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

import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

__all__ = [
    "ChatMessage", "ChatCompletionRequest", "ChatCompletionResponse",
    "ChatCompletionChunk", "CompletionRequest", "CompletionResponse",
    "EmbeddingRequest", "EmbeddingResponse", "ModelList", "ModelCard",
    "Usage", "ErrorResponse", "new_id",
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


class _SamplingFields(BaseModel):
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
