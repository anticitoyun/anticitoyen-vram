"""Le serveur HTTP compatible avec l'API OpenAI.

Le moteur est monothread et synchrone ; l'API est asynchrone et concurrente. La
jonction entre les deux est un unique fil d'arrière-plan qui fait tourner la
boucle d'étapes du moteur et pousse les sorties dans des files asyncio propres à
chaque requête, via ``loop.call_soon_threadsafe``. Les requêtes ne touchent
jamais le modèle directement, et c'est ce qui permet à une douzaine de clients
en flux de partager un seul pipeline réparti sur deux GPU et la mémoire vive.
"""

from __future__ import annotations

import asyncio
import os
import json
import threading
import time
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .. import __version__
from .console import PAGE
from ..engine.runner import Engine, GenerationOutput
from ..engine.sampler import SamplingParams
from .chat import Tokenizer, render_chat
from .chat import extraire_appels, messages_pour_gabarit
from .protocol import (ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
                       ChatCompletionResponse, ChoiceMessage, ChunkChoice,
                       CompletionChoice, CompletionRequest, CompletionResponse,
                       DeltaMessage, EmbeddingData, EmbeddingRequest,
                       EmbeddingResponse, ErrorResponse, ModelCard, ModelList,
                       Usage, new_id)

__all__ = ["create_app", "EngineService"]


class EngineService:
    """Anime le moteur depuis un fil d'arrière-plan et redistribue les résultats."""

    def __init__(self, engine: Engine, tokenizer: Optional[Tokenizer],
                 model_name: str) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._queues: dict[str, asyncio.Queue] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._loop = loop
            self._thread = threading.Thread(target=self._run,
                                            name="acvram-engine", daemon=True)
            self._thread.start()

    def ensure_started(self) -> None:
        """Démarre à la première requête si le crochet de cycle de vie n'a jamais
        été déclenché.

        Certains hôtes ASGI — et tout harnais de test qui instancie
        l'application sans entrer dans son cycle de vie — sautent les événements
        de démarrage. Un serveur dont le fil moteur ne démarre jamais accepte les
        requêtes puis reste bloqué indéfiniment, ce qui est une panne bien pire
        que de démarrer paresseusement un fil.
        """
        if self._thread is None:
            self.start(asyncio.get_running_loop())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        trace = bool(os.environ.get("ACVRAM_TRACE_STEPS"))
        durees: list[float] = []
        if trace:
            import gc
            debut = {}

            def _gc_cb(phase: str, info: dict) -> None:
                if phase == "start":
                    debut["t"] = time.perf_counter()
                else:
                    d = (time.perf_counter() - debut.get("t", time.perf_counter())) * 1000
                    if d > 5:
                        print(f"[gc] gen{info.get('generation')} {d:.1f} ms "
                              f"({info.get('collected')} objets)", flush=True)
            gc.callbacks.append(_gc_cb)
        while not self._stop.is_set():
            if self.engine.idle:
                time.sleep(0.002)
                continue
            try:
                t0 = time.perf_counter()
                outputs = self.engine.step()
                if trace:
                    durees.append((time.perf_counter() - t0) * 1000)
                    if len(durees) >= 100:
                        durees.sort()
                        print(f"[pas] med {durees[50]:.1f} p90 {durees[90]:.1f} "
                              f"max {durees[-1]:.1f} ms", flush=True)
                        durees.clear()
            except Exception as exc:                 # noqa: BLE001
                self._broadcast_error(exc)
                continue
            t1 = time.perf_counter()
            for out in outputs:
                self._deliver(out)
            if trace and (time.perf_counter() - t1) * 1000 > 2:
                print(f"[livraison] {(time.perf_counter() - t1)*1000:.1f} ms", flush=True)

    def _deliver(self, out: GenerationOutput) -> None:
        with self._lock:
            q = self._queues.get(out.request_id)
        if q is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(q.put_nowait, out)

    def _broadcast_error(self, exc: Exception) -> None:
        with self._lock:
            queues = list(self._queues.values())
        if self._loop is None:
            return
        for q in queues:
            self._loop.call_soon_threadsafe(q.put_nowait, exc)

    # -- request lifecycle -------------------------------------------------
    async def submit(self, prompt_ids: list[int], params: SamplingParams
                     ) -> tuple[str, asyncio.Queue]:
        self.ensure_started()
        request_id = new_id("req")
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._queues[request_id] = q
        self.engine.add_request(prompt_ids, params, request_id)
        return request_id, q

    def release(self, request_id: str) -> None:
        with self._lock:
            self._queues.pop(request_id, None)

    async def collect(self, request_id: str, q: asyncio.Queue
                      ) -> AsyncIterator[GenerationOutput]:
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            self.release(request_id)


def _params_from(req: Any, default_max: int) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=req.min_p,
        repetition_penalty=req.repetition_penalty,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        max_tokens=req.token_budget(default_max),
        stop=req.stop_list(),
        seed=req.seed,
        n=req.n,
    )


def create_app(engine: Engine, tokenizer: Optional[Tokenizer],
               model_name: str, served_paths: Optional[dict] = None) -> FastAPI:
    service = EngineService(engine, tokenizer, model_name)
    app = FastAPI(title="acvram", version="0.1.0",
                  description="anticitoyen VRAM/RAM — inférence compatible OpenAI")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    app.state.service = service
    app.state.info = served_paths or {}

    @app.on_event("startup")
    async def _startup() -> None:
        service.start(asyncio.get_running_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        service.stop()

    # -- console ----------------------------------------------------------
    # Le paquet s'installait sans rien de visible : ni entree de menu, ni
    # icone, ni page. Un serveur qui ne repond qu'a des clients OpenAI est
    # muet pour qui vient de l'installer et veut verifier qu'il tourne.
    # La console ne recalcule rien : elle affiche ce que /metrics rend.
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def console() -> str:
        return PAGE

    # -- introspection ----------------------------------------------------
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": model_name}

    @app.get("/metrics")
    async def metrics() -> dict:
        # `version` sert a la console, qui l'affiche en tete : sans elle on ne
        # sait pas quelle version repond, et deux versions ont deja coexiste
        # sur cette machine (paquet 0.5.0, venv 0.2.0).
        return {"engine": engine.stats.to_dict(),
                "version": __version__, **app.state.info}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        plan = engine.loaded.plan
        return ModelList(data=[ModelCard(
            id=model_name,
            acvram={
                "formats": sorted({lp.fmt for lp in plan.layers}),
                "devices": sorted({lp.exec_device for lp in plan.layers}),
                "max_model_len": engine.max_model_len,
                "kv_tokens": plan.kv_max_tokens,
                "chat_template": tokenizer.template_source if tokenizer else None,
            })])

    # -- chat -------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw: Request):
        messages = messages_pour_gabarit(req.messages)
        extra = dict(req.chat_template_kwargs or {})
        if req.tools:
            extra["tools"] = req.tools          # les gabarits HF les rendent eux-mêmes
        prompt = render_chat(tokenizer, messages, req.add_generation_prompt, extra)
        prompt_ids = _encode(tokenizer, prompt)
        params = _params_from(req, 512)
        request_id, q = await service.submit(prompt_ids, params)

        if req.stream:
            return StreamingResponse(
                _stream_chat(service, request_id, q, model_name, len(prompt_ids),
                             bool((req.stream_options or {}).get("include_usage")),
                             bool(req.tools)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        text, reason, n_out = "", "stop", 0
        async for out in service.collect(request_id, q):
            text += out.text_delta
            n_out = out.completion_tokens
            if out.finished:
                reason = out.finish_reason or "stop"
        return ChatCompletionResponse(
            model=model_name,
            choices=[ChatChoice(message=_message_finale(text, bool(req.tools)),
                                finish_reason=_raison_finale(text, reason, bool(req.tools)))],
            usage=Usage(prompt_tokens=len(prompt_ids), completion_tokens=n_out,
                        total_tokens=len(prompt_ids) + n_out))


    # -- API Anthropic (/v1/messages) --------------------------------------
    # Claude Code parle cette API-là, pas celle d'OpenAI : l'exposer permet
    # aux menus locaux de brancher Claude directement sur ce serveur. Champs
    # couverts : system, messages (texte ou blocs), stop_sequences,
    # temperature/top_p/top_k, stream (evenements message_start,
    # content_block_delta, message_delta, message_stop).
    @app.post("/v1/messages")
    async def anthropic_messages(raw: Request):
        req = await raw.json()
        messages = []
        sys_prompt = req.get("system")
        if sys_prompt:
            if isinstance(sys_prompt, list):
                sys_prompt = "".join(b.get("text", "") for b in sys_prompt)
            messages.append({"role": "system", "content": sys_prompt})
        for m in req.get("messages", []):
            contenu = m.get("content", "")
            if isinstance(contenu, list):
                contenu = "".join(b.get("text", "") for b in contenu
                                  if isinstance(b, dict)
                                  and b.get("type") == "text")
            messages.append({"role": m.get("role", "user"),
                             "content": contenu})
        prompt = render_chat(tokenizer, messages, True)
        prompt_ids = _encode(tokenizer, prompt)
        params = SamplingParams(
            temperature=float(req.get("temperature", 1.0)),
            top_p=float(req.get("top_p", 1.0)),
            top_k=int(req.get("top_k", 0) or 0),
            max_tokens=int(req.get("max_tokens", 512)),
            stop=list(req.get("stop_sequences") or []),
        )
        request_id, q = await service.submit(prompt_ids, params)
        mid = new_id("msg")

        def stop_reason(r: str) -> str:
            return {"stop": "end_turn", "length": "max_tokens"}.get(r, "end_turn")

        if req.get("stream"):
            async def flux():
                def ev(nom: str, data: dict) -> str:
                    return (f"event: {nom}\n"
                            f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
                yield ev("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant",
                    "content": [], "model": model_name, "stop_reason": None,
                    "usage": {"input_tokens": len(prompt_ids),
                              "output_tokens": 0}}})
                yield ev("content_block_start",
                         {"type": "content_block_start", "index": 0,
                          "content_block": {"type": "text", "text": ""}})
                n_out, raison = 0, "end_turn"
                async for out in service.collect(request_id, q):
                    n_out = out.completion_tokens
                    if out.text_delta:
                        yield ev("content_block_delta",
                                 {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta",
                                            "text": out.text_delta}})
                    if out.finished:
                        raison = stop_reason(out.finish_reason or "stop")
                yield ev("content_block_stop",
                         {"type": "content_block_stop", "index": 0})
                yield ev("message_delta", {"type": "message_delta",
                         "delta": {"stop_reason": raison,
                                   "stop_sequence": None},
                         "usage": {"output_tokens": n_out}})
                yield ev("message_stop", {"type": "message_stop"})
            return StreamingResponse(
                flux(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache",
                         "X-Accel-Buffering": "no"})

        text, raison, n_out = "", "end_turn", 0
        async for out in service.collect(request_id, q):
            text += out.text_delta
            n_out = out.completion_tokens
            if out.finished:
                raison = stop_reason(out.finish_reason or "stop")
        return {"id": mid, "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": model_name, "stop_reason": raison,
                "stop_sequence": None,
                "usage": {"input_tokens": len(prompt_ids),
                          "output_tokens": n_out}}

    # -- legacy completions ------------------------------------------------
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt_ids = list(prompt)
            prompt_text = ""
        else:
            prompt_text = prompt[0] if isinstance(prompt, list) else prompt
            prompt_ids = _encode(tokenizer, str(prompt_text))
        params = _params_from(req, 256)
        request_id, q = await service.submit(prompt_ids, params)

        if req.stream:
            return StreamingResponse(
                _stream_completion(service, request_id, q, model_name,
                                   len(prompt_ids),
                                   bool((req.stream_options or {}).get("include_usage"))),
                media_type="text/event-stream")

        text, reason, n_out = "", "stop", 0
        async for out in service.collect(request_id, q):
            text += out.text_delta
            n_out = out.completion_tokens
            if out.finished:
                reason = out.finish_reason or "stop"
        if req.echo:
            text = str(prompt_text) + text
        return CompletionResponse(
            model=model_name,
            choices=[CompletionChoice(text=text, finish_reason=reason)],
            usage=Usage(prompt_tokens=len(prompt_ids), completion_tokens=n_out,
                        total_tokens=len(prompt_ids) + n_out))

    # -- embeddings --------------------------------------------------------
    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest):
        import torch
        from ..engine.model import ForwardBatch
        from ..memory.kvcache import BLOCK_SIZE

        inputs = req.input if isinstance(req.input, list) else [req.input]
        if inputs and isinstance(inputs[0], int):
            inputs = [inputs]

        data, total = [], 0
        for i, item in enumerate(inputs):
            ids = item if isinstance(item, list) else _encode(tokenizer, str(item))
            ids = ids[: engine.max_model_len]
            if not ids:
                raise HTTPException(400, "entrée vide")
            total += len(ids)
            vec = await asyncio.to_thread(_embed_once, engine, ids)
            if req.dimensions:
                vec = vec[: req.dimensions]
            data.append(EmbeddingData(index=i, embedding=vec))
        return EmbeddingResponse(data=data, model=model_name,
                                 usage=Usage(prompt_tokens=total,
                                             total_tokens=total))

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400,
                            content=ErrorResponse.make(str(exc)).model_dump())

    return app


def _encode(tokenizer: Optional[Tokenizer], text: str) -> list[int]:
    if tokenizer is None:
        raise HTTPException(500, "aucun tokeniseur trouvé dans le répertoire du modèle")
    ids = tokenizer.encode(text)
    if not ids:
        raise HTTPException(400, "l'invite s'est encodée en zéro jeton")
    return ids


def _embed_once(engine: Engine, ids: list[int]) -> list[float]:
    """Moyenne les états cachés finaux, puis normalise en L2.

    S'exécute hors de la boucle de lot : une requête de plongement est un simple
    prefill sans KV à conserver, elle n'a donc pas besoin d'un emplacement de
    cache et ne doit pas en retirer un à une génération qui, elle, en a besoin.
    """
    import torch
    from ..engine.model import ForwardBatch
    from ..memory.kvcache import BLOCK_SIZE

    n = len(ids)
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks = engine.allocator.allocate(n_blocks)
    try:
        slots = torch.tensor(
            [blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
             for i in range(n)], dtype=torch.long)
        batch = ForwardBatch(
            tokens=torch.tensor(ids, dtype=torch.long),
            positions=torch.arange(n, dtype=torch.long),
            seq_lens=[n], query_lens=[n],
            block_tables=[torch.tensor(blocks, dtype=torch.long)],
            slot_mapping=slots, is_prefill=True)
        hidden = engine.model(batch, return_hidden=True)
        pooled = hidden.to(torch.float32).mean(dim=0)
        pooled = pooled / pooled.norm().clamp(min=1e-12)
        return pooled.tolist()
    finally:
        engine.allocator.free(blocks)


def _message_finale(text: str, outils: bool) -> ChoiceMessage:
    if outils:
        reste, appels = extraire_appels(text)
        if appels:
            return ChoiceMessage(content=reste or None, tool_calls=appels)
    return ChoiceMessage(content=text)


def _raison_finale(text: str, reason: str, outils: bool) -> str:
    return "tool_calls" if outils and extraire_appels(text)[1] else reason


async def _stream_chat(service: EngineService, request_id: str,
                       q: asyncio.Queue, model: str, prompt_tokens: int,
                       include_usage: bool, outils: bool = False) -> AsyncIterator[str]:
    cid = new_id("chatcmpl")
    first = ChatCompletionChunk(
        id=cid, model=model,
        choices=[ChunkChoice(delta=DeltaMessage(role="assistant", content=""))])
    yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"

    n_out = 0
    # Avec des outils, tout ce qui suit un « <tool_call> » est retenu et rendu
    # à la fin en appels structurés ; un « < » isolé attend un peu, le temps
    # de savoir s'il ouvre la balise.
    total, pend, retenu = "", "", False
    premier_jeton = None
    try:
        async for out in service.collect(request_id, q):
            n_out = out.completion_tokens
            if out.text_delta and premier_jeton is None:
                # Journalisé pour les mesures d'énergie et de bus : la fenêtre
                # d'un instrument extérieur se borne sur ces deux instants,
                # sans messagerie ni horloge partagée (protocole du 8/09).
                import datetime as _dt
                premier_jeton = _dt.datetime.now()
                print(f"[mesure] {cid} premier jeton "
                      f"{premier_jeton.strftime('%H:%M:%S.%f')[:-3]}", flush=True)
            if out.text_delta:
                total += out.text_delta
                if not outils:
                    delta = out.text_delta
                elif retenu:
                    delta = ""
                else:
                    pend += out.text_delta
                    if "<tool_call>" in pend:
                        retenu = True
                        delta, pend = pend[:pend.index("<tool_call>")], ""
                    elif "<" in pend and len(pend) < 64:
                        delta = ""
                    else:
                        delta, pend = pend, ""
                if delta:
                    chunk = ChatCompletionChunk(
                        id=cid, model=model,
                        choices=[ChunkChoice(delta=DeltaMessage(content=delta))])
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            if out.finished:
                raison = out.finish_reason or "stop"
                if outils:
                    reste, appels = extraire_appels(total)
                    if appels:
                        chunk = ChatCompletionChunk(
                            id=cid, model=model,
                            choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[
                                {"index": i, **a} for i, a in enumerate(appels)]))])
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        raison = "tool_calls"
                    elif pend:
                        chunk = ChatCompletionChunk(
                            id=cid, model=model,
                            choices=[ChunkChoice(delta=DeltaMessage(content=pend))])
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                done = ChatCompletionChunk(
                    id=cid, model=model,
                    choices=[ChunkChoice(delta=DeltaMessage(), finish_reason=raison)])
                if include_usage:
                    done.usage = Usage(prompt_tokens=prompt_tokens,
                                       completion_tokens=n_out,
                                       total_tokens=prompt_tokens + n_out)
                yield f"data: {done.model_dump_json(exclude_none=True)}\n\n"
    except Exception as exc:                          # noqa: BLE001
        # Le message seul ne dit pas d'où vient une erreur CUDA : sans la
        # pile, quatre séries de banc ont été perdues le 7/09 à deviner.
        import traceback
        traceback.print_exc()
        err = ErrorResponse.make(str(exc), "server_error")
        yield f"data: {json.dumps(err.model_dump())}\n\n"
    if premier_jeton is not None:
        import datetime as _dt
        print(f"[mesure] {cid} dernier jeton "
              f"{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
              f"({n_out} jetons)", flush=True)
    yield "data: [DONE]\n\n"


async def _stream_completion(service: EngineService, request_id: str,
                             q: asyncio.Queue, model: str, prompt_tokens: int = 0,
                             include_usage: bool = False) -> AsyncIterator[str]:
    cid = new_id("cmpl")
    try:
        async for out in service.collect(request_id, q):
            resp = CompletionResponse(
                id=cid, model=model,
                choices=[CompletionChoice(
                    text=out.text_delta,
                    finish_reason=out.finish_reason if out.finished else None)])
            if out.finished and include_usage:
                # même contrat que le flux de chat : l'usage sur le dernier morceau
                resp.usage = Usage(prompt_tokens=prompt_tokens,
                                   completion_tokens=out.completion_tokens,
                                   total_tokens=prompt_tokens + out.completion_tokens)
            yield f"data: {resp.model_dump_json(exclude_none=True)}\n\n"
    except Exception as exc:                          # noqa: BLE001
        # Le message seul ne dit pas d'où vient une erreur CUDA : sans la
        # pile, quatre séries de banc ont été perdues le 7/09 à deviner.
        import traceback
        traceback.print_exc()
        err = ErrorResponse.make(str(exc), "server_error")
        yield f"data: {json.dumps(err.model_dump())}\n\n"
    yield "data: [DONE]\n\n"
