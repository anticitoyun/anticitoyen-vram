"""The OpenAI-compatible HTTP server.

The engine is single-threaded and synchronous; the API is asynchronous and
concurrent. The join between them is one background thread running the
engine's step loop and pushing outputs into per-request asyncio queues via
``loop.call_soon_threadsafe``. Requests never touch the model directly, which
is what lets a dozen streaming clients share one pipeline spread across two
GPUs and host RAM.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..engine.runner import Engine, GenerationOutput
from ..engine.sampler import SamplingParams
from .chat import Tokenizer, render_chat
from .protocol import (ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
                       ChatCompletionResponse, ChoiceMessage, ChunkChoice,
                       CompletionChoice, CompletionRequest, CompletionResponse,
                       DeltaMessage, EmbeddingData, EmbeddingRequest,
                       EmbeddingResponse, ErrorResponse, ModelCard, ModelList,
                       Usage, new_id)

__all__ = ["create_app", "EngineService"]


class EngineService:
    """Drives the engine from a background thread and fans results out."""

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
        """Start on first request if the lifespan hook never fired.

        Some ASGI hosts (and every test harness that instantiates the app
        without entering its lifespan) skip startup events. A server whose
        engine thread never starts accepts requests and then hangs forever,
        which is a far worse failure than starting one thread lazily.
        """
        if self._thread is None:
            self.start(asyncio.get_running_loop())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.engine.idle:
                time.sleep(0.002)
                continue
            try:
                outputs = self.engine.step()
            except Exception as exc:                 # noqa: BLE001
                self._broadcast_error(exc)
                continue
            for out in outputs:
                self._deliver(out)

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
                  description="anticitoyen VRAM/RAM - OpenAI-compatible inference")
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

    # -- introspection ----------------------------------------------------
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": model_name}

    @app.get("/metrics")
    async def metrics() -> dict:
        return {"engine": engine.stats.to_dict(), **app.state.info}

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
        messages = [{"role": m.role, "content": m.text()} for m in req.messages]
        prompt = render_chat(tokenizer, messages, req.add_generation_prompt)
        prompt_ids = _encode(tokenizer, prompt)
        params = _params_from(req, 512)
        request_id, q = await service.submit(prompt_ids, params)

        if req.stream:
            return StreamingResponse(
                _stream_chat(service, request_id, q, model_name, len(prompt_ids),
                             bool((req.stream_options or {}).get("include_usage"))),
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
            choices=[ChatChoice(message=ChoiceMessage(content=text),
                                finish_reason=reason)],
            usage=Usage(prompt_tokens=len(prompt_ids), completion_tokens=n_out,
                        total_tokens=len(prompt_ids) + n_out))

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
                _stream_completion(service, request_id, q, model_name),
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
                raise HTTPException(400, "empty input")
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
        raise HTTPException(500, "no tokenizer was found in the model directory")
    ids = tokenizer.encode(text)
    if not ids:
        raise HTTPException(400, "prompt encoded to zero tokens")
    return ids


def _embed_once(engine: Engine, ids: list[int]) -> list[float]:
    """Mean-pool the final hidden states, then L2-normalise.

    Runs outside the batching loop: an embedding request is a single prefill
    with no KV to keep, so it does not need a cache slot and must not take one
    away from a generation that does.
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


async def _stream_chat(service: EngineService, request_id: str,
                       q: asyncio.Queue, model: str, prompt_tokens: int,
                       include_usage: bool) -> AsyncIterator[str]:
    cid = new_id("chatcmpl")
    first = ChatCompletionChunk(
        id=cid, model=model,
        choices=[ChunkChoice(delta=DeltaMessage(role="assistant", content=""))])
    yield f"data: {first.model_dump_json()}\n\n"

    n_out = 0
    try:
        async for out in service.collect(request_id, q):
            n_out = out.completion_tokens
            if out.text_delta:
                chunk = ChatCompletionChunk(
                    id=cid, model=model,
                    choices=[ChunkChoice(delta=DeltaMessage(content=out.text_delta))])
                yield f"data: {chunk.model_dump_json()}\n\n"
            if out.finished:
                done = ChatCompletionChunk(
                    id=cid, model=model,
                    choices=[ChunkChoice(delta=DeltaMessage(),
                                         finish_reason=out.finish_reason or "stop")])
                if include_usage:
                    done.usage = Usage(prompt_tokens=prompt_tokens,
                                       completion_tokens=n_out,
                                       total_tokens=prompt_tokens + n_out)
                yield f"data: {done.model_dump_json()}\n\n"
    except Exception as exc:                          # noqa: BLE001
        err = ErrorResponse.make(str(exc), "server_error")
        yield f"data: {json.dumps(err.model_dump())}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_completion(service: EngineService, request_id: str,
                             q: asyncio.Queue, model: str) -> AsyncIterator[str]:
    cid = new_id("cmpl")
    try:
        async for out in service.collect(request_id, q):
            resp = CompletionResponse(
                id=cid, model=model,
                choices=[CompletionChoice(
                    text=out.text_delta,
                    finish_reason=out.finish_reason if out.finished else None)])
            yield f"data: {resp.model_dump_json()}\n\n"
    except Exception as exc:                          # noqa: BLE001
        err = ErrorResponse.make(str(exc), "server_error")
        yield f"data: {json.dumps(err.model_dump())}\n\n"
    yield "data: [DONE]\n\n"
