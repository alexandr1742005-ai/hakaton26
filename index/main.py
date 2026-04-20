import asyncio
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")

SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"
UVICORN_WORKERS = 4

CHUNK_TARGET_MESSAGES = 6      
CHUNK_OVERLAP_MESSAGES = 2      
SINGLE_MESSAGE_MAX_CHARS = 4000 
SINGLE_MESSAGE_CHUNK_CHARS = 3000
SINGLE_MESSAGE_OVERLAP_CHARS = 500


DENSE_CONTENT_MAX_CHARS = 2000

RERANK_TEXT_MAX_CHARS = 3000



class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None


class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str = ""
    sender_id: str = ""
    file_snippets: str = ""
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool = False
    is_hidden: bool = False
    is_forward: bool = False
    is_quote: bool = False


class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]


class IndexAPIRequest(BaseModel):
    data: ChatData


class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]


class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]


class SparseEmbeddingRequest(BaseModel):
    texts: list[str]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]



def ts_to_str(ts: int) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def render_message_text(msg: Message) -> str:
    """Только чистый текст сообщения без метаданных."""
    parts: list[str] = []
    if msg.text:
        parts.append(msg.text)
    if msg.parts:
        for part in msg.parts:
            pt = part.get("text")
            if isinstance(pt, str) and pt.strip():
                parts.append(pt.strip())
            # forward/quote вложения
            for key in ("message", "forward_message"):
                nested = part.get(key)
                if isinstance(nested, dict):
                    nt = nested.get("text", "")
                    if nt:
                        parts.append(f"[цитата: {nt}]")
    if msg.file_snippets:
        parts.append(f"[файл: {msg.file_snippets}]")
    return "\n".join(parts).strip()


def render_message_rich(msg: Message) -> str:
    """Полное представление сообщения с метаданными для dense/page_content."""
    body = render_message_text(msg)
    if not body:
        return ""
    ts = ts_to_str(msg.time)
    flags = []
    if msg.is_forward:
        flags.append("переслано")
    if msg.is_quote:
        flags.append("цитата")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    header = f"[{ts}] {msg.sender_id}{flag_str}:"
    result = f"{header}\n{body}"
    if msg.mentions:
        result += f"\n@: {', '.join(msg.mentions)}"
    return result


def render_message_sparse(msg: Message) -> str:
    """Keyword-heavy текст для BM25."""
    parts: list[str] = []
    body = render_message_text(msg)
    if body:
        parts.append(body)
    if msg.mentions:
        parts.append(" ".join(msg.mentions))
        parts.append(" ".join(msg.mentions))
    if msg.sender_id:
        parts.append(msg.sender_id)
    return " ".join(parts).strip()


def is_useless(msg: Message) -> bool:
    if msg.is_hidden:
        return True
    if msg.is_system and not render_message_text(msg):
        return True
    if not render_message_text(msg).strip():
        return True
    return False



def build_chunk(
    msgs: list[Message],
    overlap: list[Message],
    chat: Chat,
) -> IndexAPIItem | None:
    useful = [m for m in msgs if not is_useless(m)]
    if not useful:
        return None

    chat_header = f"Чат: {chat.name} | Тип: {chat.type}"

    main_lines = [render_message_rich(m) for m in useful]
    main_block = "\n---\n".join(l for l in main_lines if l)
    dense_content = f"{chat_header}\n{main_block}"
    dense_content = dense_content[:DENSE_CONTENT_MAX_CHARS]

    overlap_useful = [m for m in overlap if not is_useless(m)]
    if overlap_useful:
        overlap_lines = [render_message_rich(m) for m in overlap_useful]
        overlap_block = "\n---\n".join(l for l in overlap_lines if l)
        page_content = f"{chat_header}\n[контекст]\n{overlap_block}\n[сообщения]\n{main_block}"
    else:
        page_content = dense_content

    sparse_parts = [render_message_sparse(m) for m in useful]
    sparse_content = f"{chat.name} {' '.join(sparse_parts)}"

    return IndexAPIItem(
        page_content=page_content[:RERANK_TEXT_MAX_CHARS],
        dense_content=dense_content,
        sparse_content=sparse_content,
        message_ids=[m.id for m in useful],
    )


def split_long_message(msg: Message, chat: Chat) -> list[IndexAPIItem]:
    """Длинное одиночное сообщение бьём на под-чанки."""
    body = render_message_rich(msg)
    result = []
    start = 0
    while start < len(body):
        chunk = body[start:start + SINGLE_MESSAGE_CHUNK_CHARS]
        if chunk.strip():
            result.append(IndexAPIItem(
                page_content=chunk,
                dense_content=chunk[:DENSE_CONTENT_MAX_CHARS],
                sparse_content=render_message_sparse(msg),
                message_ids=[msg.id],
            ))
        start += SINGLE_MESSAGE_CHUNK_CHARS - SINGLE_MESSAGE_OVERLAP_CHARS
    return result


def group_by_thread(msgs: list[Message]) -> dict[str | None, list[Message]]:
    groups: dict[str | None, list[Message]] = {}
    for m in msgs:
        groups.setdefault(m.thread_sn, []).append(m)
    return groups


def build_chunks(
    overlap_messages: list[Message],
    new_messages: list[Message],
    chat: Chat,
) -> list[IndexAPIItem]:
    result: list[IndexAPIItem] = []

    thread_groups = group_by_thread(new_messages)
    overlap_groups = group_by_thread(overlap_messages)

    for thread_sn, thread_msgs in thread_groups.items():
        useful = [m for m in thread_msgs if not is_useless(m)]
        if not useful:
            continue

        thread_overlap = overlap_groups.get(thread_sn) or overlap_messages[-CHUNK_OVERLAP_MESSAGES:]

        for i in range(0, len(useful), CHUNK_TARGET_MESSAGES):
            chunk_msgs = useful[i:i + CHUNK_TARGET_MESSAGES]
            if not chunk_msgs:
                continue

            if len(chunk_msgs) == 1 and len(render_message_text(chunk_msgs[0])) > SINGLE_MESSAGE_MAX_CHARS:
                result.extend(split_long_message(chunk_msgs[0], chat))
                continue

            if i == 0:
                ov = thread_overlap[-CHUNK_OVERLAP_MESSAGES:]
            else:
                ov = useful[max(0, i - CHUNK_OVERLAP_MESSAGES):i]

            item = build_chunk(chunk_msgs, ov, chat)
            if item:
                result.append(item)

    return result


app = FastAPI(title="Index Service", version="3.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    chunks = build_chunks(
        payload.data.overlap_messages,
        payload.data.new_messages,
        payload.data.chat,
    )
    return IndexAPIResponse(results=chunks)


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding
    logger.info("Loading sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[dict]:
    model = get_sparse_model()
    vectors = []
    for item in model.embed(texts):
        vectors.append({
            "indices": item.indices.tolist(),
            "values": item.values.tolist(),
        })
    return vectors


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, workers=UVICORN_WORKERS)


if __name__ == "__main__":
    main()
