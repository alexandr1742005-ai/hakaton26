import asyncio
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")

REQUIRED_ENV_VARS = ["EMBEDDINGS_DENSE_URL", "RERANKER_URL", "QDRANT_URL"]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")

DENSE_PREFETCH_K = 50
SPARSE_PREFETCH_K = 80
RETRIEVE_K = 100
RERANK_TOP = 30
MAX_QUERY_VARIANTS = 3

DENSE_QUERY_MAX_CHARS = 2000
RERANK_TEXT_MAX_CHARS = 3000


def validate_required_env() -> None:
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")
    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")
    missing = [n for n in REQUIRED_ENV_VARS if not os.getenv(n)]
    if missing:
        raise RuntimeError(f"Empty required env vars: {', '.join(missing)}")


validate_required_env()


def get_upstream_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}
    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return kwargs


class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=45.0)
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=API_KEY)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="3.0.0", lifespan=lifespan)


async def embed_dense_batch(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": texts,
        },
    )
    response.raise_for_status()
    payload = DenseEmbeddingResponse.model_validate(response.json())
    sorted_data = sorted(payload.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_sync(text: str) -> SparseVector:
    vectors = list(get_sparse_model().embed([text]))
    if not vectors:
        raise ValueError("Sparse embedding is empty")
    item = vectors[0]
    return SparseVector(
        indices=[int(i) for i in item.indices.tolist()],
        values=[float(v) for v in item.values.tolist()],
    )


async def embed_sparse(text: str) -> SparseVector:
    return await asyncio.to_thread(embed_sparse_sync, text)


def average_dense_vectors(vectors: list[list[float]]) -> list[float]:
    if len(vectors) == 1:
        return vectors[0]
    dim = len(vectors[0])
    result = [0.0] * dim
    for vec in vectors:
        for i, v in enumerate(vec):
            result[i] += v
    norm = sum(x * x for x in result) ** 0.5
    if norm > 0:
        result = [x / norm for x in result]
    return result


def build_query_texts(question: Question) -> list[str]:
    queries: list[str] = []
    main = (question.search_text or question.text).strip()
    if main:
        queries.append(main[:DENSE_QUERY_MAX_CHARS])
    if question.hyde:
        for h in question.hyde[:2]:
            h = h.strip()[:DENSE_QUERY_MAX_CHARS]
            if h and h not in queries:
                queries.append(h)
                if len(queries) >= MAX_QUERY_VARIANTS:
                    break
    if len(queries) < MAX_QUERY_VARIANTS and question.variants:
        for v in question.variants:
            v = v.strip()[:DENSE_QUERY_MAX_CHARS]
            if v and v not in queries:
                queries.append(v)
                if len(queries) >= MAX_QUERY_VARIANTS:
                    break
    return queries or [question.text[:DENSE_QUERY_MAX_CHARS]]


def build_sparse_query_text(question: Question) -> str:
    parts: list[str] = []
    main = (question.search_text or question.text).strip()
    if main:
        parts.append(main)
    if question.keywords:
        parts.append(" ".join(question.keywords))
    if question.entities:
        e = question.entities
        for lst in [e.people, e.names, e.documents, e.emails]:
            if lst:
                parts.append(" ".join(lst))
    return " ".join(parts)


def build_date_filter(question: Question) -> models.Filter | None:
    if not question.date_range:
        return None
    try:
        conditions = []
        if question.date_range.from_:
            conditions.append(models.FieldCondition(
                key="metadata.start",
                range=models.DatetimeRange(gte=question.date_range.from_),
            ))
        if question.date_range.to:
            conditions.append(models.FieldCondition(
                key="metadata.end",
                range=models.DatetimeRange(lte=question.date_range.to),
            ))
        return models.Filter(must=conditions) if conditions else None
    except Exception:
        return None


async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    query_filter: models.Filter | None = None,
) -> list[Any]:
    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=SPARSE_PREFETCH_K,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
    )
    return response.points or []


async def get_rerank_scores(
    client: httpx.AsyncClient,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []
    for attempt in range(5):
        response = await client.post(
            RERANKER_URL,
            **get_upstream_kwargs(),
            json={
                "model": RERANKER_MODEL,
                "encoding_format": "float",
                "text_1": label,
                "text_2": targets,
            },
        )
        if response.status_code == 429:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json().get("data") or []
        return [float(s["score"]) for s in data]
    return [0.0] * len(targets)


async def rerank_points(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[Any]:
    candidates = points[:RERANK_TOP]
    targets_raw = [(point.payload or {}).get("page_content") or "" for point in candidates]
    valid_idx = [i for i, t in enumerate(targets_raw) if t.strip()]
    valid_targets = [targets_raw[i][:RERANK_TEXT_MAX_CHARS] for i in valid_idx]
    valid_candidates = [candidates[i] for i in valid_idx]
    empty_candidates = [candidates[i] for i in range(len(candidates)) if i not in set(valid_idx)]
    if not valid_targets:
        return candidates + points[RERANK_TOP:]
    scores = await get_rerank_scores(client, query[:DENSE_QUERY_MAX_CHARS], valid_targets)
    reranked = [
        point for _, point in sorted(
            zip(scores, valid_candidates),
            key=lambda x: x[0],
            reverse=True,
        )
    ]
    return reranked + empty_candidates + points[RERANK_TOP:]


def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return [str(mid) for mid in (metadata.get("message_ids") or [])]


def deduplicate_ids(points: list[Any], limit: int = 50) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for point in points:
        for mid in extract_message_ids(point):
            if mid not in seen:
                seen.add(mid)
                result.append(mid)
            if len(result) >= limit:
                return result
    return result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    question = payload.question
    query = question.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="question.text is required")
    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant
    query_texts = build_query_texts(question)
    sparse_text = build_sparse_query_text(question)
    dense_vectors, sparse_vector = await asyncio.gather(
        embed_dense_batch(client, query_texts),
        embed_sparse(sparse_text),
    )
    merged_dense = average_dense_vectors(dense_vectors)
    date_filter = build_date_filter(question)
    points = await qdrant_search(qdrant, merged_dense, sparse_vector, date_filter)
    if not points and date_filter:
        points = await qdrant_search(qdrant, merged_dense, sparse_vector, None)
    if not points:
        return SearchAPIResponse(results=[])
    rerank_query = (question.search_text or query)
    points = await rerank_points(client, rerank_query, points)
    message_ids = deduplicate_ids(points, limit=50)
    return SearchAPIResponse(results=[SearchAPIItem(message_ids=message_ids)])


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": str(exc) or repr(exc)})


def main() -> None:
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
