"""
VectorDB Python Client Library
================================
A clean, type-safe client for VectorDB.

Sync usage:
    import os
    from client import VectorDBClient
    db = VectorDBClient("http://localhost:8000", api_key=os.environ.get("VECTORDB_API_KEY"))
    db.upsert("products", "id1", vector=[...], metadata={"name": "Widget"})
    results = db.search("products", vector=[...], top_k=5)

Async usage:
    from client import AsyncVectorDBClient
    async with AsyncVectorDBClient("http://localhost:8000") as db:
        await db.upsert("products", "id1", vector=[...])
        results = await db.search("products", vector=[...])
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Synchronous client
# ══════════════════════════════════════════════════════════════════════════════

class VectorDBClient:
    """
    Synchronous VectorDB client.

    Args:
        base_url: Server URL (default http://localhost:8000)
        api_key:  API key (default reads VECTORDB_API_KEY env var, falls back to "user-secret")
        timeout:  Request timeout seconds (default 30)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if api_key is None:
            api_key = os.environ.get("VECTORDB_API_KEY", "user-secret")
        try:
            import httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )

    # ── Collections ───────────────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        dimension: int = 384,
        distance: str = "cosine",
        index_type: Optional[str] = None,
        description: str = "",
    ) -> Dict:
        return self._post("/collections", {
            "name": name,
            "dimension": dimension,
            "distance": distance,
            **({"index_type": index_type} if index_type else {}),
            "description": description,
        })

    def list_collections(self) -> List[Dict]:
        return self._get("/collections")

    def get_collection(self, name: str) -> Dict:
        return self._get(f"/collections/{name}")

    def delete_collection(self, name: str) -> None:
        self._delete(f"/collections/{name}")

    # ── Vectors ───────────────────────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict:
        return self._post(f"/collections/{collection}/vectors", {
            "id": id,
            "vector": vector,
            "metadata": metadata or {},
            **({"ttl_seconds": ttl_seconds} if ttl_seconds else {}),
        })

    def upsert_batch(
        self,
        collection: str,
        vectors: List[Dict[str, Any]],
    ) -> Dict:
        """
        vectors: list of {"id": str, "vector": [...], "metadata": {...}}
        """
        return self._post(f"/collections/{collection}/vectors/batch", {"vectors": vectors})

    def get(
        self,
        collection: str,
        id: str,
        include_vector: bool = False,
    ) -> Optional[Dict]:
        return self._get(
            f"/collections/{collection}/vectors/{id}",
            params={"include_vector": include_vector},
        )

    def delete(self, collection: str, id: str) -> Dict:
        return self._delete(f"/collections/{collection}/vectors/{id}")

    def count(self, collection: str) -> Dict:
        return self._get(f"/collections/{collection}/vectors/count")

    def facets(self, collection: str, field: str, limit: int = 100) -> Dict:
        return self._get(
            f"/collections/{collection}/vectors/facets/{field}",
            params={"limit": limit},
        )

    def patch_metadata(self, collection: str, id: str, metadata: Dict[str, Any]) -> Dict:
        return self._patch(
            f"/collections/{collection}/vectors/{id}",
            {"metadata": metadata},
        )

    def delete_by_filter(self, collection: str, filter_dict: Dict[str, Any]) -> Dict:
        return self._delete_with_body(
            f"/collections/{collection}/vectors/by-filter",
            {"filter": filter_dict},
        )

    def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0,
        include_vector: bool = False,
    ) -> Dict:
        return self._get(
            f"/collections/{collection}/vectors/scroll",
            params={"limit": limit, "offset": offset, "include_vector": include_vector},
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        include_vector: bool = False,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        body: Dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_vector": include_vector,
        }
        if filter:
            body["filter"] = filter
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        return self._post(f"/collections/{collection}/search", body)["results"]

    def search_by_text(
        self,
        collection: str,
        text: str,
        top_k: int = 10,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        body: Dict[str, Any] = {"text": text, "top_k": top_k}
        if filter:
            body["filter"] = filter
        return self._post(f"/collections/{collection}/search/by-text", body)["results"]

    def search_by_id(
        self,
        collection: str,
        id: str,
        top_k: int = 10,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        body: Dict[str, Any] = {"id": id, "top_k": top_k}
        if filter:
            body["filter"] = filter
        return self._post(f"/collections/{collection}/search/by-id", body)["results"]

    def hybrid_search(
        self,
        collection: str,
        vector: List[float],
        text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        body: Dict[str, Any] = {
            "vector": vector,
            "text": text,
            "top_k": top_k,
            "vector_weight": vector_weight,
        }
        if filter:
            body["filter"] = filter
        return self._post(f"/collections/{collection}/search/hybrid", body)["results"]

    def batch_search(
        self,
        collection: str,
        queries: List[Dict],
        include_vector: bool = False,
    ) -> List[List[Dict]]:
        body = {"queries": queries, "include_vector": include_vector}
        return [r["results"] for r in self._post(
            f"/collections/{collection}/search/batch", body
        )["responses"]]

    # ── Admin ─────────────────────────────────────────────────────────────────

    def health(self) -> Dict:
        return self._get("/admin/health")

    def metrics(self) -> Dict:
        return self._get("/admin/metrics")

    def rebuild(self, collection: str) -> Dict:
        return self._post(f"/admin/collections/{collection}/rebuild", {})

    def force_save(self) -> Dict:
        return self._post("/admin/save", {})

    def clear_cache(self) -> Dict:
        return self._post("/admin/cache/clear", {})

    def cache_stats(self) -> Dict:
        return self._get("/admin/cache/stats")

    def create_snapshot(self, collection: str) -> Dict:
        return self._post(f"/admin/collections/{collection}/snapshot", {})

    def restore_snapshot(self, collection: str, snapshot_path: str) -> Dict:
        return self._post(
            f"/admin/collections/{collection}/restore",
            params={"snapshot_path": snapshot_path},
            body={},
        )

    def list_tasks(self) -> List[Dict]:
        return self._get("/admin/tasks")

    def get_task(self, task_id: str) -> Dict:
        return self._get(f"/admin/tasks/{task_id}")

    # ── HTTP primitives ───────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        r = self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: Dict = None, params: Optional[Dict] = None) -> Any:
        r = self._client.post(path, json=body if body is not None else {}, params=params)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, body: Dict) -> Any:
        r = self._client.patch(path, json=body)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> Any:
        r = self._client.delete(path)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {}

    def _delete_with_body(self, path: str, body: Dict) -> Any:
        r = self._client.request("DELETE", path, json=body)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {}

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Async client
# ══════════════════════════════════════════════════════════════════════════════

class AsyncVectorDBClient:
    """
    Async VectorDB client (uses httpx.AsyncClient).

    async with AsyncVectorDBClient("http://localhost:8000") as db:
        results = await db.search("products", vector=[...])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "user-secret",
        timeout: float = 30.0,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )

    async def create_collection(
        self,
        name: str,
        dimension: int = 384,
        distance: str = "cosine",
        index_type: Optional[str] = None,
        description: str = "",
    ) -> Dict:
        return await self._post("/collections", {
            "name": name,
            "dimension": dimension,
            "distance": distance,
            **({"index_type": index_type} if index_type else {}),
            "description": description,
        })

    async def list_collections(self) -> List[Dict]:
        return await self._get("/collections")

    async def get_collection(self, name: str) -> Dict:
        return await self._get(f"/collections/{name}")

    async def delete_collection(self, name: str) -> None:
        await self._delete(f"/collections/{name}")

    async def upsert(self, collection: str, id: str, vector: List[float], metadata: Optional[Dict] = None, ttl_seconds: Optional[int] = None) -> Dict:
        body: Dict[str, Any] = {"id": id, "vector": vector, "metadata": metadata or {}}
        if ttl_seconds:
            body["ttl_seconds"] = ttl_seconds
        return await self._post(f"/collections/{collection}/vectors", body)

    async def upsert_batch(self, collection: str, vectors: List[Dict]) -> Dict:
        return await self._post(f"/collections/{collection}/vectors/batch", {"vectors": vectors})

    async def get(self, collection: str, id: str, include_vector: bool = False) -> Optional[Dict]:
        return await self._get(
            f"/collections/{collection}/vectors/{id}",
            params={"include_vector": include_vector},
        )

    async def delete(self, collection: str, id: str) -> Dict:
        return await self._delete(f"/collections/{collection}/vectors/{id}")

    async def count(self, collection: str) -> Dict:
        return await self._get(f"/collections/{collection}/vectors/count")

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0,
        include_vector: bool = False,
    ) -> Dict:
        return await self._get(
            f"/collections/{collection}/vectors/scroll",
            params={"limit": limit, "offset": offset, "include_vector": include_vector},
        )

    async def search(self, collection: str, vector: List[float], top_k: int = 10, filter: Optional[Dict] = None, include_vector: bool = False) -> List[Dict]:
        body: Dict[str, Any] = {"vector": vector, "top_k": top_k, "include_vector": include_vector}
        if filter:
            body["filter"] = filter
        result = await self._post(f"/collections/{collection}/search", body)
        return result["results"]

    async def health(self) -> Dict:
        return await self._get("/admin/health")

    async def metrics(self) -> Dict:
        return await self._get("/admin/metrics")

    async def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        r = await self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, body: Dict) -> Any:
        r = await self._client.post(path, json=body)
        r.raise_for_status()
        return r.json()

    async def _delete(self, path: str) -> Any:
        r = await self._client.delete(path)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {}

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
