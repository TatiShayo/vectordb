"""
Test suite for metadata predicate filtering — both in-memory and SQLite queries.
Validates:
- Equality ($eq) & Inequality ($ne)
- Comparison ranges ($gt, $gte, $lt, $lte)
- Set membership ($in, $nin)
- Regular expressions ($regex)
- Field existence ($exists)
- Logical combinators ($and, $or, $not)
- Dynamic field names and SQL injection resilience
- Integration with CollectionDB, scroll, count, facets, delete_by_filter
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pytest

from storage.db import CollectionDB, match_filter


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_meta.db")
        db = CollectionDB(db_path)
        yield db
        db.close()


def test_match_filter_in_memory_equality():
    meta = {"genre": "sci-fi", "rating": 4.8, "year": 2024}
    assert match_filter(meta, {"genre": "sci-fi"}) is True
    assert match_filter(meta, {"genre": "fantasy"}) is False
    assert match_filter(meta, {"genre": {"$eq": "sci-fi"}}) is True
    assert match_filter(meta, {"genre": {"$ne": "fantasy"}}) is True
    assert match_filter(meta, {"genre": {"$ne": "sci-fi"}}) is False


def test_match_filter_in_memory_comparisons():
    meta = {"price": 49.99, "stock": 10}
    assert match_filter(meta, {"price": {"$gt": 40.0}}) is True
    assert match_filter(meta, {"price": {"$gt": 50.0}}) is False
    assert match_filter(meta, {"price": {"$gte": 49.99}}) is True
    assert match_filter(meta, {"stock": {"$lt": 20}}) is True
    assert match_filter(meta, {"stock": {"$lte": 10}}) is True
    assert match_filter(meta, {"stock": {"$lt": 10}}) is False


def test_match_filter_in_memory_in_and_nin():
    meta = {"tag": "electronics", "status": "published"}
    assert match_filter(meta, {"tag": {"$in": ["books", "electronics", "home"]}}) is True
    assert match_filter(meta, {"tag": {"$in": ["books", "home"]}}) is False
    assert match_filter(meta, {"status": {"$nin": ["archived", "draft"]}}) is True
    assert match_filter(meta, {"status": {"$nin": ["published"]}}) is False


def test_match_filter_in_memory_regex_and_exists():
    meta = {"email": "alice@vectordb.internal", "profile": {"age": 30}}
    assert match_filter(meta, {"email": {"$regex": r"^alice@.*\.internal$"}}) is True
    assert match_filter(meta, {"email": {"$regex": r"^bob@"}}) is False
    assert match_filter(meta, {"email": {"$exists": True}}) is True
    assert match_filter(meta, {"missing_key": {"$exists": False}}) is True
    assert match_filter(meta, {"email": {"$exists": False}}) is False


def test_match_filter_in_memory_logical_combinators():
    meta = {"category": "books", "price": 25.0, "rating": 4.5}

    # $and
    assert match_filter(meta, {"$and": [{"category": "books"}, {"price": {"$lt": 30.0}}]}) is True
    assert match_filter(meta, {"$and": [{"category": "books"}, {"price": {"$gt": 30.0}}]}) is False

    # $or
    assert match_filter(meta, {"$or": [{"category": "movies"}, {"price": {"$lt": 30.0}}]}) is True
    assert match_filter(meta, {"$or": [{"category": "movies"}, {"category": "games"}]}) is False

    # $not
    assert match_filter(meta, {"$not": {"category": "movies"}}) is True
    assert match_filter(meta, {"$not": {"category": "books"}}) is False


def test_sqlite_filtering_full_lifecycle(temp_db):
    """Test all predicate operators translated to SQLite queries in CollectionDB."""
    db = temp_db
    vec = np.zeros(8, dtype=np.float32)

    # Insert sample records
    items = [
        ("item1", 1, vec, {"category": "electronics", "price": 199.99, "brand": "Sony", "in_stock": True}, None, None),
        ("item2", 2, vec, {"category": "electronics", "price": 49.99, "brand": "Anker", "in_stock": True}, None, None),
        ("item3", 3, vec, {"category": "books", "price": 14.99, "brand": "Penguin", "in_stock": False}, None, None),
        ("item4", 4, vec, {"category": "home", "price": 89.00, "brand": "Ikea", "in_stock": True}, None, None),
    ]
    db.upsert_batch(items)
    assert db.count() == 4

    # 1. Exact match
    recs, total = db.scroll(filter_dict={"category": "electronics"})
    assert total == 2
    assert {r["id"] for r in recs} == {"item1", "item2"}

    # 2. $gte and $lte range
    recs, total = db.scroll(filter_dict={"price": {"$gte": 50.0, "$lte": 200.0}})
    assert total == 2
    assert {r["id"] for r in recs} == {"item1", "item4"}

    # 3. $in
    recs, total = db.scroll(filter_dict={"brand": {"$in": ["Sony", "Penguin"]}})
    assert total == 2
    assert {r["id"] for r in recs} == {"item1", "item3"}

    # 4. $nin
    recs, total = db.scroll(filter_dict={"category": {"$nin": ["electronics"]}})
    assert total == 2
    assert {r["id"] for r in recs} == {"item3", "item4"}

    # 5. $regex
    recs, total = db.scroll(filter_dict={"brand": {"$regex": "^[SP].*"}})
    assert total == 2
    assert {r["id"] for r in recs} == {"item1", "item3"}

    # 6. $or logical query
    recs, total = db.scroll(filter_dict={"$or": [{"category": "books"}, {"brand": "Ikea"}]})
    assert total == 2
    assert {r["id"] for r in recs} == {"item3", "item4"}

    # 7. filter_faiss_ids
    filtered_fids = db.filter_faiss_ids([1, 2, 3, 4], {"category": "electronics", "price": {"$lt": 100.0}})
    assert filtered_fids == [2]

    # 8. count_filtered
    assert db.count_filtered({"category": "electronics"}) == 2
    assert db.count_filtered({"price": {"$lt": 20.0}}) == 1

    # 9. Facets
    facet_counts = db.facets("category")
    assert facet_counts["electronics"] == 2
    assert facet_counts["books"] == 1
    assert facet_counts["home"] == 1


def test_sqlite_sql_injection_prevention(temp_db):
    """Ensure invalid / malicious metadata keys are rejected."""
    db = temp_db
    with pytest.raises(ValueError, match="Invalid metadata key"):
        db.scroll(filter_dict={"category'; DROP TABLE vectors; --": "val"})

    with pytest.raises(ValueError, match="Invalid field name"):
        db.facets("category; SELECT 1; --")
