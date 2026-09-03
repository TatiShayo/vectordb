"""
Test suite for vectordb CLI (cli.py).
Tests argument parsing, helper routines, and command formatting.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cli import _rnd_vec, _h


def test_cli_rnd_vec():
    """Verify CLI random normalized vector generator."""
    v = _rnd_vec(16)
    assert len(v) == 16
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_cli_headers():
    """Verify CLI header auth switching."""
    h_user = _h(admin=False)
    assert h_user["Content-Type"] == "application/json"
    assert "X-API-Key" in h_user

    h_admin = _h(admin=True)
    assert h_admin["X-API-Key"] == "admin-secret"


def test_cli_parser_collection_commands():
    """Verify argument parsing for collection commands."""
    from cli import cmd_col_create, cmd_col_list, cmd_col_info, cmd_col_delete

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    col_p = sub.add_parser("collections", aliases=["col"])
    col_s = col_p.add_subparsers(dest="cmd", required=True)

    c = col_s.add_parser("create")
    c.add_argument("name")
    c.add_argument("--dim", type=int, default=384)
    c.add_argument("--distance", default="cosine")
    c.add_argument("--desc", default="")

    args = parser.parse_args(["collections", "create", "testcol", "--dim", "128", "--distance", "euclidean", "--desc", "My test collection"])
    assert args.group == "collections"
    assert args.name == "testcol"
    assert args.dim == 128
    assert args.distance == "euclidean"
    assert args.desc == "My test collection"


def test_cli_parser_vector_commands():
    """Verify argument parsing for vector commands."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    vec_p = sub.add_parser("vectors", aliases=["vec"])
    vec_s = vec_p.add_subparsers(dest="cmd", required=True)

    u = vec_s.add_parser("upsert")
    u.add_argument("collection")
    u.add_argument("id")
    u.add_argument("--vector")
    u.add_argument("--meta")
    u.add_argument("--ttl", type=int)

    args = parser.parse_args(["vectors", "upsert", "products", "v1", "--vector", "0.1,0.2,0.3", "--meta", '{"k":"v"}', "--ttl", "3600"])
    assert args.collection == "products"
    assert args.id == "v1"
    assert args.vector == "0.1,0.2,0.3"
    assert args.meta == '{"k":"v"}'
    assert args.ttl == 3600


def test_cli_parser_search_commands():
    """Verify argument parsing for search commands."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="group", required=True)
    srch = sub.add_parser("search")
    srch.add_argument("collection")
    srch.add_argument("--text")
    srch.add_argument("--id")
    srch.add_argument("--k", type=int, default=5)
    srch.add_argument("--filter")

    args = parser.parse_args(["search", "products", "--text", "wireless headphones", "--k", "10", "--filter", '{"brand":"Sony"}'])
    assert args.collection == "products"
    assert args.text == "wireless headphones"
    assert args.k == 10
    assert args.filter == '{"brand":"Sony"}'
