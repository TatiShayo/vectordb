#!/usr/bin/env python3
"""
vectordb CLI — J01
Usage:
  python cli.py collections list
  python cli.py collections create products --dim 384
  python cli.py vectors upsert products vec1 --meta '{"name":"Widget"}'
  python cli.py search products --text "wireless headphones" --top-k 5
  python cli.py admin snapshot products
  python cli.py admin metrics
"""
import argparse, json, os, sys, random
import numpy as np

BASE  = os.environ.get("VDB_URL",  "http://localhost:8000")
AKEY  = os.environ.get("VDB_ADMIN_KEY", "admin-secret")
UKEY  = os.environ.get("VDB_KEY",  "user-secret")


def _h(admin=False):
    return {"X-API-Key": AKEY if admin else UKEY, "Content-Type": "application/json"}


def _req(method, path, body=None, admin=False, stream=False):
    import urllib.request, urllib.error
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=_h(admin), method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[{e.code}] {body}", file=sys.stderr)
        sys.exit(1)


def _pp(obj):
    print(json.dumps(obj, indent=2, default=str))


def _rnd_vec(dim):
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ── Collections ───────────────────────────────────────────────────────────────
def cmd_col_list(args): _pp(_req("GET", "/collections"))

def cmd_col_create(args):
    body = {"name": args.name, "dimension": args.dim,
            "distance": args.distance, "description": args.desc or ""}
    _pp(_req("POST", "/collections", body, admin=True))

def cmd_col_info(args): _pp(_req("GET", f"/collections/{args.name}"))

def cmd_col_delete(args):
    if not args.yes:
        ans = input(f"Delete '{args.name}'? [y/N] ")
        if ans.lower() != "y":
            print("Aborted."); return
    _req("DELETE", f"/collections/{args.name}", admin=True)
    print(f"Deleted '{args.name}'")

# ── Vectors ───────────────────────────────────────────────────────────────────
def cmd_vec_upsert(args):
    if args.vector:
        vector = [float(x) for x in args.vector.split(",")]
    else:
        col = _req("GET", f"/collections/{args.collection}")
        vector = _rnd_vec(col["dimension"])
        print(f"(using random {col['dimension']}-dim vector)")
    meta = json.loads(args.meta) if args.meta else {}
    body = {"id": args.id, "vector": vector, "metadata": meta}
    if args.ttl: body["ttl_seconds"] = args.ttl
    _pp(_req("POST", f"/collections/{args.collection}/vectors", body))

def cmd_vec_get(args):
    _pp(_req("GET", f"/collections/{args.collection}/vectors/{args.id}"
             + ("?include_vector=true" if args.vector else "")))

def cmd_vec_delete(args):
    _pp(_req("DELETE", f"/collections/{args.collection}/vectors/{args.id}"))

def cmd_vec_scroll(args):
    _pp(_req("GET", f"/collections/{args.collection}/vectors/scroll"
             f"?limit={args.limit}&offset={args.offset}"))

def cmd_vec_count(args):
    _pp(_req("GET", f"/collections/{args.collection}/vectors/count"))

# ── Search ────────────────────────────────────────────────────────────────────
def cmd_search(args):
    if args.text:
        body = {"text": args.text, "top_k": args.k}
        path = f"/collections/{args.collection}/search/by-text"
    elif args.id:
        body = {"id": args.id, "top_k": args.k}
        path = f"/collections/{args.collection}/search/by-id"
    else:
        col = _req("GET", f"/collections/{args.collection}")
        dim = col["dimension"]
        body = {"vector": _rnd_vec(dim), "top_k": args.k}
        path = f"/collections/{args.collection}/search"
        print(f"(searching with random {dim}-dim vector)")

    if args.filter:
        body["filter"] = json.loads(args.filter)

    result = _req("POST", path, body)
    print(f"\n{'ID':<30} {'SCORE':>8}  METADATA")
    print("-" * 72)
    for r in result.get("results", []):
        meta_str = json.dumps(r["metadata"])[:40]
        print(f"{r['id']:<30} {r['score']:>8.4f}  {meta_str}")
    print(f"\n{result.get('total_returned',0)} results in {result.get('query_time_ms',0):.1f}ms"
          + (" (cached)" if result.get("cached") else ""))

# ── Admin ─────────────────────────────────────────────────────────────────────
def cmd_health(args):   _pp(_req("GET", "/admin/health"))
def cmd_metrics(args):  _pp(_req("GET", "/admin/metrics"))
def cmd_save(args):     _pp(_req("POST", "/admin/save", admin=True))

def cmd_snapshot(args):
    _pp(_req("POST", f"/admin/collections/{args.collection}/snapshot", admin=True))

def cmd_rebuild(args):
    _pp(_req("POST", f"/admin/collections/{args.collection}/rebuild", admin=True))

def cmd_tasks(args): _pp(_req("GET", "/admin/tasks"))

def cmd_cache_clear(args): _pp(_req("POST", "/admin/cache/clear", admin=True))


# ── Parser ────────────────────────────────────────────────────────────────────
def main():
    global BASE
    p = argparse.ArgumentParser(prog="vectordb", description="VectorDB CLI")
    p.add_argument("--url", default=BASE, help="Server URL")
    sub = p.add_subparsers(dest="group", required=True)

    # collections
    col_p = sub.add_parser("collections", aliases=["col"])
    col_s = col_p.add_subparsers(dest="cmd", required=True)
    col_s.add_parser("list").set_defaults(func=cmd_col_list)
    c = col_s.add_parser("create"); c.set_defaults(func=cmd_col_create)
    c.add_argument("name"); c.add_argument("--dim", type=int, default=384)
    c.add_argument("--distance", default="cosine"); c.add_argument("--desc", default="")
    i = col_s.add_parser("info"); i.set_defaults(func=cmd_col_info); i.add_argument("name")
    d = col_s.add_parser("delete"); d.set_defaults(func=cmd_col_delete)
    d.add_argument("name"); d.add_argument("--yes", action="store_true")

    # vectors
    vec_p = sub.add_parser("vectors", aliases=["vec"])
    vec_s = vec_p.add_subparsers(dest="cmd", required=True)
    u = vec_s.add_parser("upsert"); u.set_defaults(func=cmd_vec_upsert)
    u.add_argument("collection"); u.add_argument("id")
    u.add_argument("--vector"); u.add_argument("--meta"); u.add_argument("--ttl", type=int)
    g = vec_s.add_parser("get"); g.set_defaults(func=cmd_vec_get)
    g.add_argument("collection"); g.add_argument("id"); g.add_argument("--vector", action="store_true")
    dl = vec_s.add_parser("delete"); dl.set_defaults(func=cmd_vec_delete)
    dl.add_argument("collection"); dl.add_argument("id")
    sc = vec_s.add_parser("scroll"); sc.set_defaults(func=cmd_vec_scroll)
    sc.add_argument("collection"); sc.add_argument("--limit",type=int,default=10)
    sc.add_argument("--offset",type=int,default=0)
    ct = vec_s.add_parser("count"); ct.set_defaults(func=cmd_vec_count)
    ct.add_argument("collection")

    # search
    srch = sub.add_parser("search"); srch.set_defaults(func=cmd_search)
    srch.add_argument("collection")
    srch.add_argument("--text"); srch.add_argument("--id")
    srch.add_argument("--k", type=int, default=5)
    srch.add_argument("--filter")

    # admin
    adm_p = sub.add_parser("admin")
    adm_s = adm_p.add_subparsers(dest="cmd", required=True)
    adm_s.add_parser("health").set_defaults(func=cmd_health)
    adm_s.add_parser("metrics").set_defaults(func=cmd_metrics)
    adm_s.add_parser("save").set_defaults(func=cmd_save)
    adm_s.add_parser("tasks").set_defaults(func=cmd_tasks)
    cc = adm_s.add_parser("cache-clear"); cc.set_defaults(func=cmd_cache_clear)
    sn = adm_s.add_parser("snapshot"); sn.set_defaults(func=cmd_snapshot)
    sn.add_argument("collection")
    rb = adm_s.add_parser("rebuild"); rb.set_defaults(func=cmd_rebuild)
    rb.add_argument("collection")

    args = p.parse_args()
    if args.url != BASE:
        BASE = args.url
    args.func(args)


if __name__ == "__main__":
    main()
