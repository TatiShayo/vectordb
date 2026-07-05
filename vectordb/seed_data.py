"""
seed_data.py  —  Populates a demo collection for testing.

Usage:
    python seed_data.py [--host http://localhost:8000] [--key user-secret]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import List

import numpy as np
import requests

PRODUCTS = [
    ("Mechanical Keyboard", "electronics", ["keyboard", "mechanical", "rgb"]),
    ("Noise Cancelling Headphones", "electronics", ["audio", "wireless", "anc"]),
    ("Standing Desk", "furniture", ["ergonomic", "adjustable", "office"]),
    ("Coffee Grinder", "kitchen", ["burr", "espresso", "manual"]),
    ("Running Shoes", "sports", ["marathon", "cushioned", "breathable"]),
    ("Yoga Mat", "sports", ["non-slip", "eco-friendly", "thick"]),
    ("Smartwatch", "electronics", ["fitness", "gps", "heart-rate"]),
    ("Air Purifier", "home", ["hepa", "quiet", "large-room"]),
    ("Espresso Machine", "kitchen", ["semi-auto", "steam-wand", "compact"]),
    ("Monitor 4K", "electronics", ["ips", "usb-c", "ultrawide"]),
    ("Ergonomic Chair", "furniture", ["lumbar-support", "mesh", "adjustable"]),
    ("Blender", "kitchen", ["high-speed", "smoothies", "professional"]),
    ("Water Bottle", "sports", ["insulated", "stainless-steel", "leak-proof"]),
    ("Wireless Charger", "electronics", ["fast-charge", "qi", "multi-device"]),
    ("Electric Kettle", "kitchen", ["variable-temp", "gooseneck", "pour-over"]),
    ("Laptop Stand", "electronics", ["portable", "aluminum", "adjustable"]),
    ("Resistance Bands", "sports", ["set", "latex-free", "home-gym"]),
    ("Night Light", "home", ["dimmable", "warm-white", "touch"]),
    ("USB Hub", "electronics", ["7-port", "usb-c", "data-transfer"]),
    ("Desk Lamp", "home", ["led", "colour-temperature", "study"]),
]


def random_vector(dim: int) -> List[float]:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def seed(host: str, key: str, n: int, dim: int):
    headers = {"X-API-Key": key, "Content-Type": "application/json"}
    base = host.rstrip("/")

    # 1. Create collection
    col_payload = {
        "name": "products",
        "dimension": dim,
        "distance": "cosine",
        "description": "Demo product catalogue",
    }
    r = requests.post(f"{base}/collections", json=col_payload, headers=headers)
    if r.status_code not in (201, 409):
        print(f"Failed to create collection: {r.status_code} {r.text}")
        sys.exit(1)
    if r.status_code == 409:
        print("Collection 'products' already exists — seeding into it anyway.")
    else:
        print("Created collection 'products'.")

    # 2. Batch upsert in chunks of 100
    batch_size = 100
    total_inserted = 0
    t0 = time.time()

    for i in range(0, n, batch_size):
        chunk = min(batch_size, n - i)
        vectors = []
        for j in range(chunk):
            idx = (i + j) % len(PRODUCTS)
            name, category, tags = PRODUCTS[idx]
            vectors.append({
                "id": f"prod_{i+j:05d}",
                "vector": random_vector(dim),
                "metadata": {
                    "name": f"{name} #{i+j}",
                    "category": category,
                    "tags": tags,
                    "price": round(random.uniform(9.99, 499.99), 2),
                    "in_stock": random.choice([True, False]),
                    "rating": round(random.uniform(3.0, 5.0), 1),
                },
                "ttl_seconds": None,
            })

        r = requests.post(
            f"{base}/collections/products/vectors/batch",
            json={"vectors": vectors},
            headers=headers,
        )
        if r.status_code != 200:
            print(f"Batch upsert error: {r.status_code} {r.text}")
            continue
        data = r.json()
        total_inserted += data["inserted"] + data["updated"]
        print(
            f"  Upserted {total_inserted}/{n} "
            f"({data['inserted']} new, {data['updated']} updated) …"
        )

    elapsed = time.time() - t0
    print(f"\nDone. {total_inserted} vectors in {elapsed:.2f}s "
          f"({total_inserted/elapsed:.0f} vecs/sec)\n")

    # 3. Test search
    print("Running test search …")
    query = random_vector(dim)
    r = requests.post(
        f"{base}/collections/products/search",
        json={"vector": query, "top_k": 5},
        headers=headers,
    )
    if r.status_code == 200:
        results = r.json()["results"]
        print("Top-5 results:")
        for res in results:
            print(f"  {res['id']}  score={res['score']:.4f}  name={res['metadata']['name']}")
    else:
        print(f"Search failed: {r.status_code} {r.text}")

    # 4. Test filtered search
    print("\nFiltered search (electronics only) …")
    r = requests.post(
        f"{base}/collections/products/search",
        json={"vector": query, "top_k": 3, "filter": {"category": "electronics"}},
        headers=headers,
    )
    if r.status_code == 200:
        for res in r.json()["results"]:
            print(f"  {res['id']}  {res['metadata']['name']}  cat={res['metadata']['category']}")

    print("\nSeed complete. API docs at: http://localhost:8000/docs")
    print("Admin UI at: http://localhost:8000/ui")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--key", default="user-secret")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--dim", type=int, default=384)
    args = p.parse_args()
    seed(args.host, args.key, args.n, args.dim)
