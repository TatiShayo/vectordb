import faiss
import numpy as np

d = 128
nlist = 32

centroids = np.random.random((nlist, d)).astype('float32')

quantizer = faiss.IndexFlatIP(d)
quantizer.add(centroids)

index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
print("Before setting:", index.is_trained)
index.is_trained = True
print("After setting:", index.is_trained)

# Let's add some vectors and search
vectors = np.random.random((10, d)).astype('float32')
ids = np.arange(10, dtype=np.int64)
index.add_with_ids(vectors, ids)

q = np.random.random((1, d)).astype('float32')
scores, results = index.search(q, 3)
print("Search results:", results)
