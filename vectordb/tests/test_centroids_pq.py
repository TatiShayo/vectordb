import faiss
import numpy as np

d = 128
n = 1000
nlist = 32
M = 8
nbits = 8

np.random.seed(42)
xt = np.random.random((n, d)).astype('float32')

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)
index.train(xt)
centroids = faiss.downcast_index(index.quantizer).reconstruct_n(0, nlist)

# Now, build a new index and set centroids
quantizer2 = faiss.IndexFlatIP(d)
quantizer2.add(centroids)

index2 = faiss.IndexIVFPQ(quantizer2, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)
print("index2.is_trained before train:", index2.is_trained)
# Can we set index2.is_trained = True? Or do we need to train the PQ part?
# PQ must be trained. Let's see if we can do something like:
# index2.pq.train(xt)
# Let's inspect attributes of index2 and index2.pq
print("index2 attributes:", dir(index2))
try:
    print("index2.pq attributes:", dir(index2.pq))
except AttributeError:
    print("No index2.pq")

# Let's see if we can call train on index2 and if it keeps quantizer2 size at nlist
index2.train(xt)
print("index2.is_trained after train:", index2.is_trained)
print("quantizer2 size after train:", index2.quantizer.ntotal)
# If quantizer2 size is still nlist (32), then it didn't cluster again! Or did it?
# Let's check if the centroids are the same as centroids
centroids_after = faiss.downcast_index(index2.quantizer).reconstruct_n(0, nlist)
print("Centroids unchanged?", np.allclose(centroids, centroids_after))
