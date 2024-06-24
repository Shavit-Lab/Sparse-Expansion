import time
from cuml import KMeans
import numpy as np
import cupy as cp


def make_and_apply_KMeans(inputs, n_clusters, verbose=False):
    if verbose:
        print("Making and applying KMeans with n_clusters: ", n_clusters)
        tick = time.time()
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=100,
    )
    kmeans_model.fit(inputs.reshape(-1, inputs.shape[-1]))
    if verbose:
        tock = time.time()
        print("KMeans model acquired")
        print("Time taken: ", tock - tick)
        unique, counts = np.unique(
            cp.asnumpy(kmeans_model.labels_),
            return_counts=True,
        )
        total = counts.sum()
        print(f"Proportion of each cluster:")
        for i in range(len(unique)):
            print(f"{unique[i]}: {counts[i]/total}")
    return kmeans_model
