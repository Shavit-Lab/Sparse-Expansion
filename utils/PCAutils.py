import time
from cuml import PCA


def make_and_apply_PCA(inputs, hidden_dim, reduction_factor, verbose=False):
    if verbose:
        print(
            "Making and applying PCA with reduction factor: ", reduction_factor
        )
        tick = time.time()
    pca_model = PCA(
        n_components=hidden_dim // reduction_factor,
        random_state=0,
        svd_solver="auto",
    )
    transformed_inputs = pca_model.fit_transform(
        inputs.reshape(-1, inputs.shape[-1])
    )
    if verbose:
        tock = time.time()
        print("PCA model acquired")
        print("Time taken: ", tock - tick)
        print(
            "Top 10 explained variance ratios: ",
            pca_model.explained_variance_ratio_[:10],
        )
        print(
            "Bottom 10 explained variance ratios: ",
            pca_model.explained_variance_ratio_[-10:],
        )
        print(
            "Overall explained variance ratio: ",
            pca_model.explained_variance_ratio_.sum(),
        )
    return pca_model, transformed_inputs
