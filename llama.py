from utils.modelutils import *
from utils.datautils import *
from utils.clusteringutils import *
from utils.PCAutils import *
from utils.sparsegpt import *

@torch.no_grad()
def llama_sequential(
    model,
    sparsity,
    quantize,
    bits,
    dataloader,
    dataloader_len,
    testenc,
    dev,
    no_PCA,
    pca_reduction_factor,
    num_clusters,
    verbose = False,
    ):
    return