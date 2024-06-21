import numpy as np
import cupy as cp
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPTNeoXForCausalLM, AutoModelForCausalLM

class ClusteredLinear(nn.Module):
    def __init__(self, linear=None, num_clusters=1, perc=0.0):
        super().__init__()
        self.perc = perc
        self.layers = nn.ModuleList()
        if linear is not None:
            self.layers.append(linear)
        self.freqs = np.array([0])
        self.mode = "calibrate"

        self.kmeans_model = None
        self.pca_model = None
        self.num_clusters = num_clusters
        self.batch_counter = 0

    def add_layer(self, layer):
        self.layers.append(layer)
        self.freqs = np.append(self.freqs, 0)

    def reset_counter(self):
        self.batch_counter = 0

    def forward(self, X):
        nobatch = len(X.shape) == 2
        if nobatch:
            X = X.unsqueeze(0)

        if self.mode == "calibrate":
            Y = self.layers[0](X)
            if not nobatch:
                Y = Y.unsqueeze(0)
            self.batch_counter += 1
            return Y
        
        batch_size = X.shape[0]
        sequence_length = X.shape[1]
        hidden_dimension = X.shape[2]

        if self.mode == "train":
            interm = cp.asnumpy(self.kmeans_model.labels_)
            cluster_assignment = interm.reshape(-1, sequence_length)[
                self.batch_counter
            ]

        elif self.mode == "test":
            if self.pca_model is not None:
                cluster_assignment = cp.asnumpy(
                    self.kmeans_model.predict(
                        self.pca_model.transform(
                            X.reshape(-1, hidden_dimension)
                        )
                    ).reshape(
                        sequence_length,
                    )
                )
            else:
                cluster_assignment = cp.asnumpy(
                    self.kmeans_model.predict(
                        X.reshape(-1, hidden_dimension)
                    ).reshape(
                        sequence_length,
                    )
                )

        Y = self.layers[0](X)
        Y_final = torch.zeros_like(Y[0])
        for i in range(self.num_clusters):
            idx_cluster = cluster_assignment == i

            X_cluster = X[:, idx_cluster, :]
            Y = self.layers[i + 1](X_cluster)
            Y_final[idx_cluster] = Y[0]

        self.batch_counter += 1

        if not nobatch:
            Y_final = Y_final.unsqueeze(0)

        del Y
        torch.cuda.empty_cache()
        return Y_final
    
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def make_clustered(module, names, name="", num_clusters=1):
    if isinstance(module, ClusteredLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            setattr(
                module,
                attr,
                ClusteredLinear(
                    getattr(module, attr), num_clusters=num_clusters
                ),
            )
    for name1, child in module.named_children():
        make_clustered(
            child,
            names,
            name + "." + name1 if name != "" else name1,
            num_clusters=num_clusters,
        )

def get_pythia(model_size, cache_dir = None):
    model_size = model_size.lower()
    if cache_dir is not None:
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{model_size}-deduped",
            revision="step143000",
            cache_dir=cache_dir,
        )
    else:
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{model_size}-deduped",
            revision="step143000",
        )
    return model

def get_llama(model_size, cache_dir = None):
    if model_size == "8B":
        model_id = "meta-llama/Meta-Llama-3-8B"
    else:
        model_id = f"meta-llama/Meta-Llama-2-{model_size}-hf"
    if cache_dir is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
        )
    return model