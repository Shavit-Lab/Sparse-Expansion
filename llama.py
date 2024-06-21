import copy
import gc
import time

import cupy as cp
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.clusteringutils import *
from utils.datautils import *
from utils.modelutils import *
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
    print("Starting")

    testenc = testenc.input_ids
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size

    layers_to_work_on = find_layers(model, layers=[nn.Linear])
    final_layers = {}
    for layer in layers_to_work_on:
        if "mlp" in layer:
            final_layers[layer] = layers_to_work_on[layer]

    make_clustered(model, final_layers, num_clusters=num_clusters)

    layers = model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False
    CONTEXT_LENGTH = model.seqlen

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    inps = []

    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    for i in range(testenc.numel() // CONTEXT_LENGTH):
        batch = testenc[:, (i*CONTEXT_LENGTH):((i+1)*CONTEXT_LENGTH)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    attention_mask = cache["attention_mask"]

    def get_kmeans_input(name, batch_number, train_inputs):
        def tmp(_, inp, out):
            train_inputs[batch_number[0]] = inp[0].data
            batch_number[0] += 1

        return tmp

    def add_batch(name, batch_number, subset, gpts):
        def tmp(_, inp, out):
            input_mask = cp.asnumpy(subset[name].kmeans_model.labels_)[
                CONTEXT_LENGTH*batch_number[0]:CONTEXT_LENGTH*(batch_number[0]+1)
            ]
            for cluster in range(num_clusters):
                cluster_mask = input_mask == cluster
                if inp[0].data[0].ndim == 3:
                    gpts[name][cluster].add_batch(
                        inp[0].data[0][0][cluster_mask, :], torch.tensor([])
                    )
                else:
                    gpts[name][cluster].add_batch(
                        inp[0].data[0][cluster_mask, :], torch.tensor([])
                    )
            batch_number[0] += 1

        return tmp
    
    for i in tqdm(range(len(layers))):
        if verbose: print(f"Working on layer {i}")
        layer = layers[i].to(dev)
        subset = find_layers(layer, layers=[ClusteredLinear])

        if verbose: print(f"Working on gate and up proj")
        batch_number = [0]
        train_inputs = torch.empty(
            (dataloader_len, CONTEXT_LENGTH, hidden_dim), 
            device=dev,
        )

        handles = []
        for name in subset:
            if "gate_proj" in name:
                handles.append(
                    subset[name].register_forward_hook(
                        get_kmeans_input(name, batch_number, train_inputs)
                    )
                )
    
        for j in range(len(dataloader)):
            layer(inps[j], attention_mask=attention_mask)
    
        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()

        up_gate_clustering_inputs = train_inputs

        if not no_PCA:
            pca_up_gate_clustering, transformed_up_gate_inputs = make_and_apply_PCA(
                up_gate_clustering_inputs, 
                hidden_dim, 
                pca_reduction_factor, 
                verbose,
            )
        else:
            pca_up_gate_clustering = None
            transformed_up_gate_inputs = up_gate_clustering_inputs.reshape(
                -1, 
                up_gate_clustering_inputs.shape[-1],
            )
            if verbose: print("PCA skipped")
        
        del up_gate_clustering_inputs
        del train_inputs
        gc.collect()
        torch.cuda.empty_cache()

        kmeans_up_gate = make_and_apply_KMeans(
            transformed_up_gate_inputs, 
            num_clusters, 
            verbose,
        )

        del transformed_up_gate_inputs
        gc.collect()
        torch.cuda.empty_cache()

        gpts = {}
        for name in subset:
            if "gate_proj" in name or "up_proj" in name:
                gpts[name] = {}
                subset[name].kmeans_model = kmeans_up_gate
                subset[name].pca_model = pca_up_gate_clustering
                for cluster in range(num_clusters):
                    subset[name].add_layer(copy.deepcopy(subset[name].layers[0]))
                    gpts[name][cluster] = SparseGPT(subset[name].layers[-1])
        
        batch_number_up = [0]
        batch_number_gate = [0]

        handles = []
        for name in gpts:
            if "gate_proj" in name:
                handles.append(
                    subset[name].register_forward_hook(
                        add_batch(name, batch_number_gate, subset, gpts)
                    )
                )
            else:
                handles.append(
                    subset[name].register_forward_hook(
                        add_batch(name, batch_number_up, subset, gpts)
                    )
                )
        
        for j in range(len(dataloader)):
            layer(inps[j], attention_mask=attention_mask)
    
        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()

        if verbose: print(f"Pruning")
        for name in gpts:
            for cluster in range(num_clusters):
                if verbose: print(f"Cluster {cluster}")
                if quantize:
                    gpts[name][cluster].quantizer = Quantizer()
                    gpts[name][cluster].quantizer.configure(bits = bits,
                                                            perchannel = True,
                                                            sym = False,
                                                            mse = False,
                    )
                if isinstance(sparsity, tuple):
                    gpts[name][cluster].fasterprune(
                        sparsity=0,
                        prunen=sparsity[0],
                        prunem=sparsity[1],
                        percdamp=0.01,
                    )
                else:
                    gpts[name][cluster].fasterprune(
                        sparsity=sparsity, 
                        percdamp=0.01,
                    )

                gpts[name][cluster].free()
                del gpts[name][cluster]
                gc.collect()
                torch.cuda.empty_cache()
        
        if verbose: print("Finished pruning up and gate proj")
        del gpts

        gc.collect()
        torch.cuda.empty_cache()

        if verbose: print("Working on down proj")
        batch_number = [0]
        train_inputs = torch.empty(
            (dataloader_len, CONTEXT_LENGTH, intermediate_dim), 
            device=dev,
        )
        handles = []
        for name in subset:
            if "down_proj" in name:
                handles.append(
                    subset[name].register_forward_hook(
                        get_kmeans_input(name, batch_number, train_inputs)
                    )
                )
        for j in range(len(dataloader)):
            layer(inps[j], attention_mask=attention_mask)
        
        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()
        
        down_clustering_inputs = train_inputs

        if not no_PCA:
            pca_down_clustering, transformed_down_inputs = make_and_apply_PCA(
                down_clustering_inputs, 
                intermediate_dim, 
                pca_reduction_factor, 
                verbose,
            )
        else:
            pca_down_clustering = None
            transformed_down_inputs = down_clustering_inputs.reshape(
                -1, 
                down_clustering_inputs.shape[-1],
            )
            if verbose: print("PCA skipped")
        
        del down_clustering_inputs
        del train_inputs
        gc.collect()
        torch.cuda.empty_cache()

        kmeans_down = make_and_apply_KMeans(
            transformed_down_inputs, 
            num_clusters, 
            verbose,
        )

        del transformed_down_inputs
        gc.collect()
        torch.cuda.empty_cache()

        gpts = {}
        for name in subset:
            if "down_proj" in name:
                gpts[name] = {}
                subset[name].kmeans_model = kmeans_down
                subset[name].pca_model = pca_down_clustering
                for cluster in range(num_clusters):
                    subset[name].add_layer(copy.deepcopy(subset[name].layers[0]))
                    gpts[name][cluster] = SparseGPT(subset[name].layers[-1])

        batch_number_down = [0]

        handles = []
        for name in gpts:
            handles.append(
                subset[name].register_forward_hook(
                    add_batch(name, batch_number_down, subset, gpts)
                )
            )
        
        for j in range(len(dataloader)):
            layer(inps[j], attention_mask=attention_mask)

        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()

        if verbose: print(f"Pruning")
        for name in gpts:
            for cluster in range(num_clusters):
                if verbose: print(f"Cluster {cluster}")
                if quantize:
                    gpts[name][cluster].quantizer = Quantizer()
                    gpts[name][cluster].quantizer.configure(bits = bits,
                                                            perchannel = True,
                                                            sym = False,
                                                            mse = False,
                    )
                if isinstance(sparsity, tuple):
                    gpts[name][cluster].fasterprune(
                        sparsity=0,
                        prunen=sparsity[0],
                        prunem=sparsity[1],
                        percdamp=0.01,
                    )
                else:
                    gpts[name][cluster].fasterprune(
                        sparsity=sparsity, 
                        percdamp=0.01,
                    )

                gpts[name][cluster].free()
                del gpts[name][cluster]
                gc.collect()
                torch.cuda.empty_cache()

        if verbose: print("Finished pruning down proj")
        del gpts

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()
            subset[name].mode = "train"
        
        if verbose:
            print("Starting training run")
            tick = time.time()
        
        for j in range(len(dataloader)):
            inps[j] = layer(inps[j], attention_mask=attention_mask)[0]

        if verbose:
            tock = time.time()
            print(f"Time taken to train layer: {tock - tick} seconds")
        
        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()
            subset[name].mode = "test"

        if verbose:
            print("Starting test run")
            tick = time.time()

        for j in range(len(dataloader), len(inps)):
            inps[j] = layer(inps[j],attention_mask=attention_mask)[0]

        if verbose:
            tock = time.time()
            print(f"Time taken to test layer: {tock - tick} seconds")

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()
        
        if len(subset) != 0:
            for name in subset:
                while len(subset[name].layers) > 1:
                    del subset[name].layers[-1]
        layers[i] = layer.cpu()
        del layer
        for name in subset:
            del subset[name].kmeans_model
            del subset[name].pca_model

        del subset
        gc.collect()
        torch.cuda.empty_cache()

        if verbose: print(f"Done with layer {i}")

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm.tqdm(range(len(dataloader), len(inps))):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        j = i - len(dataloader)
        shift_labels = testenc[:, (j*CONTEXT_LENGTH):((j+1)*CONTEXT_LENGTH)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * CONTEXT_LENGTH
        nlls.append(neg_log_likelihood)
    mean_nll = torch.stack(nlls).sum() / (len(nlls) * CONTEXT_LENGTH)
    ppl = torch.exp(mean_nll)
    model.config.use_cache = use_cache
    return mean_nll, ppl