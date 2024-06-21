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
def pythia_sequential(
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

    layers_to_work_on = find_layers(model, layers = [nn.Linear])
    final_layers = {}
    for layer in layers_to_work_on:
        if "mlp.dense_h_to_4h" in layer:
            final_layers[layer] = layers_to_work_on[layer]
        if "mlp.dense_4h_to_h" in layer:
            final_layers[layer] = layers_to_work_on[layer]
    
    make_clustered(model, final_layers, num_clusters=num_clusters)

    layers = model.gpt_neox.layers
    model.config.use_cache = False
    CONTEXT_LENGTH = model.seqlen

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
    model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(dev)

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
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    for i in range(testenc.numel() // CONTEXT_LENGTH):
        batch = testenc[
            :, (i * CONTEXT_LENGTH) : ((i + 1) * CONTEXT_LENGTH)
        ].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
    model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.cpu()
    position_ids = cache["position_ids"]

    def get_kmeans_input(name, batch_number, train_inputs):
        def tmp(_, inp, out):
            train_inputs[batch_number[0]] = inp[0].data
            batch_number[0] += 1

        return tmp
    
    def add_batch(name, batch_number, subset, gpts):
        def tmp(_, inp, out):
            input_mask = cp.asnumpy(subset[name].kmeans_model.labels_)[
                CONTEXT_LENGTH
                * batch_number[0] : CONTEXT_LENGTH
                * (batch_number[0] + 1)
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

        if verbose: print(f"Working on h to 4h")

        batch_number = [0]
        train_inputs = torch.empty(
            (dataloader_len, CONTEXT_LENGTH, hidden_dim), 
            device=dev,
        )

        handles = []
        for name in subset:
            if "dense_h_to_4h" in name:
                handles.append(
                    subset[name].register_forward_hook(
                        get_kmeans_input(name, batch_number, train_inputs)
                    )
                )
        
        for j in range(len(dataloader)):
            layer(inps[j], position_ids=position_ids)

        for h in handles:
            h.remove()
        
        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()
            
        dense_h_to_4h_inputs = train_inputs

        if not no_PCA:
            pca_h_to_4h, transformed_dense_h_to_4h_inputs = make_and_apply_PCA(
                dense_h_to_4h_inputs, 
                hidden_dim, 
                pca_reduction_factor, 
                verbose,
            )
        else:
            pca_h_to_4h = None
            transformed_dense_h_to_4h_inputs = dense_h_to_4h_inputs.reshape(
                    -1, 
                    dense_h_to_4h_inputs.shape[-1],
            )
            if verbose: print("PCA skipped")
        
        del dense_h_to_4h_inputs
        del train_inputs
        gc.collect()
        torch.cuda.empty_cache()

        kmeans_h_to_4h = make_and_apply_KMeans(
            transformed_dense_h_to_4h_inputs, 
            num_clusters, 
            verbose,
        )

        del transformed_dense_h_to_4h_inputs
        gc.collect()
        torch.cuda.empty_cache()

        gpts = {}

        for name in subset:
            if "dense_h_to_4h" in name:
                gpts[name] = {}
                subset[name].kmeans_model = kmeans_h_to_4h
                subset[name].pca_model = pca_h_to_4h
                for cluster in range(num_clusters):
                    subset[name].add_layer(copy.deepcopy(subset[name].layers[0]))
                    gpts[name][cluster] = SparseGPT(subset[name].layers[-1])
        batch_number = [0]

        handles = []
        for name in gpts:
            handles.append(
                subset[name].register_forward_hook(
                    add_batch(name, batch_number, subset, gpts)
                )
            )
        
        for j in range(len(dataloader)):
            layer(inps[j], position_ids=position_ids)  
        
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

        if verbose: print("Finished pruning dense_h_to_4h")
        del gpts

        gc.collect()
        torch.cuda.empty_cache()

        if verbose: print(f"Working on 4h to h")

        batch_number = [0]
        train_inputs = torch.empty(
            (dataloader_len, CONTEXT_LENGTH, intermediate_dim), 
            device=dev,
        )

        handles = []
        for name in subset:
            if "dense_4h_to_h" in name:
                handles.append(
                    subset[name].register_forward_hook(
                        get_kmeans_input(name, batch_number, train_inputs)
                    )
                )
            
        for j in range(len(dataloader)):
            layer(inps[j], position_ids=position_ids)
        
        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()
        
        dense_4h_to_h_inputs = train_inputs

        if not no_PCA:
            pca_4h_to_h, transformed_dense_4h_to_h_inputs = make_and_apply_PCA(
                dense_4h_to_h_inputs, 
                intermediate_dim, 
                pca_reduction_factor, 
                verbose,
            )
        else:
            pca_4h_to_h = None
            transformed_dense_4h_to_h_inputs = dense_4h_to_h_inputs.reshape(
                    -1, 
                    dense_4h_to_h_inputs.shape[-1],
            )
            if verbose: print("PCA skipped")

        del dense_4h_to_h_inputs
        del train_inputs
        gc.collect()
        torch.cuda.empty_cache()

        kmeans_4h_to_h = make_and_apply_KMeans(
            transformed_dense_4h_to_h_inputs, 
            num_clusters, 
            verbose,
        )

        del transformed_dense_4h_to_h_inputs
        gc.collect()
        torch.cuda.empty_cache()

        gpts = {}

        for name in subset:
            if "dense_4h_to_h" in name:
                gpts[name] = {}
                subset[name].kmeans_model = kmeans_4h_to_h
                subset[name].pca_model = pca_4h_to_h
                for cluster in range(num_clusters):
                    subset[name].add_layer(copy.deepcopy(subset[name].layers[0]))
                    gpts[name][cluster] = SparseGPT(subset[name].layers[-1])
        batch_number = [0]

        handles = []
        for name in gpts:
            handles.append(
                subset[name].register_forward_hook(
                    add_batch(name, batch_number, subset, gpts)
                )
            )
        
        for j in range(len(dataloader)):
            layer(inps[j], position_ids=position_ids)
        
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

        if verbose: print("Finished pruning dense_4h_to_h")
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
            inps[j] = layer(inps[j], position_ids=position_ids)[0]
        
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
            inps[j] = layer(inps[j], position_ids=position_ids)[0]

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

    if model.gpt_neox.final_layer_norm is not None:
        model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.to(dev)
    model.embed_out = model.embed_out.to(dev)
    testenc = testenc.to(dev)
    nlls = []
    final_losses = []
    for i in range(len(dataloader), len(inps)):
        hidden_states = inps[i]
        if model.gpt_neox.final_layer_norm is not None:
            hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
        lm_logits = model.embed_out(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        j = i - len(dataloader)
        shift_labels = testenc[:, (j * CONTEXT_LENGTH):((j + 1) * CONTEXT_LENGTH)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * CONTEXT_LENGTH
        nlls.append(neg_log_likelihood)
        final_losses.append(loss.float())
    mean_nll = torch.stack(nlls).sum()/(len(nlls)*CONTEXT_LENGTH)
    ppl = torch.exp(torch.stack(nlls).sum()/(len(nlls)*CONTEXT_LENGTH))
    for i in range(len(inps)):
        del inps[0]
    del inps
    gc.collect()
    torch.cuda.empty_cache()
    return mean_nll, ppl