import copy
import gc
import time

import cupy as cp
import numpy as np
import sys,os
import torch
import torch.nn as nn
# from tqdm import tqdm
import tqdm

from utils.clusteringutils import *
from utils.datautils import *
from utils.modelutils import *
from utils.PCAutils import *
from utils.sparsegpt import *
from typing import List, Sequence, Optional
from argparse import Namespace
from torch.nn import functional as F

def maybe_0th_element(x):
    if isinstance(x, Sequence):
        return x[0]
    return x

def make_batch_iterator(*tensor_lists, batch_size: int):
    all_batch_indices = []
    dataset_size = len(tensor_lists[0])

    while True:
        if len(all_batch_indices) == 0:
            all_batch_indices = list(torch.randperm(dataset_size).chunk(dataset_size // batch_size))
        batch_indices = all_batch_indices.pop(0)

            

        yield [torch.cat([tensor_list[i] for i in batch_indices], dim=0) for tensor_list in tensor_lists]


@torch.enable_grad()
def finetune_block(
    layer: nn.Linear,
    inps: List[torch.Tensor],
    # attention_masks: List[torch.Tensor],
    # position_ids: List[torch.Tensor],
    targets: List[torch.Tensor],
    args: Namespace,
):
    print("Finetuning ...")
    dtype = next(layer.parameters()).dtype
    layer.train()
    # cast to float32
    layer.float()

    DEV = args.device

    steps_per_epoch = len(inps) // args.batch_size
    # print(len(inps), len(targets))
    batch_iterator = make_batch_iterator(inps, targets, batch_size=args.batch_size)
    # init optimizer
    optimizer = torch.optim.Adam(layer.parameters(), lr=args.lr)
    # init scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(args.finetune_epochs):
        epoch_loss = 0
        for step in range(steps_per_epoch):
            inps_, targets_ = next(batch_iterator)
            # print(inps_.shape, targets_.shape)
            inps_, targets_ = (
                inps_.to(DEV, torch.float32),
                targets_.to(DEV, torch.float32),
                # attention_mask_.to(DEV, torch.float32),
                # position_ids_.to(DEV),
            )
            with torch.autocast(device_type="cuda", enabled=args.amp):
                out = maybe_0th_element(layer(inps_.unsqueeze(0)))[0]
            loss = F.mse_loss(out, targets_)
            # scaler and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss += loss.item() / steps_per_epoch
        print(f"Epoch {epoch}. Train loss: {epoch_loss:.2e}")

    layer = layer.to(dtype)
    layer.eval()



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
    verbose=False,
    finetune=False,
):
    print("Starting")
    print("Sparsity", sparsity)

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

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

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
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    attention_mask = cache["attention_mask"]
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

    for layer_idx in tqdm.tqdm(range(len(layers))):
        if verbose:
            print(f"Working on layer {i}")
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer, layers=[ClusteredLinear])

        if verbose:
            print(f"Working on gate and up proj")
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
            layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()


        # randomly choose 128 samples for clustering
        

        # up_gate_clustering_idx = torch.randperm(dataloader_len)[:128]
        # up_gate_clustering_inputs = train_inputs[up_gate_clustering_idx]
        up_gate_clustering_inputs = train_inputs[:min(len(train_inputs), 128)]

        if not no_PCA:
            pca_up_gate_clustering, transformed_up_gate_inputs = (
                make_and_apply_PCA(
                    up_gate_clustering_inputs,
                    hidden_dim,
                    pca_reduction_factor,
                    verbose,
                )
            )
            del transformed_up_gate_inputs
            # collect it in batches of 256
            transformed_up_gate_inputs = []
            for k in range(0, len(train_inputs), 256):
                transformed_up_gate_inputs.append(
                    pca_up_gate_clustering.transform(
                        train_inputs[k : k + 256].reshape(-1, train_inputs.shape[-1])
                    )
                )

            transformed_up_gate_inputs = np.concatenate(transformed_up_gate_inputs, axis=0)

            

            # transformed_up_gate_inputs = pca_up_gate_clustering.transform(
            #     train_inputs.reshape(-1, train_inputs.shape[-1])
            # )

            print(transformed_up_gate_inputs.shape)
            # sys.exit(0)

        else:
            pca_up_gate_clustering = None
            # transformed_up_gate_inputs = up_gate_clustering_inputs.reshape(
            #     -1,
            #     up_gate_clustering_inputs.shape[-1],
            # )
            transformed_up_gate_inputs = train_inputs.reshape(
                -1,
                train_inputs.shape[-1],
            )
            if verbose:
                print("PCA skipped")

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
                    subset[name].add_layer(
                        copy.deepcopy(subset[name].layers[0])
                    )
                    if finetune:
                        gpts[name][cluster] = SparseGPT(subset[name].layers[-1],
                                                    finetune=True)
                    else:
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
        if finetune:
            for name in gpts:
                subset[name].collect = True
            # for cluster in range(num_clusters):



        for j in range(len(dataloader)):
            layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        for name in gpts:
            subset[name].collect = False

        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()

        if verbose:
            print(f"Pruning")
        for name in gpts:
            for cluster in range(num_clusters):
                if verbose:
                    print(f"Cluster {cluster}")
                if quantize:
                    gpts[name][cluster].quantizer = Quantizer()
                    gpts[name][cluster].quantizer.configure(
                        bits=bits,
                        perchannel=True,
                        sym=False,
                        mse=False,
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
                # del gpts[name][cluster]
                # gc.collect()
                # torch.cuda.empty_cache()

        if finetune:
            for name in gpts:
                for cluster in range(num_clusters):

                    print(f"Finetuning cluster {cluster} for {name}")

                    dense_outputs = np.array(subset[name].dense_outputs)
                    dense_outputs = dense_outputs.squeeze()
                    sequence_length = dense_outputs.shape[1]
                    targets = []
                    for batch_number in range(dense_outputs.shape[0]):

                        interm = cp.asnumpy(subset[name].kmeans_model.labels_)[
                            sequence_length
                            * (batch_number) : sequence_length
                            * (batch_number + 1)
                        ]
                        cluster_assignment = interm

                        
                        cluster_mask = cluster_assignment == cluster
                        targets.append(torch.tensor(dense_outputs[batch_number][cluster_mask]))

                    args = Namespace()
                    args.device = dev
                    args.batch_size = 4
                    args.lr = 1e-4
                    args.amp = False
                    args.finetune_epochs = 5
                    finetune_block(
                        gpts[name][cluster].layer,
                        gpts[name][cluster].inputs_for_finetuning,
                        targets,
                        args,
                    )
            
        # sys.exit(0)
        

        if verbose:
            print("Finished pruning up and gate proj")

        for name in gpts:
            for cluster in range(num_clusters):
                gpts[name][cluster].free()
                del gpts[name][cluster]
        del gpts

        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print("Working on down proj")
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
            layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()

        down_clustering_inputs = train_inputs[:min(len(train_inputs), 128)]

        if not no_PCA:
            pca_down_clustering, transformed_down_inputs = make_and_apply_PCA(
                down_clustering_inputs,
                intermediate_dim,
                pca_reduction_factor,
                verbose,
            )
            del transformed_down_inputs
            # transformed_down_inputs = pca_down_clustering.transform(
            #     train_inputs.reshape(-1, train_inputs.shape[-1])
            # )
             # collect it in batches of 256
            transformed_down_inputs = []
            for k in range(0, len(train_inputs), 256):
                transformed_down_inputs.append(
                    pca_down_clustering.transform(
                        train_inputs[k : k + 256].reshape(-1, train_inputs.shape[-1])
                    )
                )

            transformed_down_inputs = np.concatenate(transformed_down_inputs, axis=0)
        else:
            pca_down_clustering = None
            # transformed_down_inputs = down_clustering_inputs.reshape(
            #     -1,
            #     down_clustering_inputs.shape[-1],
            # )
            transformed_down_inputs = train_inputs.reshape(
                -1,
                train_inputs.shape[-1],
            )
            if verbose:
                print("PCA skipped")

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
                    subset[name].add_layer(
                        copy.deepcopy(subset[name].layers[0])
                    )
                    if finetune:
                        gpts[name][cluster] = SparseGPT(subset[name].layers[-1], finetune=True)
                    else:
                        gpts[name][cluster] = SparseGPT(subset[name].layers[-1])

        batch_number_down = [0]

        handles = []
        for name in gpts:
            handles.append(
                subset[name].register_forward_hook(
                    add_batch(name, batch_number_down, subset, gpts)
                )
            )
        if finetune:
            for name in gpts:
                subset[name].collect = True

        for j in range(len(dataloader)):
            layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        for name in gpts:
            subset[name].collect = True

        for h in handles:
            h.remove()

        gc.collect()
        torch.cuda.empty_cache()

        for name in subset:
            subset[name].reset_counter()

        if verbose:
            print(f"Pruning")
        for name in gpts:
            for cluster in range(num_clusters):
                if verbose:
                    print(f"Cluster {cluster}")
                if quantize:
                    gpts[name][cluster].quantizer = Quantizer()
                    gpts[name][cluster].quantizer.configure(
                        bits=bits,
                        perchannel=True,
                        sym=False,
                        mse=False,
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

                # gpts[name][cluster].free()
                # del gpts[name][cluster]
                # gc.collect()
                torch.cuda.empty_cache()

        if finetune:
            for name in gpts:
                for cluster in range(num_clusters):

                    print(f"Finetuning cluster {cluster} for {name}")

                    dense_outputs = np.array(subset[name].dense_outputs)
                    dense_outputs = dense_outputs.squeeze()
                    sequence_length = dense_outputs.shape[1]
                    targets = []
                    for batch_number in range(dense_outputs.shape[0]):

                        interm = cp.asnumpy(subset[name].kmeans_model.labels_)[
                            sequence_length
                            * (batch_number) : sequence_length
                            * (batch_number + 1)
                        ]
                        cluster_assignment = interm

                        
                        cluster_mask = cluster_assignment == cluster
                        targets.append(torch.tensor(dense_outputs[batch_number][cluster_mask]))

                    args = Namespace()
                    args.device = dev
                    args.batch_size = 4
                    args.lr = 1e-4
                    args.amp = False
                    args.finetune_epochs = 5
                    finetune_block(
                        gpts[name][cluster].layer,
                        gpts[name][cluster].inputs_for_finetuning,
                        targets,
                        args,
                    )
        

        if verbose:
            print("Finished pruning down proj")

        for name in gpts:
            for cluster in range(num_clusters):
                gpts[name][cluster].free()
                del gpts[name][cluster]
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
            inps[j] = layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

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
            inps[j] = layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

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
        layers[layer_idx] = layer.cpu()
        del layer
        for name in subset:
            del subset[name].kmeans_model
            del subset[name].pca_model

        del subset
        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print(f"Done with layer {i}")

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
        shift_labels = testenc[
            :, (j * CONTEXT_LENGTH) : ((j + 1) * CONTEXT_LENGTH)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * CONTEXT_LENGTH
        nlls.append(neg_log_likelihood)
    mean_nll = torch.stack(nlls).sum() / (len(nlls) * CONTEXT_LENGTH)
    ppl = torch.exp(mean_nll)
    model.config.use_cache = use_cache
    return mean_nll, ppl
