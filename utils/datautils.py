import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_tokenizer(model_id, cache_dir):
    if "llama" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=False, cache_dir=cache_dir
        )
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    elif "pythia" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision="step143000",
            cache_dir=cache_dir,
        )

    return tokenizer


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name,
    model_size,
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    cache_dir="~/.cache/",
):
    if model == "pythia":
        model_id = f"EleutherAI/pythia-{model_size}-deduped"
    elif model == "llama" and model_size == "8B":
        model_id = "meta-llama/Meta-Llama-3-8B"
    elif model == "llama" and model_size == "1.1B":
        # use 7b tokenizer because 1.1B doesn't have it's own
        model_id = f"meta-llama/Llama-2-7b-hf"
    elif model == "llama":
        model_id = f"meta-llama/Llama-2-{model_size}-hf"
    else:
        raise ValueError(f"Unknown model {model}")

    print(f"Loading model {model_id}")
    tokenizer = get_tokenizer(model_id, cache_dir)

    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
