import collections
import copy
import dataclasses
import gc
import os
import pickle
import sys

from functools import partial
from pathlib import Path

import cupy as cp
# import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

import copy
import time

import cudf
import sklearn
import sklearn.preprocessing
import tqdm
from cuml import PCA, KMeans
from cuml.cluster import KMeans
from cuml.decomposition import PCA, IncrementalPCA

from utils.modelutils import *
from utils.datautils import *
from utils.clusteringutils import *
from utils.PCAutils import *
from utils.sparsegpt import *

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPTNeoXForCausalLM

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
    return