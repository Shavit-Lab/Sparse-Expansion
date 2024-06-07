import math
import time

import torch
import torch.nn as nn
import transformers
import pickle

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(self, layer, mode = "default"):
        self.mode = mode
        mode_choices = ["default", "2:4+default", "2:4+SSPN", "2:4+DSPN", "SSPN", "DSPN", "mode1", "mode2", "mode3", "mode4"]

        # SSPN (Same Sparsity Per Neuron): Adaptive per row sparsification with different masks between clustered and non-clustered layers
        # DSPN (Different Sparsities Per Neuron): Adaptive per row sparsification with different masks between clustered and non-clustered layers, where sparsity is set per neuron

        # mode1: Nonadaptive per row sparsification with the same mask between clustered and non-clustered layers
        # mode2: Nonadaptive per row sparsification with different masks between clustered and non-clustered layers
        # mode3: Nonadaptive whole matrix sparsification with the same mask between clustered and non-clustered layers
        # mode4: Nonadaptive whole matrix sparsification with different masks between clustered and non-clustered layers

        assert self.mode in mode_choices, f"Mode must be one of {mode_choices}"
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.0
    ):
        if prunen != 0:
            assert self.mode == "default", "NM pruning only supported in default mode"
        if self.mode == "2:4+default" or self.mode == "2:4+SSPN" or self.mode == "2:4+DSPN":
            assert sparsity >= 0.5, "Sparsity must be greater than 0.5 for 2:4+ mode"
            prunen = 2
            prunem = 4
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()
        
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        inc_damp = damp / 10
        inc_count = 0
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        needs_more_damping = True

        tick = time.time()

        while needs_more_damping:
            H_testing = H.clone()
            try:
                H_testing = torch.linalg.cholesky(H_testing)
                H_testing = torch.cholesky_inverse(H_testing)
                H_testing = torch.linalg.cholesky(H_testing, upper=True)
                if not torch.isnan(H_testing).any():
                    needs_more_damping = False
                else:
                    H[diag, diag] += inc_damp
                    inc_count += 1
            except:
                H[diag, diag] += inc_damp
                inc_count += 1
            del H_testing

        if inc_count > 0:
            print("Hessian not PSD, added damping", inc_count, "times")

        self.H = H.clone()

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        # does H have any NaNs?
        if torch.isnan(H).any():
            print("H has NaNs, aborting")
            return

        Hinv = H

        mask = None

        # if self.mode is not default or or if self.mode is not SSPN or DSPN
        if self.mode != "default" and self.mode != "SSPN" and self.mode != "DSPN":
            # if mode is mode1 or mode2
            if self.mode in ["mode1", "mode2"]:
                if self.mode == "mode1":
                    tmp = W ** 2
                elif self.mode == "mode2":
                    tmp = W ** 2 / (torch.diag(Hinv).reshape((1, -1))) ** 2

                thresh = torch.sort(tmp, dim=1)[0][:, int(tmp.shape[1] * sparsity)]
                mask = torch.le(tmp, thresh.unsqueeze(1))
            
            elif self.mode in ["mode3", "mode4"]:
                if self.mode == "mode3":
                    tmp = W ** 2
                elif self.mode == "mode4":
                    tmp = W ** 2 / (torch.diag(Hinv).reshape((1, -1))) ** 2

                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask = torch.le(tmp, thresh)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    if self.mode == "default":
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        mask1 = tmp <= thresh
                    elif self.mode == "SSPN":
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp, dim=1)[0][:, int(tmp.shape[1] * sparsity)]
                        mask1 = torch.le(tmp, thresh.unsqueeze(1))
                    elif self.mode == "DSPN":
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp, dim=1)[0]
                        indices = torch.tensor(tmp.shape[1] * sparsity, dtype=torch.int64).to(self.dev)
                        # convert indices, an np array, to a tensor with int64 dtype
                        selected = torch.gather(thresh, 1, indices.unsqueeze(1)).squeeze()
                        mask1 = torch.le(tmp, selected.unsqueeze(1))
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            
            if self.mode == "2:4+default" or self.mode == "2:4+SSPN" or self.mode == "2:4+DSPN":
                if self.mode == "2:4+default":
                    tmp = Q1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
                elif self.mode == "2:4+SSPN":
                    tmp = Q1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp, dim=1)[0][:, int(tmp.shape[1] * sparsity)]
                    mask1 = torch.le(tmp, thresh.unsqueeze(1))
                elif self.mode == "2:4+DSPN":
                    tmp = Q1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp, dim=1)[0]
                    indices = (tmp.shape[1] * sparsity).to(torch.int64)
                    selected = torch.gather(thresh, 1, indices.unsqueeze(1)).squeeze()
                    mask1 = torch.le(tmp, selected.unsqueeze(1))

                del W1
                del Q1
                del Err1
                del Losses1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]
                    q = w.clone()
                    q[mask1[:, i]] = 0

                    if hasattr(self, 'quantizer'):
                        q = quantize(
                            q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                        ).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        del self.H
        self.H = None
        torch.cuda.empty_cache()