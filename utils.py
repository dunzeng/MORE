import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union, List, Dict

import tqdm
import copy

import torch
import numpy as np

def numpy_sigmoid(x):
    # r_x = x - x.max()
    return 1. / (1. + np.exp(-x))
    #return 1. / (1. + np.exp(- r_x))

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def gradient_normalizer(grads, loss, type="none"):
    if type == "l2":
        norms = [torch.norm(grad, p=2, dim=0).item() for grad in grads]
        norms = np.array([1 if item == 0 else item for item in norms])
        grads = [grad / norm for grad, norm in zip(grads, norms)]  # normalize

    elif type == "loss":
        loss = np.array(loss) + np.ones_like(loss)
        loss = loss / loss.sum()
        # norms = np.array([1 if item==0 else item for item in norms])
        # print_rank_0(norms)
        grads = [grad / ls for grad, ls in zip(grads, loss)]

    elif type == "loss+":
        # print_rank_0(loss)

        norms = [torch.norm(grad, p=2, dim=0).item() for grad in grads]
        norms = np.array([1 if item == 0 else item for item in norms])
        grads = [grad / (norm * ls) for ls, grad, norm in zip(loss, grads, norms)]

    elif type == "none":
        ...
    else:
        assert False
    # print_rank_0(grads)
    return grads


def calibration_error(
    y_true,
    y_prob,
    n_bins=5,
    strategy="uniform",
):
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    # prob_true = bin_true[nonzero] / bin_total[nonzero]
    # prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    # return prob_true, prob_pred, bin_total[nonzero]
    try:
        expected_error = np.abs(bin_sums - bin_true).sum() / len(y_prob)
        average_error = (
            np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]
        ).mean()
        max_error = (
            np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]
        ).max()
    except:
        expected_error, average_error, max_error = 0.0, 0.0, 0.0
    return expected_error, average_error, max_error

def multi_reward_data_collactor(batch):
    num_task = len(batch[0])
    batch_list = []
    for task_id in range(num_task):
        scores = []
        input_ids = []
        attention_mask = []
        for item in batch:
            scores.append(item[task_id]["score"])
            input_ids.append(item[task_id]["tokens"]["input_ids"])
            attention_mask.append(item[task_id]["tokens"]["attention_mask"])
        task_batch = {
            "score": torch.Tensor(scores).float(),
            "input_ids": torch.Tensor(input_ids).long(),
            "attention_mask": torch.Tensor(attention_mask).float(),
        }
        batch_list.append(task_batch)
    return batch_list