import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, List, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
import os
from transformers import Trainer, AutoConfig
from transformers import EvalPrediction
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
import deepspeed
import json
from utils import print_rank_0
from min_norm_solvers import MinNormSolver
import numpy as np
from utils import gradient_normalizer, print_rank_0
from reward_datasets import reward_data_collactor
from sklearn.calibration import calibration_curve
import datetime

from utils import gradient_normalizer, print_rank_0, calibration_error, numpy_sigmoid

from copy import deepcopy


def rm_calibration_curve(labels, probs, masks, num_bins):
    label_list = labels.view(-1).tolist()
    prob_list = probs.view(-1).tolist()
    mask_list = masks.view(-1).tolist()

    y_true, y_prob = [], []
    for label, prob, mask in zip(label_list, prob_list, mask_list):
        if mask:
            y_true.append(label)
            y_prob.append(prob)
    return calibration_curve(np.array(y_true), np.array(y_prob), n_bins=num_bins)

def rm_calibration_errors(labels, probs, masks, num_bins):
    label_list = labels.reshape(-1).tolist()
    prob_list = probs.reshape(-1).tolist()
    mask_list = masks.reshape(-1).tolist()

    y_true, y_prob = [], []
    for label, prob, mask in zip(label_list, prob_list, mask_list):
        if mask:
            y_true.append(label)
            y_prob.append(prob)

    return calibration_error(np.array(y_true), np.array(y_prob), n_bins=num_bins)


def more_batch_creator(group_inputs):
    scores = []
    input_ids = []
    attention_mask = []
    for item in group_inputs:
        scores.append(item['score'])
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])

    return {
        "score": torch.cat(scores, dim=0).float(),
        "input_ids": torch.cat(input_ids, dim=0).long(),
        "attention_mask": torch.cat(attention_mask, dim=0).float()
    }

def compute_metrics(prediction: EvalPrediction):
    logits = torch.from_numpy(prediction.predictions)
    scores = torch.from_numpy(prediction.label_ids)
    
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)  # [batch_size, num_sample, num_sample]

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    # calculate accuracy...
    pred_compare = (logits_diff.detach() > 0.) * 1.
    # pred_compare = (logits_diff.detach() * score_mask > 0.) * 1.
    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
    correct_compare = (pred_compare == score_mask_larger) * total_mask
    
    all_acc = correct_compare.sum() / total_mask.sum()
    first_two_acc =  (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum() 

    calibration_bins = [5, 10, 20]
    calibration_errors = {}
    for num_bins in calibration_bins:
        expected_error, average_error, max_error = rm_calibration_errors(
            labels=score_mask_larger,
            probs=numpy_sigmoid(logits_diff),
            masks=total_mask,
            num_bins=num_bins,
        )
        calibration_errors[f"calibration_ECE_bin{num_bins}"] = expected_error
        #calibration_errors[f"calibration_ACE_bin{num_bins}"] = average_error
        #calibration_errors[f"calibration_MCE_bin{num_bins}"] = max_error

    log = {"Preference total Acc": all_acc.item(), 
           "Test Data Size": len(score_mask_larger),
           **calibration_errors}
    return log

def language_modeling_loss(
    lm_logits, input_ids, scores, loss_mask, score_thresh=0.9, eps=1e-7
):
    batch_size, seq_length, vocab_size = lm_logits.shape

    lm_probs = torch.nn.functional.cross_entropy(
        input=lm_logits[:, :-1, :].reshape(-1, vocab_size),
        target=input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(batch_size, -1)

    loglikeli = (lm_probs * loss_mask[:, 1:].float()).sum(dim=-1) / loss_mask[
        :, 1:
    ].float().sum(dim=-1)
    score_mask = (scores.reshape(-1) > score_thresh).float()
    return (loglikeli * score_mask).sum() / (score_mask.sum() + eps)

