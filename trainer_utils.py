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
from reward_datasets import reward_data_collator
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
    # first_two_acc =  (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum() 

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
        calibration_errors[f"calibration_ACE_bin{num_bins}"] = average_error
        calibration_errors[f"calibration_MCE_bin{num_bins}"] = max_error

    log = {"Preference total Acc": all_acc.item(), 
           # "First-two Acc": first_two_acc.item(),
           "Test Data Size": len(score_mask_larger),
           **calibration_errors}
    return log

def compute_metrics_output_logits(prediction: EvalPrediction):
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
    # first_two_acc =  (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum() 

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

    # 
    # label_list = score_mask_larger.reshape(-1).tolist()
    # logits_list = logits_diff.reshape(-1).tolist()
    # mask_list = total_mask.reshape(-1).tolist()

    # y_true, y_logits = [], []
    # for label, logit, mask in zip(label_list, logits_list, mask_list):
    #     if mask:
    #         y_true.append(label)
    #         y_logits.append(logit)

    np_logits, np_scores = prediction.predictions, prediction.label_ids
    logits_diff = np_logits[:,0] - np_logits[:,1]
    mask = np_scores[:,0] - np_scores[:,1]
    logits_diff[mask<0] = - logits_diff[mask<0]

    logits_data = {"logits_diff": logits_diff.tolist(), "total_mask": total_mask.tolist(), "score_mask_larger": score_mask_larger.tolist(), "probs":numpy_sigmoid(logits_diff).tolist()}

    log = {"Preference total Acc": all_acc.item(), 
           "Test Data Size": len(score_mask_larger),
           "logits_data": logits_data,
           **calibration_errors}
    return log

# def compute_metrics_with_calibration(prediction: EvalPrediction):
#     logits = torch.from_numpy(prediction.predictions)
#     scores = torch.from_numpy(prediction.label_ids)

#     logits_diff = logits.unsqueeze(1) - logits.unsqueeze(
#         2
#     )  # [batch_size, num_sample, num_sample]

#     score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.0
#     score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.0
#     score_mask = score_mask_larger - score_mask_smaller
#     pad_mask = (scores >= 0).unsqueeze(1) * 1.0 * (scores >= 0).unsqueeze(2)

#     # calculate accuracy...
#     pred_compare = (logits_diff.detach() * score_mask > 0.0) * 1.0
#     total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
#     correct_compare = (pred_compare == score_mask_larger) * total_mask

#     all_acc = 1.0*correct_compare.sum() / total_mask.sum()
#     average_score = logits.mean()
#     # first_two_acc =  (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum()

#     calibration_bins = [5, 10, 20]
#     calibration_errors = {}
#     for num_bins in calibration_bins:
#         expected_error, average_error, max_error = rm_calibration_errors(
#             labels=score_mask_larger,
#             probs=numpy_sigmoid(logits_diff),
#             masks=total_mask,
#             num_bins=num_bins,
#         )
#         calibration_errors[f"calibration_ECE_bin{num_bins}"] = expected_error
#         calibration_errors[f"calibration_ACE_bin{num_bins}"] = average_error
#         calibration_errors[f"calibration_MCE_bin{num_bins}"] = max_error

#     return {
#         "Preference total Acc": all_acc.item(),
#         "Average Score": average_score.item(),
#         "Test Data Size": len(score_mask_larger),
#         **calibration_errors,
#     }


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


def multiobj_ranking_loss(logits, scores, weights):
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.0
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.0
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1.0 * (scores >= 0).unsqueeze(2)

    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask

    log_prob = torch.nn.functional.logsigmoid(logits_diff * score_mask * pad_mask)

    # # multi obj baseline
    # total_loss = - (log_prob * total_mask).sum()
    # multi obj
    # weights = np.ones_like(weights)/len(weights)

    total_loss = 0.0
    for w, tid in zip(weights, range(log_prob.shape[0])):
        total_loss += (
            -w * (log_prob[tid] * total_mask[tid]).sum()
        )  # batch level re-weight
    total_loss = total_loss * len(weights)  # rescale

    total_pairs = total_mask.sum()
    # assert total_pairs <= 0

    return total_loss / total_pairs if total_pairs > 0 else total_loss



# def augmentation(self, all_train_data):
#         short_text_indices = []
#         for i, sample in enumerate(all_train_data):
#             text = sample['text']
#             if text[0].count(' ')+text[0].count('') < 1024 and text[0].count('Human')==1 and text[0].count('Assistant') == 1:
#                 short_text_indices.append(i)

#         print_rank_0(f">>> One-round text indices: {len(short_text_indices)}")

#         augment_data = []
#         for i in short_text_indices:
#             # obtain i
#             sample = all_train_data[i]
#             text, score = sample['text'], sample['score']
#             chat_win, chat_lose = [x for x in text[0].split('\n') if x], [x for x in text[1].split('\n') if x]
#             if score[0] < score[1]:
#                 chat_win, chat_lose = chat_lose, chat_win
#             query_i, res_i_win, res_i_lose = chat_win[0], chat_win[1], chat_lose[1]
#             assert chat_win[0] == chat_lose[0]
            
#             # obtain j
#             j = random.sample([x for x in short_text_indices if x != i], 1)[0]
#             sample = all_train_data[j]
#             text, score = sample['text'], sample['score']
#             chat_win, chat_lose = [x for x in text[0].split('\n') if x], [x for x in text[1].split('\n') if x]
#             if score[0] < score[1]:
#                 chat_win, chat_lose = chat_lose, chat_win
#             query_j, res_j_win, res_j_lose = chat_win[0], chat_win[1], chat_lose[1]
#             assert chat_win[0] == chat_lose[0]

#             # augment 1
#             prompt = f"Human: please answer two questions. \nQuestion 1: {query_i}\nQuestion 2: {query_j}"    
#             response_win = f"Assistant: sure. Answer to question 1: {res_i_win}\n Answer to question 2: {res_j_win}"
#             response_lose = f"Assistant: sure. Answer to question 1: {res_i_win}\n Answer to question 2: {res_j_lose}"
#             new_sample = deepcopy(sample)
#             new_sample['text'] = [prompt+'\n\n'+response_win, prompt+'\n\n'+response_lose]
#             new_sample['score'] = [1, 0]
#             augment_data.append(new_sample)

#             # augment 2
#             prompt = f"Human: please answer two questions. \nQuestion 1: {query_i}\nQuestion 2: {query_j}"    
#             response_win = f"Assistant: sure. Answer to question 1: {res_i_win}\n Answer to question 2: {res_j_win}"
#             response_lose = f"Assistant: sure. Answer to question 1: {res_i_lose}\n Answer to question 2: {res_j_win}"
#             new_sample = deepcopy(sample)
#             new_sample['text'] = [prompt+'\n\n'+response_win, prompt+'\n\n'+response_lose]
#             new_sample['score'] = [1, 0]
#             augment_data.append(new_sample)


#             # augment 3
#             prompt = f"Human: please answer two questions. \nQuestion 1: {query_i}\nQuestion 2: {query_j}"    
#             response_win = f"Assistant: sure. Answer to question 1: {res_i_lose}\n Answer to question 2: {res_j_win}"
#             response_lose = f"Assistant: sure. Answer to question 1: {res_i_lose}\n Answer to question 2: {res_j_lose}"
#             new_sample = deepcopy(sample)
#             new_sample['text'] = [prompt+'\n\n'+response_win, prompt+'\n\n'+response_lose]
#             new_sample['score'] = [1, 0]
#             augment_data.append(new_sample)

#             # augment 4
#             prompt = f"Human: please answer two questions. \nQuestion 1: {query_i}\nQuestion 2: {query_j}"    
#             response_win = f"Assistant: sure. Answer to question 1: {res_i_win}\n Answer to question 2: {res_j_lose}"
#             response_lose = f"Assistant: sure. Answer to question 1: {res_i_lose}\n Answer to question 2: {res_j_lose}"
#             new_sample = deepcopy(sample)
#             new_sample['text'] = [prompt+'\n\n'+response_win, prompt+'\n\n'+response_lose]
#             new_sample['score'] = [1, 0]
#             augment_data.append(new_sample)
        
#         # add to main data set
#         random.shuffle(augment_data)
#         augment_size = int(min(self.args.augment_rate*len(all_train_data), len(augment_data)))
#         all_train_data.extend(augment_data[0:augment_size])
#         print_rank_0(f">>> Augment data size: {augment_size}")

#         return all_train_data
