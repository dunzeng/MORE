import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, List, Tuple, Union
import datetime
from copy import deepcopy

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

from reward_datasets import reward_data_collator, more_data_collator
from sklearn.calibration import calibration_curve
from utils import gradient_normalizer, print_rank_0, calibration_error, numpy_sigmoid
from trainer_utils import compute_metrics, calibration_curve, language_modeling_loss, more_batch_creator


def full_batch_creator(group_inputs):
    scores = []
    input_ids = []
    attention_mask = []
    for item in group_inputs:
        scores.append(item["score"])
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])

    return {
        "score": torch.cat(scores, dim=0).float(),
        "input_ids": torch.cat(input_ids, dim=0).long(),
        "attention_mask": torch.cat(attention_mask, dim=0).float(),
    }

def more_ranking_loss(logits, scores, weights, resampling, task_mask):
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.0
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.0
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1.0 * (scores >= 0).unsqueeze(2)

    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask

    log_prob = torch.nn.functional.logsigmoid(logits_diff * score_mask * pad_mask)

    if resampling:
        total_loss = 0.0
        for w, tid in zip(weights, range(log_prob.shape[0])):
            total_loss += - w*(log_prob[tid] * total_mask[tid]).sum() # batch level re-weight
        total_loss = total_loss*len(weights) #rescale
        total_pairs = total_mask.sum()
        # assert total_pairs <= 0
    else:
        total_loss = 0.0
        for w, tid in zip(weights, range(log_prob.shape[0])):
            total_loss += - w*(log_prob[tid] * total_mask[tid]).sum() # batch level re-weight
        total_loss = total_loss * len(set(task_mask))  # rescale
        total_pairs = total_mask.sum()
    return  total_loss / total_pairs  if total_pairs > 0 else total_loss


def ranking_loss(logits, scores):  # with shape [bs, r]
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.0
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.0
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1.0 * (scores >= 0).unsqueeze(2)

    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask

    log_prob = torch.nn.functional.logsigmoid(logits_diff * score_mask * pad_mask)

    total_loss = -(log_prob * total_mask).sum()
    total_pairs = total_mask.sum()

    return total_loss / total_pairs if total_pairs > 0 else total_loss


def gather_all_with_local_grad(tensor, dim=0):
    local_rank = torch.distributed.get_rank()

    with torch.no_grad():
        all_tensors = [torch.zero_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(all_tensors, tensor)
    all_tensors[local_rank] = tensor

    return torch.stack(all_tensors, dim=dim)


def copy_last_layers(model, num_layers):
    layers = list(model.children())
    print_rank_0(layers)
    last_layers = deepcopy(layers[-num_layers:])
    new_model = nn.Sequential(*last_layers).cpu()
    print_rank_0(list(new_model.children()))
    return new_model


def serialize_model_parameters(model):
    params = model.parameters()
    params_vector = torch.cat([param.detach().view(-1) for param in params])
    return params_vector


def deserialize_model_parameters(model, params_vector):
    idx = 0
    for param in model.parameters():
        num_param_elements = param.numel()
        param_values = params_vector[idx : idx + num_param_elements]
        param_values = param_values.view(param.shape)
        param.data.copy_(param_values)
        idx += num_param_elements


class RewardModelTrainer(Trainer):
    def init_multiobj(self):
        self.lambda_ = np.ones_like(self.args.task_num) / self.args.task_num

        self.more_base = nn.Linear(self.model.config.hidden_size, 1, bias=False).cpu()
        # self.grad_m = [torch.zeros_like(self.more_base.weight.data.view(-1)) for i in range(self.args.task_num)]

        # self.more_base = copy_last_layers(self.model, 1)
        self.grad_m = [
            serialize_model_parameters(self.more_base).detach().cpu()
            for i in range(self.args.task_num)
        ]

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[List[str]] = None,
    ):
        device = model.device
        labels = inputs["score"].to(device)

        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def compute_lambda(self, weight, task_num, rm_embeddings, scores, batch_size, sample_num):
        # load
        deserialize_model_parameters(self.more_base, deepcopy(weight))
        self.more_base.requires_grad = True
        scores = scores.to("cpu")

        # compute
        grads = []
        loss_data = []
        rm_logits = self.more_base(rm_embeddings).view(batch_size, sample_num)
        for i in range(task_num):
            if self.args.debug_mode:
                print_rank_0(f">>> rm embedding {rm_embeddings.shape}")
                print_rank_0(f">>> rm logits {rm_logits.shape}")
                print_rank_0(f">>> rm logits {rm_logits[i].shape} >>> score {scores[i].shape}")

            rm_loss = ranking_loss(rm_logits[[i]], scores[[i]])
            rm_loss.backward(retain_graph=True)
            #grads.append(deepcopy(self.more_base.weight.grad.data.detach().cpu().view(-1)))
            grad = deepcopy(self.more_base.weight.grad.data.detach().cpu().view(-1))
            self.grad_m[i] = (1-self.args.alpha)*self.grad_m[i]+ self.args.alpha*grad
            loss_data.append(rm_loss.detach().item())
            self.more_base.weight.grad.data.zero_()
        
        grads = gradient_normalizer(self.grad_m, loss_data, self.args.normalize)
        lambda_, _ = MinNormSolver.find_min_norm_element_FW(grads)
        return lambda_

    def compute_lambda_noresampling(self, head_weight, task_mask, embeddings, scores, batch_size, sample_num):
        deserialize_model_parameters(self.more_base, head_weight)
        self.more_base.requires_grad = True
        scores = scores.to("cpu")

        rm_logits = self.more_base(embeddings).view(batch_size, sample_num) # (batch_size)
        loss_data = []
        for task_id in set(task_mask):
            t_mask = (torch.tensor(task_mask) == task_id)
            if self.args.debug_mode:
                print_rank_0(">>> task mask {}".format(task_mask))
                print_rank_0(">>> rm_logits {}, scores {}".format(rm_logits[t_mask], scores[t_mask]))
                # assert len(set(t_mask)) == self.args.task_num
            
            rm_loss = ranking_loss(rm_logits[t_mask], scores[t_mask])
            rm_loss.backward(retain_graph=True)
            
            grad = deepcopy(self.more_base.weight.grad.data.detach().cpu().view(-1))
            if self.args.debug_mode:
                print_rank_0(">>> grad {}".format(grad))
            self.more_base.weight.grad.data.zero_()
            
            self.grad_m[task_id] = (1 - self.args.alpha) * self.grad_m[task_id] + self.args.alpha * grad
            loss_data.append(rm_loss.detach().item())

        if len(set(task_mask)) > 1:
            grads = [self.grad_m[tid] for tid in set(task_mask)]
            grads = gradient_normalizer(grads, loss_data, self.args.normalize)
            lambda_, _ = MinNormSolver.find_min_norm_element_FW(grads)
        else:
            lambda_ = [1.0]
        # if self.args.debug_mode:
        #     print_rank_0(">>> LAMBDA {}".format(lambda_))
        return lambda_

    def compute_more_loss(self, model, inputs, return_outputs=False):
        full_batch, task_mask = inputs
        device = model.device
        if self.args.debug_mode:
            print_rank_0(">>> task mask {}".format(task_mask))

        # loss computing
        total_loss = 0.0
        # to device
        device = model.device

        if self.args.resampling:
            full_batch = full_batch_creator(full_batch)
        
        scores = full_batch["score"].to(device)
        input_ids = full_batch["input_ids"].to(device)
        attention_mask = full_batch["attention_mask"].to(device)

        batch_size, sample_num, seq_length = input_ids.shape  # batch_size
        outputs = model(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length),
            padding_side=self.args.padding_side,
            pooling_type=self.args.pooling_type,
        )
        rm_embeddings = (outputs["rm_embeddings"].view(batch_size, sample_num, -1).detach().cpu())
        batch_logits = outputs["rm_logits"].view(batch_size, sample_num)

        # computing reweighting factors
        if self.args.reweight:
            for n, lp in model.named_parameters():
                if n == "module.reward_head.weight":
                    model_weight = deepspeed.utils.safe_get_full_fp32_param(lp) # get head weight
    
            if self.args.resampling:
                # full task
                lambda_ = self.compute_lambda(model_weight.cpu(), self.args.task_num, rm_embeddings, deepcopy(scores), batch_size, sample_num)
            else:
                # partial task
                lambda_ = self.compute_lambda_noresampling(model_weight.cpu(), task_mask, rm_embeddings, deepcopy(scores), batch_size, sample_num)
                lambda_ = {tid: w for w, tid in zip(lambda_, list(set(task_mask)))}
                lambda_ = torch.Tensor([lambda_[tid] for tid in task_mask])
        else:
            if self.args.resampling:
                task_num = float(len(full_batch)) 
                lambda_ = np.ones(shape=(int(task_num),))/task_num
            else:  
                lambda_ = {tid: 1.0 / len(set(task_mask)) for tid in task_mask}
        
        if self.args.debug_mode:
            print_rank_0(f">>> applied lambda {lambda_}")
            print_rank_0(f">>> gradients {self.grad_m}")

        rm_loss = more_ranking_loss(batch_logits, scores, lambda_, self.args.resampling, task_mask)
        total_loss = rm_loss

        # print_rank_0(total_loss)
        if self.args.debug_mode:
            print_rank_0(f">>> debug")
            print_rank_0(f">>> MORE Ranking loss {rm_loss}")

        return (total_loss, batch_logits) if return_outputs else total_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.more and model.training:
            if self.args.debug_mode:
                print_rank_0("-----Running MORE-----")
            more_loss = self.compute_more_loss(model, inputs, return_outputs)
            return more_loss

        elif not self.args.more or not self.model.training:  # vanilla loss
            # inputs, task_mask = inputs
            device = model.device
            scores = inputs["score"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            batch_size, sample_num, seq_length = input_ids.shape

            if self.args.debug_mode:
                print(f">>> input_ids shape {input_ids.shape}")

            outputs = model(
                input_ids=input_ids.view(-1, seq_length),
                attention_mask=attention_mask.view(-1, seq_length),
                padding_side=self.args.padding_side,
                pooling_type=self.args.pooling_type,
            )

            hidden_states = outputs["hidden_states"]  # shape [bs*r, seq_length, dim]

            batch_logits = outputs["rm_logits"].view(batch_size, sample_num)

            rm_loss = ranking_loss(batch_logits, scores)

            lm_loss = 0

            total_loss = rm_loss + self.args.lm_loss_coeff * lm_loss
            # print_rank_0(total_loss)
            if self.args.debug_mode:
                print_rank_0(f">>> debug")
                print_rank_0(f">>> Language modeling loss {lm_loss}")
                print_rank_0(f">>> Ranking loss {rm_loss}")

            return (total_loss, batch_logits) if return_outputs else total_loss

        else:
            assert False

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        if self.args.more:
            self.data_collator = reward_data_collator
            tmp = super().get_eval_dataloader(eval_dataset)
            self.data_collator = more_data_collator
            return tmp
        else:
            return super().get_eval_dataloader(eval_dataset)
