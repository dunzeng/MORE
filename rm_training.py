import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json

import torch
import transformers

from datasets import interleave_datasets
from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig
from transformers import EvalPrediction


from model import LlamaRewardModel, BertRewardModel, PythiaRewardModel
from utils import print_rank_0
from reward_datasets import TextRewardDataset, MultiRewardTextDataset, reward_data_collator, more_data_collator, more_data_collator_without_resampling
from reward_datasets import load_text_score_dataset
from arguments import CustomTrainingArguments
from trainer import RewardModelTrainer
from trainer_utils import compute_metrics, compute_metrics_output_logits
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.set_seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_eval_datasets(args, tokenizer):
    data_dict = {}

    for data_path in args.eval_data_path:
        eval_data_list = load_text_score_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            debug=args.debug_mode,
            padding=not args.per_device_eval_batch_size == 1
        )
        eval_dataset = TextRewardDataset([eval_data_list])
        data_dict[data_path] = eval_dataset
        print_rank_0(f">>> train datasets from {args.eval_data_path} - has the number of samples {len(eval_dataset)}")

    return data_dict

def set_llama_tokenizer(model, tokenizer):
    tokenizer.pad_token_id = 3
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 0
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print_rank_0(tokenizer)
    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)
    

    # setup model
    #---------------------------------------------------------------------------------
    print_rank_0(f"Begin loading model from {args.model_name_or_path}")
    if args.model_type == "llama":
        model = LlamaRewardModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    elif args.model_type == "pythia":
        model = PythiaRewardModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,)
    else:
        model = BertRewardModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )

    print_rank_0(model)
    print_rank_0(f"Finished loading model from {args.model_name_or_path}")

    model.is_parallelizable = True
    model.model_parallel = True

    # active_modules = args.active_module_name.split(',')
    # for name, param in model.named_parameters():
    #     if any(nd in name for nd in active_modules) and "lm_head" not in name:
    #         param.requires_grad = True
    #         print_rank_0(f"layer {name} activated")
    #     else:
    #         param.requires_grad = False
    #         print_rank_0(f"layer {name} freezed")
    # setup tokenizer
    #---------------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.max_length,        
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        use_fast=False,
    )

    if args.model_type in ["llama", "pythia"]:
        model, tokenizer = set_llama_tokenizer(model=model, tokenizer=tokenizer)
        print_rank_0(f"check tokenizer length {len(tokenizer)}")

    # load data
    #---------------------------------------------------------------------------------    
    if args.do_train:
        train_dataset = MultiRewardTextDataset(tokenizer, args)
    else:
        train_dataset = None
    eval_dataset_dict = get_eval_datasets(args, tokenizer)

    # build trainer
    #---------------------------------------------------------------------------------
    if args.more:
        if args.resampling:
            collator = more_data_collator
        else:
            collator = more_data_collator_without_resampling
    else:
        collator = reward_data_collator
    trainer = RewardModelTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=args,
        compute_metrics=compute_metrics if args.do_train else compute_metrics_output_logits,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=collator,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False

    if args.more:
        trainer.init_multiobj()

    if args.do_train:
        if args.eval_at_start:
            for eval_set_name, eval_dataset in eval_dataset_dict.items():
                eval_result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval_"+eval_set_name)
                print_rank_0(eval_result)

        with torch.autocast("cuda"): 
            if args.resume_from_checkpoint:
                train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            else:
                train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)

    final_eval_results ={}
    for eval_set_name, eval_dataset in eval_dataset_dict.items():
        metric_key_prefix="eval_"+eval_set_name.split('/')[-1]
        eval_result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix)
        if trainer.compute_metrics is compute_metrics_output_logits:
            if not os.path.exists(f"{args.output_dir}/reward_logs"):
                os.makedirs(f"{args.output_dir}/reward_logs", exist_ok=True)
            logits_data = eval_result.pop(f"{metric_key_prefix}_logits_data")
            # diff_mask = eval_result.pop(f"{metric_key_prefix}_scores")
            with open(f"{args.output_dir}/reward_logs/testdata-{eval_set_name.split('/')[-1]}.json", 'w') as f:
                json.dump(logits_data, f, ensure_ascii=False)
        print_rank_0(eval_result)
        final_eval_results[eval_set_name] = eval_result
        
    with open(f"{args.output_dir}/final_eval_results.json", 'w') as f:
        json.dump(final_eval_results, f, ensure_ascii=False)

    with open(f"{args.output_dir}/args.json", 'w') as f:
        # json.dump(vars(args), f, ensure_ascii=False)
        f.write(str(args))
        
if __name__ == "__main__":
    train()
