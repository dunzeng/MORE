from arguments import TrainingArguments, CustomTrainingArguments
from transformers import HfArgumentParser
from utils import print_object_on_main_process, print_rank_0, getDataset, loadTokenizerAndModel
from typing import Dict
import os
import json
from collator import rjs_data_collator
from utils import load_data_from_paths, load_jsonl_data, read_json_or_jsonl_data
from datasets import Dataset
from transformers import Trainer


def main():
    parser = HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)
    print_object_on_main_process("Arguments", args)

    print_rank_0("Loading data>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # data_list = load_jsonl_data(args.data_path)
    # train_dataset = Dataset.from_list(data_list)
    
    full_data = []
    for data_path in args.train_data_path:
        data_list = read_json_or_jsonl_data(data_path)
        full_data.extend(data_list)
    
    train_dataset = Dataset.from_list(full_data)
    
    # eval_dataset = getDataset(args, type='eval')
    print_object_on_main_process("training set", train_dataset, split_line_color="green", object_color="cyan")
    # print_object_on_main_process("evaluation set", eval_dataset, split_line_color="green", object_color="cyan")

    tokenizer, model = loadTokenizerAndModel(args)
    print_object_on_main_process("tokenizer", tokenizer, split_line_color="green", object_color="cyan")
    print_object_on_main_process("model", model, split_line_color="green", object_color="cyan")

    print_rank_0("Using rejection sampling data collator")
    data_collator = rjs_data_collator(tokenizer, args)
    compute_metrics = None

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        if args.save_training_states:
            trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

if __name__ == '__main__':
    main()