import json
from tqdm import tqdm
import gzip
import random
from copy import deepcopy
import argparse

from utils import print_rank_0
from pprint import pprint
import numpy as np

from datasets import interleave_datasets

import torch
from torch.utils.data import Dataset

QUERY_PROMPT = "## Human:\n{request}\n\n## Assistant:\n{response}"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_data_iter(data_list, debug=False):
    if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
        return tqdm(data_list)
    else:
        return data_list

def reward_data_collactor(batch):
    scores = []
    input_ids = []
    attention_mask = []
    # task_mask = [item["task_id"] for item in batch]
    for item in batch:
        scores.append(item["score"])
        input_ids.append(item["tokens"]["input_ids"])
        attention_mask.append(item["tokens"]["attention_mask"])
    full_batch = {
            "score": torch.Tensor(scores).float(),
            "input_ids": torch.Tensor(input_ids).long(),
            "attention_mask": torch.Tensor(attention_mask).float()
        }
    return full_batch

def more_data_collactor_without_resampling(batch):
    # full batch
    scores = []
    input_ids = []
    attention_mask = []
    task_mask = [item["task_id"] for item in batch]
    for item in batch:
        scores.append(item["score"])
        input_ids.append(item["tokens"]["input_ids"])
        attention_mask.append(item["tokens"]["attention_mask"])
    full_batch = {
            "score": torch.Tensor(scores).float(),
            "input_ids": torch.Tensor(input_ids).long(),
            "attention_mask": torch.Tensor(attention_mask).float()
        }
    return full_batch, task_mask

def more_data_collactor(batch):
    # interleaves
    num_task = len(batch[0])
    batch_list = []
    for task_id in range(num_task):
        scores = []
        input_ids = []
        attention_mask = []
        for item in batch:
            scores.append(item[task_id]['score'])
            input_ids.append(item[task_id]['tokens']['input_ids'])
            attention_mask.append(item[task_id]['tokens']['attention_mask'])
        task_batch = {
                "score": torch.Tensor(scores).float(),
                "input_ids": torch.Tensor(input_ids).long(),
                "attention_mask": torch.Tensor(attention_mask).float()
            }
        batch_list.append(task_batch)        
    task_mask = [item for item in range(num_task)]
    return batch_list, task_mask

def reward_tokenize(sentences, tokenizer, padding="longest"):
    if isinstance(sentences, str):
        sentences = [sentences]
    input_ids = [
        [tokenizer.bos_token_id]
        + tokenizer.encode(sent, add_special_tokens=False)
        + [tokenizer.eos_token_id]
        for sent in sentences
    ]

    if padding == "longest":
        max_input_length = max([len(inp_ids) for inp_ids in input_ids])
        max_length = min(tokenizer.model_max_length, max_input_length)
    else:
        max_length = tokenizer.model_max_length

    outputs = {"input_ids": [], "attention_mask": []}
    for inp_ids in input_ids:
        attn_mask = [1] * len(inp_ids)
        if len(inp_ids) >= max_length:
            if tokenizer.truncation_side == "left":
                inp_ids = inp_ids[-max_length:]
                attn_mask = attn_mask[-max_length:]
            else:
                inp_ids = inp_ids[:max_length]
                attn_mask = attn_mask[:max_length]
        else:
            if tokenizer.padding_side == "left":
                inp_ids = [tokenizer.pad_token_id] * (
                    max_length - len(inp_ids)
                ) + inp_ids
                attn_mask = [0] * (max_length - len(attn_mask)) + attn_mask
            else:
                inp_ids = inp_ids + [tokenizer.pad_token_id] * (
                    max_length - len(inp_ids)
                )
                attn_mask = attn_mask + [0] * (max_length - len(attn_mask))

        # inp_ids = [ele if ele is not None else 0 for ele in inp_ids] # remove none
        outputs["input_ids"].append(deepcopy(inp_ids))
        outputs["attention_mask"].append(deepcopy(attn_mask))

    return outputs

def prepare_data_item(item, tokenizer=None, padding=False, max_response_num=1):
    PAIRWISE = 2 
    new_item = deepcopy(item)
    if not len(new_item["score"]) == len(new_item["text"]):
        print_rank_0("invalid data point {}".format(new_item))
        return None

    # remove OPT-IML response from gpt4llm
    new_item["score"] = new_item["score"][0:PAIRWISE]
    new_item["text"] = new_item["text"][0:PAIRWISE]

    assert len(new_item["score"]) == PAIRWISE

    score_idx = np.argsort(new_item["score"])
    max_score = max(new_item["score"]) + 1e-5

    new_item["score"] = [
        new_item["score"][s_i] / max_score for s_i in score_idx[::-1]
    ]  # normalize the scores
    new_item["text"] = [new_item["text"][s_i] for s_i in score_idx[::-1]]

    if padding:
        new_item["text"] += [" "] * (max_response_num - len(new_item["text"]))
        new_item["score"] += [-1.0] * (max_response_num - len(new_item["score"]))

    if tokenizer is not None:
        new_item["tokens"] = reward_tokenize(
            sentences=new_item["text"],
            tokenizer=tokenizer,
            padding="max_length" if padding else "longest",
        )
    # print_rank_0(new_item)
    return new_item


def load_jsonl_data(data_path):
    print_rank_0("loading text-score dataset from: \n   {}".format(data_path))
    with open(data_path, "r") as f:
        lines = f.read().strip().split("\n")

    data_list = [json.loads(l) for l in lines]

    return data_list


def load_json_data(data_path):
    print_rank_0("loading text-score dataset from: \n   {}".format(data_path))
    with open(data_path, "r") as f:
        data_list = json.load(f)

    return data_list

def load_text_score_dataset(data_path, tokenizer=None, debug=False, padding=False):
    print_rank_0("loading text-score dataset from: \n   {}".format(data_path))
    if data_path[-4:] == "json":
        data_list = load_json_data(data_path)
        if "train" in data_path:  # sampling train data
            data_list = data_list
        if "test" in data_path and "summ" in data_path:  # sampling train data
            # random.shuffle(data_list)
            data_list = data_list[0:2000] # 5% for summarization testing set
    else:
        data_list = load_jsonl_data(data_path)

    max_response_num = 1
    if padding:
        max_response_num = max([len(item["score"]) for item in data_list])
        print_rank_0(">>> response padding number: {}".format(max_response_num))

    outputs = []
    for item in get_data_iter(data_list, debug=debug):
        new_item = prepare_data_item(
            item,
            tokenizer=tokenizer,
            padding=padding,
            max_response_num=max_response_num,
        )
        if new_item is not None:
            outputs.append(new_item)

    print_rank_0("finished processing {}  data.".format(len(outputs)))
    return outputs


class MultiRewardTextDataset(Dataset):
    def __init__(self, tokenizer, args) -> None:
        self.args = args
        all_train_data = []
        for task_id, data_path in enumerate(args.train_data_path):
            data_list = load_json_data(data_path)
            if "summ." in data_path:
                data_list = data_list[:40000]
            assert "train" in data_path and "json" in data_path
            for item in data_list:
                item["task_id"] = task_id
            all_train_data.append(data_list)
        assert self.args.task_num == len(all_train_data)
        self.task_num = self.args.task_num

        if args.debug_mode:
            for data_list in all_train_data:
                print_rank_0(f"{len(data_list)} - {data_list[0].keys()}")

        # resampling
        if args.resampling: 
            print_rank_0(f"resampling data with size {args.resampling_size}")
            all_train_data = self.resampling(all_train_data)
    
        # tokenizing
        all_train_data = self.preprocess(all_train_data, tokenizer)
    
        if args.more:
            if args.resampling:
                self.dataset = []
                for index in range(args.resampling_size):
                    group_data = []
                    for task_id in range(self.task_num):
                        item = all_train_data[task_id][index]
                        group_data.append(item)  # [sample_1, ..., sample_task_num]
                    self.dataset.append(group_data)      
            else:
                self.dataset = []
                for task_data in all_train_data:
                    self.dataset.extend(task_data)
                random.shuffle(self.dataset)
        else:
            self.dataset = []
            for task_data in all_train_data:
                self.dataset.extend(task_data)

        if args.shuffle:
            random.shuffle(self.dataset)
        print_rank_0(">>>  Total data size {}".format(len(self.dataset)))

    def preprocess(self, all_train_data, tokenizer):
        tokenized_train_data = []
        for data_list in all_train_data:
            padding = self.args.more or not self.args.per_device_train_batch_size == 1
            max_response_num = 1
            if padding:
                max_response_num = max([len(item["score"]) for item in data_list])
                print_rank_0(">>> response padding number: {}".format(max_response_num))

            outputs = []
            for item in get_data_iter(data_list, debug=self.args.debug_mode):
                new_item = prepare_data_item(
                    item,
                    tokenizer=tokenizer,
                    padding=padding,
                    max_response_num=max_response_num,
                )
                if new_item is not None:
                    outputs.append(new_item)
            print_rank_0(">>> finished processing {}  data.".format(len(outputs)))
            tokenized_train_data.append(outputs)
        return tokenized_train_data

    def resampling(self, all_train_data):
        args = self.args
        # resampling - balance dataset across preferences
        for task_id in range(self.task_num):
            data_size = len(all_train_data[task_id])
            if data_size < args.resampling_size:
                ext_size = args.resampling_size - len(all_train_data[task_id])
                if ext_size > len(all_train_data[task_id]):
                    pool_size = len(all_train_data[task_id])
                    enlarge = int(ext_size/pool_size) + (1 if ext_size%pool_size > 1 else 0)
                    sample_pool = enlarge * all_train_data[task_id]
                ext_data = deepcopy(random.sample(sample_pool, ext_size))
                # ext_data = deepcopy(all_train_data[task_id][0:ext_size])
                all_train_data[task_id].extend(ext_data)
            else:
                all_train_data[task_id] = all_train_data[task_id][
                    0 : args.resampling_size
                ]
        return all_train_data

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    

class TextRewardDataset(Dataset):
    # for test dataset only
    def __init__(self, data):
        all_train_data = []
        for domain_data in data:
            all_train_data.extend(domain_data)
        self.data = all_train_data 

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self,):
        return len(self.data)

def get_reward_data(data_path):
    with open(data_path, "r") as f:
        lines = f.read().strip().split("\n")

    data = [json.loads(line) for line in lines]
    print_rank_0("finished loading data with {} lines".format(len(data)))

    for item in data:
        answers = item["answers"]
        for style, value in answers.items():
            if isinstance(value, str):
                continue
            elif isinstance(value, dict):
                if "choices" in value:
                    answers[style] = value["choices"][0]["message"]["content"]
                elif "content" in value:
                    answers[style] = value["content"]
                else:
                    print("check this value")
                    print(value)

    return data


if __name__ == "__main__":
    ...
