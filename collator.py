import torch
from arguments import TrainingArguments
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from utils import print_rank_0
from datasets import Dataset
from random import sample

def _llm_tokenize(prompts: List[str], texts: List[str], tokenizer: PreTrainedTokenizer, args: TrainingArguments) -> Dict[str, torch.Tensor]:
    if prompts is None:
        input_ids = []
        labels = []
        for text in zip(texts):
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            label = deepcopy(text_ids)

            input_ids.append(torch.tensor(text_ids[-args.max_length:]))
            labels.append(torch.tensor(label[-args.max_length:]))
            
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        if args.pad_labels_with_ignore:
            labels = pad_sequence(labels, batch_first=True, padding_value=args.ignore_token_id)
        else:
            labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)
    else:
        input_ids = []
        labels = []
        for prompt, text in zip(prompts, texts):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            label = deepcopy(text_ids)
            response_start_idx = len(prompt_ids)
            if prompt_ids != text_ids[:response_start_idx]:
                response_start_idx -= 1
                prompt_ids = text_ids[:response_start_idx]
            text_ids = [tokenizer.bos_token_id] + text_ids + [tokenizer.eos_token_id]
            label = [tokenizer.bos_token_id] + label + [tokenizer.eos_token_id]
            if args.only_predict_answer:
                label[:len(prompt_ids) + 1] = [args.ignore_token_id] * (len(prompt_ids) + 1)

            input_ids.append(torch.tensor(text_ids[-args.max_length:]))
            labels.append(torch.tensor(label[-args.max_length:]))
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        if args.pad_labels_with_ignore:
            labels = pad_sequence(labels, batch_first=True, padding_value=args.ignore_token_id)
        else:
            labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    

def classfication_data_collator(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
        texts = []
        labels = []
        for example in examples:
            texts.append(example['text'])
            labels.append(args.label2id[example['label']])
                
        encodings = tokenizer(texts, padding=True, truncation=True, add_special_tokens=True)

        return {
            "input_ids": torch.tensor(encodings['input_ids']),
            "attention_mask": torch.tensor(encodings['attention_mask']),
            "labels": torch.tensor(labels)
        }
    
    return collator
        

def reward_data_collator(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples: List[Dict[str, List]]):
        batch_size = len(examples)
        num_sample = max([len(example['texts']) for example in examples])
        all_texts = []
        all_scores = []
        for example in examples:
            if len(example['texts']) < num_sample:
                example['texts'].extend([' ']*(num_sample - len(example['texts'])))
                example['scores'].extend([-100]*(num_sample - len(example['scores'])))
            all_texts.extend(example['texts'])
            all_scores.extend(example['scores'])

        encodings = tokenizer(all_texts, padding=True, truncation=True, add_special_tokens=True)

        return {
            "input_ids": torch.tensor(encodings['input_ids']).reshape(batch_size, num_sample, -1),
            "attention_mask": torch.tensor(encodings['attention_mask']).reshape(batch_size, num_sample, -1),
            "scores": torch.tensor(all_scores).reshape(batch_size, -1)
        }

    return collator

    
def sft_data_collator(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
        texts = []
        prompts = []
        for example in examples:
            text = example['prompt'] + example['answer']
            texts.append(text)
            prompts.append(example['prompt'])

        return _llm_tokenize(prompts, texts, tokenizer, args)
    
    return collator

def replace_last_occurrence(text):
    # 找到最后一次出现'Assistant:'的位置
    last_occurrence = text.rfind('Assistant:')
    if last_occurrence != -1:
        # 将最后一次出现的'Assistant:'替换为'Assistant:<sep>'
        text = text[:last_occurrence] + 'Assistant:<sep>' + text[last_occurrence + len('Assistant:'):]
    return text

def rjs_data_collator(tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    def collator(examples):
        best_texts: List[str] = []
        for example in examples:
            texts = example['text']
            scores = example['score']
            best_text = texts[torch.argmax(torch.tensor(scores))]
            best_texts.append(best_text)
        
        if not args.only_predict_answer:
            prompts = None
            texts = best_texts
        else:
            prompts = []
            texts = []
            for text in best_texts:
                if "<sep>" not in text:
                    text = replace_last_occurrence(text)
                prompt, answer = text.split(args.sep_token)
                prompts.append(prompt)
                texts.append(prompt + answer)

        return _llm_tokenize(prompts, texts, tokenizer, args)

    return collator
        
def multi_reward_data_collator(batch):
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