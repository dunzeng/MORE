from model import PythiaRewardModel, LlamaRewardModel
from transformers import AutoTokenizer
from accelerate import PartialState
from accelerate.utils import gather_object
import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from utils import numpy_sigmoid, print_rank_0

def load_dataset(data_path):
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def load_tokenizer_and_model(args):
    if args.model_type == 'pythia':
        tokenizer =  AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side='left', trust_remote_code=True)
        model = PythiaRewardModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    elif args.model_type == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trucation_side='left')
        model = LlamaRewardModel.from_pretrained(args.model_name_or_path)
        print(tokenizer.pad_token)

    return tokenizer, model


def main(args):
    distributed_state = PartialState()
    tokenizer, model = load_tokenizer_and_model(args)
    model.to(distributed_state.device)
    dataset = load_dataset(args.data_path)
    sub_dataset_size = 16
    final_dataset_count = 0
    for i in tqdm(range(0, len(dataset), sub_dataset_size), desc="Running"):
        new_dataset = []
        sub_dataset = dataset[i:i+sub_dataset_size]
        with distributed_state.split_between_processes(sub_dataset) as sd:
            n = distributed_state.process_index
            # with tqdm(total=len(sd), desc=f'rank: {n}', position=n+1) as pbar:
            for data in sd:
                if args.task == 'rjs':
                    query = data['query']
                    answers = [data['sample_0'], data['sample_1'], data['sample_2'], data['sample_3']]
                    texts = [query+answer for answer in answers]
                    input_ids = tokenizer(texts, padding=True, truncation=True)['input_ids']
                    with torch.no_grad():
                        rewards: torch.Tensor = model(input_ids=torch.tensor(input_ids).to(model.device))['rm_logits']
                    new_data = {
                        "text": [query+'<sep>'+answer for answer in answers],
                        "score": rewards.tolist()
                    }
                elif args.task == 'dpo':
                    text = data['text']
                    score = data['score']
                    input_ids = tokenizer(text, padding=True, truncation=True)['input_ids']
                    with torch.no_grad():
                        rewards: torch.Tensor = model(input_ids=torch.tensor(input_ids).to(model.device))['rm_logits']
                    conf = torch.Tensor(rewards).view(-1).cpu()
                    conf = numpy_sigmoid(conf[0].item() - conf[1].item())
                    assert score[0] > score[1]
                    new_data = {
                        "texts": text,
                        "scores": score,
                        "conf": conf
                    }
                new_dataset.append(new_data)
                # pbar.update(1)
                        
        new_dataset = gather_object(new_dataset)
        final_dataset_count += len(new_dataset)
        # print_rank_0(f">>> Debug data size: {final_dataset_count}")

        if distributed_state.is_main_process:
            with open(args.save_path, 'a+') as f:
                for data in new_dataset:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_name_or_path', type=str,
    )
    parser.add_argument(
        '--model_type', type=str, choices=['llama', 'pythia']
    )
    parser.add_argument(
        '--data_path', type=str
    )
    parser.add_argument(
        '--save_path', type=str
    )
    parser.add_argument(
        '--task', type=str, default="rjs"
    ) # rjs, dpo
    args = parser.parse_args()
    main(args)
