from model import PythiaRewardModel, LlamaRewardModel
from transformers import AutoTokenizer
from accelerate import PartialState
from accelerate.utils import gather_object
import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser


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
    for i in range(0, len(dataset), sub_dataset_size):
        new_dataset = []
        sub_dataset = dataset[i:i+sub_dataset_size]
        with distributed_state.split_between_processes(sub_dataset) as sd:
            n = distributed_state.process_index
            with tqdm(total=len(sd), desc=f'rank: {n}', position=n+1) as pbar:
                for data in sd:
                    query = data['query']
                    answers = [data['sample_0'], data['sample_1'], data['sample_2'], data['sample_3']]
                    texts = [query+answer for answer in answers]
                    input_ids = tokenizer(texts, padding=True, truncation=True)['input_ids']
                    with torch.no_grad():
                        rewards: torch.Tensor = model(input_ids=torch.tensor(input_ids).to(model.device))['rm_logits']
                    new_data = {
                        "texts": [query+'<sep>'+answer for answer in answers],
                        "scores": rewards.tolist()
                    }
                    new_dataset.append(new_data)
                    pbar.update(1)

        new_dataset = gather_object(new_dataset)

        with open(args.save_path, 'a+') as f:
            for new_data in new_dataset:
                s = json.dumps(new_data)
                f.write(s+'\n')

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
    args = parser.parse_args()
    main(args)
