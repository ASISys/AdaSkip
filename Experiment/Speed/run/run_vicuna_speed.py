import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
from model.vicuna_speed import LlamaForCausalLM, MyVicunaConfig
import numpy as np
import random
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="vicuna-SPEED")
    return parser.parse_args(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(path, device):
    config = MyVicunaConfig.from_pretrained(path)
    model = LlamaForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model


def prepare_random_input(batch_size, seq_length, vocab_size=32000):
    random_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        dtype=torch.long,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    attention_mask = torch.ones(
        (batch_size, seq_length),
        dtype=torch.long,
        device=random_ids.device
    )
    return {
        'input_ids': random_ids,
        'attention_mask': attention_mask
    }


if __name__ == '__main__':
    seed_everything(2025)
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    grand_dir = os.path.dirname(os.path.dirname(current_dir))
    greatgrand_dir = os.path.dirname(grand_dir)
    model2path = json.load(open(f"{greatgrand_dir}/config/model2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    model_path = model2path["vicuna"]
    model = load_model(model_path, device)
    DEC_LEN = 10
    BS = 1
    SEQ_LEN = 8000
    input_data = prepare_random_input(batch_size=BS, seq_length=SEQ_LEN)
    output = model.generate(
        **input_data,
        max_new_tokens=DEC_LEN,
        min_new_tokens=DEC_LEN,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
    )[0]
