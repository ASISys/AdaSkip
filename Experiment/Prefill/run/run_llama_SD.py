import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer
from model.llama_SD import LlamaForCausalLM, MyLlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_full_layer_num', type=int, default=4)
    parser.add_argument('--model', type=str, default="llama-EE-PREFILL")
    return parser.parse_args(args)


def get_pred(target_datasets, dataset2prompt, dataset2maxlen, device, model_name, model_path, output_dir, SKIP_FULL_LAYER_NUM):
    if not os.path.exists(f"{output_dir}/{model_name}"):
        os.makedirs(f"{output_dir}/{model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_path, SKIP_FULL_LAYER_NUM, device)
    
    for target_dataset in target_datasets:
        out_path = f"{output_dir}/{model_name}/{target_dataset}.jsonl"
        prompt_format = dataset2prompt[target_dataset]
        max_gen = dataset2maxlen[target_dataset]
        target_data = load_dataset('THUDM/LongBench', f"{target_dataset}_e", split='test')
        target_data_all = [data_sample for data_sample in target_data]
        for json_obj in tqdm(target_data_all):
            if json_obj["length"] <= 4000:
                continue
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            if target_dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, SKIP_FULL_LAYER_NUM, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = MyLlamaConfig.from_pretrained(path, SKIP_FULL_LAYER_NUM=SKIP_FULL_LAYER_NUM)
    model = LlamaForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer


if __name__ == '__main__':
    seed_everything(2025)
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    grand_dir = os.path.dirname(os.path.dirname(current_dir))
    greatgrand_dir = os.path.dirname(grand_dir)
    model2path = json.load(open(f"{greatgrand_dir}/config/model2path.json", "r"))
    model2maxlen = json.load(open(f"{greatgrand_dir}/config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SKIP_FULL_LAYER_NUM = args.skip_full_layer_num
    model_name = args.model + "-SKIP-FU-" + str(SKIP_FULL_LAYER_NUM)
    max_length = model2maxlen["llama"]
    model_path = model2path["llama"]
    # Prefill
    target_datasets = ["multifieldqa_en", "triviaqa", "trec"]
    dataset2prompt = json.load(open(f"{greatgrand_dir}/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{greatgrand_dir}/config/dataset2maxlen.json", "r"))
    output_dir = os.path.join(grand_dir, "Result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    get_pred(target_datasets, dataset2prompt, dataset2maxlen, device, model_name, model_path, output_dir, SKIP_FULL_LAYER_NUM)