import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer
from hf_model.internlm import InternLMForCausalLM
from hf_model.llama import LlamaForCausalLM
from hf_model.vicuna import VicunaForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["internlm", "llama", "vicuna"])
    return parser.parse_args(args)

def build_chat(prompt, model_name):
    if "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    cnt = 0
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        output = model.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        if cnt%10==0:
            with open(out_path, "a", encoding="utf-8") as f:
                for idx, _ in enumerate(model.model.layers):
                    print(f'UntilSample{cnt}-layer{idx}: attn_sim_sum_prefill is {model.model.IO_attn_sum_prefill[idx]}', file=f)
                    print(f'UntilSample{cnt}-layer{idx}: attn_sim_sum_decode is {model.model.IO_attn_sum_decode[idx]}', file=f)
                    print(f'UntilSample{cnt}-layer{idx}: ffn_sim_sum_prefill is {model.model.IO_mlp_sum_prefill[idx]}', file=f)
                    print(f'UntilSample{cnt}-layer{idx}: ffn_sim_sum_decode is {model.model.IO_mlp_sum_decode[idx]}', file=f)
                    
                sorted_indices = [i for i, _ in sorted(enumerate(range(len(model.model.IO_attn_sum_prefill))), key=lambda x: model.model.IO_attn_sum_prefill[x[0]], reverse=True)]
                print(f'UntilSample{cnt}-AttnSimSortPrefill: {sorted_indices}', file=f)
                formatted_list = [num for num in model.model.IO_attn_avg_prefill]
                print(f'UntilSample{cnt}-AttnSimAvgPrefill: {formatted_list}', file=f)

                sorted_indices = [i for i, _ in sorted(enumerate(range(len(model.model.IO_attn_sum_decode))), key=lambda x: model.model.IO_attn_sum_decode[x[0]], reverse=True)]
                print(f'UntilSample{cnt}-AttnSimSortDecode: {sorted_indices}', file=f)
                formatted_list = [num for num in model.model.IO_attn_avg_decode]
                print(f'UntilSample{cnt}-AttnSimAvgDecode: {formatted_list}', file=f)

                sorted_indices = [i for i, _ in sorted(enumerate(range(len(model.model.IO_mlp_sum_prefill))), key=lambda x: model.model.IO_mlp_sum_prefill[x[0]], reverse=True)]
                print(f'UntilSample{cnt}-FFNSimSortPrefill: {sorted_indices}', file=f)
                formatted_list = [num for num in model.model.IO_mlp_avg_prefill]
                print(f'UntilSample{cnt}-FFNSimAvgPrefill: {formatted_list}', file=f)

                sorted_indices = [i for i, _ in sorted(enumerate(range(len(model.model.IO_mlp_sum_decode))), key=lambda x: model.model.IO_mlp_sum_decode[x[0]], reverse=True)]
                print(f'UntilSample{cnt}-FFNSimSortDecode: {sorted_indices}', file=f)
                formatted_list = [num for num in model.model.IO_mlp_avg_decode]
                print(f'UntilSample{cnt}-FFNSimAvgDecode: {formatted_list}', file=f)
                f.write('\n')
        cnt+=1


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if "internlm" in model_name:
        model = InternLMForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "vicuna" in model_name:
        model = VicunaForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model2path = json.load(open("../config/model2path.json", "r"))
    model2maxlen = json.load(open("../config/model2maxlen.json", "r"))

    model_name = args.model
    max_length = model2maxlen[model_name]
    datasets = ["trec"]
    dataset2prompt = json.load(open("../config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("../config/dataset2maxlen.json", "r"))


    if not os.path.exists("res"):
        os.makedirs("res")
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        if not os.path.exists(f"res/{model_name}"):
            os.makedirs(f"res/{model_name}")
        out_path = f"res/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data if data_sample["length"]>4000]
        get_pred(0, data_all, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path)

#how to run: python ob3.py --model internlm
# python ob3.py --model llama
# python ob3.py --model vicuna