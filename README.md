# AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
This is the implementation repository of our AAAI'25 paper: [AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference](https://arxiv.org/abs/2501.02336).

This artifact provides the source code of Adaskip and scripts to reproduce the results.

This README is specifically for artifact evaluation (AE).

## Preliminary
A single L20 GPU with CUDA version 12.1 is used as the testbed. Please check the requirement.txt for the package version.


## Observation
In **Background and Motivation**, we made three main observations.

Observation 1: The layer importance distribution exhibits significant variation across diverse models. 
```
# how to run
cd Observation1
python ob1.py --model [llama/internlm/vicuna]
```

Observation 2: The importance distributions of attention and FFN modules are different. 
```
# how to run
cd Observation2
python ob2.py --model [llama/internlm/vicuna]
```

Observation 3: The importance distribution of sublayers in the prefilling and decoding phases have similar trends but different fluctuation degrees.
```
# how to run
cd Observation3
python ob3.py --model [llama/internlm/vicuna]
```

## Experiment
### Prefill Task
For Baseline models:
```
# how to run (Taking llama as an example, the same applies to internlm and vicuna.)
cd Prefill/run
# Early Exit:
python run_llama_EE.py --model llama-EE-PREFILL --skip_full_layer_num [4/6/8]
# SkipDecode:
python run_llama_SD.py --model llama-SD-PREFILL --skip_full_layer_num [4/6/8]
# Unifed Skipping:
python run_llama_US.py --model llama-US-PREFILL --skip_full_layer_num [4/6/8]
```

For AdaSkip models:
```
# how to run (Taking llama as an example, the same applies to internlm and vicuna.)
cd Prefill/run
python run_llama_ADA.py --model llama-ADA-PREFILL --skip_sub_layer_num [8/12/16]
```

### Decode Task
For Baseline models:
```
# how to run (Taking internlm as an example, the same applies to llama and vicuna.)
cd Decode/run
# Early Exit:
python run_internlm_EE.py --model internlm-EE-DECODE --skip_full_layer_num [4/6/8]
# SkipDecode:
python run_internlm_SD.py --model internlm-SD-DECODE --skip_full_layer_num [4/6/8]
# Unifed Skipping:
python run_internlm_US.py --model internlm-US-DECODE --skip_full_layer_num [4/6/8]
```

For AdaSkip models:
```
# how to run (Taking internlm as an example, the same applies to llama and vicuna.)
cd Decode/run
python run_internlm_ADA.py --model internlm-ADA-DECODE --skip_sub_layer_num [8/12/16]
```

### E2E Task
For Baseline models:
```
# how to run (Taking vicuna as an example, the same applies to llama and internlm.)
cd E2E/run
# Early Exit:
python run_vicuna_EE.py --model vicuna-EE-E2E --skip_full_layer_num [4/6/8]
# SkipDecode:
python run_vicuna_SD.py --model vicuna-SD-E2E --skip_full_layer_num [4/6/8]
# Unifed Skipping:
python run_vicuna_US.py --model vicuna-US-E2E --skip_full_layer_num [4/6/8]
```

For AdaSkip models:
```
# how to run (Taking vicuna as an example, the same applies to llama and internlm.)
cd E2E/run
python run_vicuna_ADA.py --model vicuna-ADA-E2E --skip_sub_layer_num [8/12/16]
```
### Evaluation for Prefill/Decode/E2E Task
```
# how to run
cd Eval
python eval.py --model [The model name above (Eg: llama-EE-PREFILL)]
```

### Speed Test
```
# how to run
cd Speed/run
python run_llama_speed.py
python run_internlm_speed.py
python run_vicuna_speed.py
```

## Paper
If you think Adaskip is helpful, please cite this paper:
```
@article{he2025adaskip,
  title={AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference},
  author={He, Zhuomin and Yao, Yizhen and Zuo, Pengfei and Gao, Bin and Li, Qinya and Zheng, Zhenzhe and Wu, Fan},
  journal={arXiv preprint arXiv:2501.02336},
  year={2025}
}
```

## Acknowledgement
We really appreciate the datasets from [Longbench](https://github.com/THUDM/LongBench), and Opensource Models from Huggingface and LLaMA.