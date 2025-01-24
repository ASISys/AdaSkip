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
Experiment

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