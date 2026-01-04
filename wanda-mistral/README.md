# Pruning LLMs by Weights and Activations
Modified PyTorch implementation of **Wanda** (Pruning by **W**eights **and a**ctivations), as presented in the paper:

**A Simple and Effective Pruning Approach for Large Language Models** </br>
*Mingjie Sun\*, Zhuang Liu\*, Anna Bair, J. Zico Kolter* (* indicates equal contribution) <br>
Carnegie Mellon University, Meta AI Research and Bosch Center for AI  <br>
[Paper](https://arxiv.org/abs/2306.11695) - [Project page](https://eric-mingjie.github.io/wanda/home.html)

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

--- 
<p align="center">
<img src="https://user-images.githubusercontent.com/20168304/273351964-53c3807e-3453-49c5-b855-b620b1026466.png" width=100% height=100% 
class="center">
</p>

Compared to magnitude pruning which removes weights solely based on their magnitudes, the pruning approach **Wanda** removes weights on a *per-output* basis, by the product of weight magnitudes and input activation norms.

## Setup
Installation instructions are same as original repo and can be found in [INSTALL.md](INSTALL.md).

## Usage
The [scripts](scripts) directory contains all the bash commands to replicate the main results (Table 2) in our paper.

Below is an example command for pruning LLaMA-7B with Wanda, to achieve unstructured 50% sparsity.
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/ 
```

### Pruning LLaMA-2
For [LLaMA-2](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-2-7b-hf` (take `7b` as an example):
```sh 
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/
```
LLaMA-2 results: (LLaMA-2-34b is not released as of 9.22.2023)
|sparsity| ppl              | llama2-7b | llama2-13b | llama2-70b |
|------|------------------|----------|------------|------------|
|-| dense            | 5.12     | 4.57       | 3.12     |
|unstructured 50%| magnitude        | 14.89    | 6.37       | 4.98     |
|unstructured 50%| sparsegpt        | 6.51     | 5.63       | **3.98**  |
|unstructured 50%| wanda            | **6.42** | **5.56**   | **3.98**  |
|4:8| magnitude        | 16.48    | 6.76       | 5.58     |
|4:8| sparsegpt        | 8.12     | 6.60      | 4.59     |
|4:8| wanda            | **7.97** | **6.55**  | **4.47**     |
|2:4| magnitude        | 54.59    | 8.33       | 6.33       |
|2:4| sparsegpt        | **10.17** | 8.32       | 5.40      |
|2:4| wanda            | 11.02    | **8.27**   | **5.16**     |

### Pruning Mistral-7B
For [Mistral](https://mistral.ai/) models from Hugging Face:
```sh 
python main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/mistral_7b/unstructured/wanda/
```

Supported Mistral models:
- `mistralai/Mistral-7B-v0.1` (base model)
- `mistralai/Mistral-7B-v0.3` (latest base)
- `mistralai/Mistral-7B-Instruct-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mistral-7B-Instruct-v0.3`

For 2:4 structured sparsity (compatible with NVIDIA Tensor Cores):
```sh
python main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/mistral_7b/2-4/wanda/
```

**Note**: Mistral 7B uses the same architecture pattern as LLaMA (`model.model.layers`), so the pruning code works without modification. Key Mistral features preserved:
- Sliding Window Attention
- Grouped Query Attention (GQA)  
- SwiGLU activations


## Acknowledgement
This repository is build upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.


mingjies at cs.cmu.edu  
liuzhuangthu at gmail.com 
