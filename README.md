<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/kowndinya-renduchintala/WIT/blob/main/wit_logo.png" width="500" />
    </br>
    <br>
        <strong> Weighted Instruction Tuning </strong>
    </br>
    <br>
        (Companion Software for our TACL paper - On the Effect of Instruction Tuning Loss on Generalization)
    </br>
</p>

<p align="center">
    <a href="https://github.com/kowndinya-renduchintala/WIT/blob/main/LICENSE.txt">
        <img alt="GitHub" src="https://img.shields.io/github/license/kowndinya-renduchintala/WIT?color=blue">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/kowndinya-renduchintala/WIT">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/kowndinya-renduchintala/WIT">
    </a>
</p>

# About WIT

The conventional Instruction Tuning Loss involves backpropagating only on response tokens while zeroing out the response on prompt tokens. WIT is a better alternative to this conventional loss, which differentially weighs prompt and response tokens.

# How to Run?

This repository contains two bash scripts for finetuning auto-regressive models using the WIT loss. For smaller models such as Llama-3.2-1B or Llama-3.2-3B, run the `instruction_tuner_no_fsdp.sh` script which does not use FSDP. For larger models such as Llama-3-8B or Mistral-7B, use the `instruction_tuner_fsdp.sh` script which uses FSDP.

Before running the scripts, set the hyperparameters such as prompt token weight, response token weight etc based on your needs. 

# The WIT Loss Function

Let $`\mathcal{D} = \{(\boldsymbol{P}_i, \boldsymbol{R}_i)\}_{i=1}^{N_{\mathcal{T}}}`$ be an instruction tuning dataset with $`N_{\mathcal{T}}`$ (prompt, response) pairs. Each prompt $`\boldsymbol{P}_i`$ includes an instruction (implicit or explicit) and optionally some input, while $`\boldsymbol{R}_i`$ is the expected ground-truth response.

If $`|\boldsymbol{S}|`$ denotes the number of tokens in a sequence $`\boldsymbol{S}`$, then:

```math
\boldsymbol{P}_i = \left\{p_i^{(1)}, p_i^{(2)}, \ldots, p_i^{(|\boldsymbol{P}_i|)}\right\}
```

```math
\boldsymbol{R}_i = \left\{r_i^{(1)}, r_i^{(2)}, \ldots, r_i^{(|\boldsymbol{R}_i|)}\right\}
```

The WIT loss is given by:

```math
\mathcal{L}_{WIT} = -\frac{\sum\limits_{i=1}^{N_{\mathcal{T}}}\left[\lambda_p \cdot \sum\limits_{j=1}^{|\boldsymbol{P}_i|} \log \mathbb{P}_{\mathcal{M}}\left(p_i^{(j)} |\; p_i^{(1)},\ldots, p_i^{(j-1)} \right) + \lambda_r \cdot \sum\limits_{j=1}^{|\boldsymbol{R}_i|} \log \mathbb{P}_{\mathcal{M}}\left(r_i^{(j)} |\; r_i^{(1)},\ldots, r_i^{(j-1)} \right)\right]}{\sum\limits_{i=1}^{N_{\mathcal{T}}}\Big(\mathbb{I}{(\lambda_p \neq 0)}\cdot| \boldsymbol{P}_i| +  \mathbb{I}{(\lambda_r \neq 0)}\cdot |\boldsymbol{R}_i|\Big)}
```

where $`\mathbb{I}(\cdot)`$ is the indicator function, $`\lambda_p`$ is the prompt token weight, and $`\lambda_r`$ is the response token weight. $`\mathcal{L}_{WIT}`$ computes the weighted sum of log-probabilities -- scaling the log-probabilities of prompt tokens by $`\lambda_p`$ and those of response tokens by $`\lambda_r`$ -- and then normalizes by the count of tokens with non-zero weight. The indicator function ($`\mathbb{I}`$) ensures that the weighted sum is divided exactly by those tokens whose weight is non-zero. Note that the conventional instruction tuning loss $`\mathcal{L}_{IT}`$ is a special case of $`\mathcal{L}_{WIT}`$ for $`(\lambda_p, \lambda_r) = (0,1)`$ and continual pre-training is a special case of $`\mathcal{L}_{WIT}`$ for $`(\lambda_p, \lambda_r) = (1,1)`$. 

# Citation

If you use *WIT* in your research, please cite our preprint that is accepted to Transactions of the Association for Computational Linguistics (TACL) :blush: -

[On the Effect of Instruction Tuning Loss on Generalization](https://arxiv.org/abs/2507.07817) [arXiv preprint arXiv:2507.07817 (2025)]

```
@article{chatterjee2025effect,
  title={On the Effect of Instruction Tuning Loss on Generalization},
  author={Chatterjee, Anwoy and Renduchintala, HSVNS Kowndinya and Bhatia, Sumit and Chakraborty, Tanmoy},
  journal={arXiv preprint arXiv:2507.07817},
  year={2025}
}
```

# License

*WIT* is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
