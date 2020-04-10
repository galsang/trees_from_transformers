# Trees from Transformers

This repository contains the implementation for ''ARE PPE-TRAINED LANGUAGE MODELS AWARE OF PHRASES? SIMPLE BUT STRONG BASELINES FOR GRAMMAR INDCUTION''.
 
When using this code for following work, please cite our paper with the BibTex below.

	@inproceedings{
    Kim2020Are,
    title={Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction},
    author={Taeuk Kim and Jihun Choi and Daniel Edmiston and Sang-goo Lee},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=H1xPR3NtPB}
    }

## Experimental Environment

- OS: Ubuntu 16.04 LTS (64bit)
- GPU: Nvidia GTX 1080, Titan XP, and Tesla P100
- CUDA: 10.1 (Nvidia driver: 418.39), CuDNN: 7.6.4
- Python (>= 3.6.8)
- **PyTorch** (>= 1.3.1)
- Core Python library: [**Transformers by HuggingFace**](https://github.com/huggingface/transformers) (>=2.2.0)

## Pre-requisite Python Libraries

Please install the following libraries specified in the **requirements.txt** first before running our code.

    transformers==2.2.0
    numpy==1.15.4
    tqdm==4.26.0
    torch==1.3.1
    nltk==3.4
    matplotlib==2.2.3
    
## Data preparation (PTB)

Please download the PTB dataset (ptb-valid.txt, ptb-test.txt) from [Yoon Kim's repo](https://github.com/harvardnlp/compound-pcfg) and locate them in the **.data/PTB** folder.


## How to Run Code

> python run.py --help

	usage: run.py [-h] [--data-path DATA_PATH] [--result-path RESULT_PATH]
              [--from-scratch] [--gpu GPU] [--bias BIAS] [--seed SEED]
              [--token-heuristic TOKEN_HEURISTIC] [--use-coo-not-parser]

    optional arguments:
      -h, --help    show this help message and exit
      --data-path DATA_PATH
      --result-path RESULT_PATH
      --from-scratch
      --gpu GPU
      --bias BIAS   the right-branching bias hyperparameter lambda
      --seed SEED
      --token-heuristic TOKEN_HEURISTIC     Available options: mean, first, last
      --use-coo-not-parser  Turning on this option will allow you to exploit the
                            COO-NOT parser (named by Dyer et al. 2019), which has
                            been broadly adopted by recent methods for
                            unsupervised parsing. As this parser utilizes the
                            right-branching bias in its inner workings, it may
                            give rise to some unexpected gains or latent issues
                            for the resulting trees. For more details, see
                            https://arxiv.org/abs/1909.09428.


## Acknowledgments

- Some utility functions and datasets used in this repo are originally from the source code for 
**Compound Probabilistic Context-Free Grammars for Grammar Induction** (Y. Kim et al., ACL 2019).
For more details, visit [the original repo](https://github.com/harvardnlp/compound-pcfg). 