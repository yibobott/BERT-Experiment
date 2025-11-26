# Quick Start

```shell

# install dependencies
pip install -r requirements.txt

# MLM
python dynamic-masking.py
python static-masking.py

# NSP
python nsp.py

# analyse 
python analyse.py

# replace LayerNorm with RMSNorm
python -m rmsnorm.train_rmsnorm

# analyse
python rmsnorm/analyse_rmsnorm_vs_postln.py

```


# BERT Pre-training Experiments:

This repository contains a set of small, focused experiments on BERT pre‑training, including:

- Static vs. dynamic Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- RMSNorm vs. Post‑LayerNorm for BERT‑MLM

The code is implemented with HuggingFace Transformers and WikiText-103 Datasets.

