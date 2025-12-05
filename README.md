# Quick Start

```shell

# install dependencies
pip install -r requirements.txt

# MLM
python static-masking.py --config configs/static.yaml
python dynamic-masking.py --config configs/dynamic.yaml

# NSP
python nsp.py --config configs/nsp.yaml

# replace LayerNorm with RMSNorm
python rmsnorm/train_rmsnorm.py --config configs/rmsnorm_postln_dynamic.yaml

# analyse 
python analyse.py

# analyse
python rmsnorm/analyse_rmsnorm_vs_postln.py

```


# BERT Pre-training Experiments:

This repository contains a set of small, focused experiments on BERT pre‑training, including:

- Static vs. dynamic Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- RMSNorm vs. Post‑LayerNorm for BERT‑MLM

The code is implemented with HuggingFace Transformers and WikiText-103 Datasets.

