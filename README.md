# Quick Start

```shell

# install dependencies
pip install -r requirements.txt

# MLM
python static-masking.py --config configs/static.yaml
# result_dir example: ./result-mlm/20251205-151603/static
python dynamic-masking.py --config configs/dynamic.yaml
# result_dir example: ./result-mlm/20251205-152207/dynamic

# NSP
python nsp.py --config configs/nsp.yaml
# result_dir example: ./result-nsp/20251205-153055/nsp

# replace LayerNorm with RMSNorm
python -m normalization.normalize --config configs/normalization.yaml
# result_dir example: ./result-normalization/20251205-160341/postln-dynamic

# analyse 
python analyse.py

# analyse
python normalization/analyse_rmsnorm_vs_postln.py

```


# BERT Pre-training Experiments:

This repository contains a set of small, focused experiments on BERT pre‑training, including:

- Static vs. dynamic Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- RMSNorm vs. Post‑LayerNorm for BERT‑MLM

The code is implemented with HuggingFace Transformers and WikiText-103 Datasets.

