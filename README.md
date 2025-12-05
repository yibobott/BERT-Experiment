# BERT Pretraining Experiments

This repository contains a set of small, focused experiments on BERT pretraining.
The main goals are to:

- Compare **static vs. dynamic** Masked Language Modeling (MLM)
- Study **Next Sentence Prediction (NSP)** with MLM
- Compare **Post-LayerNorm vs. RMSNorm** for BERT-MLM

---

## 1. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
```

Run all commands from the project root directory (`BERT/`).

### 1.1 MLM: Static and Dynamic Masking

```bash
# Static masking MLM
python static-masking.py --config configs/static.yaml
# result_dir example: ./result-mlm/20251205-151603/static

# Dynamic masking MLM
python dynamic-masking.py --config configs/dynamic.yaml
# result_dir example: ./result-mlm/20251205-152207/dynamic
```

### 1.2 NSP with MLM

```bash
python nsp.py --config configs/nsp.yaml
# result_dir example: ./result-nsp/20251205-153055/nsp
```

### 1.3 Normalization: Post-LayerNorm vs. RMSNorm

```bash
python -m normalization.normalize --config configs/normalization.yaml
# result_dir example: ./result-normalization/20251205-160341/postln-dynamic
```

The normalization experiment is controlled by:

- `experiment.norm_type`: `"postln"` or `"rmsnorm"`
- `experiment.masking_type`: `"static"` or `"dynamic"`

in `configs/normalization.yaml`.
The output directory structure is determined by `output.base_dir` and `output.sub_dir`.

### 1.4 Analysis Scripts

```bash
# Aggregate and visualize MLM / NSP results
python analyse.py

# Analyse RMSNorm vs Post-LN experiments
python normalization/analyse_rmsnorm_vs_postln.py
```

---

## 2. Project Structure (core files)

```text
BERT/
  config.py                     # Global config loader (YAML -> dataclass)
  utils.py                      # Utility functions (e.g., load_yaml_config)

  configs/
    static.yaml                 # Config for static masking MLM
    dynamic.yaml                # Config for dynamic masking MLM
    nsp.yaml                    # Config for NSP + MLM
    normalization.yaml          # Config for normalization experiments (PostLN / RMSNorm)

  static-masking.py             # Static masking MLM training script
  dynamic-masking.py            # Dynamic masking MLM training script
  nsp.py                        # NSP + MLM training script
  analyse.py                    # Shared analysis/plotting for MLM / NSP

  normalization/
    normalize.py                # Normalization experiment: PostLN vs RMSNorm (MLM)
    rmsnorm_utils.py            # RMSNorm-based BERT implementation
    analyse_rmsnorm_vs_postln.py# Analysis of normalization experiments
```

---

## 3. Configuration System

All experiments use a unified configuration system implemented in `config.py`:

- **Static MLM**  
  - Dataclass: `StaticMaskingConfig`  
  - YAML: `configs/static.yaml`  
  - Loader: `load_static_config()`

- **Dynamic MLM**  
  - Dataclass: `DynamicMaskingConfig`  
  - YAML: `configs/dynamic.yaml`  
  - Loader: `load_dynamic_config()`

- **NSP**  
  - Dataclass: `NSPConfig`  
  - YAML: `configs/nsp.yaml`  
  - Loader: `load_nsp_config()`

- **Normalization (PostLN vs RMSNorm)**  
  - Dataclass: `RMSNormConfig`  
  - YAML: `configs/normalization.yaml`  
  - Loader: `load_normalization_config()`

Each training script accepts:

```bash
--config path/to/config.yaml
```

to specify the experiment configuration, and `config.py` converts the YAML into a strongly-typed dataclass instance that is passed through the code.

---

## 4. Dependencies

Key Python dependencies (see `requirements.txt` for exact versions):

- `transformers`
- `datasets`
- `torch`
- `pandas`
- `matplotlib`
- `scipy` (for statistical tests in normalization analysis)

---

## 5. Reproducibility

- Random seeds for each experiment are controlled through the YAML config (e.g., `training.seeds`).
- Output directories are timestamped to avoid overwriting previous runs.
- All loss logs and summary plots are saved under the corresponding `result_dir` so that runs are easy to compare afterwards.

