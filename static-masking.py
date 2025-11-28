from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    Trainer,
    TrainingArguments
)
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
import os
from datasets import DatasetDict
from datetime import datetime
"""
Config
"""
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
RESULT_DIR = "./result-" + timestamp + "/static"
os.makedirs(RESULT_DIR, exist_ok=True)

# MODEL_NAME = "prajjwal1/bert-small"
MODEL_NAME = "bert-base-uncased"

DATASET_NAME = ("wikitext", "wikitext-103-raw-v1")
# TRAIN_SAMPLE_SIZE = 100000 # 100k
TRAIN_SAMPLE_SIZE = None

MAX_LENGTH = 64
BATCH_SIZE = 16
# BATCH_SIZE = 8
# MAX_STEPS = 3000
MAX_STEPS = 50000 # 5w
# GRAD_ACC_STEPS = 1
GRAD_ACC_STEPS = 2 

LR = 5e-5
# WARMUP_STEPS = 100
WARMUP_STEPS = 1000
# EVAL_STEPS = 100
EVAL_STEPS = 500
# LOGGING_STEPS = 10
LOGGING_STEPS = 50

SEEDS = [42, 43, 44]


def load_dataset_demo():
    """
    Load dataset once (fixed data across seeds)
    """
    original_dataset = load_dataset(*DATASET_NAME)

    train_full = original_dataset["train"].shuffle(seed=42)
    if TRAIN_SAMPLE_SIZE is not None and TRAIN_SAMPLE_SIZE < len(train_full):
        train = train_full.select(range(TRAIN_SAMPLE_SIZE))
    else:
        train = train_full

    # filter empty / whitespace-only lines
    train = train.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)
    
    validation = original_dataset["validation"]
    validation = validation.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)

    dataset = DatasetDict({
        "train": train,
        "validation": validation
    })
    return dataset

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


def static_masking(example, mlm_prob=0.15):
    """
    Static Masking for BERT-style MLM pretraining.

    - First, follow BERT paper exactly: choose ~15% token positions
      by independent Bernoulli sampling.
    - Rare-case fix: if no token is selected (often for very short / empty lines),
      we randomly force-mask ONE valid token to avoid degenerate batches
      with zero MLM targets.
    - Then apply BERT 80/10/10 replacement rule.
    - Only masked positions contribute to loss (others set to -100).
    """

    # Convert to tensor
    input_ids = torch.tensor(example["input_ids"])
    labels = input_ids.clone()

    # Special tokens should never be masked
    special_ids = set(tokenizer.all_special_ids)

    # 1) Collect candidate positions (non-special tokens)
    cand_positions = [
        i for i, tok_id in enumerate(input_ids)
        if int(tok_id) not in special_ids
    ]

    # If the sequence has no valid tokens (e.g., empty after tokenization),
    # then we cannot do MLM on it -> ignore all positions in loss.
    if len(cand_positions) == 0:
        labels[:] = -100
        return {"input_ids": input_ids.tolist(), "labels": labels.tolist()}

    # 2) BERT-style Bernoulli sampling: each token has p=0.15 to be masked
    mask_positions = [
        i for i in cand_positions
        if random.random() < mlm_prob
    ]

    # 3) Rare-case fix:
    # If nothing is masked (common for very short sentences),
    # randomly select ONE token to mask to avoid loss=0 batches.
    if len(mask_positions) == 0:
        mask_positions = [random.choice(cand_positions)]

    mask_positions = torch.tensor(mask_positions, dtype=torch.long)

    # 4) Apply BERT 80/10/10 replacement strategy
    rand = torch.rand(len(mask_positions))

    # (1) 80% -> replace with [MASK]
    mask80 = rand < 0.8
    input_ids[mask_positions[mask80]] = tokenizer.mask_token_id

    # (2) 10% -> replace with random token
    mask10 = (rand >= 0.8) & (rand < 0.9)
    random_tokens = torch.randint(
        low=0, high=tokenizer.vocab_size, size=(mask10.sum(),)
    )
    input_ids[mask_positions[mask10]] = random_tokens

    # (3) 10% -> keep unchanged (do nothing)

    # 5) Only compute loss on masked positions
    labels[:] = -100
    labels[mask_positions] = torch.tensor(example["input_ids"])[mask_positions]

    return {"input_ids": input_ids.tolist(), "labels": labels.tolist()}


def prepare_tokenized_static():
    dataset = load_dataset_demo()

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # filter sequences that are basically only CLS/SEP/PAD
    tokenized = tokenized.filter(lambda ex: sum(ex["attention_mask"]) > 2)

    # static masking
    tokenized_static = tokenized.map(static_masking, load_from_cache_file=False)
    return tokenized_static


def get_training_args(output_dir, seed):
    """
    We reduce both train and evaluation batch size to 8 to make the MLM pretraining more stable and efficient on Apple MPS, 
    which does not provide the same level of optimization as CUDA GPUs.
    """
    return TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,

        # —— 控制训练长度 ——
        max_steps=MAX_STEPS,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,    

        # —— Evaluation 设置 ——
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        # —— Logging（记录训练 loss） ——
        logging_strategy="steps",      
        logging_steps=LOGGING_STEPS,

        # —— Checkpoint（关闭省空间） ——
        save_strategy="no", 

        # —— 其他设置 ——
        fp16=True,
        report_to="none",
        seed=seed
    )


class LossRecorder(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.logs.append({
                "step": state.global_step,
                "loss": logs["loss"],
                "type": "train"
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            self.logs.append({
                "step": state.global_step,
                "loss": metrics["eval_loss"],
                "type": "eval"
            })                

    def to_dataframe(self):
        return pd.DataFrame(self.logs)


def run_and_get_losses(seed, train_dataset ,eval_dataset):
    """
    Run training for one seed
    """
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    loss_recorder = LossRecorder()
    args = get_training_args(output_dir=f"{RESULT_DIR}/static_{seed}", seed=seed)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[loss_recorder]
    )
    trainer.train()

    df = loss_recorder.to_dataframe()
    file_name = f"{RESULT_DIR}/static_loss_seed_{seed}.csv"
    df.to_csv(file_name, index=False)
    print(f"\n[Static] Running seed={seed}: Saved to {file_name}")

    df["seed"] = seed
    return df

def plot_static_loss(all_losses: pd.DataFrame):
    """
    collums in all_losses:
      - step
      - loss
      - type ('train' / 'eval')
      - seed
    """

    # plot train loss
    plt.figure()
    for seed in SEEDS:
        df_seed_train = all_losses[
            (all_losses["seed"] == seed) & (all_losses["type"] == "train")
        ]
        if df_seed_train.empty:
            continue
        plt.plot(
            df_seed_train["step"],
            df_seed_train["loss"],
            label=f"seed={seed} - train"
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Static Masking - Training Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/static_train_loss_multi_seeds.png")
    plt.close()

    # plot eval loss
    plt.figure()
    for seed in SEEDS:
        df_seed_eval = all_losses[
            (all_losses["seed"] == seed) & (all_losses["type"] == "eval")
        ]
        if df_seed_eval.empty:
            continue
        plt.plot(
            df_seed_eval["step"],
            df_seed_eval["loss"],
            label=f"seed={seed} - eval"
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Static Masking - Evaluation Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/static_eval_loss_multi_seeds.png")
    plt.close()


def main():
    # data prepare
    tokenized_static = prepare_tokenized_static()
    all_losses = []

    for seed in SEEDS:
        # Run training and save losses
        print(f"\n START: [Static] Running seed={seed}")
        loss_static = run_and_get_losses(
            seed,
            tokenized_static["train"],
            tokenized_static["validation"],
        )
        all_losses.append(loss_static)
        print(f"\n FINISH: [Static] Running seed={seed}")
    
    all_losses_df = pd.concat(all_losses, ignore_index=True)
    all_losses_df.to_csv(f"{RESULT_DIR}/static_loss_all_seeds.csv", index=False)

    plot_static_loss(all_losses_df)


if __name__ == "__main__":
    main()