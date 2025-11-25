from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForPreTraining,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

"""
Config
"""
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
RESULT_DIR = "./result-nsp-" + timestamp
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
# MODEL_NAME = "prajjwal1/bert-small"
DATASET_NAME = ("wikitext", "wikitext-103-raw-v1")
TRAIN_SAMPLE_SIZE = None
VALID_SAMPLE_SIZE = 50000
# TRAIN_SAMPLE_SIZE = 100000
# VALID_SAMPLE_SIZE = 20000
MAX_LENGTH = 64

BATCH_SIZE = 8
MAX_STEPS = 3000
LR = 5e-5
WARMUP_STEPS = 100
EVAL_STEPS = 100
LOGGING_STEPS = 10

SEEDS = [42, 43, 44]


def load_wikitext_pairs(train_sample_size=None, seed=42):
    """Construct NSP training/validation sentence pairs from wikitext-103-raw-v1.

    - Positive examples: (sent[i], sent[i + 1]), label = 0
    - Negative examples: (sent[i], sent[j]), where j is random and j != i + 1, label = 1

    Note: use the given seed to fix Python's random state for reproducibility.
    """
    random.seed(seed)
    original = load_dataset(*DATASET_NAME)

    def collect_sentences(split):
        # Filter out empty lines or lines that contain only whitespace
        return [t for t in original[split]["text"] if t and t.strip()]

    train_sents = collect_sentences("train")
    valid_sents = collect_sentences("validation")

    # For efficiency, optionally keep only the first train_sample_size + 1 sentences to build pairs
    if train_sample_size is not None:
        train_sents = train_sents[: train_sample_size + 1]

    def make_nsp_pairs(sent_list, max_pairs=None):
        data_a, data_b, labels = [], [], []
        n = len(sent_list)

        for i in range(n - 1):
            # 50% positive examples: the next sentence
            if random.random() < 0.5:
                a = sent_list[i]
                b = sent_list[i + 1]
                label = 0  # is_next
            else:
                # 50% negative examples: a random sentence
                a = sent_list[i]
                j = random.randint(0, n - 1)
                # Avoid accidentally sampling the true next sentence
                if j == i + 1:
                    j = (j + 1) % n
                b = sent_list[j]
                label = 1  # random

            data_a.append(a)
            data_b.append(b)
            labels.append(label)

            if max_pairs and len(data_a) >= max_pairs:
                break

        return Dataset.from_dict(
            {
                "sentence_a": data_a,
                "sentence_b": data_b,
                "next_sentence_label": labels,
            }
        )

    train_pairs = make_nsp_pairs(train_sents)
    valid_pairs = make_nsp_pairs(valid_sents, max_pairs=VALID_SAMPLE_SIZE)  # Validation set does not need to be very large

    dataset = DatasetDict({"train": train_pairs, "validation": valid_pairs})
    return dataset


# Tokenizer 
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize_pair(examples):
    """Encode (sentence_a, sentence_b) into input_ids, token_type_ids and attention_mask."""
    return tokenizer(
        examples["sentence_a"],
        examples["sentence_b"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def static_masking(example, mlm_prob=0.15):
    """Static masking for the BERT NSP setting.

    - Apply BERT-style MLM on input_ids:
      * Among non-special tokens, select positions with probability 0.15
      * 80% -> replace with [MASK]
      * 10% -> replace with a random token
      * 10% -> keep the original token
    - In labels, keep the original token id only at masked positions; set all other positions to -100.
    """
    input_ids = torch.tensor(example["input_ids"])
    labels = input_ids.clone()

    special_ids = set(tokenizer.all_special_ids)

    # Candidate mask positions: non-special tokens only
    cand_positions = [
        i for i, tok_id in enumerate(input_ids)
        if int(tok_id) not in special_ids
    ]

    if len(cand_positions) == 0:
        labels[:] = -100
        example["labels"] = labels.tolist()
        return example

    # Sample mask positions with probability mlm_prob (default 0.15)
    mask_positions = [
        i for i in cand_positions if random.random() < mlm_prob
    ]

    # If none are selected, force-mask one position to avoid degenerate loss = 0
    if len(mask_positions) == 0:
        mask_positions = [random.choice(cand_positions)]

    mask_positions = torch.tensor(mask_positions, dtype=torch.long)
    rand = torch.rand(len(mask_positions))

    # 80% -> replace with [MASK]
    mask80 = rand < 0.8
    input_ids[mask_positions[mask80]] = tokenizer.mask_token_id

    # 10% -> replace with a random token
    mask10 = (rand >= 0.8) & (rand < 0.9)
    random_tokens = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(mask10.sum(),),
    )
    input_ids[mask_positions[mask10]] = random_tokens

    # 10% -> keep unchanged (do not modify input_ids)

    # labels: only masked positions participate in the MLM loss
    labels[:] = -100
    orig_ids = torch.tensor(example["input_ids"])
    labels[mask_positions] = orig_ids[mask_positions]

    example["input_ids"] = input_ids.tolist()
    example["labels"] = labels.tolist()
    return example


class LossRecorder(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Training loss
        if logs is not None and "loss" in logs:
            self.logs.append(
                {
                    "step": state.global_step,
                    "loss": logs["loss"],
                    "type": "train",
                }
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Evaluation loss
        if metrics is not None and "eval_loss" in metrics:
            self.logs.append(
                {
                    "step": state.global_step,
                    "loss": metrics["eval_loss"],
                    "type": "eval",
                }
            )

    def to_dataframe(self):
        return pd.DataFrame(self.logs)


def run_for_seed(seed: int) -> pd.DataFrame:
    """Run training for a single random seed."""
    print(f"\n========== START: Running NSP baseline with SEED = {seed} ==========\n")

    # Random seeds
    random.seed(seed)
    torch.manual_seed(seed)

    # 1) Construct NSP dataset
    dataset = load_wikitext_pairs(
        train_sample_size=TRAIN_SAMPLE_SIZE,
        seed=seed,
    )

    # 2) Tokenize sentence pairs
    tokenized = dataset.map(tokenize_pair, batched=True)
    # tokenized = dataset.map(tokenize_pair, batched=True, num_proc=12, batch_size=4000)
    tokenized = tokenized.remove_columns(["sentence_a", "sentence_b"])

    # 3) Apply static masking (MLM)
    tokenized_static = tokenized.map(static_masking)

    # 4) Set up TrainingArguments
    args = TrainingArguments(
        output_dir=f"{RESULT_DIR}/bert_nsp_seed_{seed}",

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        max_steps=MAX_STEPS,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        save_strategy="no",

        fp16=False,
        report_to="none",
        seed=seed,
    )

    # 5) Model + Trainer
    model = BertForPreTraining.from_pretrained(MODEL_NAME)
    loss_recorder = LossRecorder()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_static["train"],
        eval_dataset=tokenized_static["validation"],
        callbacks=[loss_recorder],
    )

    trainer.train()

    df = loss_recorder.to_dataframe()
    df["seed"] = seed
    # Save a separate CSV for each seed
    per_seed_csv = os.path.join(RESULT_DIR, f"nsp_baseline_loss_seed{seed}.csv")
    df.to_csv(per_seed_csv, index=False)
    print(f"Saved per-seed loss to: {per_seed_csv}")
    print(f"\n========== FINISH! Running NSP baseline with SEED = {seed} ==========\n")

    return df


def plot_static_loss(all_losses: pd.DataFrame):
    """Columns in all_losses:

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
    plt.title("NSP Baseline - Training Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/nsp_baseline_train_loss_multi_seeds.png")
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
    plt.title("NSP Baseline - Evaluation Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/nsp_baseline_eval_loss_multi_seeds.png")
    plt.close()

def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu or MPS")
    all_dfs = []
    for seed in SEEDS:
        df = run_for_seed(seed)
        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_path = os.path.join(RESULT_DIR, "nsp_baseline_loss_all_seeds.csv")
    all_df.to_csv(all_path, index=False)
    print(f"\nSaved combined multi-seed loss to: {all_path}\n")

    plot_static_loss(all_df)


if __name__ == "__main__":
    main()
