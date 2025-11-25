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

"""
Config
"""
RESULT_DIR = "./result-nsp"
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
# MODEL_NAME = "prajjwal1/bert-small"
DATASET_NAME = ("wikitext", "wikitext-103-raw-v1")
# TRAIN_SAMPLE_SIZE = None
# VALID_SAMPLE_SIZE = 50000
TRAIN_SAMPLE_SIZE = 100000
VALID_SAMPLE_SIZE = 20000
MAX_LENGTH = 64

BATCH_SIZE = 8
MAX_STEPS = 3000
LR = 5e-5
WARMUP_STEPS = 100
EVAL_STEPS = 100
LOGGING_STEPS = 10

SEEDS = [42, 43, 44]


def load_wikitext_pairs(train_sample_size=None, seed=42):
    """
    从 wikitext-103-raw-v1 构造 NSP 训练/验证对:
    - 正样本: (sent[i], sent[i+1]), label=0
    - 负样本: (sent[i], sent[j]), j 随机且 != i+1, label=1

    注意：使用 seed 固定 random 以便多次实验可复现。
    """
    random.seed(seed)
    original = load_dataset(*DATASET_NAME)

    def collect_sentences(split):
        # 过滤空行 / 只包含空白字符的行
        return [t for t in original[split]["text"] if t and t.strip()]

    train_sents = collect_sentences("train")
    valid_sents = collect_sentences("validation")

    # 为了效率，只取前 train_sample_size+1 个构造 pair
    if train_sample_size is not None:
        train_sents = train_sents[: train_sample_size + 1]

    def make_nsp_pairs(sent_list, max_pairs=None):
        data_a, data_b, labels = [], [], []
        n = len(sent_list)

        for i in range(n - 1):
            # 50% 正样本: 下一句
            if random.random() < 0.5:
                a = sent_list[i]
                b = sent_list[i + 1]
                label = 0  # is_next
            else:
                # 50% 负样本: 随机句子
                a = sent_list[i]
                j = random.randint(0, n - 1)
                # 避免刚好选到下一句
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
    valid_pairs = make_nsp_pairs(valid_sents, max_pairs=VALID_SAMPLE_SIZE)  # 验证集不用太大

    dataset = DatasetDict({"train": train_pairs, "validation": valid_pairs})
    return dataset


# Tokenizer 
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize_pair(examples):
    """
    把 (sentence_a, sentence_b) 编码成:
    input_ids, token_type_ids, attention_mask
    """
    return tokenizer(
        examples["sentence_a"],
        examples["sentence_b"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def static_masking(example, mlm_prob=0.15):
    """
    Static Masking for BERT NSP setting:

    - 对 input_ids 做 BERT-style MLM:
      * 从非 special token 中，以 0.15 概率选 mask 位置
      * 80% -> [MASK]
      * 10% -> random token
      * 10% -> 保持原样
    - labels 只在 mask 的位置保留原 token id，其余设为 -100
    """
    input_ids = torch.tensor(example["input_ids"])
    labels = input_ids.clone()

    special_ids = set(tokenizer.all_special_ids)

    # 候选 mask 位置: 非 special token
    cand_positions = [
        i for i, tok_id in enumerate(input_ids)
        if int(tok_id) not in special_ids
    ]

    if len(cand_positions) == 0:
        labels[:] = -100
        example["labels"] = labels.tolist()
        return example

    # 15% 概率采样 mask 位置
    mask_positions = [
        i for i in cand_positions if random.random() < mlm_prob
    ]

    # 如果一个都没采到，强制选一个，避免 loss=0
    if len(mask_positions) == 0:
        mask_positions = [random.choice(cand_positions)]

    mask_positions = torch.tensor(mask_positions, dtype=torch.long)
    rand = torch.rand(len(mask_positions))

    # 80% -> [MASK]
    mask80 = rand < 0.8
    input_ids[mask_positions[mask80]] = tokenizer.mask_token_id

    # 10% -> random token
    mask10 = (rand >= 0.8) & (rand < 0.9)
    random_tokens = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(mask10.sum(),),
    )
    input_ids[mask_positions[mask10]] = random_tokens

    # 10% -> 保持原样（不改 input_ids）

    # labels: 只在 mask 的位置参与 MLM loss
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
        # 训练时的 loss
        if logs is not None and "loss" in logs:
            self.logs.append(
                {
                    "step": state.global_step,
                    "loss": logs["loss"],
                    "type": "train",
                }
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # eval 的 loss
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
    """
    Run training for a single seed
    """
    print(f"\n========== START: Running NSP baseline with SEED = {seed} ==========\n")

    # 随机种子
    random.seed(seed)
    torch.manual_seed(seed)

    # 1) 构造 NSP 数据
    dataset = load_wikitext_pairs(
        train_sample_size=TRAIN_SAMPLE_SIZE,
        seed=seed,
    )

    # 2) tokenize
    tokenized = dataset.map(tokenize_pair, batched=True)
    tokenized = tokenized.remove_columns(["sentence_a", "sentence_b"])

    # 3) static masking
    tokenized_static = tokenized.map(static_masking)

    # 4) TrainingArguments
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

    # 5) 模型 + Trainer
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
    # 每个 seed 单独存一份
    per_seed_csv = os.path.join(RESULT_DIR, f"nsp_baseline_loss_seed{seed}.csv")
    df.to_csv(per_seed_csv, index=False)
    print(f"Saved per-seed loss to: {per_seed_csv}")
    print(f"\n========== FINISH! Running NSP baseline with SEED = {seed} ==========\n")

    return df


# ========== 6. 多 seed 循环 & 合并结果 ==========

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
