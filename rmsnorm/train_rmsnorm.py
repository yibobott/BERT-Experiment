from datasets import load_dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers import TrainerCallback
import torch
import random
import numpy as np
import os
import pandas as pd
from datetime import datetime

from rmsnorm.rmsnorm_utils import RMSNormBertForMaskedLM

"""
Config
"""
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# NORM_TYPE = "rmsnorm"
NORM_TYPE = "postln"

# MASKING_TYPE = "static"
MASKING_TYPE = "dynamic"

RESULT_DIR = f"./result-rmsnorm-{timestamp}-{NORM_TYPE}-{MASKING_TYPE}"
os.makedirs(RESULT_DIR, exist_ok=True)

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "prajjwal1/bert-small"

DATASET_NAME = ("wikitext", "wikitext-103-raw-v1")
# TRAIN_SAMPLE_SIZE = None
TRAIN_SAMPLE_SIZE = 100000
MAX_LENGTH = 64

BATCH_SIZE = 8
MAX_STEPS = 3000
LR = 5e-5
WARMUP_STEPS = 100
EVAL_STEPS = 100
LOGGING_STEPS = 10

SEEDS = [42, 43, 44]


def load_dataset_demo():
    original_dataset = load_dataset(*DATASET_NAME)

    if TRAIN_SAMPLE_SIZE is not None:
        train = original_dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLE_SIZE))
    else:
        train = original_dataset["train"]
    train = train.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)
    
    validation = original_dataset["validation"]
    validation = validation.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)
    
    return DatasetDict({"train": train, "validation": validation})


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
    和你之前 static-masking.py 中的逻辑一致：
    做一次性静态掩码，并只在被 mask 的位置上计算 loss。
    """
    input_ids = torch.tensor(example["input_ids"])
    labels = input_ids.clone()

    special_ids = set(tokenizer.all_special_ids)
    cand_positions = [i for i, tok_id in enumerate(input_ids)
                      if int(tok_id) not in special_ids]

    if len(cand_positions) == 0:
        labels[:] = -100
        return {"input_ids": input_ids.tolist(), "labels": labels.tolist()}

    mask_positions = [i for i in cand_positions if random.random() < mlm_prob]
    if len(mask_positions) == 0:
        mask_positions = [random.choice(cand_positions)]
    mask_positions = torch.tensor(mask_positions, dtype=torch.long)

    rand = torch.rand(len(mask_positions))
    # 80% -> [MASK]
    mask80 = rand < 0.8
    input_ids[mask_positions[mask80]] = tokenizer.mask_token_id
    # 10% -> random token
    mask10 = (rand >= 0.8) & (rand < 0.9)
    random_tokens = torch.randint(low=0, high=tokenizer.vocab_size, size=(mask10.sum(),))
    input_ids[mask_positions[mask10]] = random_tokens
    # 10% -> keep

    labels[:] = -100
    labels[mask_positions] = torch.tensor(example["input_ids"])[mask_positions]

    return {"input_ids": input_ids.tolist(), "labels": labels.tolist()}


def prepare_static_dataset():
    dataset = load_dataset_demo()
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.filter(lambda ex: sum(ex["attention_mask"]) > 2)
    static_ds = tokenized.map(static_masking, load_from_cache_file=False)
    return static_ds


def prepare_dynamic_dataset():
    dataset = load_dataset_demo()
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.filter(lambda ex: sum(ex["attention_mask"]) > 2)
    return tokenized


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


def get_training_args(output_dir, seed):
    return TrainingArguments(
        output_dir=output_dir,
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


def build_model(norm_type: str):
    """
    norm_type: "postln" -> 标准 BertForMaskedLM
               "rmsnorm" -> 替换所有 LayerNorm 为 RMSNorm
    注意：这里都是从 config 初始化（不加载预训练权重），保证结构公平对比。
    """
    from transformers import BertConfig
    config = BertConfig.from_pretrained(MODEL_NAME)
    if norm_type == "postln":
        model = BertForMaskedLM(config)
    elif norm_type == "rmsnorm":
        model = RMSNormBertForMaskedLM(config)
    else:
        raise ValueError(f"Unknown norm_type={norm_type}")
    return model

def assert_no_layernorm(model):
    import torch.nn as nn
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            raise RuntimeError(f"Found LayerNorm not replaced at: {name}")
    print("[Check] No LayerNorm found. All replaced by RMSNorm.")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果你使用 CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 如果你使用 MPS
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # 为了更确定性（可选）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 准备数据
    if MASKING_TYPE == "static":
        dataset = prepare_static_dataset()
        train_ds = dataset["train"]
        eval_ds = dataset["validation"]
        data_collator = None  # 已经在 dataset 中构造好了 labels
    else:
        dataset = prepare_dynamic_dataset()
        train_ds = dataset["train"]
        eval_ds = dataset["validation"]
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

    all_losses = []

    for seed in SEEDS:
        print(f"\n==== Start training: norm={NORM_TYPE}, masking={MASKING_TYPE}, seed={seed} ====")
        set_seed(seed)

        model = build_model(NORM_TYPE)
        # Check if all LayerNorm are replaced
        if NORM_TYPE == "rmsnorm":
            assert_no_layernorm(model)

        loss_recorder = LossRecorder()
        args = get_training_args(
            output_dir=f"{RESULT_DIR}/{NORM_TYPE}_{MASKING_TYPE}_seed_{seed}",
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=[loss_recorder],
        )
        trainer.train()

        df = loss_recorder.to_dataframe()
        df["seed"] = seed
        csv_path = f"{RESULT_DIR}/{NORM_TYPE}_{MASKING_TYPE}_loss_seed_{seed}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Saved] {csv_path}")
        all_losses.append(df)

    all_df = pd.concat(all_losses, ignore_index=True)
    all_csv = f"{RESULT_DIR}/{NORM_TYPE}_{MASKING_TYPE}_loss_all_seeds.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"[Saved all seeds] {all_csv}")


if __name__ == "__main__":
    main()
