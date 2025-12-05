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
from config import RMSNormConfig, load_rmsnorm_config

from rmsnorm.rmsnorm_utils import RMSNormBertForMaskedLM


def preprocess_dataset(config: RMSNormConfig):
    original_dataset = load_dataset(*config.dataset_name)

    if config.train_sample_size is not None:
        train = original_dataset["train"].shuffle(seed=42).select(range(config.train_sample_size))
    else:
        train = original_dataset["train"]
    train = train.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)
    
    validation = original_dataset["validation"]
    validation = validation.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)
    
    return DatasetDict({"train": train, "validation": validation})


def tokenize_fn(examples, config: RMSNormConfig, tokenizer: BertTokenizerFast):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=config.max_length
    )


def static_masking(example, tokenizer: BertTokenizerFast, mlm_prob=0.15):
    """One-shot static masking, same logic as in static-masking.py.

    Perform static masking once during dataset construction, and compute MLM
    loss only on the masked positions.
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


def prepare_static_dataset(config: RMSNormConfig, tokenizer: BertTokenizerFast):
    dataset = preprocess_dataset(config)
    tokenized = dataset.map(
        lambda examples: tokenize_fn(examples, config, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized = tokenized.filter(lambda ex: sum(ex["attention_mask"]) > 2)
    static_ds = tokenized.map(lambda example: static_masking(example, tokenizer), load_from_cache_file=False)
    return static_ds


def prepare_dynamic_dataset(config: RMSNormConfig, tokenizer: BertTokenizerFast):
    dataset = preprocess_dataset(config)
    tokenized = dataset.map(
        lambda examples: tokenize_fn(examples, config, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
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


def get_training_args(config: RMSNormConfig, output_dir, seed):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        max_steps=config.max_steps,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="no",
        fp16=False,
        report_to="none",
        seed=seed,
    )


def build_model(config: RMSNormConfig):
    """Build a BERT-MLM model under different normalization schemes.

    norm_type:
        "postln"  -> standard BertForMaskedLM (Post-LayerNorm)
        "rmsnorm" -> BertForMaskedLM with all LayerNorm replaced by RMSNorm

    Note: both variants are initialized from a BertConfig (no pretrained
    checkpoint) to ensure a fair comparison of architectures.
    """
    from transformers import BertConfig
    from transformers import BertConfig
    bert_config = BertConfig.from_pretrained(config.model_name)
    if config.norm_type == "postln":
        model = BertForMaskedLM(bert_config)
    elif config.norm_type == "rmsnorm":
        model = RMSNormBertForMaskedLM(bert_config)
    else:
        raise ValueError(f"Unknown norm_type={config.norm_type}")
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

    # If CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # If you are using MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # For more deterministic behavior in CuDNN (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu or MPS")

    # Initialize config
    config = load_rmsnorm_config()

    # Prepare tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

    # Prepare datasets
    if config.masking_type == "static":
        dataset = prepare_static_dataset(config, tokenizer)
        train_ds = dataset["train"]
        eval_ds = dataset["validation"]
        data_collator = None  # labels have already been constructed in the dataset
    else:
        dataset = prepare_dynamic_dataset(config, tokenizer)
        train_ds = dataset["train"]
        eval_ds = dataset["validation"]
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

    all_losses = []

    for seed in config.seeds:
        print(f"\n==== Start training: norm={config.norm_type}, masking={config.masking_type}, seed={seed} ====")
        set_seed(seed)

        model = build_model(config)
        # Check if all LayerNorm are replaced
        if config.norm_type == "rmsnorm":
            assert_no_layernorm(model)

        loss_recorder = LossRecorder()
        args = get_training_args(
            config,
            output_dir=f"{config.result_dir}/{config.norm_type}_{config.masking_type}_seed_{seed}",
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
        csv_path = f"{config.result_dir}/{config.norm_type}_{config.masking_type}_loss_seed_{seed}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Saved] {csv_path}")
        all_losses.append(df)

    all_df = pd.concat(all_losses, ignore_index=True)
    all_csv = f"{config.result_dir}/{config.norm_type}_{config.masking_type}_loss_all_seeds.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"[Saved all seeds] {all_csv}")


if __name__ == "__main__":
    main()
