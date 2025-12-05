from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import DatasetDict
from config import DynamicMaskingConfig, load_dynamic_config


def preprocess_dataset(config: DynamicMaskingConfig):
    original_dataset = load_dataset(*config.dataset_name)

    train_full = original_dataset["train"].shuffle(seed=42)
    if config.train_sample_size is not None and config.train_sample_size < len(
        train_full
    ):
        train = train_full.select(range(config.train_sample_size))
    else:
        train = train_full

    # filter empty / whitespace-only lines
    train = train.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)

    validation = original_dataset["validation"]
    validation = validation.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)

    dataset = DatasetDict({"train": train, "validation": validation})
    return dataset


def tokenize_fn(examples, config: DynamicMaskingConfig, tokenizer: BertTokenizerFast):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
    )


def prepare_tokenized_dynamic(config: DynamicMaskingConfig, tokenizer: BertTokenizerFast):
    """
    Load dataset and tokenize
    """
    dataset = preprocess_dataset(config)
    tokenized = dataset.map(
        lambda examples: tokenize_fn(examples, config, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    # filter sequences that are basically only CLS/SEP/PAD
    tokenized = tokenized.filter(lambda ex: sum(ex["attention_mask"]) > 2)
    return tokenized


def get_training_args(config: DynamicMaskingConfig, output_dir, seed):
    """
    Training args (small demo)
    We reduce both train and evaluation batch size to 8 to make the MLM pretraining more stable and efficient on Apple MPS,
    which does not provide the same level of optimization as CUDA GPUs.
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_acc_steps,
        max_steps=config.max_steps,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="no",
        fp16=True,
        report_to="none",
        seed=seed,
    )


class LossRecorder(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            if "loss" in logs:
                self.logs.append({"step": step, "loss": logs["loss"], "type": "train"})
            if "eval_loss" in logs:
                self.logs.append(
                    {"step": step, "loss": logs["eval_loss"], "type": "eval"}
                )

    def to_dataframe(self):
        return pd.DataFrame(self.logs)


def run_and_get_losses(config: DynamicMaskingConfig, seed, train_dataset, eval_dataset, data_collator):
    """
    Run training for one seed
    """
    args = get_training_args(
        config, output_dir=f"{config.result_dir}/dynamic_{seed}", seed=seed
    )
    model = BertForMaskedLM.from_pretrained(config.model_name)
    loss_recorder = LossRecorder()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[loss_recorder],
    )
    trainer.train()

    df = loss_recorder.to_dataframe()
    file_name = f"{config.result_dir}/dynamic_loss_seed_{seed}.csv"
    df.to_csv(file_name, index=False)
    print(f"\n[Dynamic] Running seed={seed}: Saved to {file_name}")

    df["seed"] = seed
    return df


def plot_dynamic_loss(config: DynamicMaskingConfig, all_losses: pd.DataFrame):
    """
    collums in all_losses:
      - step
      - loss
      - type ('train' / 'eval')
      - seed
    """

    # plot train loss
    plt.figure()
    for seed in config.seeds:
        df_seed_train = all_losses[
            (all_losses["seed"] == seed) & (all_losses["type"] == "train")
        ]
        if df_seed_train.empty:
            continue
        plt.plot(
            df_seed_train["step"], df_seed_train["loss"], label=f"seed={seed} - train"
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Dynamic Masking - Training Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.result_dir}/dynamic_train_loss_multi_seeds.png")
    plt.close()

    # plot eval loss
    plt.figure()
    for seed in config.seeds:
        df_seed_eval = all_losses[
            (all_losses["seed"] == seed) & (all_losses["type"] == "eval")
        ]
        if df_seed_eval.empty:
            continue
        plt.plot(
            df_seed_eval["step"], df_seed_eval["loss"], label=f"seed={seed} - eval"
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Dynamic Masking - Evaluation Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.result_dir}/dynamic_eval_loss_multi_seeds.png")
    plt.close()


def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu or MPS")

    # Initialize config
    config = load_dynamic_config()

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

    # Dynamic Masking collator
    data_collator_dynamic = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15  # same as BERT
    )
    """
    # test dynamic masking
    sample = tokenizer("I am doing the term project of C6009 Machine Learning at Scale. 55555", return_tensors="pt")
    for _ in range(10):
        batch = data_collator_dynamic([sample["input_ids"][0]])
        print(tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))

    """

    # data prepare
    tokenized = prepare_tokenized_dynamic(config, tokenizer)

    all_losses = []
    for seed in SEEDS:
        print(f"\nSTART: [Dynamic] Running seed={seed}")
        loss = run_and_get_losses(
            config,
            seed,
            tokenized["train"],
            tokenized["validation"],
            data_collator_dynamic,
        )
        all_losses.append(loss)
    all_losses_df = pd.concat(all_losses, ignore_index=True)
    all_losses_df.to_csv(f"{config.result_dir}/dynamic_loss_all_seeds.csv", index=False)

    plot_dynamic_loss(config, all_losses_df)


if __name__ == "__main__":
    main()
