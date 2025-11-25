from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import pandas as pd
import os
from datasets import DatasetDict
from datetime import datetime

"""
Config
"""
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
RESULT_DIR = "./result-" + timestamp + "/dynamic"
os.makedirs(RESULT_DIR, exist_ok=True)

# bert-base-uncased: the original BERT-base model
# prajjwal1/bert-small: a smaller version of BERT-base (6 layers)
# MODEL_NAME = "prajjwal1/bert-small"
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = ("wikitext", "wikitext-103-raw-v1")
TRAIN_SAMPLE_SIZE = 100000 # 100k lines for MPS
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

    # We only need a subset (e.g., 100k lines) for efficient experimentation on MPS
    train = original_dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLE_SIZE))
    # train = original_dataset["train"]
    # filter empty / whitespace-only lines
    train = train.filter(lambda x: x["text"] and len(x["text"].strip()) > 0)
    
    # keep the validation set from WikiText-103
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

def prepare_tokenized_dynamic():
    """
    Load dataset and tokenize
    """
    dataset = load_dataset_demo()
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # filter sequences that are basically only CLS/SEP/PAD
    tokenized = tokenized.filter(lambda ex: sum(ex["attention_mask"]) > 2)
    return tokenized

# Dynamic Masking collator
data_collator_dynamic = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15   # same as BERT
)


"""
# test dynamic masking
sample = tokenizer("I am doing the term project of C6009 Machine Learning at Scale. 55555", return_tensors="pt")
for _ in range(10):
    batch = data_collator_dynamic([sample["input_ids"][0]])
    print(tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))

"""

def get_training_args(output_dir, seed):
    """
    Training args (small demo)
    We reduce both train and evaluation batch size to 8 to make the MLM pretraining more stable and efficient on Apple MPS, 
    which does not provide the same level of optimization as CUDA GPUs.
    """
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
        seed=seed
    )

class LossRecorder(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            if "loss" in logs:
                self.logs.append({
                    "step": step,
                    "loss": logs["loss"],
                    "type": "train"
                })
            if "eval_loss" in logs:
                self.logs.append({
                    "step": step,
                    "loss": logs["eval_loss"],
                    "type": "eval"
                })

    def to_dataframe(self):
        return pd.DataFrame(self.logs)


def run_and_get_losses(seed, train_dataset, eval_dataset, data_collator):
    """
    Run training for one seed
    """
    args = get_training_args(output_dir=f"{RESULT_DIR}/dynamic_{seed}", seed=seed)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    loss_recorder = LossRecorder()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[loss_recorder]
    )
    trainer.train()

    df = loss_recorder.to_dataframe()
    file_name = f"{RESULT_DIR}/dynamic_loss_seed_{seed}.csv"
    df.to_csv(file_name, index=False)
    print(f"\n[Dynamic] Running seed={seed}: Saved to {file_name}")

    df["seed"] = seed
    return df


def plot_dynamic_loss(all_losses: pd.DataFrame):
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
    plt.title("Dynamic Masking - Training Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/dynamic_train_loss_multi_seeds.png")
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
    plt.title("Dynamic Masking - Evaluation Loss (multi seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/dynamic_eval_loss_multi_seeds.png")
    plt.close()


def main():
    # data prepare
    tokenized = prepare_tokenized_dynamic()
    all_losses = []

    for seed in SEEDS:
        print(f"\nSTART: [Dynamic] Running seed={seed}")
        loss = run_and_get_losses(
            seed,
            tokenized["train"],
            tokenized["validation"],
            data_collator_dynamic,
        )
        all_losses.append(loss)

    all_losses_df = pd.concat(all_losses, ignore_index=True)
    all_losses_df.to_csv(f"{RESULT_DIR}/dynamic_loss_all_seeds.csv", index=False)

    plot_dynamic_loss(all_losses_df)

if __name__ == "__main__":
    main()