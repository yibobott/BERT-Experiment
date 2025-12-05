from dataclasses import dataclass, asdict
from datetime import datetime
import os

from utils import load_yaml_config


@dataclass
class StaticMaskingConfig:
    result_dir: str
    model_name: str
    dataset_name: tuple
    train_sample_size: int | None
    max_length: int
    batch_size: int
    max_steps: int
    grad_acc_steps: int
    lr: float
    warmup_steps: int
    eval_steps: int
    logging_steps: int
    seeds: list[int]


def load_static_config() -> StaticMaskingConfig:
    cfg = load_yaml_config()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _base_dir = cfg["output"]["base_dir"]
    _sub_dir = cfg["output"]["sub_dir"]
    result_dir = os.path.join(_base_dir, timestamp, _sub_dir)
    os.makedirs(result_dir, exist_ok=True)

    model_name = cfg["model"]["name"]

    dataset_name = (cfg["dataset"]["name"], cfg["dataset"]["config"])
    train_sample_size = cfg["dataset"]["train_sample_size"]
    max_length = cfg["dataset"]["max_length"]

    batch_size = cfg["training"]["batch_size"]
    max_steps = cfg["training"]["max_steps"]
    grad_acc_steps = cfg["training"]["grad_acc_steps"]
    lr = cfg["training"]["lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    eval_steps = cfg["training"]["eval_steps"]
    logging_steps = cfg["training"]["logging_steps"]
    seeds = cfg["training"]["seeds"]

    config = StaticMaskingConfig(
        result_dir=result_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        train_sample_size=train_sample_size,
        max_length=max_length,
        batch_size=batch_size,
        max_steps=max_steps,
        grad_acc_steps=grad_acc_steps,
        lr=lr,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seeds=seeds,
    )

    print("[Config] Loaded StaticMaskingConfig:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    return config


@dataclass
class DynamicMaskingConfig:
    result_dir: str
    model_name: str
    dataset_name: tuple
    train_sample_size: int | None
    max_length: int
    batch_size: int
    max_steps: int
    grad_acc_steps: int
    lr: float
    warmup_steps: int
    eval_steps: int
    logging_steps: int
    seeds: list[int]


def load_dynamic_config() -> DynamicMaskingConfig:
    cfg = load_yaml_config()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _base_dir = cfg["output"]["base_dir"]
    _sub_dir = cfg["output"]["sub_dir"]
    result_dir = os.path.join(_base_dir, timestamp, _sub_dir)
    os.makedirs(result_dir, exist_ok=True)

    model_name = cfg["model"]["name"]

    dataset_name = (cfg["dataset"]["name"], cfg["dataset"]["config"])
    train_sample_size = cfg["dataset"]["train_sample_size"]
    max_length = cfg["dataset"]["max_length"]

    batch_size = cfg["training"]["batch_size"]
    max_steps = cfg["training"]["max_steps"]
    grad_acc_steps = cfg["training"]["grad_acc_steps"]
    lr = cfg["training"]["lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    eval_steps = cfg["training"]["eval_steps"]
    logging_steps = cfg["training"]["logging_steps"]
    seeds = cfg["training"]["seeds"]

    config = DynamicMaskingConfig(
        result_dir=result_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        train_sample_size=train_sample_size,
        max_length=max_length,
        batch_size=batch_size,
        max_steps=max_steps,
        grad_acc_steps=grad_acc_steps,
        lr=lr,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seeds=seeds,
    )

    print("[Config] Loaded DynamicMaskingConfig:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    return config


@dataclass
class NSPConfig:
    result_dir: str
    model_name: str
    dataset_name: tuple
    train_sample_size: int | None
    valid_sample_size: int | None
    max_length: int
    batch_size: int
    max_steps: int
    lr: float
    warmup_steps: int
    eval_steps: int
    logging_steps: int
    seeds: list[int]


def load_nsp_config() -> NSPConfig:
    cfg = load_yaml_config()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _base_dir = cfg["output"]["base_dir"]
    result_dir = os.path.join(_base_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    model_name = cfg["model"]["name"]

    dataset_name = (cfg["dataset"]["name"], cfg["dataset"]["config"])
    train_sample_size = cfg["dataset"]["train_sample_size"]
    valid_sample_size = cfg["dataset"].get("valid_sample_size")
    max_length = cfg["dataset"]["max_length"]

    batch_size = cfg["training"]["batch_size"]
    max_steps = cfg["training"]["max_steps"]
    lr = cfg["training"]["lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    eval_steps = cfg["training"]["eval_steps"]
    logging_steps = cfg["training"]["logging_steps"]
    seeds = cfg["training"]["seeds"]

    config = NSPConfig(
        result_dir=result_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        train_sample_size=train_sample_size,
        valid_sample_size=valid_sample_size,
        max_length=max_length,
        batch_size=batch_size,
        max_steps=max_steps,
        lr=lr,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seeds=seeds,
    )

    print("[Config] Loaded NSPConfig:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    return config


@dataclass
class RMSNormConfig:
    norm_type: str
    masking_type: str
    result_dir: str
    model_name: str
    dataset_name: tuple
    train_sample_size: int | None
    max_length: int
    batch_size: int
    max_steps: int
    lr: float
    warmup_steps: int
    eval_steps: int
    logging_steps: int
    seeds: list[int]


def load_rmsnorm_config() -> RMSNormConfig:
    cfg = load_yaml_config()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    norm_type = cfg["experiment"]["norm_type"]
    masking_type = cfg["experiment"]["masking_type"]

    _base_dir = cfg["output"]["base_dir"]
    result_dir = os.path.join(_base_dir, f"{timestamp}-{norm_type}-{masking_type}")
    os.makedirs(result_dir, exist_ok=True)

    model_name = cfg["model"]["name"]

    dataset_name = (cfg["dataset"]["name"], cfg["dataset"]["config"])
    train_sample_size = cfg["dataset"]["train_sample_size"]
    max_length = cfg["dataset"]["max_length"]

    batch_size = cfg["training"]["batch_size"]
    max_steps = cfg["training"]["max_steps"]
    lr = cfg["training"]["lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    eval_steps = cfg["training"]["eval_steps"]
    logging_steps = cfg["training"]["logging_steps"]
    seeds = cfg["training"]["seeds"]

    config = RMSNormConfig(
        norm_type=norm_type,
        masking_type=masking_type,
        result_dir=result_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        train_sample_size=train_sample_size,
        max_length=max_length,
        batch_size=batch_size,
        max_steps=max_steps,
        lr=lr,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seeds=seeds,
    )

    print("[Config] Loaded RMSNormConfig:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    return config
