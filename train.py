from typing import Callable
import importlib
import yaml
from argparse import ArgumentParser
import os

ROOT_DIR = os.path.basename(os.path.dirname(__file__))


def get_training_fn(id: str) -> Callable:
    module_name, fn_name = id.rsplit(".", 1)
    module = importlib.import_module("models." + module_name, ROOT_DIR)
    return getattr(module, fn_name)


def get_config(filepath: str) -> dict:
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Trains models on the dance dataset and saves weights."
    )
    parser.add_argument(
        "--config",
        help="Path to the yaml file that defines the training configuration.",
        default="models/config/train_local.yaml",
    )
    args = parser.parse_args()
    config = get_config(args.config)
    training_fn_path = config["training_fn"]
    print(f"Config: {args.config}\nTrainer Id: {training_fn_path}")
    train = get_training_fn(training_fn_path)
    train(config)
