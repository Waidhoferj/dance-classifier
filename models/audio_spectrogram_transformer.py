from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    ASTConfig,
    ASTFeatureExtractor,
    ASTForAudioClassification,
)
import torch
from torch import nn
from models.training_environment import TrainingEnvironment
from preprocessing.pipelines import WaveformTrainingPipeline

from preprocessing.dataset import (
    DanceDataModule,
    HuggingFaceDatasetWrapper,
    get_datasets,
)
from .utils import LabelWeightedBCELoss, get_id_label_mapping, compute_hf_metrics

import pytorch_lightning as pl
from pytorch_lightning import callbacks as cb

MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


class AST(nn.Module):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        id2label, label2id = get_id_label_mapping(labels)
        config = ASTConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=4,
            id2label=id2label,
            label2id=label2id,
            num_labels=len(label2id),
            ignore_mismatched_sizes=True,
        )
        self.model = ASTForAudioClassification(config)

    def forward(self, x):
        return self.model(x).logits


# TODO: Remove waveform normalization from ASTFeatureExtractor.
# Find correct mean and std dev
# Find correct max length
class ASTExtractorWrapper:
    def __init__(self, sampling_rate=16000, return_tensors="pt") -> None:
        max_length = 1024
        self.extractor = ASTFeatureExtractor(max_length=max_length, do_normalize=True)
        self.sampling_rate = sampling_rate
        self.return_tensors = return_tensors
        self.waveform_pipeline = WaveformTrainingPipeline()  # TODO configure from yaml

    def __call__(self, x) -> Any:
        x = self.waveform_pipeline(x)
        device = x.device
        x = x.squeeze(0).numpy()
        x = self.extractor(
            x, return_tensors=self.return_tensors, sampling_rate=self.sampling_rate
        )

        x = x["input_values"].squeeze(0).to(device)
        return x


def train_lightning_ast(config: dict):
    """
    work on integration between waveform dataset and environment. Should work for both HF and PTL.
    """
    TARGET_CLASSES = config["dance_ids"]
    DEVICE = config["device"]
    SEED = config["seed"]
    pl.seed_everything(SEED, workers=True)
    feature_extractor = ASTExtractorWrapper()
    dataset = get_datasets(config["datasets"], feature_extractor)
    data = DanceDataModule(
        dataset,
        target_classes=TARGET_CLASSES,
        **config["data_module"],
    )
    model = AST(TARGET_CLASSES).to(DEVICE)
    label_weights = data.get_label_weights().to(DEVICE)
    criterion = LabelWeightedBCELoss(label_weights)
    if "checkpoint" in config:
        train_env = TrainingEnvironment.load_from_checkpoint(
            config["checkpoint"], criterion=criterion, model=model, config=config
        )
    else:
        train_env = TrainingEnvironment(model, criterion, config)
    callbacks = [
        cb.EarlyStopping("val/loss", patience=2),
        cb.RichProgressBar(),
    ]
    trainer = pl.Trainer(callbacks=callbacks, **config["trainer"])
    trainer.fit(train_env, datamodule=data)
    trainer.test(train_env, datamodule=data)


def train_huggingface_ast(config: dict):
    TARGET_CLASSES = config["dance_ids"]
    DEVICE = config["device"]
    SEED = config["seed"]
    OUTPUT_DIR = "models/weights/ast"
    batch_size = config["data_module"]["batch_size"]
    epochs = config["data_module"]["min_epochs"]
    test_proportion = config["data_module"].get("test_proportion", 0.2)
    pl.seed_everything(SEED, workers=True)
    dataset = get_datasets(config["datasets"])
    hf_dataset = HuggingFaceDatasetWrapper(dataset)
    id2label, label2id = get_id_label_mapping(TARGET_CLASSES)
    model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    preprocess_waveform = lambda wf: feature_extractor(
        wf,
        sampling_rate=train_ds.resample_frequency,
        # padding="max_length",
        # return_tensors="pt",
    )
    hf_dataset.append_to_pipeline(preprocess_waveform)
    test_proportion = config["data_module"]["test_proportion"]
    train_proporition = 1 - test_proportion
    train_ds, test_ds = torch.utils.data.random_split(
        hf_dataset, [train_proporition, test_proportion]
    )

    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(TARGET_CLASSES),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=5,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        use_mps_device=DEVICE == "mps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_hf_metrics,
    )
    trainer.train()
    return model
