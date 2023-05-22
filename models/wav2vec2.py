import os
from typing import Any
import pytorch_lightning as pl
from torch.utils.data import random_split
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

from preprocessing.dataset import (
    HuggingFaceDatasetWrapper,
    BestBallroomDataset,
    get_datasets,
)
from preprocessing.pipelines import WaveformTrainingPipeline

from .utils import get_id_label_mapping, compute_hf_metrics

MODEL_CHECKPOINT = "facebook/wav2vec2-base"


class Wav2VecFeatureExtractor:
    def __init__(self) -> None:
        self.waveform_pipeline = WaveformTrainingPipeline()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_CHECKPOINT,
        )

    def __call__(self, waveform) -> Any:
        waveform = self.waveform_pipeline(waveform)
        return self.feature_extractor(
            waveform, sampling_rate=self.feature_extractor.sampling_rate
        )

    def __getattr__(self, attr):
        return getattr(self.feature_extractor, attr)


def train_wav_model(config: dict):
    TARGET_CLASSES = config["dance_ids"]
    DEVICE = config["device"]
    SEED = config["seed"]
    OUTPUT_DIR = "models/weights/wav2vec2"
    batch_size = config["data_module"]["batch_size"]
    epochs = config["trainer"]["min_epochs"]
    test_proportion = config["data_module"].get("test_proportion", 0.2)
    pl.seed_everything(SEED, workers=True)
    dataset = get_datasets(config["datasets"])
    id2label, label2id = get_id_label_mapping(TARGET_CLASSES)
    test_proportion = config["data_module"]["test_proportion"]
    train_proporition = 1 - test_proportion
    train_ds, test_ds = random_split(dataset, [train_proporition, test_proportion])
    feature_extractor = Wav2VecFeatureExtractor()
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(TARGET_CLASSES),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
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
