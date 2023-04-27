from torch.utils.data import DataLoader
import pandas as pd
from typing import Callable
from torch import nn
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning import callbacks as cb
from models.utils import LabelWeightedBCELoss
from models.audio_spectrogram_transformer import train as train_audio_spectrogram_transformer, get_id_label_mapping
from preprocessing.dataset import SongDataset, WaveformTrainingEnvironment
from preprocessing.preprocess import get_examples
from models.residual import ResidualDancer, TrainingEnvironment
import yaml
from preprocessing.dataset import DanceDataModule, WaveformSongDataset, HuggingFaceWaveformSongDataset
from torch.utils.data import random_split
from wakepy import keepawake
import numpy as np
from transformers import ASTFeatureExtractor, AutoFeatureExtractor, ASTConfig, AutoModelForAudioClassification
from argparse import ArgumentParser



import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight

def get_training_fn(id:str) -> Callable:
    match id:
        case "ast_ptl":
            return train_ast_lightning
        case "ast_hf":
            return train_ast
        case "residual_dancer":
            return train_model
        case _:
            raise Exception(f"Couldn't find a training function for '{id}'.")

def get_config(filepath:str) -> dict:
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config


def cross_validation(config, k=5):
    df = pd.read_csv("data/songs.csv")
    g_config = config["global"]
    batch_size = config["data_module"]["batch_size"]
    x,y = get_examples(df, "data/samples",class_list=g_config["dance_ids"])
    dataset = SongDataset(x,y)
    splits=KFold(n_splits=k,shuffle=True,random_state=g_config["seed"])
    trainer = pl.Trainer(accelerator=g_config["device"])
    for fold, (train_idx,val_idx) in enumerate(splits.split(x,y)):
        print(f"Fold {fold+1}")
        model = ResidualDancer(n_classes=len(g_config["dance_ids"]))
        train_env = TrainingEnvironment(model,nn.BCELoss())
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        trainer.fit(train_env, train_loader)
        trainer.test(train_env, test_loader)


def train_model(config:dict):
    TARGET_CLASSES = config["global"]["dance_ids"]
    DEVICE = config["global"]["device"]
    SEED = config["global"]["seed"]
    pl.seed_everything(SEED, workers=True)
    data = DanceDataModule(target_classes=TARGET_CLASSES, **config['data_module'])
    model = ResidualDancer(n_classes=len(TARGET_CLASSES), **config['model'])
    label_weights = data.get_label_weights().to(DEVICE)
    criterion = LabelWeightedBCELoss(label_weights) #nn.CrossEntropyLoss(label_weights)
    train_env = TrainingEnvironment(model, criterion, config)
    callbacks = [
        # cb.LearningRateFinder(update_attr=True),
        cb.EarlyStopping("val/loss", patience=5),
        cb.StochasticWeightAveraging(1e-2),
        cb.RichProgressBar(),
        cb.DeviceStatsMonitor(),
    ]
    trainer = pl.Trainer(
        callbacks=callbacks, 
        **config["trainer"]
        )
    trainer.fit(train_env, datamodule=data)
    trainer.test(train_env, datamodule=data)


def train_ast(
    config:dict
):
    TARGET_CLASSES = config["global"]["dance_ids"]
    DEVICE = config["global"]["device"]
    SEED = config["global"]["seed"]
    dataset_kwargs = config["data_module"]["dataset_kwargs"]
    test_proportion = config["data_module"].get("test_proportion", 0.2)
    train_proportion = 1. - test_proportion
    song_data_path="data/songs_cleaned.csv"
    song_audio_path = "data/samples"
    pl.seed_everything(SEED, workers=True)

    df = pd.read_csv(song_data_path)
    x, y = get_examples(df, song_audio_path,class_list=TARGET_CLASSES, multi_label=True)
    train_i, test_i = random_split(np.arange(len(x)), [train_proportion, test_proportion])
    train_ds = HuggingFaceWaveformSongDataset(x[train_i], y[train_i], **dataset_kwargs, resample_frequency=16000)
    test_ds = HuggingFaceWaveformSongDataset(x[test_i], y[test_i], **dataset_kwargs, resample_frequency=16000)
    train_audio_spectrogram_transformer(TARGET_CLASSES, train_ds, test_ds, device=DEVICE)


def train_ast_lightning(config:dict):
    """
    work on integration between waveform dataset and environment. Should work for both HF and PTL.
    """
    TARGET_CLASSES = config["global"]["dance_ids"]
    DEVICE = config["global"]["device"]
    SEED = config["global"]["seed"]
    pl.seed_everything(SEED, workers=True)
    data = DanceDataModule(target_classes=TARGET_CLASSES, dataset_cls=WaveformSongDataset, **config['data_module'])
    id2label, label2id = get_id_label_mapping(TARGET_CLASSES)
    model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, 
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
).to(DEVICE)
    label_weights = data.get_label_weights().to(DEVICE)
    criterion = LabelWeightedBCELoss(label_weights) #nn.CrossEntropyLoss(label_weights)
    train_env = WaveformTrainingEnvironment(model, criterion,feature_extractor, config)
    callbacks = [
        # cb.LearningRateFinder(update_attr=True),
        cb.EarlyStopping("val/loss", patience=5),
        cb.StochasticWeightAveraging(1e-2),
        cb.RichProgressBar()
    ]
    trainer = pl.Trainer(
        callbacks=callbacks, 
        **config["trainer"]
        )
    trainer.fit(train_env, datamodule=data)
    trainer.test(train_env, datamodule=data)

if __name__ == "__main__":
    parser = ArgumentParser(description="Trains models on the dance dataset and saves weights.")
    parser.add_argument("--config", help="Path to the yaml file that defines the training configuration.", default="models/config/train.yaml")
    args = parser.parse_args()
    config = get_config(args.config)
    training_id = config["global"]["id"]
    train = get_training_fn(training_id)
    with keepawake():
        train(config)