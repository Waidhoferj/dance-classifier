from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning import callbacks as cb
from models.utils import LabelWeightedBCELoss
from preprocessing.dataset import SongDataset
from preprocessing.preprocess import get_examples
from models.residual import ResidualDancer, TrainingEnvironment
import yaml
from preprocessing.dataset import DanceDataModule
from wakepy import keepawake

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
        cb.RichProgressBar()
    ]
    trainer = pl.Trainer(
        callbacks=callbacks, 
        **config["trainer"]
        )
    trainer.fit(train_env, datamodule=data)
    trainer.test(train_env, datamodule=data)



if __name__ == "__main__":
    config = get_config("models/config/train.yaml")
    with keepawake():
        train_model(config)