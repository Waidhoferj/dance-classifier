import datetime
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import random_split, SubsetRandomSampler
import json
from sklearn.model_selection import KFold

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from preprocessing.dataset import SongDataset
from preprocessing.preprocess import get_examples
from models.residual import ResidualDancer

DEVICE = "mps"
SEED = 42
TARGET_CLASSES = ['ATN',
        'BBA',
        'BCH',
        'BLU',
        'CHA',
        'CMB',
        'CSG',
        'ECS',
        'HST',
        'JIV',
        'LHP',
        'QST',
        'RMB',
        'SFT',
        'SLS',
        'SMB',
        'SWZ',
        'TGO',
        'VWZ',
        'WCS']

def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

class EarlyStopping:
    def __init__(self, patience=0):
        self.patience = patience
        self.last_measure = np.inf
        self.consecutive_increase = 0
    
    def step(self, val) -> bool:
        if self.last_measure <= val:
            self.consecutive_increase +=1
        else:
            self.consecutive_increase = 0
        self.last_measure = val

        return self.patience < self.consecutive_increase



def calculate_metrics(pred, target, threshold=0.5, prefix=""):
    target = target.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = np.array(pred > threshold, dtype=float)
    metrics= {
            'precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'accuracy': accuracy_score(y_true=target, y_pred=pred),
            }
    if prefix != "":
        metrics = {prefix + k : v for k, v in metrics.items()}
    
    return metrics


def evaluate(model:nn.Module, data_loader:DataLoader, criterion, device="mps") -> pd.Series:
    val_metrics = []
    for features, labels in (prog_bar := tqdm(data_loader)):
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(features)
            loss = criterion(outputs, labels)
        batch_metrics = calculate_metrics(outputs, labels, prefix="val_")
        batch_metrics["val_loss"] = loss.item()
        prog_bar.set_description(f'Validation - Loss: {batch_metrics["val_loss"]:.2f}, Accuracy: {batch_metrics["val_accuracy"]:.2f}')
        val_metrics.append(batch_metrics)
    return pd.DataFrame(val_metrics).mean()

    

def train(
    model: nn.Module,
    data_loader: DataLoader,
    val_loader=None,
    epochs=3,
    lr=1e-3,
    device="mps"):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    early_stop = EarlyStopping(1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    steps_per_epoch=int(len(data_loader)),
                                                    epochs=epochs,
                                                    anneal_strategy='linear')
    metrics = []
    for epoch in range(1,epochs+1):
        train_metrics = []
        prog_bar = tqdm(data_loader)
        for features, labels in prog_bar:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_metrics = calculate_metrics(outputs, labels)
            batch_metrics["loss"] = loss.item()
            train_metrics.append(batch_metrics)
            prog_bar.set_description(f'Training - Epoch: {epoch}/{epochs}, Loss: {batch_metrics["loss"]:.2f}, Accuracy: {batch_metrics["accuracy"]:.2f}')
        train_metrics = pd.DataFrame(train_metrics).mean()
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion)
            if early_stop.step(val_metrics["val_f1"]):
                break
            epoch_metrics = pd.concat([train_metrics, val_metrics], axis=0)
        else:
            epoch_metrics = train_metrics
        metrics.append(dict(epoch_metrics))

    return model, metrics
        

def cross_validation(seed=42, batch_size=64, k=5, device="mps"):
    df = pd.read_csv("data/songs.csv")
    x,y = get_examples(df, "data/samples",class_list=TARGET_CLASSES)
    
    dataset = SongDataset(x,y)
    splits=KFold(n_splits=k,shuffle=True,random_state=seed)
    metrics = []
    for fold, (train_idx,val_idx) in enumerate(splits.split(x,y)):
        print(f"Fold {fold+1}")

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        n_classes = len(y[0])
        model = ResidualDancer(n_classes=n_classes).to(device)
        model, _ = train(model,train_loader, epochs=2, device=device)
        val_metrics = evaluate(model, test_loader, nn.BCELoss())
        metrics.append(val_metrics)
    metrics = pd.DataFrame(metrics)
    log_dir = os.path.join(
    "logs", get_timestamp()
    )
    os.makedirs(log_dir, exist_ok=True)
    
    metrics.to_csv(model.state_dict(), os.path.join(log_dir, "cross_val.csv"))
                                                                                                             


def train_model():

    df = pd.read_csv("data/songs.csv")
    x,y = get_examples(df, "data/samples",class_list=TARGET_CLASSES)
    dataset = SongDataset(x,y)
    train_count = int(len(dataset) * 0.9)
    datasets = random_split(dataset, [train_count, len(dataset) - train_count], torch.Generator().manual_seed(SEED))
    data_loaders = [DataLoader(data, batch_size=64, shuffle=True) for data in datasets]
    train_data, val_data = data_loaders
    example_spec, example_label = dataset[0]
    n_classes = len(example_label)
    model = ResidualDancer(n_classes=n_classes).to(DEVICE)
    model, metrics = train(model,train_data, val_data, epochs=3, device=DEVICE)
    
    log_dir = os.path.join(
    "logs", get_timestamp()
    )
    os.makedirs(log_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(log_dir, "residual_dancer.pt"))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(log_dir, "metrics.csv"))
    config = {
        "classes": TARGET_CLASSES
    }
    with open(os.path.join(log_dir, "config.json")) as f:
        json.dump(config, f)
    print("Training information saved!")

if __name__ == "__main__":
    cross_validation()