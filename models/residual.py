import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import numpy as np
import torchaudio
import yaml
from .utils import calculate_metrics
from preprocessing.pipelines import WaveformPreprocessing, AudioToSpectrogram

# Architecture based on: https://github.com/minzwon/sota-music-tagging-models/blob/36aa13b7205ff156cf4dcab60fd69957da453151/training/model.py

class ResidualDancer(nn.Module):
    def __init__(self,n_channels=128, n_classes=50):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Spectrogram
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.res_layers = nn.Sequential(
            ResBlock(1, n_channels, stride=2),
            ResBlock(n_channels, n_channels, stride=2),
            ResBlock(n_channels, n_channels*2, stride=2),
            ResBlock(n_channels*2, n_channels*2, stride=2),
            ResBlock(n_channels*2, n_channels*2, stride=2),
            ResBlock(n_channels*2, n_channels*2, stride=2),
            ResBlock(n_channels*2, n_channels*4, stride=2)
        )

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.spec_bn(x)

        # CNN
        x = self.res_layers(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x
        

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super().__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out

class TrainingEnvironment(pl.LightningModule):

    def __init__(self, model: nn.Module, criterion: nn.Module, config:dict, learning_rate=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.config=config
        self.save_hyperparameters({
            "model": type(model).__name__,
            "loss": type(criterion).__name__,
            "config": config,
             **kwargs
            })

    def training_step(self, batch: tuple[torch.Tensor, torch.TensorType], batch_index: int) -> torch.Tensor:
        features, labels = batch
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        metrics = calculate_metrics(outputs, labels, prefix="train/", multi_label=True)
        self.log_dict(metrics, prog_bar=True)
        # Log spectrograms
        if batch_index % 100 == 0:
            tensorboard = self.logger.experiment
            img_index = torch.randint(0, len(features), (1,)).item()
            img = features[img_index][0]
            img = (img - img.min()) / (img.max() - img.min())
            tensorboard.add_image(f"batch: {batch_index}, element: {img_index}", img, 0, dataformats='HW')
        return loss


    def validation_step(self, batch:tuple[torch.Tensor, torch.TensorType], batch_index:int):
        x, y = batch
        preds = self.model(x)
        metrics = calculate_metrics(preds, y, prefix="val/", multi_label=True)
        metrics["val/loss"] = self.criterion(preds, y)
        self.log_dict(metrics,prog_bar=True)

    def test_step(self, batch:tuple[torch.Tensor, torch.TensorType], batch_index:int):
        x, y = batch
        preds = self.model(x)
        self.log_dict(calculate_metrics(preds, y, prefix="test/", multi_label=True), prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') {"scheduler": scheduler, "monitor": "val/loss"}
        return [optimizer] 
    
        

class DancePredictor:
    def __init__(
        self, 
        weight_path:str,
        labels:list[str],
        expected_duration=6, 
        threshold=0.5,
        resample_frequency=16000,
        device="cpu"):

        super().__init__()
        
        self.expected_duration = expected_duration
        self.threshold = threshold
        self.resample_frequency = resample_frequency
        self.preprocess_waveform = WaveformPreprocessing(resample_frequency * expected_duration)
        self.audio_to_spectrogram = AudioToSpectrogram(resample_frequency)
        self.labels = np.array(labels)
        self.device = device
        self.model = self.get_model(weight_path)

    
    def get_model(self, weight_path:str) -> nn.Module:
        weights = torch.load(weight_path, map_location=self.device)["state_dict"]
        model = ResidualDancer(n_classes=len(self.labels))
        for key in list(weights):
            weights[key.replace("model.", "")] = weights.pop(key)
        model.load_state_dict(weights)
        return model.to(self.device).eval()

    @classmethod
    def from_config(cls, config_path:str) -> "DancePredictor":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return DancePredictor(**config)

    @torch.no_grad()
    def __call__(self, waveform: np.ndarray, sample_rate:int) -> dict[str,float]:
        if len(waveform.shape) > 1 and waveform.shape[1] < waveform.shape[0]:
            waveform = waveform.transpose(1,0)
        elif len(waveform.shape) == 1:
            waveform = np.expand_dims(waveform, 0)
        waveform = torch.from_numpy(waveform.astype("int16"))
        waveform = torchaudio.functional.apply_codec(waveform,sample_rate, "wav", channels_first=True)

        waveform = torchaudio.functional.resample(waveform, sample_rate,self.resample_frequency)
        waveform = self.preprocess_waveform(waveform)
        spectrogram = self.audio_to_spectrogram(waveform)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)

        results = self.model(spectrogram)
        results = results.squeeze(0).detach().cpu().numpy()
        result_mask = results > self.threshold
        probs = results[result_mask]
        dances = self.labels[result_mask]
        
        return {dance:float(prob) for dance, prob in zip(dances, probs)}



