import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import torchaudio as ta
from .pipelines import AudioTrainingPipeline
import pytorch_lightning as pl
from .preprocess import get_examples
from sklearn.model_selection import train_test_split
from torchaudio import transforms as taT
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



class SongDataset(Dataset):
    def __init__(self, 
    audio_paths: list[str], 
    dance_labels: list[np.ndarray],
    audio_duration=30, # seconds
    audio_window_duration=6, # seconds
    audio_window_jitter=0.0, # seconds
    audio_pipeline_kwargs={},
    resample_frequency=16000
    ):
        assert audio_duration % audio_window_duration == 0, "Audio window should divide duration evenly."
        assert audio_window_duration > audio_window_jitter, "Jitter should be a small fraction of the audio window duration."

        self.audio_paths = audio_paths
        self.dance_labels = dance_labels
        audio_info = ta.info(audio_paths[0])
        self.sample_rate = audio_info.sample_rate
        self.audio_window_duration = int(audio_window_duration)
        self.audio_window_jitter = audio_window_jitter
        self.audio_duration = int(audio_duration)

        self.audio_pipeline = AudioTrainingPipeline(self.sample_rate, resample_frequency, audio_window_duration, **audio_pipeline_kwargs)

    def __len__(self):
        return len(self.audio_paths) * self.audio_duration // self.audio_window_duration 

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self._waveform_from_index(idx)
        assert waveform.shape[1] > 10, f"No data found: {self._backtrace_audio_path(idx)}"
        spectrogram = self.audio_pipeline(waveform)

        dance_labels = self._label_from_index(idx)

        example_is_valid = self._validate_output(spectrogram, dance_labels)
        if example_is_valid:
            return spectrogram, dance_labels
        else:
            # Try the previous one
            # This happens when some of the audio recordings are really quiet
            # This WILL NOT leak into other data partitions because songs belong entirely to a partition
            return self[idx-1]

    def _convert_idx(self,idx:int) -> int:
        return idx * self.audio_window_duration // self.audio_duration

    def _backtrace_audio_path(self, index:int) -> str:
        return self.audio_paths[self._convert_idx(index)]

    def _validate_output(self,x,y):
        is_finite =  not torch.any(torch.isinf(x))
        is_numerical = not torch.any(torch.isnan(x))
        has_data = torch.any(x != 0.0)
        is_binary = len(torch.unique(y)) < 3
        return all((is_finite,is_numerical, has_data, is_binary))

    def _waveform_from_index(self, idx:int) -> torch.Tensor:
        audio_filepath = self.audio_paths[self._convert_idx(idx)]
        num_windows = self.audio_duration // self.audio_window_duration
        frame_index = idx % num_windows
        jitter_start = -self.audio_window_jitter if frame_index > 0 else 0.0
        jitter_end = self.audio_window_jitter if frame_index != num_windows - 1 else 0.0
        jitter = int(torch.FloatTensor(1).uniform_(jitter_start, jitter_end) * self.sample_rate)
        frame_offset = frame_index * self.audio_window_duration * self.sample_rate + jitter
        num_frames = self.sample_rate * self.audio_window_duration
        waveform, sample_rate = ta.load(audio_filepath, frame_offset=frame_offset, num_frames=num_frames)
        assert sample_rate == self.sample_rate, f"Expected sample rate of {self.sample_rate}. Found {sample_rate}"        
        return waveform


    def _label_from_index(self, idx:int) -> torch.Tensor:
        return torch.from_numpy(self.dance_labels[self._convert_idx(idx)])
    

class WaveformSongDataset(SongDataset):
    """
    Outputs raw waveforms of the data instead of a spectrogram.
    """

    def __init__(self, *args,resample_frequency=16000, **kwargs):
        super().__init__(*args, **kwargs)
        self.resample_frequency = resample_frequency
        self.resampler = taT.Resample(self.sample_rate, self.resample_frequency)
        self.pipeline = []

    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]:
        waveform = self._waveform_from_index(idx)
        assert waveform.shape[1] > 10, f"No data found: {self._backtrace_audio_path(idx)}"
        # resample the waveform
        waveform = self.resampler(waveform)
        
        waveform = waveform.mean(0)

        dance_labels = self._label_from_index(idx)
        return waveform, dance_labels
    
    


class HuggingFaceWaveformSongDataset(WaveformSongDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = []


    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]:
        x,y = super().__getitem__(idx)
        if len(self.pipeline) > 0:
            for fn in self.pipeline:
                x = fn(x)

        dance_labels = y.argmax()
        return {"input_values": x["input_values"][0] if hasattr(x, "input_values") else x, "label": dance_labels}

    def map(self,fn):
        """
        NOTE this mutates the original, doesn't return a copy like normal maps.
        """
        self.pipeline.append(fn)

class DanceDataModule(pl.LightningDataModule):
    def __init__(self, 
    song_data_path="data/songs_cleaned.csv",
    song_audio_path="data/samples",
    test_proportion=0.15,
    val_proportion=0.1,
    target_classes:list[str]=None,
    min_votes=1,
    batch_size:int=64,
    num_workers=10,
    dataset_cls = None,
    dataset_kwargs={}
    ):
        super().__init__()
        self.song_data_path = song_data_path
        self.song_audio_path = song_audio_path
        self.val_proportion=val_proportion
        self.test_proportion=test_proportion
        self.train_proportion= 1.-test_proportion-val_proportion
        self.target_classes=target_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        self.dataset_cls = dataset_cls if dataset_cls is not None else SongDataset

        df = pd.read_csv(song_data_path)
        self.x,self.y = get_examples(df, self.song_audio_path,class_list=self.target_classes, multi_label=True, min_votes=min_votes)

    def setup(self, stage: str):
        train_i, val_i, test_i = random_split(np.arange(len(self.x)), [self.train_proportion, self.val_proportion, self.test_proportion])
        self.train_ds = self._dataset_from_indices(train_i)
        self.val_ds = self._dataset_from_indices(val_i)
        self.test_ds = self._dataset_from_indices(test_i)
    
    def _dataset_from_indices(self, idx:list[int]) -> SongDataset:
        return self.dataset_cls(self.x[idx], self.y[idx], **self.dataset_kwargs)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_label_weights(self):
        n_examples, n_classes = self.y.shape
        return torch.from_numpy(n_examples / (n_classes * sum(self.y)))
    

class WaveformTrainingEnvironment(pl.LightningModule):

    def __init__(self, model: nn.Module, criterion: nn.Module, feature_extractor, config:dict, learning_rate=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.config=config
        self.feature_extractor=feature_extractor
        self.save_hyperparameters({
            "model": type(model).__name__,
            "loss": type(criterion).__name__,
            "config": config,
             **kwargs
            })

    def preprocess_inputs(self, x):
        device = x.device
        x = x.squeeze(1).cpu().numpy()
        x = self.feature_extractor(list(x),return_tensors='pt', sampling_rate=16000)
        return x["input_values"].to(device)
    
    def training_step(self, batch: tuple[torch.Tensor, torch.TensorType], batch_index: int) -> torch.Tensor:
        features, labels = batch
        features = self.preprocess_inputs(features)
        outputs = self.model(features).logits
        outputs = nn.Sigmoid()(outputs) # good for multi label classification, should be softmax otherwise
        loss = self.criterion(outputs, labels)
        metrics = calculate_metrics(outputs, labels, prefix="train/", multi_label=True)
        self.log_dict(metrics, prog_bar=True)
        return loss


    def validation_step(self, batch:tuple[torch.Tensor, torch.TensorType], batch_index:int):
        x,y = batch
        x = self.preprocess_inputs(x)
        preds = self.model(x).logits
        preds = nn.Sigmoid()(preds) 
        metrics = calculate_metrics(preds, y, prefix="val/", multi_label=True)
        metrics["val/loss"] = self.criterion(preds, y)
        self.log_dict(metrics,prog_bar=True)

    def test_step(self, batch:tuple[torch.Tensor, torch.TensorType], batch_index:int):
        x, y = batch
        x = self.preprocess_inputs(x)
        preds = self.model(x).logits
        preds = nn.Sigmoid()(preds) 
        self.log_dict(calculate_metrics(preds, y, prefix="test/", multi_label=True), prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') {"scheduler": scheduler, "monitor": "val/loss"}
        return [optimizer] 
    


def calculate_metrics(pred, target, threshold=0.5, prefix="", multi_label=True) -> dict[str, torch.Tensor]:
    target = target.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    params = {
            "y_true": target if multi_label else target.argmax(1) ,
            "y_pred": np.array(pred > threshold, dtype=float) if multi_label else pred.argmax(1), 
            "zero_division": 0,
            "average":"macro"
            }
    metrics= {
            'precision': precision_score(**params),
            'recall': recall_score(**params),
            'f1': f1_score(**params),
            'accuracy': accuracy_score(y_true=params["y_true"], y_pred=params["y_pred"]),
            }
    return {prefix + k: torch.tensor(v,dtype=torch.float32) for k,v in metrics.items()}