from utils import set_path

set_path()
from preprocessing.dataset import BestBallroomDataset
from preprocessing.pipelines import SpectrogramTrainingPipeline


def test_spectrogram_training_pipeline():
    ds = BestBallroomDataset()
    pipeline = SpectrogramTrainingPipeline()
    waveform, _ = ds[0]
    out = pipeline(waveform)
    assert len(out.shape) == 3
