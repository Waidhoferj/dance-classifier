from utils import set_path
import pytest

set_path()
from preprocessing.dataset import PipelinedDataset, BestBallroomDataset, SongDataset
import numpy as np


def test_preprocess_dataset():
    dataset = BestBallroomDataset()
    dataset = PipelinedDataset(dataset, lambda x: x * 0.0)
    assert isinstance(dataset._data.song_dataset, SongDataset)
    assert hasattr(dataset, "feature_extractor")
    features, _ = dataset[0]
    assert np.unique(features.numpy())[0] == 0.0
    with pytest.raises(AttributeError):
        dataset.foo
