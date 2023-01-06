import torchaudio
import numpy as np
from audio_utils import play_audio
from preprocessing.dataset import SongDataset

def test_audio_splitting():
    
    

    audio_paths = ["data/samples/95f2df65f7450db3b1af29aa77ba7edc6ab52075?cid=7ffadeb2e136495fb5a62d1ac9be8f62.wav"]
    labels = [np.array([1,0,1,0])]
    whole_song, sr = torchaudio.load("data/samples/95f2df65f7450db3b1af29aa77ba7edc6ab52075?cid=7ffadeb2e136495fb5a62d1ac9be8f62.wav")

    ds = SongDataset(audio_paths, labels)
    song_parts = (ds._waveform_from_index(i) for i in range(len(ds)))
    print("Sample Parts")
    for part in song_parts:
        play_audio(part,sr)


    print("Whole Sample")
    play_audio(whole_song,sr)