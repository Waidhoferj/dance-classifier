import torchaudio
from preprocessing.preprocess import AudioPipeline
from dancer_net.dancer_net import ShortChunkCNN
import torch
import numpy as np
import os
import json

if __name__ == "__main__":

    audio_file = "data/samples/mzm.iqskzxzx.aac.p.m4a.wav"
    seconds = 6
    model_path = "logs/20221226-230930"
    weights = os.path.join(model_path, "dancer_net.pt")
    config_path = os.path.join(model_path, "config.json")
    device = "mps"
    threshold = 0.5

    with open(config_path) as f:
        config = json.load(f)
    labels = np.array(sorted(config["classes"]))

    audio_pipeline = AudioPipeline()
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform[:, :seconds * sample_rate]
    spectrogram = audio_pipeline(waveform)
    spectrogram = spectrogram.unsqueeze(0).to(device)

    model = ShortChunkCNN(n_class=len(labels))
    model.load_state_dict(torch.load(weights))
    model = model.to(device).eval()

    with torch.no_grad():
        results = model(spectrogram)
    results = results.squeeze(0).detach().cpu().numpy()
    results = results > threshold
    results = labels[results]
    print(results)




    



