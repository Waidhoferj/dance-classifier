import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


def url_to_filename(url: str) -> str:
    return f"{url.split('/')[-1]}.wav"


def play_audio(waveform: np.ndarray, sample_rate: int):
    """
    Assumes that waveform is a numpy array normalized between -1 and 1.
    """
    if waveform.max() > 1.0 or waveform.min() < -1.0:
        raise ValueError("waveform must be a numpy array normalized between -1 and 1.")
    sd.play(waveform, sample_rate)
    sd.wait()


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    """
    Assumes that the spectrogram is in decibels.
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(spec, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    return fig
