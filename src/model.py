import torch
from torch import nn

CLASSES = ['yes', 'no', 'up', 'down']
SAMPLE_RATE = 16000

class SpeechCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 64),
                                        nn.ReLU(), nn.Linear(64, len(CLASSES)))

    def forward(self, x):
        return self.classifier(self.features(x))

def preprocess(waveform, sample_rate):
    import torchaudio
    import torch.nn.functional as F
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    waveform = F.pad(waveform[:, :SAMPLE_RATE], (0, max(0, SAMPLE_RATE - waveform.shape[1])))
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=400,
                                               hop_length=160, n_mels=64)(waveform)
    return torch.log(mel + 1e-9)
