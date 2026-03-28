import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy.fftpack import dct

CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

def compute_mfcc(waveform, sample_rate=16000, n_mfcc=40, n_fft=480, hop_length=160, n_mels=40):
    emphasized = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    frame_length = n_fft
    num_frames = 1 + (len(emphasized) - frame_length) // hop_length
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        frames[i] = emphasized[i * hop_length: i * hop_length + frame_length]

    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfcc.T


class SpeechCommandsDataset(Dataset):
    def __init__(self, root='./data', subset='training'):
        self.root = os.path.join(root, 'SpeechCommands', 'speech_commands_v0.02')
        self.subset = subset
        self.samples = []

        if not os.path.exists(self.root):
            raise RuntimeError(f"Dataset not found at {self.root}. Please download manually.")

        val_list = set()
        test_list = set()

        val_file = os.path.join(self.root, 'validation_list.txt')
        test_file = os.path.join(self.root, 'testing_list.txt')

        if os.path.exists(val_file):
            with open(val_file) as f:
                val_list = set(line.strip() for line in f)
        if os.path.exists(test_file):
            with open(test_file) as f:
                test_list = set(line.strip() for line in f)

        for label_dir in os.listdir(self.root):
            label_path = os.path.join(self.root, label_dir)
            if not os.path.isdir(label_path) or label_dir.startswith('_'):
                continue

            if label_dir not in CLASSES:
                continue

            for audio_file in os.listdir(label_path):
                if not audio_file.endswith('.wav'):
                    continue

                rel_path = os.path.join(label_dir, audio_file)

                if subset == 'validation' and rel_path in val_list:
                    self.samples.append((os.path.join(label_path, audio_file), label_dir))
                elif subset == 'testing' and rel_path in test_list:
                    self.samples.append((os.path.join(label_path, audio_file), label_dir))
                elif subset == 'training' and rel_path not in val_list and rel_path not in test_list:
                    self.samples.append((os.path.join(label_path, audio_file), label_dir))



        print(f"Total {len(self.samples)} samples for {subset}")

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        try:
            sample_rate, waveform = wavfile.read(audio_path)
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32768.0
            elif waveform.dtype == np.int32:
                waveform = waveform.astype(np.float32) / 2147483648.0
            else:
                waveform = waveform.astype(np.float32)

            target_length = 16000
            if len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)))
            else:
                waveform = waveform[:target_length]

            mfcc = compute_mfcc(waveform, sample_rate=16000, n_mfcc=40)
            mfcc = torch.tensor(mfcc, dtype=torch.float32)

        except Exception as e:
            print(f"[WARNING] Error processing sample {idx}: {e}, using zeros")
            mfcc = torch.zeros(40, 98, dtype=torch.float32)

        label_idx = CLASSES.index(label)
        return mfcc, label_idx
        
    def __len__(self):
        return len(self.samples)
