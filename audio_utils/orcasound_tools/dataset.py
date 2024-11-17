import torchaudio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

class OrcasoundDataset(Dataset):
    def __init__(self, accelerator, metadata_file, audio_conf, resampling_freq, audio_length, use_fbank=False, fbank_dir=None, roll_mag_aug=False,
                 mode='train'):
        self.metadata_file = metadata_file
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir
        self.audio_conf = audio_conf
        self.roll_mag_aug = roll_mag_aug
        self.mode = mode
        self.melbins = 128
        self.resampling_freq = resampling_freq
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        self.audio_length = audio_length
        self.mixup = self.audio_conf.get('mixup')
        self.norm_mean = self.metadata_file['dataset_stats']['mean']
        self.norm_std = self.metadata_file['dataset_stats']['std']
        self.noise = self.audio_conf.get('noise')
        if self.audio_length < 4:
            self.target_length = 256
        elif 4 <= self.audio_length < 7:
            self.target_length = 512
        else:
            self.target_length = 1024
        if accelerator.is_main_process:
            print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
            print(f'multilabel: {self.audio_conf.get("multilabel", False)}')
            print(f'using following mask: {self.freqm} freq, {self.timem} time')
            print(f'using mix-up with rate {self.mixup}')
            print(f'Dataset: Orcasound, mean {self.norm_mean:.3f} and std {self.norm_std:.3f}')
            print('now use noise augmentation' if self.noise else 'no noise augmentation')
            print(f'size of dataset {self.__len__()}')

    def _roll_mag_aug(self, waveform):
        waveform = waveform.numpy()
        idx = np.random.randint(len(waveform))
        rolled_waveform = np.roll(waveform, idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform * mag)

    def _wav2fbank(self, file_dict, file_dict2=None):
        if file_dict2 is None:
            file_path = file_dict['file_path']
            start_idx = file_dict['start_time']
            waveform, sr = torchaudio.load(file_path, backend="ffmpeg")
            start_sample = int(start_idx * sr)
            end_sample = int((start_idx+ self.audio_length) * sr)
            waveform = waveform[:, start_sample:end_sample]
            resampler = T.Resample(sr, self.resampling_freq)
            waveform = resampler(waveform)
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        else:
            file_path1 = file_dict['file_path']
            file_path2 = file_dict['file_path']
            waveform1, sr = torchaudio.load(file_path1)
            waveform2, _ = torchaudio.load(file_path2)
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()
            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[0] != waveform2.shape[0]:
                if waveform1.shape[0] > waveform2.shape[0]:
                    # If waveform1 has more channels, duplicate waveform2's channel
                    temp_wav = torch.zeros((waveform1.shape[0], waveform2.shape[1]))
                    # Repeat waveform2 across the first dimension to match the number of channels in waveform1
                    temp_wav[:waveform2.shape[0], :] = waveform2
                    waveform2 = temp_wav
                else:
                    # If waveform2 has more channels, truncate waveform2 to match the number of channels in waveform1
                    waveform2 = waveform2[:waveform1.shape[0], :waveform1.shape[1]]

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # Create a zero tensor with the same shape as waveform1
                    temp_wav = torch.zeros_like(waveform1)

                    temp_wav[0, :waveform2.shape[1]] = waveform2[0, :waveform2.shape[1]]
                    waveform2 = temp_wav

                else:
                    # Truncate waveform2 to match waveform1's length
                    waveform2 = waveform2[:, :waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.resampling_freq, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                  frame_shift=10)

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
        if file_dict2 is None:
            return fbank, 0
        else:
            return fbank, mix_lambda




    def __getitem__(self, index):

        file_dict = self.metadata_file['metadata'][self.mode][index]

        if not self.use_fbank:
            fbank, mix_lambda = self._wav2fbank(file_dict)
        else:
            fbank, mix_lambda = self._fbank(file_dict)

        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0, 1).unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank.squeeze(), 0, 1)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return fbank.unsqueeze(0), file_dict


    def __len__(self):
        return len(self.metadata_file['metadata'][self.mode])
