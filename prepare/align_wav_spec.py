import numpy as np
import torch
import torch.utils.data

from mel_processing import spectrogram_torch
from utils import load_wav_to_torch
import scipy.io.wavfile as sciwav


class Align:
    def __init__(
        self, max_wav_value, sampling_rate, filter_length, hop_length, win_length
    ):
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

    def align_wav_spec(self, filename, phone_dur):
        phone_dur = np.int32(phone_dur)
        phone_dur = torch.Tensor(phone_dur).to(torch.int32)
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")

        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        # align mel and wave
        phone_dur_sum = torch.sum(phone_dur).item()
        spec_length = spec.shape[2]

        if spec_length > phone_dur_sum:
            spec = spec[:, :, :phone_dur_sum]
        elif spec_length < phone_dur_sum:
            pad_length = phone_dur_sum - spec_length
            spec = torch.nn.functional.pad(
                input=spec, pad=(0, pad_length, 0, 0), mode="constant", value=0
            )
        assert spec.shape[2] == phone_dur_sum

        # align wav
        fixed_wav_len = phone_dur_sum * self.hop_length
        if audio_norm.shape[1] > fixed_wav_len:
            audio_norm = audio_norm[:, :fixed_wav_len]
        elif audio_norm.shape[1] < fixed_wav_len:
            pad_length = fixed_wav_len - audio_norm.shape[1]
            audio_norm = torch.nn.functional.pad(
                input=audio_norm,
                pad=(0, pad_length, 0, 0),
                mode="constant",
                value=0,
            )
        assert audio_norm.shape[1] == fixed_wav_len

        # rewrite aligned wav
        audio = (
            (audio_norm * self.max_wav_value).transpose(0, 1).numpy().astype(np.int16)
        )

        sciwav.write(
            filename,
            self.sampling_rate,
            audio,
        )
        # save spec
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filename)
        return spec.shape[1]
