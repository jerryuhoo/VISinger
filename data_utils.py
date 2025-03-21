import os
import numpy as np
import torch
import torch.utils.data

from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
import scipy.io.wavfile as sciwav


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_and_text_new = []
        lengths = []

        for (
            audiopath,
            text,
            text_dur,
            score,
            score_dur,
            pitch,
            slur,
        ) in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append(
                    [audiopath, text, text_dur, score, score_dur, pitch, slur]
                )
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        phone_dur = audiopath_and_text[2]
        score = audiopath_and_text[3]
        score_dur = audiopath_and_text[4]
        pitch = audiopath_and_text[5]
        slurs = audiopath_and_text[6]

        phone, phone_dur, score, score_dur, pitch, slurs = self.get_labels(
            phone, phone_dur, score, score_dur, pitch, slurs
        )
        spec, wav = self.get_audio(file, phone_dur)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]

        if len_phone != len_spec:
            # print("**************CareFull*******************")
            # print(f"filepath={audiopath_and_text[0]}")
            # print(f"len_text={len_phone}")
            # print(f"len_spec={len_spec}")
            if len_phone > len_spec:
                print(file)
                print("len_phone", len_phone)
                print("len_spec", len_spec)
            assert len_phone < len_spec
            # len_min = min(len_phone, len_spec)
            # amor hop_size=256
            len_wav = len_spec * self.hop_length
            # print(wav.size())
            # print(f"len_min={len_min}")
            # print(f"len_wav={len_wav}")
            # spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
        return (phone, phone_dur, score, score_dur, pitch, slurs, spec, wav)

    def get_labels(self, phone, phone_dur, score, score_dur, pitch, slurs):
        phone = np.load(phone)
        phone_dur = np.load(phone_dur)
        score = np.load(score)
        score_dur = np.load(score_dur)
        pitch = np.load(pitch)
        slurs = np.load(slurs)
        phone = torch.LongTensor(phone)
        phone_dur = torch.LongTensor(phone_dur)
        score = torch.LongTensor(score)
        score_dur = torch.LongTensor(score_dur)
        pitch = torch.FloatTensor(pitch)
        slurs = torch.LongTensor(slurs)
        return phone, phone_dur, score, score_dur, pitch, slurs

    def get_audio(self, filename, phone_dur):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            print("please run data_vits_phn.py first")
            assert FileExistsError
        # else:
        #     spec = spectrogram_torch(
        #         audio_norm,
        #         self.filter_length,
        #         self.sampling_rate,
        #         self.hop_length,
        #         self.win_length,
        #         center=False,
        #     )
        #     # align mel and wave
        #     phone_dur_sum = torch.sum(phone_dur).item()
        #     spec_length = spec.shape[2]

        #     if spec_length > phone_dur_sum:
        #         spec = spec[:, :, :phone_dur_sum]
        #     elif spec_length < phone_dur_sum:
        #         pad_length = phone_dur_sum - spec_length
        #         spec = torch.nn.functional.pad(
        #             input=spec, pad=(0, pad_length, 0, 0), mode="constant", value=0
        #         )
        #     assert spec.shape[2] == phone_dur_sum

        #     # align wav
        #     fixed_wav_len = phone_dur_sum * self.hop_length
        #     if audio_norm.shape[1] > fixed_wav_len:
        #         audio_norm = audio_norm[:, :fixed_wav_len]
        #     elif audio_norm.shape[1] < fixed_wav_len:
        #         pad_length = fixed_wav_len - audio_norm.shape[1]
        #         audio_norm = torch.nn.functional.pad(
        #             input=audio_norm,
        #             pad=(0, pad_length, 0, 0),
        #             mode="constant",
        #             value=0,
        #         )
        #     assert audio_norm.shape[1] == fixed_wav_len

        #     # rewrite aligned wav
        #     audio = (audio_norm * self.max_wav_value).transpose(0, 1).numpy().astype(np.int16)

        #     sciwav.write(
        #         filename,
        #         self.sampling_rate,
        #         audio,
        #     )
        #     # save spec
        #     spec = torch.squeeze(spec, 0)
        #     torch.save(spec, spec_filename)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[6].size(1) for x in batch]), dim=0, descending=True
        )

        max_phone_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[6].size(1) for x in batch])
        max_wave_len = max([x[7].size(1) for x in batch])

        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.LongTensor(len(batch), max_phone_len)
        phone_dur_padded = torch.LongTensor(len(batch), max_phone_len)
        score_padded = torch.LongTensor(len(batch), max_phone_len)
        score_dur_padded = torch.LongTensor(len(batch), max_phone_len)
        pitch_padded = torch.FloatTensor(len(batch), max_spec_len)
        slurs_padded = torch.LongTensor(len(batch), max_phone_len)
        phone_padded.zero_()
        phone_dur_padded.zero_()
        score_padded.zero_()
        score_dur_padded.zero_()
        pitch_padded.zero_()
        slurs_padded.zero_()

        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][6].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            phone = row[0]
            phone_padded[i, : phone.size(0)] = phone
            phone_lengths[i] = phone.size(0)

            phone_dur = row[1]
            phone_dur_padded[i, : phone_dur.size(0)] = phone_dur

            score = row[2]
            score_padded[i, : score.size(0)] = score

            score_dur = row[3]
            score_dur_padded[i, : score_dur.size(0)] = score_dur

            pitch = row[4]
            pitch_padded[i, : pitch.size(0)] = pitch

            slurs = row[5]
            slurs_padded[i, : slurs.size(0)] = slurs

            spec = row[6]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[7]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

        return (
            phone_padded,
            phone_lengths,
            phone_dur_padded,
            score_padded,
            score_dur_padded,
            pitch_padded,
            slurs_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
