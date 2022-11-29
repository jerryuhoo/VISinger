import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from time import *

import torch
import utils
from models import SynthesizerTrn
from prepare.data_vits_phn_ofuton import SingInput
from prepare.data_vits_phn_ofuton import FeatureInput
from prepare.phone_map import get_vocab_size


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


use_cuda = True

# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/singing_base.json")

vocab_size = get_vocab_size()

net_g = SynthesizerTrn(
    vocab_size,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
)  # .cuda()

if use_cuda:
    net_g = net_g.cuda()

_ = utils.load_checkpoint("./logs/singing_base/G_40000.pth", net_g, None)
net_g.eval()
# net_g.remove_weight_norm()

singInput = SingInput(hps.data.sampling_rate, hps.data.hop_length)
featureInput = FeatureInput(
    "../VISinger_data/wav_dump_16k/", hps.data.sampling_rate, hps.data.hop_length
)

# check directory existence
if not os.path.exists("./singing_out"):
    os.makedirs("./singing_out")

fo = open("./vsinging_infer_jp.txt", "r+")
while True:
    try:
        message = fo.readline().strip()
    except Exception as e:
        print("nothing of except:", e)
        break
    if message == None:
        break
    if message == "":
        break
    print(message)
    (
        file,
        labels_ids,
        labels_frames,
        scores_ids,
        scores_dur,
        labels_slr,
        # labels_uvs,
    ) = singInput.parseInput(message)

    phone = torch.LongTensor(labels_ids)
    score = torch.LongTensor(scores_ids)
    score_dur = torch.LongTensor(scores_dur)
    slurs = torch.LongTensor(labels_slr)

    phone_lengths = phone.size()[0]

    begin_time = time()
    with torch.no_grad():
        if use_cuda:
            phone = phone.cuda().unsqueeze(0)
            score = score.cuda().unsqueeze(0)
            score_dur = score_dur.cuda().unsqueeze(0)
            slurs = slurs.cuda().unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths]).cuda()
        else:
            phone = phone.unsqueeze(0)
            score = score.unsqueeze(0)
            score_dur = score_dur.unsqueeze(0)
            slurs = slurs.unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths])
        audio = (
            net_g.infer(phone, phone_lengths, score, score_dur, slurs)[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
    end_time = time()
    run_time = end_time - begin_time
    print("Syth Time (Seconds):", run_time)
    data_len = len(audio) / hps.data.sampling_rate
    print("Wave Time (Seconds):", data_len)
    print("Real time Rate (%):", run_time / data_len)
    save_wav(audio, f"./singing_out/{file}.wav", hps.data.sampling_rate)
fo.close()
# can be deleted
os.system("chmod 777 ./singing_out -R")
