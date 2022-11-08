import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

from prepare.data_vits_phn import FeatureInput, SingInput

# setting
hop_length = 256
sample_rate = 24000
wav_name = "2001000001"
input_path = "singing_gt/2001000001_bits16.wav"

# get mel
y, sr = librosa.load(input_path, sr=sample_rate)
librosa.feature.melspectrogram(y=y, sr=sr)
D = librosa.stft(y, hop_length=hop_length)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# get f0
featureInput = FeatureInput("singing_gt/", sr, hop_length)
featur_pit = featureInput.compute_f0("2001000001_bits16.wav")

fo = open("../VISinger_data/transcriptions.txt", "r+")
# load text info

while True:
    try:
        message = fo.readline().strip()
    except Exception as e:
        print("nothing of except:", e)
    if message == None:
        break
    if message == "":
        break
    if wav_name in message:
        break
print(message)

infos = message.split("|")
file = infos[0]
hanz = infos[1]
phon = infos[2].split(" ")
note = infos[3].split(" ")
note_dur = infos[4].split(" ")
phon_dur = infos[5].split(" ")
phon_slur = infos[6].split(" ")


singInput = SingInput(sample_rate, hop_length)

(
    file,
    labels_ids,
    labels_dur,
    scores_ids,
    scores_dur,
    labels_slr,
    labels_uvs,
) = singInput.parseInput(message)
labels_uvs = np.repeat(labels_uvs, labels_dur, axis=0)
featur_pit = featur_pit[: len(labels_uvs)]
featur_pit_uv = featur_pit * labels_uvs

uv = featur_pit == 0
print("uv", uv)
print("bbb", np.where(uv)[0])
print("ccc", np.where(~uv)[0])
featur_pit_intp = np.copy(featur_pit)
featur_pit_intp[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], featur_pit[~uv])
# plot
# plt.figure()
fig = plt.figure(figsize=(15, 6))

librosa.display.specshow(
    S_db, y_axis="log", sr=sr, hop_length=hop_length, x_axis="frames"
)

(F0_ori,) = plt.plot(featur_pit.T, "r", label="F0_ori", alpha=0.9)
(F0_uv,) = plt.plot(featur_pit_uv.T, "y", label="F0_uv", alpha=0.9)
(F0_intp,) = plt.plot(featur_pit_intp.T, "b", label="F0_intp", alpha=0.9)
plt.legend([F0_ori, F0_uv, F0_intp], ["F0_ori", "F0_uv", "F0_intp"], loc="upper right")
plt.colorbar(format="%+2.0f dB")
plt.savefig("f0.png")
