# Init
Unofficial Implement of VISinger

# Reference Repos
https://github.com/jaywalnut310/vits

https://github.com/MoonInTheRiver/DiffSinger

https://wenet.org.cn/opencpop/

https://github.com/PlayVoice/VI-SVS

# Data Preprocess
```bash
export PYTHONPATH=.
```

Generate ../VISinger_data/label_vits_phn/XXX._label.npy|XXX._label_dur.npy|XXX_score.npy|XXX_score_dur.npy|XXX_pitch.npy|XXX_slurs.npy

```bash
python prepare/data_vits_phn.py
```

Generate filelists/vits_file.txt
Format: wave path|label path|label duration path|score path|score duration path|pitch path|slurs path;

```bash
python prepare/preprocess.py
```

# VISinger training

```bash
python train.py -c configs/singing_base.json -m singing_base
```

or

```bash
./train.sh
```

# Inference

```bash
./evaluate_score.sh
```

![LOSS](/resource/vising_loss.png)
![MEL](/resource/vising_mel.png)

# Samples

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000151.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000152.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000186.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000187.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000268.wav">
</audio>





