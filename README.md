# Init
Unofficial Implement of VISinger

# Reference Repos
https://github.com/jaywalnut310/vits

https://github.com/MoonInTheRiver/DiffSinger

https://wenet.org.cn/opencpop/

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

# Inference

```bash
./infer.sh
```

![LOSS值](/resource/vising_loss.png)
![MEL谱](/resource/vising_mel.png)

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/vising_sample.wav">
</audio>

# 样例音频

[vits_singing_样例.wav](/resource/vising_sample.wav)

# AI修复
https://github.com/brentspell/hifi-gan-bwe
