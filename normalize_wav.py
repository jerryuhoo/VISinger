from prepare.align_wav_spec import Align
import os
from tqdm import tqdm

align = Align(32768, 24000, 1024, 256, 1024)
output_path = "singing_gt"
input_path = "/home/yyu479/VISinger_data/wav_dump_24k"

files = os.listdir(path=input_path)
for i, wav_file in enumerate(tqdm(files)):
    suffix = os.path.splitext(os.path.split(wav_file)[-1])[1]
    if not suffix == ".wav":
        continue
    basename = os.path.splitext(os.path.split(wav_file)[-1])[0][:-7]
    align.normalize_wav(
        os.path.join(input_path, wav_file), os.path.join(output_path, wav_file)
    )
