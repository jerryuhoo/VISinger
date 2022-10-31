import argparse
import os


def process_utterance(
    audio_dir,
    wav_dumpdir,
    segment,
    tgt_sr=24000,
):
    uid, lyrics, phns, notes, syb_dur, phn_dur, keep = segment.strip().split("|")
    cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}_bits16.wav".format(
        os.path.join(audio_dir, uid),
        tgt_sr,
        os.path.join(wav_dumpdir, uid),
    )
    print("uid", uid)
    os.system(cmd)


def process_subset(args, set_name):
    with open(
        os.path.join(args.src_data, "segments", set_name + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        segments = f.read().strip().split("\n")
        for segment in segments:
            process_utterance(
                os.path.join(args.src_data, "segments", "wavs"),
                args.wav_dumpdir,
                segment,
                tgt_sr=args.sr,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Opencpop Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    for name in ["train", "test"]:
        process_subset(args, name)
