def dur_to_frame(ds, fs, hop_size):
    frames = [int(i * fs / hop_size + 0.5) for i in ds]
    return frames
