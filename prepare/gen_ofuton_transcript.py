import music21 as m21
import os
from typing import Iterable, List, Optional, Union


def pyopenjtalk_g2p(text) -> List[str]:
    import pyopenjtalk

    # phones is a str object separated by space
    phones = pyopenjtalk.g2p(text, kana=False)
    phones = phones.split(" ")
    return phones


def text2tokens_svs(syllable: str) -> List[str]:
    customed_dic = {
        "へ": ["h", "e"],
        "ヴぁ": ["v", "a"],
        "ヴぃ": ["v", "i"],
        "ヴぇ": ["v", "e"],
        "ヴぉ": ["v", "i"],
        "でぇ": ["dy", "e"],
    }
    tokens = pyopenjtalk_g2p(syllable)
    if syllable in customed_dic:
        tokens = customed_dic[syllable]
    return tokens


def note_filter(note_name, note_map):
    note_name = note_name.replace("-", "")
    if "#" in note_name:
        note_name = note_name + "/" + note_map[note_name[0]] + "b" + note_name[2]
    return note_name


# eval(valid), dev(test), train
def process(base_path, file_path):

    note_map = {
        "A": "B",
        "B": "C",
        "C": "D",
        "D": "E",
        "E": "F",
        "F": "G",
        "G": "A",
    }

    label_path = file_path + "label"
    text_path = file_path + "text"
    data = []

    for line in open(label_path, "r"):
        # add phn and phn_dur
        str_list = line.replace("\n", "").split(" ")
        name = str_list[0]
        phn_dur = []
        phn = []
        score = []
        score_dur = []

        for i in range(1, len(str_list)):
            try:
                phn_dur_ = str(round(float(str_list[i + 1]) - float(str_list[i]), 6))
                # phn_dur_ = float(str_list[i + 1]) - float(str_list[i])
            except:
                if str_list[i] != "" and str_list[i].isalpha():
                    phn.append(str_list[i])
                    phn_dict.add(str_list[i])
                continue

            phn_dur.append(phn_dur_)

        # append text
        for line2 in open(text_path, "r"):
            str_list2 = line2.replace("\n", "").split(" ")
            if str_list2[0] != name:
                continue
            else:
                text_ = str_list2[1]
                break

        # add score and score_dur
        musicxmlscp = open(os.path.join(file_path, "xml.scp"), "r", encoding="utf-8")
        for xml_line in musicxmlscp:
            xmlline = xml_line.strip().split(" ")
            recording_id = xmlline[0]
            if recording_id != name:
                continue
            else:
                path = base_path + xmlline[1]
                parse_file = m21.converter.parse(path)
                part = parse_file.parts[0].flat
                m = parse_file.metronomeMarkBoundaries()
                tempo = m[0][2]
                for part in parse_file.parts:
                    for note in part.recurse().notes:
                        note_dur_ = note.quarterLength * 60 / tempo.number
                        note_name_ = note_filter(note.nameWithOctave, note_map)
                        note_text_ = note.lyric
                        # print("note_text1", text_)
                        # print("note_text_", note_text_)
                        if not note_text_:
                            continue
                        note_phn_ = text2tokens_svs(note_text_)
                        for i in range(len(note_phn_)):
                            score.append(note_name_)
                            score_dur.append(str(note_dur_))
                        # print("note_phn", note_phn_)
                break

        # print("tempo", tempo.number)

        # TODO: add slur. currently all 0
        slur = []
        for i in range(len(phn)):
            slur.append("0")

        # add one line
        data.append(
            name
            + "|"
            + text_
            + "|"
            + " ".join(phn)
            + "|"
            + " ".join(score)
            + "|"
            + " ".join(score_dur)
            + "|"
            + " ".join(phn_dur)
            + "|"
            + " ".join(slur)
        )
        print(data)
        assert len(phn) == len(phn_dur)
        assert len(phn) == len(score)
        assert len(phn) == len(score_dur)
        assert len(phn) == len(slur)
    return data


base_path = "/home/yyu479/espnet/egs2/ofuton_p_utagoe_db/svs1/"

data = []
phn_dict = set()
data_eval = process(base_path, base_path + "dump/raw/eval/")
data_dev = process(base_path, base_path + "dump/raw/org/dev/")
data_tr_no_dev = process(base_path, base_path + "dump/raw/org/tr_no_dev/")
data = data_eval + data_dev + data_tr_no_dev

with open("transcriptions.txt", "w") as f:
    for i in data:
        f.writelines(i)
        f.write("\n")

phn_dict_sort = list(phn_dict)
phn_dict_sort.sort()
with open("dict.txt", "w") as f:
    for i in phn_dict_sort:
        f.writelines(i)
        f.write("\n")
