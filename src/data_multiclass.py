import json
import os
from random import sample

from os.path import join
from typing import Dict, List

CLASS_SIZE = 500


def prepare_class(name: str, _type: str, base_read_path: str, base_write_path: str, index: int):
    embeddings = []
    for file_name in os.listdir(join(base_read_path, _type)):
        if name not in file_name:
            continue
        with open(os.path.join(base_read_path, _type, file_name)) as f_read:
            for line in f_read.readlines():
                data = json.loads(line)
                data["name"] = name
                if data["label"] == 1:
                    data["label"] = index + 1
                    embeddings.append(data)
    embeddings = sample(embeddings, k=min(CLASS_SIZE, len(embeddings)))
    negatives = sample_negatives(_type, index)
    embeddings.extend(negatives)
    embeddings = sample(embeddings, k=len(embeddings))
    write_filename = "%s_%d.jsonl" % (_type, index)
    with open(os.path.join(base_write_path, _type, write_filename), "w") as f_write:
        for embedding in sample(embeddings, k=len(embeddings)):
            f_write.write(json.dumps(embedding) + "\n")


def sample_negatives(_type: str, index: int) -> List[Dict]:
    if index >= 1:
        return []
    path = "../t5p_small_embeddings/%s/%s_%d.jsonl" % (_type, _type, index)
    embeddings = []
    with open(path) as f:
        for line in f.readlines():
            data = json.loads(line)
            if data["label"] == 0:
                data["name"] = "nothing"
                data["label"] = 0
                embeddings.append(data)
    return sample(embeddings, k=min(CLASS_SIZE, len(embeddings)))


def main():
    classes = ["dos", "+info", "bypass", "priv", "other"]
    for i, c in enumerate(classes):
        prepare_class(c, "train", "../multiclass_embeddings/", "../t5p_small_multiclass_embeddings/", i)
        prepare_class(c, "test", "../multiclass_embeddings/", "../t5p_small_multiclass_embeddings/", i)


if __name__ == '__main__':
    main()
