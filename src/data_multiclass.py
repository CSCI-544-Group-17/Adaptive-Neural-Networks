"""import json
import os
from os.path import join
from random import sample

CLASS_SIZE = 500


def prepare_class(name: str, _type: str, base_read_path: str, base_write_path: str, index: int):
    positives = []
    negatives = []
    for file_name in os.listdir(join(base_read_path, _type)):
        if name not in file_name:
            continue
        with open(os.path.join(base_read_path, _type, file_name)) as f_read:
            for line in f_read.readlines():
                data = json.loads(line)
                data["name"] = name
                if data["label"] == 1:
                    data["label"] = index + 1
                    positives.append(data)
                else:
                    data["label"] = 0
                    negatives.append(data)
    positives = sample(positives, k=min(CLASS_SIZE, len(positives)))
    negatives = sample(negatives, k=min(CLASS_SIZE, len(negatives)))
    embeddings = positives + negatives
    embeddings = sample(embeddings, k=len(embeddings))
    write_filename = "%s_%d.jsonl" % (_type, index)
    with open(os.path.join(base_write_path, _type, write_filename), "w") as f_write:
        for embedding in sample(embeddings, k=len(embeddings)):
            f_write.write(json.dumps(embedding) + "\n")


def main():
    classes = ["dos", "+info", "bypass", "priv", "other"]
    for i, c in enumerate(classes):
        prepare_class(c, "train", "../multiclass_embeddings/", "../t5p_small_multiclass_multilabel_embeddings/", i)
        prepare_class(c, "test", "../multiclass_embeddings/", "../t5p_small_multiclass_multilabel_embeddings/", i)


if __name__ == '__main__':
    main()
"""


import json
import os
from os.path import join
from random import sample

CLASS_SIZE = 500


def prepare_class(name: str, _type: str, base_read_path: str, base_write_path: str, index: int):
    positives = []
    negatives = []
    for file_name in os.listdir(join(base_read_path, _type)):
        if name not in file_name:
            continue
        with open(os.path.join(base_read_path, _type, file_name)) as f_read:
            for line in f_read.readlines():
                data = json.loads(line)
                data["name"] = name
                if data["label"] == 1:
                    data["label"] = index + 1
                    positives.append(data)
                else:
                    data["label"] = 0
                    negatives.append(data)
    positives = sample(positives, k=min(CLASS_SIZE, len(positives)))
    negatives = sample(negatives, k=min(CLASS_SIZE, len(negatives)))
    embeddings = positives + negatives
    embeddings = sample(embeddings, k=len(embeddings))
    write_filename = "%s_%d.jsonl" % (_type, index)
    with open(os.path.join(base_write_path, _type, write_filename), "w") as f_write:
        for embedding in sample(embeddings, k=len(embeddings)):
            f_write.write(json.dumps(embedding) + "\n")


def main():
    classes = ["dos", "+info", "bypass", "priv", "other"]
    for i, c in enumerate(classes):
        prepare_class(c, "train", "../multiclass_embeddings/", "../t5p_small_multiclass_multilabel_embeddings/", i)
        prepare_class(c, "test", "../multiclass_embeddings/", "../t5p_small_multiclass_multilabel_embeddings/", i)


if __name__ == '__main__':
    main()