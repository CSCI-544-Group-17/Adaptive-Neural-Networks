import json
import os
from random import sample

from os.path import join

SIZE = 6


def prepare_class(name: str, _type: str, base_read_path: str, base_write_path: str, index: int):
    first = index == 0
    for file_name in os.listdir(join(base_read_path, _type)):
        if name not in file_name:
            continue
        with open(os.path.join(base_read_path, _type, file_name)) as f_read:
            embeddings = []
            for line in f_read.readlines():
                data = json.loads(line)
                data["name"] = name
                if data["label"] == 1:
                    data["label"] = [0] * SIZE
                    data["label"][index + 1] = 1
                else:
                    data["label"] = [0] * SIZE
                    data["label"][0] = 1
                embeddings.append(data)
            write_filename = "%s.jsonl" % _type if first else "%s_%d.jsonl" % (_type, index - 1)
            with open(os.path.join(base_write_path, _type, write_filename), "w") as f_write:
                for embedding in sample(embeddings, k=len(embeddings)):
                    f_write.write(json.dumps(embedding) + "\n")


def main():
    classes = ["dos", "+info", "bypass", "priv", "other"]
    for i, c in enumerate(classes):
        prepare_class(c, "train", "../multiclass_embeddings/", "../t5p_small_multiclass_embeddings/", i)
        prepare_class(c, "test", "../multiclass_embeddings/", "../t5p_small_multiclass_embeddings/", i)


if __name__ == '__main__':
    main()
