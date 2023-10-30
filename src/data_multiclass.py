import json
import os
from random import sample

SIZE = 6


def prepare_class(name: str, base_read_path: str, base_write_path: str, index: int):
    for file_name in os.listdir(base_read_path):
        if name not in file_name:
            continue
        with open(os.path.join(base_read_path, file_name)) as f_read:
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
            with open(os.path.join(base_write_path, "%d.jsonl" % index), "w") as f_write:
                for embedding in sample(embeddings, k=len(embeddings)):
                    f_write.write(json.dumps(embedding) + "\n")


def main():
    classes = ["dos", "+info", "bypass", "priv", "other"]
    for i, c in enumerate(classes):
        prepare_class(c, "../multiclass_embeddings/train", "../t5p_small_multiclass_embeddings/train", i)
        prepare_class(c, "../multiclass_embeddings/test", "../t5p_small_multiclass_embeddings/test", i)


if __name__ == '__main__':
    main()
