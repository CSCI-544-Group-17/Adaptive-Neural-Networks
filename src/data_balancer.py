import json
import os
from os.path import join
from random import sample
from typing import List

CLASS_SIZE = 500


class DataBalancer:
    _TYPE_TRAIN = "train"
    _TYPE_TEST = "test"

    def __init__(self, name, classes: List[str], base_read_path: str, base_write_path: str, size: int, nothing: str,
                 other: str):
        self._name = name
        self._classes = classes
        self._base_read_path = base_read_path
        self._base_write_path = base_write_path
        self._size = size
        self._nothing = nothing
        self._other = other

    def balance(self):
        raise NotImplementedError("'balance' not implemented")

    def _load_file(self, clazz: str, file_name: str, _type: str, value: int):
        embeddings = []
        with open(os.path.join(self._base_read_path, _type, file_name)) as f_read:
            for line in f_read.readlines():
                data = json.loads(line)
                data["class"] = clazz
                if data["label"] == 1:
                    data["label"] = value
                    embeddings.append(data)
        return embeddings


class PNNDataBalancer(DataBalancer):
    """
    Balances data among classes and puts it in a desired location. Different ways of sampling the nothing class and vulnerability
    classes can be defined
    """

    def __init__(self, classes: List[str], base_read_path: str, base_write_path: str, size: int, nothing: str,
                 other: str):
        super().__init__("PNN Data Balancer", classes, base_read_path, base_write_path, size, nothing, other)

    def balance(self):
        for i, clazz in enumerate(self._classes):
            self._balance_class(clazz, DataBalancer._TYPE_TRAIN)
            self._balance_class(clazz, DataBalancer._TYPE_TEST)

    def _balance_class(self, clazz: str, _type: str):
        if clazz == self._nothing:
            positives, negatives = self._sample_nothing(_type)
        else:
            positives, negatives = self._sample_class(clazz, _type)
        embeddings = positives + negatives
        embeddings = sample(embeddings, k=len(embeddings))
        write_filename = "%s.jsonl" % clazz
        with open(os.path.join(self._base_write_path, _type, write_filename), "w") as f_write:
            for embedding in sample(embeddings, k=len(embeddings)):
                f_write.write(json.dumps(embedding) + "\n")

    def _sample_nothing(self, _type):
        positives = []
        negatives = []
        for file_name in os.listdir(join(self._base_read_path, _type)):
            if self._nothing in file_name:
                with open(os.path.join(self._base_read_path, _type, file_name)) as f_read:
                    for line in f_read.readlines():
                        data = json.loads(line)
                        data["class"] = self._nothing
                        if data["label"] == 0:
                            data["label"] = 1
                            positives.append(data)
            elif self._other in file_name:
                with open(os.path.join(self._base_read_path, _type, file_name)) as f_read:
                    for line in f_read.readlines():
                        data = json.loads(line)
                        data["class"] = self._nothing
                        if data["label"] == 1:
                            data["label"] = 0
                            negatives.append(data)
        positives = sample(positives, k=min(self._size, len(positives)))
        negatives = sample(negatives, k=min(self._size, len(negatives)))
        return positives, negatives

    def _sample_class(self, clazz: str, _type: str):
        positives = []
        negatives = []
        for file_name in os.listdir(join(self._base_read_path, _type)):
            if clazz not in file_name:
                continue
            with open(os.path.join(self._base_read_path, _type, file_name)) as f_read:
                for line in f_read.readlines():
                    data = json.loads(line)
                    data["class"] = clazz
                    if data["label"] == 1:
                        data["label"] = 1
                        positives.append(data)
                    else:
                        data["label"] = 0
                        negatives.append(data)
        positives = sample(positives, k=min(self._size, len(positives)))
        negatives = sample(negatives, k=min(self._size, len(negatives)))
        return positives, negatives


class FNNDataBalancer(DataBalancer):
    def __init__(self, classes: List[str], base_read_path: str, base_write_path: str, size: int, nothing: str,
                 other: str):
        super().__init__("FNN Data Balancer", classes, base_read_path, base_write_path, size, nothing, other)

    def balance(self):
        for index, _ in enumerate(self._classes):
            self._balance_class(index, DataBalancer._TYPE_TRAIN)
            self._balance_class(index, DataBalancer._TYPE_TEST)

    def _balance_class(self, index: int, _type: str):
        if index == 0:
            embeddings = self.__sample_nothing(_type)
        else:
            embeddings = self.__sample_class(index, _type)
        embeddings = sample(embeddings, k=len(embeddings))
        write_filename = "%s_%d.jsonl" % (_type, index)
        with open(os.path.join(self._base_write_path, _type, write_filename), "w") as f_write:
            for embedding in sample(embeddings, k=len(embeddings)):
                f_write.write(json.dumps(embedding) + "\n")

    def __sample_nothing(self, _type):
        embeddings = []
        for file_name in os.listdir(join(self._base_read_path, _type)):
            if self._nothing in file_name:
                with open(os.path.join(self._base_read_path, _type, file_name)) as f_read:
                    for line in f_read.readlines():
                        data = json.loads(line)
                        data["class"] = self._nothing
                        if data["label"] == 0:
                            data["label"] = 0
                            embeddings.append(data)
        embeddings = sample(embeddings, min(self._size, len(embeddings)))
        return embeddings

    def __sample_class(self, index: int, _type: str):
        embeddings = []
        for file_name in os.listdir(join(self._base_read_path, _type)):
            if self._classes[index] not in file_name:
                continue
            with open(os.path.join(self._base_read_path, _type, file_name)) as f_read:
                for line in f_read.readlines():
                    data = json.loads(line)
                    data["class"] = self._classes[index]
                    if data["label"] == 1:
                        data["label"] = index
                        embeddings.append(data)
        embeddings = sample(embeddings, min(self._size, len(embeddings)))
        return embeddings


def main():
    classes = ["nothing", "dos", "+info", "bypass", "priv"]
    balancer = PNNDataBalancer(classes, "../embeddings", "../data/pnn", 500, "nothing", "other")
    balancer.balance()


def main2():
    classes = ["nothing", "dos", "+info", "bypass", "priv"]
    balancer = FNNDataBalancer(classes, "../embeddings", "../data/fnn", 500, "nothing", "other")
    balancer.balance()


if __name__ == '__main__':
    main()
