import json
import os
import re

from random import sample
from typing import Dict, Tuple
from progress.bar import Bar

KEY_ZEROS = "zeros"

KEY_ONES = "ones"

BASE_DATA_PATH = "./defect/"
BASE_SAMPLES_PATH = "./sample/"


def get_balanced_data():
    for file_name in os.listdir(BASE_DATA_PATH):
        if "train" in file_name:
            with open(BASE_DATA_PATH + file_name, "r") as f_read:
                ones = []
                zeros = []
                for line in f_read.readlines():
                    data = json.loads(line)
                    if data["vul"] == "0":
                        zeros.append(data)
                    elif data["vul"] == "1":
                        ones.append(data)
                    data.pop("lang")
                    data.pop("project")
                    data.pop("Publish Date")
                size = min(len(ones), len(zeros))
                sampled_ones = sample(ones, k=size)
                sampled_zeros = sample(zeros, k=size)
                sampled = sampled_ones + sampled_zeros
                with open(BASE_SAMPLES_PATH + file_name, "w") as f_write:
                    for _sample in sampled:
                        f_write.write(json.dumps(_sample) + "\n")


def get_directory_size() -> int:
    return len(os.listdir(BASE_DATA_PATH))


def main():
    get_balanced_data()


if __name__ == '__main__':
    main()
