import json
import time

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

CHECKPOINT = "Salesforce/codet5p-110m-embedding"
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
MODEL = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).to("cpu")
DEVICE = "cpu"

BASE_READ_PATH = "./sample/"
EMBEDDINGS_WRITE_PATH = "./embeddings/"


def get_code_embedding(code: str):
    inputs = TOKENIZER.encode(code, return_tensors="pt").to(DEVICE)
    embedding = MODEL(inputs)[0]
    embedding = embedding.reshape(-1, embedding.shape[0]).detach().numpy()
    return embedding


def get_and_save_embeddings(file_name):
    with open(BASE_READ_PATH + file_name, "r") as f_read:
        num_lines = get_file_length(BASE_READ_PATH + file_name)
        i = 0
        with open(EMBEDDINGS_WRITE_PATH + file_name, "w") as f_write:
            for line in f_read.readlines():
                data = json.loads(line)
                code = data["func_before"]
                embeddings = get_code_embedding(code)
                label = int(data["vul"])
                result = {"embeddings": embeddings.tolist(), "label": label}
                f_write.write(json.dumps(result) + "\n")
                print("%d/%d" % (i, num_lines))
                i += 1


def get_file_length(file_path):
    count = 0
    with open(file_path, "r") as f:
        line = f.readline()
        while True:
            if not line:
                break
            count += 1
            line = f.readline()
    return count


if __name__ == '__main__':
    get_and_save_embeddings("train.jsonl")
    # for i in range(int(10e7)):
    #     print("%d" % i, end="\r")

    # for _ in tqdm(range(1000)):
    #     print("akss")
    #     time.sleep(0.1)
