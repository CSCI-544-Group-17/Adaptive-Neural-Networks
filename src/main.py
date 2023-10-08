import json
import time

import numpy as np
import torch
from progress.bar import Bar
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

CHECKPOINT = "Salesforce/codet5p-110m-embedding"
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
MODEL = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).to("cpu")
DEVICE = "cpu"

BASE_READ_PATH = "./sample/"


def generate_code_embedding(code: str):
    inputs = TOKENIZER.encode(code, return_tensors="pt").to(DEVICE)
    embedding = MODEL(inputs)[0]
    return embedding


def get_embeddings(file_name, max_count):
    embeddings = []
    with open(BASE_READ_PATH + file_name, "r") as f:
        i = 0
        bar = Bar(max=get_file_length(file_name))
        bar.start()
        for line in f.readlines():
            if i == max_count:
                break
            try:
                data = json.loads(line)
                code = data["func_before"]
                embedding = generate_code_embedding(code)
                embeddings.append(embedding.detach().numpy().reshape(-1, 256))
            except Exception as e:
                print(e)
                print(i, embedding.shape)
            finally:
                i += 1
                bar.next()
    return np.array(embeddings)


def get_file_length(file_name):
    with open("./defect/%s" % file_name, "r") as f:
        return len(f.readlines())


def main():
    embeddings = get_embeddings("train_0.jsonl", 100)
    return embeddings


def run_kmeans(embeddings: np.ndarray):
    start = time.time()
    # embeddings = np.load("./embeddings.npy")
    embeddings = embeddings.reshape(-1, embeddings.shape[2])
    kmeans = KMeans(n_clusters=5, verbose=True)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    print(len(labels))
    cluster_centers = kmeans.cluster_centers_
    print(labels)
    print("time taken: %f" % (time.time() - start))

    import matplotlib.pyplot as plt

    # Assuming you have cluster centers from K-Means clustering in 'cluster_centers'
    # 'cluster_centers' should be a 2D NumPy array where each row represents a cluster centroid

    # Create a scatter plot to visualize the cluster centroids
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', label='Cluster Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Cluster Centroids')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()
    e = main()
    run_kmeans(e)
    # for file_name in os.listdir("./defect"):
    #     if "train" not in file_name:
    #         print(file_name)
    #         continue
    #     with open("./defect/%s" % file_name, "r") as f:
    #         i = 0
    #         for line in f.readlines():
    #             data = json.loads(line)
    #             code = data["func_before"]
    #             embedding = generate_code_embedding(code)
    #             print(i, embedding.shape)
    #             i += 1
