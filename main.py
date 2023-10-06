import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import progress
from progress.bar import Bar
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import os
import json

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


def generate_code_embedding(code: str):
    tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")
    model = AutoModelForMaskedLM.from_pretrained("neulab/codebert-cpp")
    tokens_ids = tokenizer.encode(code, add_special_tokens=True, max_length=512, padding='max_length')[:512]
    context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    return context_embeddings


def get_embeddings(file_name, max_count):
    embeddings = []
    with open("./defect/%s" % file_name, "r") as f:
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
                embeddings.append(embedding.detach().numpy().reshape(-1, 50265))
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
    # print(embeddings.shape)
    # np.save("./embeddings.npy", embeddings)
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
