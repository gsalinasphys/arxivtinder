import argparse
import os
from functools import cache

import numpy as np
import torch
from sentence_transformers import util


@cache
def get_ids(filename: str) -> np.ndarray:
    return np.load(filename + "_ids.npy", allow_pickle=True)


@cache
def get_embeddings(filename: str) -> np.ndarray:
    return np.load(filename + "_emb.npy")


def find_row_numbers(chosen_ids: list, ids: np.ndarray) -> np.ndarray:
    return [np.where(ids == id)[0][0] for id in chosen_ids]


def top_hits(
    embeddings: np.ndarray,
    rows: list,
    n: int = 10,
) -> np.ndarray:
    """Top n hits that match a row in embeddings, returns their indices
    and cosine similarities"""
    corpus_embeddings = torch.from_numpy(embeddings)
    query_embedding = torch.from_numpy(embeddings[rows])

    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=n)
    return [search_hit[1:] for search_hit in search_hits]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_database")
    args = parser.parse_args()

    filename, _ = os.path.splitext(args.path_to_database)
    ids = get_ids(filename)
    embeddings = get_embeddings(filename)

    stay = True
    while stay:
        chosen_ids = input(
            "Enter arXiv ids, separated by spaces (type 'exit' to leave): "
        ).split()
        if chosen_ids == "exit":
            break

        rows = find_row_numbers(chosen_ids, ids)
        tophits = top_hits(embeddings, rows)

        for ii, id in enumerate(chosen_ids):
            print("Paper arXiv id: ", id)
            print("Best matches: ")
            print("Score    arXiv id")

            for hit in tophits[ii]:
                print(f"{hit['score']:.3f}  {ids[hit['corpus_id']]}")
