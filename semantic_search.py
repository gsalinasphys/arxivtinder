import argparse
import os
from functools import cache

import numpy as np
import torch
from sentence_transformers import util


@cache
def get_title_embeddings(filename: str) -> np.ndarray:
    return np.load(filename + "_title_emb.npy")


@cache
def get_abs_embeddings(filename: str) -> np.ndarray:
    return np.load(filename + "_abs_emb.npy")


@cache
def get_cat_ohe(filename: str) -> np.ndarray:
    return np.load(filename + "_cat_ohe.npy")


def find_row_number(id: str, ids: np.ndarray) -> np.ndarray:
    return np.where(ids == id)[0][0]


def top_hits(
    title_embeddings: np.ndarray,
    abs_embeddings: np.ndarray,
    cat_ohe: np.ndarray,
    rows: list,
    n: int = 10,
) -> np.ndarray:
    assert (
        title_embeddings.shape[0] == abs_embeddings.shape[0]
    ), "Not the same number of titles and abstracts"

    """Top n hits that match a row in embeddings, returns their indices
    and cosine similarities"""
    corpus_embeddings = np.concatenate(
        (title_embeddings, abs_embeddings, cat_ohe), axis=1
    )
    corpus_embeddings = torch.from_numpy(corpus_embeddings)

    query_embedding = np.concatenate(
        (title_embeddings[rows], abs_embeddings[rows], cat_ohe[rows])
    )
    query_embedding = torch.from_numpy(query_embedding)

    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=n)
    search_hits = search_hits[0]  # Get the hits for the first query

    print(search_hits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_database")
    args = parser.parse_args()

    filename, _ = os.path.splitext(args.path_to_database)
    arxiv_ids = np.load(filename + "_ids.npy", allow_pickle=True)

    title_embeddings = get_title_embeddings(filename)
    abs_embeddings = get_abs_embeddings(filename)
    cat_ohe = get_cat_ohe(filename)

    stay = True
    while stay:
        id = input("Enter arXiv id (type 'exit' to leave): ")
        if id == "exit":
            break
        row = find_row_number(id, arxiv_ids)

        top_hits(title_embeddings, abs_embeddings, cat_ohe, row)
