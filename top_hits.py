import argparse
import os
from functools import cache

import numpy as np


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


def cosine_similarities(
    title_embeddings: np.ndarray, abs_embeddings: np.ndarray, row: int
) -> tuple:
    return np.sum(title_embeddings[row] * title_embeddings, axis=1), np.sum(
        abs_embeddings[row] * abs_embeddings, axis=1
    )


def top_hits(
    title_embeddings: np.ndarray,
    abs_embeddings: np.ndarray,
    cat_ohe: np.ndarray,
    row: int,
    n: int = 10,
) -> np.ndarray:
    assert (
        title_embeddings.shape[0] == abs_embeddings.shape[0]
    ), "Not the same number of titles and abstracts"

    """Top n hits that match a row in embeddings, returns their indices
    and cosine similarities"""
    cos_similarities = cosine_similarities(title_embeddings, abs_embeddings, row)
    ohe_scalar_prods = np.sum(cat_ohe[row] * cat_ohe, axis=1)

    closeness_measure = (sum(cos_similarities) + ohe_scalar_prods) / 3

    indices = np.flip(np.argsort(closeness_measure))[1 : n + 1]  # noqa: E203
    return indices, closeness_measure[indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_database")
    args = parser.parse_args()

    filename, _ = os.path.splitext(args.path_to_database)
    ids = np.load(filename + "_ids.npy", allow_pickle=True)

    title_embeddings = get_title_embeddings(filename)
    abs_embeddings = get_abs_embeddings(filename)
    cat_ohe = get_cat_ohe(filename)

    stay = True
    while stay:
        id = input("Enter arXiv id (type 'exit' to leave): ")
        if id == "exit":
            break
        row = find_row_number(id, ids)

        indices, closeness_measures = top_hits(
            title_embeddings, abs_embeddings, cat_ohe, row
        )

        for id, closeness_measure in list(zip(ids[indices], closeness_measures)):
            print(f"{id}: {closeness_measure:.3f}")
