import argparse
import os
import time
from functools import cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@cache
def get_model(model_name: str):
    return SentenceTransformer("sentence-transformers/" + model_name)


def get_title_embedding_model():
    return get_model("all-MiniLM-L6-v2")


def get_abstract_embedding_model():
    return get_model("all-MiniLM-L6-v2")


def encode(sentences: np.ndarray, model) -> np.ndarray:
    """Generates the embeddings of sentences using a
    sentence-transformers model."""
    embeddings = model.encode(sentences)

    return embeddings


def get_df_from_json(filepath: str) -> pd.DataFrame:
    return pd.read_json(filepath, lines=True)


def normalize_vecs(vs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vs, axis=1)
    return np.divide(vs.T, norms).T


def encode_df(df: pd.DataFrame) -> None:
    """Generates the embeddings of titles, abstract and one-hot encodes
    categories from arXiv data."""
    title_embeddings = encode(df.title.values, get_title_embedding_model())
    abstract_embeddings = encode(df.abstract.values, get_abstract_embedding_model())
    category_ohe = df.categories.str.get_dummies(sep=" ").values

    filename, _ = os.path.splitext(args.filepath)
    np.save(filename + "_title_emb", title_embeddings)
    np.save(filename + "_abs_emb", abstract_embeddings)
    np.save(filename + "_cat_ohe", normalize_vecs(category_ohe))

    np.save(filename + "_ids", df.id.values.astype("str"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()

    df = get_df_from_json(args.filepath)

    t0 = time.perf_counter()
    encode_df(df)
    print("Time in minutes: ", (time.perf_counter() - t0) / 60)
