import argparse
import os
import time
from functools import cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@cache
def get_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer("sentence-transformers/" + model_name)


def encode(sentences: np.ndarray, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generates the embeddings of sentences using a
    sentence-transformers model."""
    model = get_model(model_name)
    embeddings = model.encode(sentences)

    return embeddings


def get_df_from_json(filepath: str) -> pd.DataFrame:
    return pd.read_json(filepath, lines=True)


def encode_df(df: pd.DataFrame) -> None:
    """Generates the embeddings of titles, abstract and one-hot encodes
    categories from arXiv data."""
    title_embeddings = encode(df.title.values)
    abstract_embeddings = encode(df.abstract.values)
    category_ohe = df.categories.str.get_dummies(sep=" ").values

    filename, _ = os.path.splitext(args.filepath)
    np.save(filename + "_title_emb", title_embeddings)
    np.save(filename + "_abs_emb", abstract_embeddings)
    np.save(filename + "_cat_ohe", category_ohe)

    np.save(filename + "_ids", df.id.values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()

    df = get_df_from_json(args.filepath)

    t0 = time.perf_counter()
    encode_df(df)
    print("Time in minutes: ", (time.perf_counter() - t0) / 60)
