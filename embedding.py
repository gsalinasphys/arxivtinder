import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def encode(sentences: np.ndarray, model: str = "all-MiniLM-L6-v2") -> None:
    """Generates the embeddings of sentences using a
    sentence-transformers model."""

    model = SentenceTransformer("sentence-transformers/" + model)
    embeddings = model.encode(sentences)

    return embeddings


def get_df_from_json(filepath: str) -> pd.DataFrame:
    return pd.read_json(filepath, lines=True)


def main():
    filepath = "static/arxiv-small.json"
    df = get_df_from_json(filepath)

    title_embeddings = encode(df.title.values)
    abstract_embeddings = encode(df.abstract.values)
    category_ohe = df.categories.str.get_dummies(sep=" ").values

    filename, _ = os.path.splitext(filepath)
    np.save(filename + "_title_emb", title_embeddings)
    np.save(filename + "_abs_emb", abstract_embeddings)
    np.save(filename + "_cat_ohe", category_ohe)

    np.save(filename + "_ids", df.id.values)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print("Time in minutes: ", (time.perf_counter() - t0) / 60)
