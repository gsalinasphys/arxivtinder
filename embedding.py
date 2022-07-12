import multiprocessing
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def encode(
    df: pd.DataFrame, model: str = "all-MiniLM-L6-v2"
) -> None:  # Also try 'allenai-specter', but very slow
    """Encodes the abstracts from papers in a dataframe using a
    sentence-transformers model."""
    abstracts = list(df["abstract"])

    model = SentenceTransformer("sentence-transformers/" + model)
    embeddings = model.encode(abstracts)

    return embeddings


def main():
    filepath = "static/arxiv-metadata-oai-snapshot_hepthph.json"
    df = pd.read_json(filepath)

    ncores = multiprocessing.cpu_count() - 1
    df_splits = np.array_split(df, ncores)
    with multiprocessing.Pool(processes=ncores) as pool:
        embeddings = pool.map(encode, df_splits)

    filename, _ = os.path.splitext(filepath)
    np.save(filename + "_emb", embeddings)


if __name__ == "__main__":
    main()
