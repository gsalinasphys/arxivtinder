import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def encode(
    df: pd.DataFrame, model: str = "all-MiniLM-L6-v2"
) -> None:  # Also try 'allenai-specter', but very slow
    """Encodes the abstracts from papers in a dataframe using a
    sentence-transformers model."""
    abstracts = df["abstract"].values

    model = SentenceTransformer("sentence-transformers/" + model)
    embeddings = model.encode(abstracts)

    return embeddings


def main():
    filepath = "static/arxiv-metadata-oai-snapshot_hepthph.json"
    df = pd.read_json(filepath, lines=True)

    embeddings = encode(df)

    filename, _ = os.path.splitext(filepath)
    np.save(filename + "_emb", embeddings)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print("Time in minutes: ", (time.perf_counter() - t0) / 60)
