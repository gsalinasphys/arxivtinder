import argparse
import json
import os
import time
from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer


@cache
def get_model():
    return SentenceTransformer('allenai-specter')

def get_title_abs(filepath: str) -> list:
    papers = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            paper = json.loads(line)
            papers.append(paper['title'] + '[SEP]' + paper['abstract'].strip())
    
    return papers

def encode(sentences: np.ndarray, model) -> np.ndarray:
    """Generates the embeddings of sentences using a
    sentence-transformers model."""
    embeddings = model.encode(sentences)

    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()

    filename, _ = os.path.splitext(args.filepath)
    papers = get_title_abs(args.filepath)
    print(len(papers), "papers loaded.")

    model = get_model()

    start = time.perf_counter()
    embeddings = encode(papers, model)
    print('Elapsed time (minutes): ', round((time.perf_counter() - start) / 60, 1))

    np.save(filename + '_emb', embeddings)
