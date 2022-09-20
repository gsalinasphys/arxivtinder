import numpy as np


def top_hits(
    title_embeddings: np.ndarray,
    abs_embeddings: np.ndarray,
    cat_ohe: np.ndarray,
    row: int,
    n: int,
) -> np.ndarray:
    assert (
        title_embeddings.shape[0] == abs_embeddings.shape[0]
    ), "Not the same number of titles and abstracts"

    """Top n hits that match a row in embeddings, returns their indices
    and cosine similarities"""
    cosine_similarities = np.sum(
        title_embeddings[row] * title_embeddings, axis=1
    ) + np.sum(abs_embeddings[row] * abs_embeddings, axis=1)

    indices = np.flip(np.argsort(cosine_similarities))[1 : n + 1]  # noqa: E203
    return indices, cosine_similarities[indices]
