import numpy as np


def find_row_number(id: str) -> np.ndarray:
    pass


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
    n: int,
) -> np.ndarray:
    assert (
        title_embeddings.shape[0] == abs_embeddings.shape[0]
    ), "Not the same number of titles and abstracts"

    """Top n hits that match a row in embeddings, returns their indices
    and cosine similarities"""
    cos_similarities = cosine_similarities(title_embeddings, abs_embeddings, row)
    ohe_scalar_prods = np.sum(cat_ohe[row] * cat_ohe, axis=1)

    closeness_measure = sum(cos_similarities) / 2 + ohe_scalar_prods

    indices = np.flip(np.argsort(closeness_measure))[1 : n + 1]  # noqa: E203
    return indices, closeness_measure[indices]


def recommend_from_one(
    title_embeddings: np.ndarray,
    abs_embeddings: np.ndarray,
    cat_ohe: np.ndarray,
    row: int,
    n_reach: int = 10,
    temperature: int = 10,
) -> int:
    """Recommend a paper given a single paper as input.

    Args:
        embeddings:     An array with the embeddings of all papers
                        in a database.
        row:            The row of the input paper
        n_reach:        How many top hits to look for
        temperature:    Temperature for the probability distribution.
                        Larger values look further from the top hits.

    """
    indices, cos_sims = top_hits(
        title_embeddings, abs_embeddings, cat_ohe, row, n_reach
    )
    index = np.random.choice(
        indices,
        replace=False,
        p=cos_sims ** (100 / temperature) / sum(cos_sims ** (100 / temperature)),
    )
    return index, cos_sims[index]


if __name__ == "__main__":
    pass
