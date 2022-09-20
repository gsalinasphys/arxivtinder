import numpy as np

from top_hits import cosine_similarity, top_hits


def find_row_number(id: str) -> np.ndarray:
    pass


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
    tophits = top_hits(title_embeddings, abs_embeddings, cat_ohe, row, n_reach)
    index = np.random.choice(
        tophits[0],
        replace=False,
        p=tophits[1] ** (100 / temperature) / sum(tophits[1] ** (100 / temperature)),
    )
    return index, cosine_similarity(
        title_embeddings[row], title_embeddings[index]
    ) + cosine_similarity(abs_embeddings[row], abs_embeddings[index])


if __name__ == "__main__":
    pass
