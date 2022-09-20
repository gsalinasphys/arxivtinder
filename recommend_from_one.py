import numpy as np

from top_hits import top_hits


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
