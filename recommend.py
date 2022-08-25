import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)


def top_hits(
    embeddings: np.ndarray, row: int, n: int, n_exclude: int = 1
) -> np.ndarray:
    """Top n hits that match a row in embeddings, returns their indices
    and cosine similarities"""
    cosine_similarities = np.array(
        [
            cosine_similarity(embeddings[row], embeddings[ii])
            for ii in range(len(embeddings))
        ]
    )
    indices = np.flip(np.argsort(cosine_similarities))[
        n_exclude : n + n_exclude  # noqa: E203
    ]
    return indices, cosine_similarities[indices]


def recommend_from_one(
    embeddings: np.ndarray,
    row: int,
    n_reach: int = 10,
    temperature: int = 10,
    n_exclude: int = 1,
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
    tophits = top_hits(embeddings, row, n_reach, n_exclude)
    index = np.random.choice(
        tophits[0],
        replace=False,
        p=tophits[1] ** (100 / temperature) / sum(tophits[1] ** (100 / temperature)),
    )
    return index, cosine_similarity(embeddings[row], embeddings[index])


def recommend_from_many(
    embeddings: np.ndarray, rows: np.ndarray, n_reach: int = 10, temperature: int = 10
) -> int:
    embeddings = np.append(embeddings, [np.mean(embeddings[rows], axis=0)], axis=0)
    return recommend_from_one(
        embeddings,
        -1,
        n_reach=n_reach,
        temperature=temperature,
        n_exclude=len(rows) + 1,
    )
