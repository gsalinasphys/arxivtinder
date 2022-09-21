# import numpy as np

# def recommend_from_many(
#     embeddings: np.ndarray, rows: np.ndarray, n_reach: int = 10, temperature: int = 10
# ) -> int:
#     embeddings = np.append(embeddings, [np.mean(embeddings[rows], axis=0)], axis=0)
#     return recommend_from_one(
#         embeddings,
#         -1,
#         n_reach=n_reach,
#         temperature=temperature,
#         n_exclude=len(rows) + 1,
#     )
