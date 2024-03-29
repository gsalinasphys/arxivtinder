import json
from typing import List

import pandas as pd
import tqdm


def filter_arxiv_file(
    input_filename: str,
    output_filename: str,
    categories: List[str] = None,
    start_year: int = None,
    end_year: int = None,
):
    """Filters an arXiv database from a directory, based on the year of last update
    and categories.
    """
    with open(output_filename, "w") as f:
        for chunk in tqdm.tqdm(
            pd.read_json(input_filename, lines=True, chunksize=1000)
        ):
            filtered_chunk = _filter_categories(
                _filter_timerange(chunk, start_year, end_year), categories
            )

            f.writelines(
                json.dumps(d) + "\n" for d in filtered_chunk.to_dict(orient="records")
            )


def _filter_categories(chunk: pd.DataFrame, categories: List[str]):
    if len(chunk) == 0:
        return chunk
    if categories is None:
        return chunk
    category_filter = pd.Series(data=len(chunk) * [False], index=chunk.index)
    for category in categories:
        category_filter |= chunk.categories.str.contains(category)
    return chunk[category_filter]


def _filter_timerange(chunk: pd.DataFrame, start_year: int, end_year: int):
    if start_year is not None:
        chunk = chunk[chunk.update_date >= str(start_year)]
    if end_year is not None:
        chunk = chunk[chunk.update_date < str(end_year + 1)]
    return chunk
