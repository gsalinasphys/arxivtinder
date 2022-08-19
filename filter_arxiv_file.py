import pandas as pd
import json
import tqdm
from typing import List


def filter_arxiv_file(
    input_filename: str,
    output_filename: str,
    categories: List[str],
    start_year: int,
    end_year: int,
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
    category_filter = pd.Series(data=len(chunk) * [False], index=chunk.index)
    for category in categories:
        category_filter |= chunk.categories.str.contains(category)
    return chunk[category_filter]


def _filter_timerange(chunk: pd.DataFrame, start_year: int, end_year: int):
    return chunk[
        (chunk.update_date >= str(start_year)) & (chunk.update_date <= str(end_year))
    ]
