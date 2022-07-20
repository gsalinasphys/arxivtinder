# Filtering arxiv dataset
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Comment
# More comments


def arxiv_filter(
    directory: str, yr_range: Tuple[float, float], categories: List[str] = None
) -> pd.DataFrame:
    """Filters an arXiv database from a directory, based on the year of last update
    and categories.
    Files should be divided into chunks named in the format "chunk*.json".
    """
    files = Path(directory).glob("chunk*.json")

    frames = []
    for file in files:
        achunk = pd.read_json(file)

        if categories:
            achunk_filter = achunk.loc[
                (achunk["update_date"] >= str(yr_range[0]))
                & (achunk["update_date"] < str(yr_range[1] + 1))
                & (achunk["categories"].isin(categories))
            ].copy()
        else:
            achunk_filter = achunk.loc[
                (achunk["update_date"] >= str(yr_range[0]))
                & (achunk["update_date"] < str(yr_range[1] + 1))
            ].copy()

        frames.append(achunk_filter)

    return pd.concat(frames)
