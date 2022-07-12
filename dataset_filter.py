# Filtering arxiv dataset
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def arxiv_filter(
    directory: str, yr_range: Tuple[float, float], categories: List[str] = None
) -> pd.DataFrame:
    files = Path(directory).glob("chunk*.json")  # Files have to be named chunk*.json

    frames = []
    for file in files:
        achunk = pd.read_json(file)

        achunk["year"] = [int(date[:4]) for date in achunk["update_date"]]

        if categories:
            achunk_filter = achunk.loc[
                (achunk["year"] >= yr_range[0])
                & (achunk["year"] <= yr_range[1])
                & (achunk["categories"].isin(categories))
            ].copy()
        else:
            achunk_filter = achunk.loc[
                (achunk["year"] >= yr_range[0]) & (achunk["year"] <= yr_range[1])
            ].copy()
        achunk_filter.drop("year", axis=1, inplace=True)

        frames.append(achunk_filter)

    return pd.concat(frames)
