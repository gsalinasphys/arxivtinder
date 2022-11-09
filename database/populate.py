from datetime import datetime
from typing import List

import pandas as pd
from orm.article import Article
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import DATABASE_URI


def _authors_fmt(authors_parsed) -> str:
    return "\n".join([" ".join(author[:-1]) for author in authors_parsed])

def get_articles(arxiv_json_path: str = "../local_files/arxiv-metadata-oai-snapshot.json") -> List:
    arxiv_df = pd.read_json(arxiv_json_path, lines=True).drop_duplicates(subset=["id"], keep="last")

    ids = arxiv_df["id"].values
    titles = arxiv_df["title"].values
    abstracts = arxiv_df["abstract"].values
    authors = [_authors_fmt(author_list) for author_list in arxiv_df["authors_parsed"]]
    categories = arxiv_df["categories"].values
    update_dates = [datetime.strptime(updated_at, "%Y-%m-%d") for updated_at in arxiv_df["update_date"]]
    journal_refs = arxiv_df["journal-ref"].values
    dois = arxiv_df["doi"].values
    submitters = arxiv_df["submitter"].values

    articles = []
    for ii in range(len(arxiv_df)):
        article = Article(id=ids[ii],
                        title=titles[ii],
                        abstract=abstracts[ii],
                        authors=authors[ii],
                        categories=categories[ii],
                        updated_at=update_dates[ii],
                        journal_ref=journal_refs[ii],
                        doi=dois[ii],
                        submitter=submitters[ii])
        articles.append(article)

    return articles

def populate_db(arxiv_json_path: str = "../local_files/arxiv-metadata-oai-snapshot.json"):
    articles = get_articles(arxiv_json_path)

    DATABASE_URI = DATABASE_URI.split("///")[0] + "///../" + DATABASE_URI.split("///")[1]
    session_maker = sessionmaker(bind=create_engine(DATABASE_URI))

    with session_maker() as session:
        session.add_all(articles)
        session.commit()

    print(f"{len(articles)} added to the DB.")

if __name__ == '__main__':
    populate_db()