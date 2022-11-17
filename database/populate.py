from datetime import datetime
from functools import cache
from typing import List

import pandas as pd
from orm.article import Article, CategoryTag, CategoryTagArticle
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from database import engine


@cache
def get_df(filename: str = "local_files/arxiv-metadata-oai-snapshot.json"):
    return pd.read_json(filename, lines=True).drop_duplicates(subset=["id"], keep="last")

def _authors_fmt(authors_parsed) -> str:
    return "\n".join([" ".join(author[:-1]) for author in authors_parsed])

def get_articles_categories() -> tuple:
    arxiv_df = get_df()

    ids = arxiv_df["id"].to_numpy()
    titles = arxiv_df["title"].to_numpy()
    abstracts = arxiv_df["abstract"].to_numpy()
    authors = [_authors_fmt(author_list) for author_list in arxiv_df["authors_parsed"]]
    update_dates = [datetime.strptime(updated_at, "%Y-%m-%d") for updated_at in arxiv_df["update_date"]]
    journal_refs = arxiv_df["journal-ref"].to_numpy()
    dois = arxiv_df["doi"].to_numpy()
    submitters = arxiv_df["submitter"].to_numpy()
    categories_grouped = arxiv_df["categories"].str.split().to_numpy()

    articles, categories = [], {}
    for ii in tqdm(range(len(ids)), desc='Loading papers'):
        tags = []
        for category in categories_grouped[ii]:
            if category not in categories:
                categories[category] = CategoryTag(tag=category, name=category)
            tags.append(categories[category])

        article = Article(id=ids[ii],
                        title=titles[ii],
                        abstract=abstracts[ii],
                        authors=authors[ii],
                        updated_at=update_dates[ii],
                        journal_ref=journal_refs[ii],
                        doi=dois[ii],
                        submitter=submitters[ii],
                        tags=tags)
        articles.append(article)

    return articles

def populate_db():
    articles = get_articles_categories()

    session_maker = sessionmaker(bind=engine)
    with session_maker() as session:
        session.add_all(articles)
        session.commit()    

    print(f"{len(articles)} articles added to the DB.")

def wipe_db(table: object):
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        n_deleted = session.query(table).delete()
        session.commit()

    print(f"{n_deleted} papers removed from DB.")

if __name__ == '__main__':
    populate_db()