from datetime import datetime
from functools import cache
from typing import List

import pandas as pd
from orm.article import Article, CategoryTag, CategoryTagArticle
from sqlalchemy.orm import sessionmaker

from database import engine


@cache
def get_df(filename: str = "../local_files/arxiv-metadata-oai-snapshot.json"):
    return pd.read_json(filename, lines=True).drop_duplicates(subset=["id"], keep="last")

def _authors_fmt(authors_parsed) -> str:
    return "\n".join([" ".join(author[:-1]) for author in authors_parsed])

def get_categories() -> List:
    arxiv_df = get_df()

    categories_grouped = arxiv_df["categories"].str.split().values
    categories_flatten = [category for categories in categories_grouped for category in categories]
    categories_set = sorted(list(set(categories_flatten)))

    categories = []
    for ii, category_name in enumerate(categories_set):
        categorytag = CategoryTag(tag=ii, name=category_name)
        categories.append(categorytag)

    return categories

def get_article_categories() -> List:
    arxiv_df = get_df()

    ids = arxiv_df["id"].values
    categories_grouped = arxiv_df["categories"].str.split().values

    categorytags = []
    for ii, id in enumerate(ids):
        for category in categories_grouped[ii]:
            categorytag = CategoryTagArticle(article_id=id, category_tag_id=category)
            categorytags.append(categorytag)

    return categorytags

def get_articles() -> List:
    arxiv_df = get_df()

    ids = arxiv_df["id"].values
    titles = arxiv_df["title"].values
    abstracts = arxiv_df["abstract"].values
    authors = [_authors_fmt(author_list) for author_list in arxiv_df["authors_parsed"]]
    update_dates = [datetime.strptime(updated_at, "%Y-%m-%d") for updated_at in arxiv_df["update_date"]]
    journal_refs = arxiv_df["journal-ref"].values
    dois = arxiv_df["doi"].values
    submitters = arxiv_df["submitter"].values

    articles = []
    for ii, id in enumerate(ids):
        article = Article(id=id,
                        title=titles[ii],
                        abstract=abstracts[ii],
                        authors=authors[ii],
                        updated_at=update_dates[ii],
                        journal_ref=journal_refs[ii],
                        doi=dois[ii],
                        submitter=submitters[ii])
        articles.append(article)

    return articles

def populate_db():
    articles = get_articles()
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        session.add_all(articles)
        session.commit()

    print(f"{len(articles)} articles added to the DB.")

def populate_categories_db():
    categories = get_categories()
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        session.add_all(categories)
        session.commit()

    print(f"{len(categories)} category tags added to the DB.")

def populate_article_categories_db():
    article_categories = get_article_categories()
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        session.add_all(article_categories)
        session.commit()

    print(f"{len(article_categories)} article categories added to the DB.")

def wipe_db(table: object):
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        n_deleted = session.query(table).delete()
        session.commit()

    print(f"{n_deleted} papers removed from DB.")

if __name__ == '__main__':
    # populate_db()
    # populate_categories_db()
    populate_article_categories_db()