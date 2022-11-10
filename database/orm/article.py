from sqlalchemy import BLOB, Column, Date, ForeignKey, String
from sqlalchemy.orm import deferred, relationship

from .base import Base


class CategoryTagArticle(Base):
    __tablename__ = "category_tags_articles"
    category_tag_id = Column(ForeignKey("category_tags.tag"), primary_key=True)
    article_id = Column(ForeignKey("articles.id"), primary_key=True)


class Article(Base):
    __tablename__ = "articles"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    abstract = Column(String, nullable=False)
    authors = Column(String)
    updated_at = Column(Date, nullable=False)
    journal_ref = Column(String)
    doi = Column(String)
    submitter = Column(String)

    tags = relationship(
        "CategoryTag",
        secondary=CategoryTagArticle.__tablename__,
        back_populates="articles",
    )


class CategoryTag(Base):
    __tablename__ = "category_tags"
    tag = Column(String, primary_key=True)
    name = Column(String, nullable=False)

    articles = relationship(
        "Article",
        secondary=CategoryTagArticle.__tablename__,
        back_populates="tags",
    )
