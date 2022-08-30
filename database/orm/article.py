from sqlalchemy import Column, Date, ForeignKey, String
from sqlalchemy.dialects.mysql import MEDIUMBLOB, MEDIUMTEXT
from sqlalchemy.orm import relationship

from .base import Base


class CategoryTagArticle(Base):
    __tablename__ = "category_tags_articles"
    category_tag_id = Column(ForeignKey("category_tags.tag"), primary_key=True)
    article_id = Column(ForeignKey("articles.id"), primary_key=True)


class Article(Base):
    __tablename__ = "articles"
    id = Column(String, primary_key=True)
    abstract = Column(MEDIUMTEXT(unicode=True))
    abstract_embedding = Column(MEDIUMBLOB)
    title = Column(String)
    submitter = Column(String)
    journal_ref = Column(String)
    doi = Column(String)
    authors = Column(String)
    updated_at = Column(Date)

    tags = relationship(
        "CategoryTag",
        secondary=CategoryTagArticle.__tablename__,
        back_populates="articles",
    )


class CategoryTag(Base):
    __tablename__ = "category_tags"
    tag = Column(String, primary_key=True)
    name = Column(String)

    articles = relationship(
        "Article",
        secondary=CategoryTagArticle.__tablename__,
        back_populates="category_tags",
    )
