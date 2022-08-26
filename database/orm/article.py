from sqlalchemy import Column, String

from .base import Base


class Article(Base):
    __tablename__ = "articles"
    id = Column(String, primary_key=True)
