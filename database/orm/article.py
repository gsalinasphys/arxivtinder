from sqlalchemy import Column, String

from .base import Base


class Article(Base):
    id = Column(String, primary_key=True)
