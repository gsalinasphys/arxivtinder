from sqlalchemy import Column, String

from .base import Base


class User(Base):
    id = Column(String, primary_key=True)
