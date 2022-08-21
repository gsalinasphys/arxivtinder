from sqlalchemy import Column, String

from .base import Base


class User(Base):
    __tablename__ = "user"
    id = Column(String, primary_key=True)