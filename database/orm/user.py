import datetime

from sqlalchemy import Column, DateTime, Integer, String

from .base import Base


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    full_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now)
