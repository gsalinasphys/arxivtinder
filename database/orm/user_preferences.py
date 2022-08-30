import enum

from sqlalchemy import Column, Enum, ForeignKey

from .base import Base


class UserSentiment(enum.Enum):
    like = 0
    dislike = 1


class UserArticlePreference(Base):
    __tablename__ = "user_article_preferences"
    article_id = Column(ForeignKey("articles.id"), primary_key=True)
    user_id = Column(ForeignKey("user.id"), primary_key=True)
    sentiment = Column(Enum(UserSentiment))
