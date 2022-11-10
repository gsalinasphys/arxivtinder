"""Removed categories from Article, populating CategoryTag

Revision ID: ab400bb83afe
Revises: 2a7789884916
Create Date: 2022-11-10 17:15:11.577266

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ab400bb83afe'
down_revision = '2a7789884916'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
