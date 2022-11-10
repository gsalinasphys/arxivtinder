"""Removed categories from Article, populating CategoryTag

Revision ID: 2a7789884916
Revises: df57a16cf7b9
Create Date: 2022-11-10 17:12:25.839385

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2a7789884916'
down_revision = 'df57a16cf7b9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
