"""merge heads audit and settings

Revision ID: 6a1b
Revises: 36019b0ef251, 6a1a
Create Date: 2025-08-10
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6a1b'
down_revision = ('36019b0ef251', '6a1a')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

