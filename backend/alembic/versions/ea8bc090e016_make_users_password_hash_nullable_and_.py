"""make users.password_hash nullable and backfill

Revision ID: ea8bc090e016
Revises: 42553d8e7aab
Create Date: 2025-08-10 11:51:08.464184

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ea8bc090e016'
down_revision: Union[str, None] = '42553d8e7aab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Backfill password_hash from hashed_password where missing; then relax constraint
    op.execute(
        "UPDATE users SET password_hash = hashed_password WHERE password_hash IS NULL AND hashed_password IS NOT NULL"
    )
    # Make password_hash nullable to align with ORM usage of hashed_password
    op.alter_column('users', 'password_hash', existing_type=sa.VARCHAR(length=255), nullable=True)


def downgrade() -> None:
    # Best effort: set empty string where null then make not null
    op.execute("UPDATE users SET password_hash = '' WHERE password_hash IS NULL")
    op.alter_column('users', 'password_hash', existing_type=sa.VARCHAR(length=255), nullable=False)
