"""add app_settings table for runtime config

Revision ID: 36019b0ef251
Revises: 487530638893
Create Date: 2025-08-10 12:21:28.437386

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '36019b0ef251'
down_revision: Union[str, None] = '487530638893'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Minimal: create app_settings table only
    op.create_table('app_settings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('key', sa.String(length=100), nullable=False),
    sa.Column('value', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('key', name='uq_app_settings_key')
    )
    op.create_index(op.f('ix_app_settings_id'), 'app_settings', ['id'], unique=False)
    op.create_index(op.f('ix_app_settings_key'), 'app_settings', ['key'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_app_settings_key'), table_name='app_settings')
    op.drop_index(op.f('ix_app_settings_id'), table_name='app_settings')
    op.drop_table('app_settings')
