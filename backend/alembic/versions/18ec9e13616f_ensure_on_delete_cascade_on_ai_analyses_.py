"""ensure ON DELETE CASCADE on ai_analyses.thermal_scan_id FK

Revision ID: 18ec9e13616f
Revises: 5c673726ae36
Create Date: 2025-08-10 11:56:48.454040

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '18ec9e13616f'
down_revision: Union[str, None] = '5c673726ae36'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing FK and recreate with ON DELETE CASCADE
    op.drop_constraint('ai_analyses_thermal_scan_id_fkey', 'ai_analyses', type_='foreignkey')
    op.create_foreign_key(
        'ai_analyses_thermal_scan_id_fkey',
        'ai_analyses', 'thermal_scans',
        ['thermal_scan_id'], ['id'], ondelete='CASCADE'
    )


def downgrade() -> None:
    op.drop_constraint('ai_analyses_thermal_scan_id_fkey', 'ai_analyses', type_='foreignkey')
    op.create_foreign_key(
        'ai_analyses_thermal_scan_id_fkey',
        'ai_analyses', 'thermal_scans',
        ['thermal_scan_id'], ['id']
    )
