"""set ai_analyses.thermal_scan_id on delete cascade and nullable for delete path

Revision ID: 5c673726ae36
Revises: ea8bc090e016
Create Date: 2025-08-10 11:52:09.037143

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5c673726ae36'
down_revision: Union[str, None] = 'ea8bc090e016'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing FK and recreate with ON DELETE CASCADE
    op.execute(
        """
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name='ai_analyses' AND constraint_name='fk_ai_analyses_thermal_scan_id'
          ) THEN
            ALTER TABLE ai_analyses DROP CONSTRAINT fk_ai_analyses_thermal_scan_id;
          END IF;
        END $$;
        """
    )
    # Recreate FK with cascade
    op.create_foreign_key(
        'fk_ai_analyses_thermal_scan_id',
        'ai_analyses', 'thermal_scans',
        ['thermal_scan_id'], ['id'], ondelete='CASCADE'
    )


def downgrade() -> None:
    # Drop and recreate without cascade
    op.drop_constraint('fk_ai_analyses_thermal_scan_id', 'ai_analyses', type_='foreignkey')
    op.create_foreign_key(
        'fk_ai_analyses_thermal_scan_id',
        'ai_analyses', 'thermal_scans',
        ['thermal_scan_id'], ['id']
    )
