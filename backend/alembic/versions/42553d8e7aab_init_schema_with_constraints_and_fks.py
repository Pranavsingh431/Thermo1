"""init schema with constraints and fks

Revision ID: 42553d8e7aab
Revises: 
Create Date: 2025-08-10 11:45:46.520710

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '42553d8e7aab'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users: add missing columns/constraints if not exist
    op.execute("""
    DO $$ BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='hashed_password'
      ) THEN
        ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255);
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='full_name'
      ) THEN
        ALTER TABLE users ADD COLUMN full_name VARCHAR(255);
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='is_verified'
      ) THEN
        ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT FALSE NOT NULL;
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='updated_at'
      ) THEN
        ALTER TABLE users ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='login_count'
      ) THEN
        ALTER TABLE users ADD COLUMN login_count INTEGER DEFAULT 0;
      END IF;
    END $$;
    """)

    # Substations: add voltage_level and other columns if missing
    op.execute("""
    DO $$ BEGIN
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='substations' AND column_name='voltage_level') THEN
        ALTER TABLE substations ADD COLUMN voltage_level VARCHAR(50);
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='substations' AND column_name='boundary_coordinates') THEN
        ALTER TABLE substations ADD COLUMN boundary_coordinates JSONB;
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='substations' AND column_name='inspection_radius') THEN
        ALTER TABLE substations ADD COLUMN inspection_radius DOUBLE PRECISION DEFAULT 500.0;
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='substations' AND column_name='updated_at') THEN
        ALTER TABLE substations ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
      END IF;
    END $$;
    """)

    # Thermal scans: unique hash, FKs, indexes
    op.execute("""
    DO $$ BEGIN
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='thermal_scans' AND column_name='file_hash') THEN
        ALTER TABLE thermal_scans ADD COLUMN file_hash VARCHAR(64);
      END IF;
      IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='thermal_scans' AND indexname='uq_thermal_scans_file_hash') THEN
        CREATE UNIQUE INDEX uq_thermal_scans_file_hash ON thermal_scans (file_hash);
      END IF;
      -- batch_id/batch_sequence composite index
      IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='thermal_scans' AND indexname='idx_thermal_scans_batch_seq') THEN
        CREATE INDEX idx_thermal_scans_batch_seq ON thermal_scans (batch_id, batch_sequence);
      END IF;
      -- Add uploaded_by FK if not present
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name='thermal_scans' AND constraint_name='fk_thermal_scans_uploaded_by_users'
      ) THEN
        ALTER TABLE thermal_scans
          ADD CONSTRAINT fk_thermal_scans_uploaded_by_users
          FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL;
      END IF;
      -- Add substation FK if not present
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name='thermal_scans' AND constraint_name='fk_thermal_scans_substation_id_substations'
      ) THEN
        ALTER TABLE thermal_scans
          ADD CONSTRAINT fk_thermal_scans_substation_id_substations
          FOREIGN KEY (substation_id) REFERENCES substations(id) ON DELETE SET NULL;
      END IF;
    END $$;
    """)

    # AI analyses: ensure FK to thermal_scans
    op.execute("""
    DO $$ BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name='ai_analyses' AND constraint_name='fk_ai_analyses_thermal_scan_id'
      ) THEN
        ALTER TABLE ai_analyses
          ADD CONSTRAINT fk_ai_analyses_thermal_scan_id
          FOREIGN KEY (thermal_scan_id) REFERENCES thermal_scans(id) ON DELETE CASCADE;
      END IF;
    END $$;
    """)


def downgrade() -> None:
    # Best-effort downgrade: drop added constraints/indexes
    op.execute("""
    DO $$ BEGIN
      IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name='ai_analyses' AND constraint_name='fk_ai_analyses_thermal_scan_id') THEN
        ALTER TABLE ai_analyses DROP CONSTRAINT fk_ai_analyses_thermal_scan_id;
      END IF;
      IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name='thermal_scans' AND constraint_name='fk_thermal_scans_uploaded_by_users') THEN
        ALTER TABLE thermal_scans DROP CONSTRAINT fk_thermal_scans_uploaded_by_users;
      END IF;
      IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name='thermal_scans' AND constraint_name='fk_thermal_scans_substation_id_substations') THEN
        ALTER TABLE thermal_scans DROP CONSTRAINT fk_thermal_scans_substation_id_substations;
      END IF;
      IF EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='thermal_scans' AND indexname='uq_thermal_scans_file_hash') THEN
        DROP INDEX uq_thermal_scans_file_hash;
      END IF;
      IF EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='thermal_scans' AND indexname='idx_thermal_scans_batch_seq') THEN
        DROP INDEX idx_thermal_scans_batch_seq;
      END IF;
    END $$;
    """)
