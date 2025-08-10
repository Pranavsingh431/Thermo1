"""add task_runs table for per-task metrics

Revision ID: 487530638893
Revises: 18ec9e13616f
Create Date: 2025-08-10 12:01:56.507424

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '487530638893'
down_revision: Union[str, None] = '18ec9e13616f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Minimal migration: create task_runs table and indexes only
    op.create_table(
        'task_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('task_name', sa.String(length=200), nullable=False),
    sa.Column('task_id', sa.String(length=100), nullable=False),
    sa.Column('args', sa.JSON(), nullable=True),
    sa.Column('kwargs', sa.JSON(), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('duration_seconds', sa.Float(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('worker_hostname', sa.String(length=200), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_task_runs_id'), 'task_runs', ['id'], unique=False)
    op.create_index(op.f('ix_task_runs_status'), 'task_runs', ['status'], unique=False)
    op.create_index(op.f('ix_task_runs_task_id'), 'task_runs', ['task_id'], unique=False)
    op.create_index(op.f('ix_task_runs_task_name'), 'task_runs', ['task_name'], unique=False)


def downgrade() -> None:
    # Minimal downgrade: drop task_runs indexes and table only
    op.drop_index(op.f('ix_task_runs_task_name'), table_name='task_runs')
    op.drop_index(op.f('ix_task_runs_task_id'), table_name='task_runs')
    op.drop_index(op.f('ix_task_runs_status'), table_name='task_runs')
    op.drop_index(op.f('ix_task_runs_id'), table_name='task_runs')
    op.drop_table('task_runs')
