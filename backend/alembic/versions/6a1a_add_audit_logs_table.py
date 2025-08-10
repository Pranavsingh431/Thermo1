"""add audit logs table

Revision ID: 6a1a
Revises: 487530638893
Create Date: 2025-08-10
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6a1a'
down_revision = '487530638893'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=100), nullable=False),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('method', sa.String(length=10), nullable=True),
        sa.Column('path', sa.String(length=500), nullable=True),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('ip_address', sa.String(length=100), nullable=True),
        sa.Column('user_agent', sa.String(length=300), nullable=True),
        sa.Column('extra', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_resource', 'audit_logs', ['resource_type', 'resource_id'])
    op.create_index('ix_audit_logs_created', 'audit_logs', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_audit_logs_created', table_name='audit_logs')
    op.drop_index('ix_audit_logs_resource', table_name='audit_logs')
    op.drop_index('ix_audit_logs_action', table_name='audit_logs')
    op.drop_index('ix_audit_logs_user_id', table_name='audit_logs')
    op.drop_table('audit_logs')

