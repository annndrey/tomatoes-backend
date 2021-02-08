"""init

Revision ID: ff2309d90654
Revises: 
Create Date: 2019-03-25 15:06:18.402350

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ff2309d90654'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('login', sa.String(length=400), nullable=True),
    sa.Column('name', sa.String(length=400), nullable=True),
    sa.Column('password_hash', sa.String(length=400), nullable=True),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('is_confirmed', sa.Boolean(), nullable=True),
    sa.Column('confirmed_on', sa.DateTime(), nullable=True),
    sa.Column('registered_on', sa.DateTime(), nullable=True),
    sa.Column('phone', sa.String(length=400), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user_query',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('local_name', sa.Text(), nullable=True),
    sa.Column('orig_name', sa.Text(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('result', sa.Text(), nullable=True),
    sa.Column('fsize', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user_query')
    op.drop_table('user')
    # ### end Alembic commands ###