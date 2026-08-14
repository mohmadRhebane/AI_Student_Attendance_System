import asyncio
from logging.config import fileConfig
import app.models
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from app.config import DATABASE_URL

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import Base

from app.models.supervisor import Supervisor
from app.models.classroom import Classroom
from app.models.student import Student
from app.models.session import Session
from app.models.face_embedding import FaceEmbedding
from app.models.attendance_logs import AttendanceLogs
from app.models.system_setting import SystemSetting
from app.models.attendance_record import AttendanceRecord
from app.models.scheduler_logs import SchedulerLog
from app.models.videos import Video
from app.models.scan_job import ScanJob
config = context.config

config.set_main_option("sqlalchemy.url", DATABASE_URL)


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# def do_run_migrations(connection: Connection) -> None:
#     context.configure(connection=connection, target_metadata=target_metadata)
#
#     with context.begin_transaction():
#         context.run_migrations()
#
#
# async def run_async_migrations() -> None:
#     """In this scenario we need to create an Engine
#     and associate a connection with the context.
#
#     """
#
#     connectable = async_engine_from_config(
#         config.get_section(config.config_ini_section, {}),
#         prefix="sqlalchemy.",
#         poolclass=pool.NullPool,
#     )
#
#     async with connectable.connect() as connection:
#         await connection.run_sync(do_run_migrations)
#
#     await connectable.dispose()



def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata,include_schemas=True,)
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """في هذه الدالة نقوم بالاتصال غير المتزامن بقاعدة البيانات."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """هذه هي الدالة الرئيسية التي يستدعيها Alembic."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
