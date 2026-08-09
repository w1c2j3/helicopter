from __future__ import annotations

from tortoise import Tortoise

from .schema import apply_schema_sql
from .settings import DatabaseSettings


async def init_db(settings: DatabaseSettings | None = None, *, generate_schemas: bool = False) -> None:
    if Tortoise._inited:
        return
    resolved = settings or DatabaseSettings.from_env()
    await Tortoise.init(config=resolved.tortoise_config())
    if generate_schemas:
        await Tortoise.generate_schemas(safe=True)
        await apply_schema_sql()


async def close_db() -> None:
    if not Tortoise._inited:
        return
    await Tortoise.close_connections()
    Tortoise.apps.clear()
    Tortoise._inited = False


async def check_db_readiness(settings: DatabaseSettings | None = None) -> None:
    """Verify the database connection and a core scoreboard table are readable."""

    await init_db(settings)
    connection = Tortoise.get_connection("default")
    await connection.execute_query("SELECT 1")

    # Import lazily so the connection module remains safe to import while the
    # model package is being initialized.  LIMIT 1 verifies the schema/table
    # without loading scoreboard data into the health-check process.
    from .models import Task

    await Task.all().limit(1).values_list("task_id", flat=True)
