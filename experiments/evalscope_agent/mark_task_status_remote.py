from __future__ import annotations

import asyncio
import sys
from pathlib import Path


async def main() -> None:
    root = Path("/home/rwkv/chase/EvalScope")
    scoreboard_path = root / "src" / "scoreboard-server"
    sys.path.insert(0, str(scoreboard_path))
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        for task_id in sys.argv[1:]:
            await store.update_task_status(task_id=task_id, status="Failed")
            print(f"task={task_id} status=Failed")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
