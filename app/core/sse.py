import asyncio
from typing import AsyncIterator

class EventBus:
    def __init__(self) -> None:
        self._channels: dict[str, asyncio.Queue[str]] = {}

    def channel(self, job_id: str) -> asyncio.Queue[str]:
        if job_id not in self._channels:
            self._channels[job_id] = asyncio.Queue()
        return self._channels[job_id]

    async def publish(self, job_id: str, msg: str) -> None:
        await self.channel(job_id).put(msg)

    async def stream(self, job_id: str) -> AsyncIterator[str]:
        q = self.channel(job_id)
        while True:
            msg = await q.get()
            yield f"data: {msg}\n\n"

event_bus = EventBus()
