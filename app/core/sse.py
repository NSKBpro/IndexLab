import asyncio
from contextlib import suppress
from typing import Dict

class EventBus:
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._loop = asyncio.get_event_loop()

    def _queue(self, job_id: str) -> asyncio.Queue:
        q = self._queues.get(job_id)
        if q is None:
            q = self._queues[job_id] = asyncio.Queue(maxsize=1000)
        return q

    # Async publishers can use this
    async def publish(self, job_id: str, data: str):
        await self._queue(job_id).put(str(data))

    # Use this inside threads / sync code
    def publish_threadsafe(self, job_id: str, data: str):
        asyncio.run_coroutine_threadsafe(self.publish(job_id, str(data)), self._loop)

    async def stream(self, job_id: str):
        q = self._queue(job_id)
        try:
            yield {"event": "message", "data": f"START {job_id}"}
            while True:
                msg = await q.get()
                yield {"event": "message", "data": str(msg)}
                if str(msg) == "DONE" or str(msg).startswith("ERROR"):
                    break
        finally:
            with suppress(Exception):
                self._queues.pop(job_id, None)

event_bus = EventBus()
