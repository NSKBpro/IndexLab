from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import asyncio
from ..core.sse import event_bus

router = APIRouter()
HEARTBEAT_SECS = 10

@router.get("/events/{job_id}")
async def sse_events(job_id: str):
    async def gen():
        yield {"retry": 2000}
        agen = event_bus.stream(job_id).__aiter__()
        while True:
            try:
                item = await asyncio.wait_for(agen.__anext__(), timeout=HEARTBEAT_SECS)
                yield item
            except asyncio.TimeoutError:
                # keep-alive so proxies/browsers don’t buffer or close
                yield {"event": "ping", "data": "keepalive"}
            except StopAsyncIteration:
                break

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # nginx hint
        "Connection": "keep-alive",
    }
    return EventSourceResponse(gen(), headers=headers, media_type="text/event-stream")
