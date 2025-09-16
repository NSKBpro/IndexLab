from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from ..core.sse import event_bus

router = APIRouter()

@router.get("/events/{job_id}")
async def sse_events(job_id: str):
    return EventSourceResponse(event_bus.stream(job_id))
