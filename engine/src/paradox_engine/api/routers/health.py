from fastapi import APIRouter

from paradox_engine.api.schemas import StatusResponse

router = APIRouter()


@router.get("/", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(message="PARADOX ENGINE: Status operational.")
