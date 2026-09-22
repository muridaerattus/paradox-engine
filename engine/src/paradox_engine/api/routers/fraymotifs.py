from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from paradox_engine.api.dependencies import resources
from paradox_engine.api.schemas import FraymotifRequest, FraymotifResponse
from paradox_engine.fraymotifs.models import Title
from paradox_engine.fraymotifs.utils import split_titles
from paradox_engine.runtime import RuntimeResources

router = APIRouter(tags=["fraymotifs"])
Resources = Annotated[RuntimeResources, Depends(resources)]


@router.post("/fraymotif", response_model=FraymotifResponse)
async def fraymotif(
    request: FraymotifRequest,
    runtime: Resources,
) -> FraymotifResponse:
    try:
        classes, aspects = split_titles(request.players)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    titles = [
        Title(title_class=title_class, title_aspect=aspect)
        for title_class, aspect in zip(classes, aspects)
    ]
    if not titles:
        raise HTTPException(status_code=400, detail="At least one title is required")
    result = await runtime.fraymotifs.create_fraymotif(
        titles,
        request.memory,
        request.additional_info,
    )
    return FraymotifResponse.model_validate(result, from_attributes=True)
