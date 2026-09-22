from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from paradox_engine.api.dependencies import resources
from paradox_engine.api.schemas import AlchemizeRequest, ItemResponse
from paradox_engine.runtime import RuntimeResources

router = APIRouter(prefix="/alchemy", tags=["alchemy"])
Resources = Annotated[RuntimeResources, Depends(resources)]


@router.post("/alchemize", response_model=ItemResponse)
async def alchemize(request: AlchemizeRequest, runtime: Resources) -> ItemResponse:
    item = await runtime.alchemy.alchemize_items(
        request.item_one,
        request.item_two,
        request.operation,
    )
    return ItemResponse.model_validate(item, from_attributes=True)


@router.get("/captchalogue", response_model=ItemResponse)
async def captchalogue(
    runtime: Resources, code: Annotated[str, Query(min_length=8, max_length=8)]
) -> ItemResponse:
    item = await runtime.alchemy.repository.get_item_by_code(code)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemResponse.model_validate(item, from_attributes=True)
