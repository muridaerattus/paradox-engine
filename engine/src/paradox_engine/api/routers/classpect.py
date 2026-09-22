from typing import Annotated

from fastapi import APIRouter, Depends

from paradox_engine.api.dependencies import resources
from paradox_engine.api.schemas import ClasspectRequest, ClasspectResponse
from paradox_engine.runtime import RuntimeResources

router = APIRouter(tags=["classpect"])
Resources = Annotated[RuntimeResources, Depends(resources)]


@router.post("/classpect", response_model=ClasspectResponse)
async def classpect(
    request: ClasspectRequest, runtime: Resources
) -> ClasspectResponse:
    result = await runtime.classpect.calculate_title(
        request.personality,
        runtime.class_quiz,
        runtime.aspect_quiz,
    )
    return ClasspectResponse(
        **{
            "class": result.class_result,
            "aspect": result.aspect_result,
            "result": result.llm_response,
        }
    )
