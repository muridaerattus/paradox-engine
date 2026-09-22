from fastapi import Request

from paradox_engine.runtime import RuntimeResources


def resources(request: Request) -> RuntimeResources:
    return request.app.state.resources
