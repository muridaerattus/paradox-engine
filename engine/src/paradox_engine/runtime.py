from dataclasses import dataclass

from paradox_engine.alchemy.service import AlchemyService
from paradox_engine.classpect.service import ClasspectService
from paradox_engine.classpect.thread_service import ClasspectThreadService
from paradox_engine.fraymotifs.service import FraymotifService


@dataclass
class RuntimeResources:
    alchemy: AlchemyService
    classpect: ClasspectService
    classpect_threads: ClasspectThreadService
    fraymotifs: FraymotifService
    class_quiz: list[dict]
    aspect_quiz: list[dict]
