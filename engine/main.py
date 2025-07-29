from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from engine.classpect.service import calculate_title
from alchemy.service import alchemize_items
from alchemy.models import Operation
from fraymotifs.models import Title
from fraymotifs.service import create_fraymotif
from fraymotifs.utils import split_titles
from database.alchemy_database import get_item_by_code
from settings import CLASS_QUIZ_FILENAME, ASPECT_QUIZ_FILENAME
import json

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"http://localhost:\d+",  # come on, vite
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ClasspectRequest(BaseModel):
    personality: str


class AlchemizeRequest(BaseModel):
    item_one: str
    item_two: str
    operation: Operation


class FraymotifRequest(BaseModel):
    players: str
    memory: str
    additional_info: str


class_quiz_json = json.load(open(CLASS_QUIZ_FILENAME, "r"))
aspect_quiz_json = json.load(open(ASPECT_QUIZ_FILENAME, "r"))


@app.get("/")
async def root():
    return {"message": "PARADOX ENGINE: Status operational."}


@app.post("/classpect")
async def classpect(req: ClasspectRequest):
    try:
        result = await calculate_title(
            req.personality, class_quiz_json, aspect_quiz_json
        )
        return {"result": result}
    except Exception as e:
        logging.error(f"Error in /classpect: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alchemy/alchemize")
async def alchemize(req: AlchemizeRequest):
    try:
        combined_item = await alchemize_items(req.item_one, req.item_two, req.operation)
        return {
            "name": combined_item.name,
            "code": combined_item.code,
            "description": combined_item.description,
        }
    except Exception as e:
        logging.error(f"Error in /alchemy/alchemize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alchemy/captchalogue")
async def captchalogue(code: str):
    try:
        item = await get_item_by_code(code)
        if not item:
            logging.warning(f"Item not found for code: {code}")
            raise HTTPException(status_code=404, detail="Item not found")
        return {"name": item.name, "code": item.code, "description": item.description}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error in /alchemy/captchalogue: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fraymotif")
async def fraymotif(req: FraymotifRequest):
    try:
        classes, aspects = split_titles(req.players)
        if not classes or not aspects:
            logging.warning(f"Invalid titles in /fraymotif: {req.players}")
            raise HTTPException(
                status_code=400,
                detail="Valid titles not detected. Use the form 'Class of Aspect, Class of Aspect'.",
            )
        titles = [
            Title(title_class=cls, title_aspect=asp)
            for cls, asp in zip(classes, aspects)
        ]
        fraymotif = await create_fraymotif(titles, req.memory, req.additional_info)
        return {
            "visual_description": fraymotif.visual_description,
            "name": fraymotif.name,
            "mechanical_description": fraymotif.mechanical_description,
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error in /fraymotif: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
