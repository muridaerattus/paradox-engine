from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paradox_engine import calculate_title
from alchemy.service import alchemize_items
from alchemy.models import Operation
from fraymotifs.models import Title
from fraymotifs.service import create_fraymotif
from fraymotifs.utils import split_titles
from database.alchemy_database import get_item_by_code
from settings import CLASS_QUIZ_FILENAME, ASPECT_QUIZ_FILENAME
import json

app = FastAPI()

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

class_quiz_json = json.load(open(CLASS_QUIZ_FILENAME, 'r'))
aspect_quiz_json = json.load(open(ASPECT_QUIZ_FILENAME, 'r'))

@app.post("/classpect")
def classpect(req: ClasspectRequest):
    try:
        result = calculate_title(req.personality, class_quiz_json, aspect_quiz_json)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alchemy/alchemize")
def alchemize(req: AlchemizeRequest):
    try:
        combined_item = alchemize_items(req.item_one, req.item_two, req.operation)
        return {
            "name": combined_item.name,
            "code": combined_item.code,
            "description": combined_item.description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alchemy/captchalogue")
def captchalogue(code: str):
    try:
        item = get_item_by_code(code)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return {
            "name": item.name,
            "code": item.code,
            "description": item.description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fraymotif")
def fraymotif(req: FraymotifRequest):
    try:
        classes, aspects = split_titles(req.players)
        if not classes or not aspects:
            raise HTTPException(status_code=400, detail="Valid titles not detected. Use the form 'Class of Aspect, Class of Aspect'.")
        titles = [Title(title_class=cls, title_aspect=asp) for cls, asp in zip(classes, aspects)]
        fraymotif = create_fraymotif(titles, req.memory, req.additional_info)
        return {
            "visual_description": fraymotif.visual_description,
            "name": fraymotif.name,
            "mechanical_description": fraymotif.mechanical_description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
