import warnings
from typing import List

from fastapi import FastAPI, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api_models import PredictionResponse, UIDesignPattern
from src.get_annotations import generate_annotations

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


app = FastAPI(
    title="Akin API",
    description=(
        """Web API for Akin, a UI wireframe generator that allows designers 
        to chose a UI design pattern and provides them with multiple 
        UI wireframes for a given UI design pattern."""
    ),
    version="1.0.0",
)

origins = [
    "https://akin.blackbox-toolkit.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def redirect_to_home():
    return RedirectResponse("/docs")


@app.post(
    "/generate",
    response_model=List[List[PredictionResponse]],
    status_code=status.HTTP_200_OK,
    response_description=(
        "Responds with a list of UI wireframe annotations"
        " with their location, and their prediction certainty in JSON format. This JSON file"
        " contains a list of UI element categories from the generated wireframe as JSON objects. Each JSON object"
        " contains predicted bounding box position (top left x,y coordinates) and its dimensions"
        " (width, height)"
    ),
    tags=["Generate UI wireframes"],
    description="Generate UI wireframe annotations for the given UI design pattern",
)
async def generate_wireframes(
    ui_design_pattern_type: UIDesignPattern = Query(
        ...,
        description="UI Design pattern type",
    ),
):

    response = generate_annotations(ui_design_pattern_type, sample_num=8)

    return response
