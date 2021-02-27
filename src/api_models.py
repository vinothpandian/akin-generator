from enum import Enum, IntEnum

from pydantic import BaseModel


class PositionSchema(BaseModel):
    x: int
    y: int


class DimensionSchema(BaseModel):
    width: int
    height: int


class PredictionResponse(BaseModel):
    name: str
    position: PositionSchema
    dimension: DimensionSchema


class UIDesignPattern(IntEnum, Enum):
    login = 1
    account_creation = 2
    product_listing = 3
    product_description = 4
    splash = 5
