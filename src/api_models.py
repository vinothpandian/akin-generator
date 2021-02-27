from enum import Enum

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


class UIDesignPattern(Enum):
    login = "login"
    account_creation = "account_creation"
    product_listing = "product_listing"
    product_description = "product_description"
    splash = "splash"
