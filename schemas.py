from pydantic import BaseModel
from typing import List

class DayData(BaseModel):
    humidity: float
    wind_speed: float
    day_of_year: int
    temperature: float

class ForecastRequest(BaseModel):
    recent_days: List[DayData]
