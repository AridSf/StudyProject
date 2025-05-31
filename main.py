from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from services.get_days import get_days
import numpy as np
from model import load_model, forecast
from datetime import datetime

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model, scaler_X, scaler_y = load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    
    page_data = {
        "header": "Прогнозирование погоды с использованием больших данных",
    }
    
    with open("weather_30_days.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return templates.TemplateResponse("index.html", {"request": request, "all_data": data, "page_data": page_data})



@app.get("/app", response_class=HTMLResponse)
def app_root(request: Request):
    
    page_data = {
        "header": "Приложение",
    }
    return templates.TemplateResponse("app.html", {"request": request, "page_data": page_data, "forecast": None})

@app.post("/forecast", response_class=HTMLResponse)
async def post_forecast(request: Request):
    
    page_data = {
        "header": "Приложение",
    }

    # Данные о погоде за 7 дней
    recent_days = get_days()

    
    def date_to_day_of_year(date_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.timetuple().tm_yday
    
    
    input_data = [
        [d['humidity'], d['wind_speed'], date_to_day_of_year(d['date']), d['temperature']] for d in recent_days
    ]

    forecast_result = forecast(model, scaler_X, scaler_y, input_data)
    forecast_result = forecast_result.tolist()

    return templates.TemplateResponse("app.html", {"request": request, "page_data": page_data, "forecast": forecast_result})