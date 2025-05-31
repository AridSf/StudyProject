import requests
import json
from datetime import datetime, timedelta

api_key = "e263a83db5904ec2bcc124553252905"
city = "Izhevsk"
url = "http://api.weatherapi.com/v1/history.json"

# Сколько дней брать назад
days = 730
today = datetime.today()

# Сюда буду сохранять все данные
all_data = []

for i in range(days):
    date = (today - timedelta(days=i)).strftime("%Y-%m-%d")

    params = {
        "key": api_key,
        "q": city,
        "dt": date,
        "lang": "ru",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        all_data.append(data)

        print(f"Данные за {date} получены")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении данных за {date}:", e)

# Сохраняем все данные в один файл
with open("weather_30_days.json", "w", encoding="utf-8") as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)

print("Все данные сохранены в файл weather_30_days.json")