import requests
from datetime import datetime, timedelta


def get_days():
    api_key = "643d822c4ddf4055a8522522251305"
    city = "Izhevsk"
    url = "http://api.weatherapi.com/v1/history.json"

    # Сколько дней брать назад
    days = 7
    today = datetime.today()

    # Сюда буду сохранять все данные
    all_data = []
    file_data = []

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
        
    for i in all_data:
        for j in i['forecast']['forecastday']:
            file_data.append({ "date":j['date'], "humidity":j['day']['avghumidity'], "wind_speed":j['day']['maxwind_kph'], "temperature":j['day']['avgtemp_c'] })
    return file_data