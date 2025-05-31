import json

# Чтение данных из исходного файла
with open('weather_30_days.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

file_data = []

for i in data:
    
    for j in i['forecast']['forecastday']:
        file_data.append({ "date":j['date'], "humidity":j['day']['avghumidity'], "wind_speed":j['day']['maxwind_kph'], "temperature":j['day']['avgtemp_c'] })




# Сохранение данных в новый файл
with open('output.json', 'w', encoding='utf-8') as outfile:
    json.dump(file_data, outfile, ensure_ascii=False, indent=4)



