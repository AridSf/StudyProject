import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Константы
INPUT_DAYS = 7
FORECAST_DAYS = 7
BATCH_SIZE = 64
EPOCHS = 1000
HIDDEN_SIZE = 128

# Загрузка и подготовка данных из JSON
def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['day_of_year'] = df['date'].dt.dayofyear
    df.drop(columns=['date'], inplace=True)
    features = ['humidity', 'wind_speed', 'day_of_year', 'temperature']
    data = df[features].values

    X, y = [], []
    for i in range(len(data) - INPUT_DAYS - FORECAST_DAYS):
        X.append(data[i:i+INPUT_DAYS])
        y.append(data[i+INPUT_DAYS:i+INPUT_DAYS+FORECAST_DAYS, -1])  # температура

    return np.array(X), np.array(y)

# Масштабирование и подготовка
def preprocess(X, y):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_flat = X.reshape(-1, X.shape[2])
    X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            scaler_X,
            scaler_y)

# Модель 
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_days, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_days)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)


# Обучение
def train_model(model, loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# Сохранение и загрузка
def save_model(model, scaler_X, scaler_y, model_path='weather_model.pt'):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_X, 'scaler_X.save')
    joblib.dump(scaler_y, 'scaler_y.save')

def load_model(input_size, hidden_size, output_days, model_path='weather_model.pt'):
    model = WeatherLSTM(input_size, hidden_size, output_days)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler_X = joblib.load('scaler_X.save')
    scaler_y = joblib.load('scaler_y.save')
    return model, scaler_X, scaler_y

# Прогноз
def forecast_next_days(model, scaler_X, scaler_y, recent_days):
    scaled = scaler_X.transform(recent_days)
    input_tensor = torch.tensor(scaled.reshape(1, INPUT_DAYS, -1), dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
        forecast = scaler_y.inverse_transform(pred_scaled)
    return forecast[0]

# Главная логика
def main():
    data_path = "output.json"
    if not os.path.exists(data_path):
        print(f"Файл {data_path} не найден.")
        return

    X, y = load_json_data(data_path)
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = preprocess(X, y)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = WeatherLSTM(input_size=X.shape[2], hidden_size=HIDDEN_SIZE, output_days=FORECAST_DAYS)
    train_model(model, loader, epochs=EPOCHS)

    save_model(model, scaler_X, scaler_y)

    # Используем последние 7 дней
    with open(data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    df_recent = pd.DataFrame(json_data).tail(INPUT_DAYS).copy()
    df_recent['date'] = pd.to_datetime(df_recent['date'])
    df_recent['day_of_year'] = df_recent['date'].dt.dayofyear
    recent_input = df_recent[['humidity', 'wind_speed', 'day_of_year', 'temperature']].values

    model, scaler_X, scaler_y = load_model(
    input_size=4,
    hidden_size=HIDDEN_SIZE,
    output_days=FORECAST_DAYS
    )
    forecast = forecast_next_days(model, scaler_X, scaler_y, recent_input)

    print(f"\n📈 Прогноз температуры на следующие {INPUT_DAYS} дней:")
    for i, temp in enumerate(forecast, 1):
        print(f"День {i}: {temp:.2f} °C")

if __name__ == "__main__":
    main()
