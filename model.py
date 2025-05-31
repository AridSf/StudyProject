import torch
import numpy as np
import joblib
from torch import nn


class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_days, num_layers=2):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_days)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)


# Константы — должны совпадать с обучающей моделью
INPUT_SIZE = 4
HIDDEN_SIZE = 128
OUTPUT_DAYS = 7
INPUT_DAYS = 7
NUM_LAYERS = 2

# Загрузка модели и масштабировщиков
def load_model():
    model = WeatherLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_DAYS, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load("weather_model.pt", map_location=torch.device("cpu")))
    model.eval()
    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")
    return model, scaler_X, scaler_y

# Прогнозирование
def forecast(model, scaler_X, scaler_y, recent_days):
    recent_days = np.array(recent_days)  # <- добавьте эту строку
    assert recent_days.shape == (INPUT_DAYS, recent_days.shape[1]), f"Неверная форма recent_days: {recent_days.shape}"

    scaled = scaler_X.transform(recent_days)
    input_tensor = torch.tensor(scaled.reshape(1, INPUT_DAYS, -1), dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
        forecast = scaler_y.inverse_transform(pred_scaled)

    return forecast[0]
