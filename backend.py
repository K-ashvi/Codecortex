from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import os

app = FastAPI()

# Allow frontend to access the backend
origins = [
    "http://localhost",
    "http://127.0.0.1:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve front2.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_frontend():
    # Serve the front2.html file
    with open("static/front2.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/predict")
def predict(stock_name: str = Query(..., description="Stock ticker symbol")):
    end_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = yf.download(stock_name, start='2010-01-01', end=end_date)

    if not stock_data.empty:
        # Preprocessing
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        train_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_len]
        x_train, y_train = [], []

        for i in range(60, len(train_data) - 7):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i+7, 0])

        x_train = np.array(x_train).reshape(-1, 60, 1)
        y_train = np.array(y_train)

        # Model training
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=64, epochs=10)

        test_data = scaled_data[train_len - 60:]
        x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
        x_test = np.array(x_test).reshape(-1, 60, 1)

        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        actual_prices = close_prices[train_len:]
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

        x_future = test_data[-60:].reshape(1, 60, 1)
        future_predictions = []
        for _ in range(7):
            pred = model.predict(x_future)
            future_predictions.append(pred[0, 0])
            x_future = np.append(x_future[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]

        min_future_price = future_predictions.min()
        max_future_price = future_predictions.max()
        min_future_date = future_dates[np.argmin(future_predictions)]
        max_future_date = future_dates[np.argmax(future_predictions)]

        currency_symbol = "₹" if stock_name.endswith(".NS") else "$"

        return {
            "predicted_prices": predicted_prices.tolist(),
            "future_predictions": future_predictions.tolist(),
            "min_future_price": f"{currency_symbol}{min_future_price:.2f}",
            "min_future_date": str(min_future_date.date()),
            "max_future_price": f"{currency_symbol}{max_future_price:.2f}",
            "max_future_date": str(max_future_date.date()),
            "rmse": f"{rmse:.2f}"
        }
    else:
        return {"error": f"Could not retrieve data for stock: {stock_name}"}
