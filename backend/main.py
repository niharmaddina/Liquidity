# backend/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from data_processing import get_historical_data, latest_feature_row
from model_inference import load_model, load_scalers, make_prediction

app = FastAPI()

# Allow CORS for frontend
origins = [
    "http://localhost:3000",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model('lnn_ode_model.pth')
scalers = load_scalers('scalers.joblib')

# Global variables
symbol = 'AAPL'  # Default stock symbol
interval = 60     # Interval in seconds for real-time updates

@app.get('/historical-data')
async def historical_data(symbol: str = symbol):
    data = get_historical_data(symbol)
    return data.to_dict(orient='records')

@app.get('/predict')
async def predict(symbol: str = symbol):
    prediction = make_prediction(model, latest_feature_row(symbol), scalers)
    return {'symbol': symbol, 'predicted_next_close': prediction}

@app.websocket('/ws/predictions')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            prediction = make_prediction(model, latest_feature_row(symbol), scalers)
            await websocket.send_json({'symbol': symbol, 'predicted_next_close': prediction})
            await asyncio.sleep(interval)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(interval)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
