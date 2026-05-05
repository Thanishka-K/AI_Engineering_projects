from fastapi import FastAPI
from pydantic import BaseModel
from ml_logic import predict_sentiment

app = FastAPI(title="Sentiment Analysis API")

class TextData(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "NLP API is Online"}

@app.post("/predict")
def analyze_text(data: TextData):
    sentiment = predict_sentiment(data.text)
    return {
        "input_text": data.text,
        "sentiment": sentiment
    }
