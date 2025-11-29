from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import torch
import json
from bot import model, le, vectorizer, antworten, train_model, X_train_t, y_train_t

# FastAPI Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class Message(BaseModel):
    msg: str

# Endpoint
@app.post("/chat")
def chat(message: Message):
    text = message.msg.lower()
    
    # Bag-of-Words
    X = torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = le.inverse_transform(predicted.detach().numpy())[0]
    
    # Antwort
    if tag in antworten:
        return {"reply": random.choice(antworten[tag])}
    else:
        return {"reply": "Ich kenne das noch nicht. Kannst du mir sagen, wie ich darauf antworten soll?"}
