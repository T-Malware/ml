import json
import random
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trainingsdaten
with open("td.json", "r", encoding="utf-8") as f:
    daten = json.load(f)

fragen = daten["fragen"]
labels = daten["labels"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(fragen).toarray()

le = LabelEncoder()
y_train = le.fit_transform(labels)

# Neural Network
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(set(labels))

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

def train_model(X, y, epochs=50):
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(X_train_t, y_train_t, epochs=100)

# Antworten
antworten = {
    "greeting": ["Hallo! Wie kann ich dir helfen?", "Hi! Schön, dass du da bist!", "Hey!"],
    "smalltalk": ["Mir geht's gut! Und dir?", "Alles bestens!", "Danke der Nachfrage!"],
    "help": ["Natürlich! Wobei brauchst du Hilfe?", "Klar, frag mich einfach!", "Ich helfe dir gerne."],
    "bye": ["Tschüss! Bis bald!", "Auf Wiedersehen!", "Mach's gut!"],
}

def bag_of_words(text):
    return torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)

@app.get("/")
def get_index():
    return FileResponse("../frontend/index.html")

@app.get("/chat")
def chat(msg: str):
    X = bag_of_words(msg)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = le.inverse_transform(predicted.detach().numpy())[0]
    if tag in antworten:
        return {"response": random.choice(antworten[tag])}
    else:
        return {"response": "Das habe ich noch nicht gelernt."}
