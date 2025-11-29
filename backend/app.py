from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import json
import random
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Trainingsdaten laden
# ------------------------------

with open("td.json", "r", encoding="utf-8") as f:
    daten = json.load(f)

fragen = daten["fragen"]
labels = daten["labels"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(fragen).toarray()

le = LabelEncoder()
y_train = le.fit_transform(labels)

input_size = X_train.shape[1]
hidden_size = 16
output_size = len(set(labels))

# ------------------------------
# Neuronales Netzwerk
# ------------------------------

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Modell erstellen
model = NeuralNet(input_size, hidden_size, output_size)

# Dummy-Training (damit das Modell initialisiert wird)
X_t = torch.tensor(X_train, dtype=torch.float32)
y_t = torch.tensor(y_train, dtype=torch.long)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for _ in range(50):
    output = model(X_t)
    loss = criterion(output, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ------------------------------
# Antworten
# ------------------------------

antworten = {
    "greeting": ["Hallo!", "Hi!", "Hey!"],
    "bye": ["Tschüss!", "Bis bald!"],
    "help": ["Wie kann ich dir helfen?"],
}

# ------------------------------
# Helper
# ------------------------------

def bag_of_words(text):
    return torch.tensor(
        vectorizer.transform([text]).toarray(), dtype=torch.float32
    )

# ------------------------------
# Flask App
# ------------------------------

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Chatbot läuft!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("msg", "")

    X = bag_of_words(msg)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = le.inverse_transform(predicted.detach().numpy())[0]

    if tag in antworten:
        antwort = random.choice(antworten[tag])
    else:
        antwort = "Das habe ich nicht gelernt."

    return jsonify({"response": antwort})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
