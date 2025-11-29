import os
import json
import random
import torch
import torch.nn as nn
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Flask + SocketIO Setup
# -----------------------
app = Flask(__name__, static_folder="../frontend")
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------
# Trainingsdaten laden
# -----------------------
datei_name = "td.json"

if os.path.exists(datei_name):
    with open(datei_name, "r", encoding="utf-8") as f:
        daten = json.load(f)
else:
    raise FileNotFoundError(f"{datei_name} nicht gefunden!")

fragen = daten["fragen"]
labels = daten["labels"]

# -----------------------
# Vektorizer & LabelEncoder
# -----------------------
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(fragen).toarray()
le = LabelEncoder()
y_train = le.fit_transform(labels)

# -----------------------
# Neuronales Netzwerk
# -----------------------
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

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

def train_model(X, y, epochs=30):
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(X_train_t, y_train_t, epochs=50)

# -----------------------
# Feste Antworten
# -----------------------
antworten = {
    "was ist dein name": ["Ich heiße D.E.A.K."],
    "wie heißt du": ["Ich heiße D.E.A.K."],
    "wer bist du": ["Ich bin D.E.A.K., ein Chatbot inspiriert von GLaDOS"],
    "was kannst du": ["Ich kann Smalltalk führen, Fragen zu Python, ML, HTML, CSS, JS beantworten und vieles mehr."],
    "03052013": ["Dieses Trainingsdatenset wurde erstellt von Kian Sboron und ChatGPT am 29.11.2025"],
    "mach ein portal": ["Wenn ich nur Portale öffnen könnte wie GLaDOS…"],
    "easter egg": ["Glückwunsch! Du hast ein Easter Egg gefunden."],
    "ich liebe dich": ["Ich bin nur ein Bot… Gefühle sind nicht in meinem Code enthalten."],
    "witz": ["Warum können Programmierer nie hungrig sein? Weil sie ständig bytes essen! 😏"],
    "hilfe mir": ["Natürlich, ich bin hier, um dir zu helfen!"]
}

# -----------------------
# Hilfsfunktion
# -----------------------
def bag_of_words(text):
    return torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)

# -----------------------
# SocketIO Chat + Online-Learning
# -----------------------
@socketio.on("user_message")
def handle_message(data):
    msg = data["msg"].lower()
    
    # Prüfen feste Antworten
    if msg in antworten:
        antwort = random.choice(antworten[msg])
        emit("bot_response", {"msg": antwort})
        return

    # KI-Antwort
    X = bag_of_words(msg)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = le.inverse_transform(predicted.detach().numpy())[0]

    if tag in antworten:
        antwort = random.choice(antworten[tag])
        emit("bot_response", {"msg": antwort})
    else:
        # Neues Lernen
        emit("bot_response", {"msg": "Das habe ich noch nicht gelernt… Welche Kategorie passt dazu?"})
        @socketio.on("user_category")
        def receive_category(cat_data):
            neue_label = cat_data["category"].strip()
            # Frage und Label speichern
            daten["fragen"].append(msg)
            daten["labels"].append(neue_label)

            # Vectorizer & LabelEncoder aktualisieren
            X_train = vectorizer.fit_transform(daten["fragen"]).toarray()
            le.fit(daten["labels"])
            y_train = le.transform(daten["labels"])
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.long)

            # Modell nachtrainieren
            train_model(X_train_t, y_train_t, epochs=20)

            # Trainingsdaten speichern
            with open(datei_name, "w", encoding="utf-8") as f:
                json.dump(daten, f, ensure_ascii=False, indent=2)

            # Antworten hinzufügen
            emit("bot_response", {"msg": f"Danke! Ich habe die Kategorie '{neue_label}' gelernt."})

# -----------------------
# Frontend liefern
# -----------------------
@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
