from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import json, torch, random
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_folder="../frontend")
socketio = SocketIO(app, cors_allowed_origins="*")

# Trainingsdaten laden
with open("td.json", "r", encoding="utf-8") as f:
    daten = json.load(f)

fragen = daten["fragen"]
labels = daten["labels"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(fragen).toarray()
le = LabelEncoder()
y_train = le.fit_transform(labels)

# Modell
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

def train_model(X, y, epochs=50):
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(X_train_t, y_train_t, epochs=100)

antworten = {
    "greeting": ["Hallo! Wie kann ich dir helfen?", "Hi! Schön, dass du da bist!", "Hey!"],
    "smalltalk": ["Mir geht's gut! Und dir?", "Alles bestens!", "Danke der Nachfrage!"],
    "help": ["Natürlich! Wobei brauchst du Hilfe?", "Klar, frag mich einfach!", "Ich helfe dir gerne."],
    "bye": ["Tschüss! Bis bald!", "Auf Wiedersehen!", "Mach's gut!"]
}

def bag_of_words(text):
    return torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)

# SocketIO Events
@socketio.on("user_message")
def handle_message(data):
    msg = data["msg"]
    X = bag_of_words(msg)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = le.inverse_transform(predicted.detach().numpy())[0]

    if tag in antworten:
        socketio.emit("bot_response", {"msg": random.choice(antworten[tag])})
    else:
        socketio.emit("bot_response", {"msg": "Das habe ich noch nicht gelernt. Welche Kategorie passt dazu?"})

@socketio.on("user_category")
def handle_category(data):
    neue_label = data["category"]
    # Trainingsdaten erweitern, optional Modell trainieren
    socketio.emit("bot_response", {"msg": f"Neue Kategorie '{neue_label}' gespeichert!"})

# Frontend ausliefern
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
