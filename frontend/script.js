async function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const message = input.value;
    if(!message) return;

    chatBox.innerHTML += `<div class="user-msg"><b>Du:</b> ${message}</div>`;
    input.value = "";

    const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({msg: message})
    });
    const data = await response.json();
    chatBox.innerHTML += `<div class="bot-msg"><b>Bot:</b> ${data.reply}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
}
