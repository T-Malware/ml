async function sendMessage() {
    const msgInput = document.getElementById("msg");
    const msg = msgInput.value;
    msgInput.value = "";

    const chatbox = document.getElementById("chatbox");
    chatbox.innerHTML += "<p><b>Du:</b> " + msg + "</p>";

    const res = await fetch(`/chat?msg=${encodeURIComponent(msg)}`);
    const data = await res.json();

    chatbox.innerHTML += "<p><b>Bot:</b> " + data.response + "</p>";
}
