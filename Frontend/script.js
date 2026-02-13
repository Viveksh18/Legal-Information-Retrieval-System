const input = document.getElementById("userInput");
const chatBox = document.getElementById("chatBox");
const sendBtn = document.getElementById("sendBtn");

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", e => {
  if (e.key === "Enter") sendMessage();
});

function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

function formatAnswer(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- (.*)$/gm, "• $1")
    .replace(/\n\n/g, "<br><br>")
    .replace(/\n/g, "<br>");
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  const typing = addMessage("Thinking...", "bot typing");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text })
    });

    const data = await res.json();
    typing.remove();

    if (!data.answer) {
      typeEffect("No response from server.", "bot");
    } else {
      typeEffect(data.answer, "bot");
    }

  } catch (err) {
    typing.remove();
    addMessage("⚠️ Server error. Please try again.", "bot");
  }
}

function typeEffect(text, type) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  chatBox.appendChild(div);

  const formatted = formatAnswer(text);
  let i = 0;

  const interval = setInterval(() => {
    div.innerHTML = formatted.substring(0, i);
    i++;
    chatBox.scrollTop = chatBox.scrollHeight;
    if (i >= formatted.length) clearInterval(interval);
  }, 5);
}
