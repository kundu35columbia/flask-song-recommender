document.addEventListener("DOMContentLoaded", () => {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");

    const appendMessage = (text, sender) => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.innerHTML = text;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    };

    const sendMessage = () => {
        const userMessage = userInput.value.trim();
        if (userMessage === "") return;
        appendMessage(userMessage, "user");
        userInput.value = "";

        fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
            .then(response => response.json())
            .then(data => {
                appendMessage(data.response, "bot");
            })
            .catch(error => {
                appendMessage("An error occurred. Please try again.", "bot");
                console.error("Error:", error);
            });
    };

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });

    // 初始化时添加完整开场白，但只计入后半部分到聊天记录
    appendMessage("Hello! I am your personal music recommendation chatbot. I can analyze your mood based on our chat in 3 dialogues and your favorite song. Now let's start! How are you doing today?", "bot");
});
