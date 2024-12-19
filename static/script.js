document.addEventListener("DOMContentLoaded", () => {
    const chatWindow = document.getElementById("chat-window"); //Change to chat window
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");
    const recommendationsContainer = document.getElementById("recommendations");
    const loadingContainer = document.getElementById("loading-container");

    let isInRecommendationStage = false;

    // Add a message to the chat window
    const appendMessage = (text, sender) => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.innerHTML = text;
        messagesContainer.appendChild(messageDiv);

        // Automatically scroll to the bottom of the chat window
        scrollToBottom();
    };

    // Automatically scroll to the bottom of the chat window
    const scrollToBottom = () => {
        setTimeout(() => {
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }, 0); // Ensure that the DOM is fully updated before executing
    };

    // Display recommended songs (pictures and links)
    const displayRecommendations = (recommendations) => {
        recommendationsContainer.innerHTML = ""; // Clear the recommendation container

        recommendations.forEach(song => {
            const songDiv = document.createElement("div");
            songDiv.classList.add("recommendation-item");

            songDiv.innerHTML = `
                <a href="${song.url}" target="_blank">
                    <img src="${song.image_url}" alt="${song.name}">
                </a>
                <a href="${song.url}" target="_blank">${song.name}</a>
            `;

            recommendationsContainer.appendChild(songDiv);
        });

        // Automatically scroll to the bottom of the chat window
        scrollToBottom();
    };

    // Determine whether to enter the recommendation stage
    const isRecommendationStep = (userMessage) => {
        const normalizedMessage = userMessage.trim().toLowerCase();
        const isQuoteFormat = normalizedMessage.startsWith("'") && normalizedMessage.endsWith("'"); // Check if there is a single quote
        return normalizedMessage === "4" || normalizedMessage.includes(" by ") || isQuoteFormat;
    };

    // Send user message
    const sendMessage = () => {
        const userMessage = userInput.value.trim();
        if (userMessage === "") return;
        appendMessage(userMessage, "user");
        userInput.value = "";

        if (isRecommendationStep(userMessage)) {
            isInRecommendationStage = true;
            loadingContainer.classList.remove("hidden");
        } else {
            isInRecommendationStage = false;
        }

        fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
            .then(response => response.json())
            .then(data => {
                if (isInRecommendationStage) {
                    loadingContainer.classList.add("hidden");

                    try {
                        const recommendations = JSON.parse(data.response.match(/{.+}/)[0]);
                        displayRecommendations(recommendations);
                        appendMessage("Here are some songs I recommend for you:", "bot");
                    } catch (error) {
                        console.error("Error parsing recommendations:", error);
                        appendMessage(data.response, "bot");
                    }
                } else {
                    appendMessage(data.response, "bot");
                }
            })
            .catch(error => {
                if (isInRecommendationStage) {
                    loadingContainer.classList.add("hidden");
                }

                appendMessage("An error occurred. Please try again.", "bot");
                console.error("Error:", error);
            });
    };

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });

    appendMessage(
        "Hello! I am your personal music recommendation chatbot. I can analyze your mood based on our chat in 3 dialogues and your favorite song. Now let's start! How are you doing today?",
        "bot"
    );
});
