document.addEventListener("DOMContentLoaded", () => {
    const chatWindow = document.getElementById("chat-window"); // 修改为聊天窗口
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");
    const recommendationsContainer = document.getElementById("recommendations");
    const loadingContainer = document.getElementById("loading-container");

    let isInRecommendationStage = false;

    // 添加消息到聊天窗口
    const appendMessage = (text, sender) => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.innerHTML = text;
        messagesContainer.appendChild(messageDiv);

        // 自动滚动到聊天窗口底部
        scrollToBottom();
    };

    // 自动滚动到聊天窗口底部
    const scrollToBottom = () => {
        setTimeout(() => {
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }, 0); // 确保 DOM 完全更新后执行
    };

    // 展示推荐的歌曲（图片和链接）
    const displayRecommendations = (recommendations) => {
        recommendationsContainer.innerHTML = ""; // 清空推荐容器

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

        // 自动滚动到聊天窗口底部
        scrollToBottom();
    };

    // 判断是否进入推荐阶段
    const isRecommendationStep = (userMessage) => {
        const normalizedMessage = userMessage.trim().toLowerCase();
        const isQuoteFormat = normalizedMessage.startsWith("'") && normalizedMessage.endsWith("'"); // 检查是否带单引号
        return normalizedMessage === "4" || normalizedMessage.includes(" by ") || isQuoteFormat;
    };

    // 发送用户消息
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
