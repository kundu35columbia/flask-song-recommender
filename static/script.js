document.addEventListener("DOMContentLoaded", () => {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");
    const recommendationsContainer = document.getElementById("recommendations");
    const loadingContainer = document.getElementById("loading-container");

    let isInRecommendationStage = false; // 是否处于推荐阶段

    // 添加消息到聊天窗口
    const appendMessage = (text, sender) => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.innerHTML = text;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
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
        if (userMessage === "") return; // 如果输入为空则不发送
        appendMessage(userMessage, "user");
        userInput.value = "";

        // 判断是否进入推荐阶段
        if (isRecommendationStep(userMessage)) {
            isInRecommendationStage = true; // 标记进入推荐阶段
            loadingContainer.classList.remove("hidden"); // 显示 loading 动画
        } else {
            isInRecommendationStage = false;
        }

        fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
            .then(response => response.json())
            .then(data => {
                if (isInRecommendationStage) {
                    // 隐藏 loading 动画
                    loadingContainer.classList.add("hidden");

                    // 展示推荐内容
                    try {
                        const recommendations = JSON.parse(data.response.match(/{.+}/)[0]);
                        displayRecommendations(recommendations);
                        appendMessage("Here are some songs I recommend for you:", "bot");
                    } catch (error) {
                        console.error("Error parsing recommendations:", error);
                        appendMessage(data.response, "bot");
                    }
                } else {
                    appendMessage(data.response, "bot"); // 普通对话阶段直接显示回复
                }
            })
            .catch(error => {
                // 如果发生错误，隐藏 loading 动画
                if (isInRecommendationStage) {
                    loadingContainer.classList.add("hidden");
                }

                appendMessage("An error occurred. Please try again.", "bot");
                console.error("Error:", error);
            });
    };

    // 按钮和键盘事件绑定
    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });

    // 初始化时添加开场白
    appendMessage(
        "Hello! I am your personal music recommendation chatbot. I can analyze your mood based on our chat in 3 dialogues and your favorite song. Now let's start! How are you doing today?",
        "bot"
    );
});
