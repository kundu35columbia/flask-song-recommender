document.addEventListener("DOMContentLoaded", () => {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");
    const recommendationsContainer = document.getElementById("recommendations"); // 新增推荐区域容器

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

    // 发送用户消息
    const sendMessage = () => {
        const userMessage = userInput.value.trim();
        if (userMessage === "") return; // 如果输入为空则不发送
        appendMessage(userMessage, "user");
        userInput.value = "";

        fetch(`/get?msg=${encodeURIComponent(userMessage)}`)
            .then(response => response.json())
            .then(data => {
                // 检查是否是推荐结果
                if (data.response.startsWith("Here are some songs you might like")) {
                    try {
                        // 解析推荐数据
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
