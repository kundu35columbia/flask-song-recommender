import os
import openai
import pandas as pd
from flask import Flask, render_template, request, jsonify
import requests
from recommendation import url_out, recommend_songs_with_main_logic

# 初始化 Flask 应用
app = Flask(__name__)

# 设置 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 使用 Hugging Face Inference API 进行情感分析
HF_API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/bert-base-uncased-emotion"
HF_API_KEY = os.getenv("HF_API_KEY")

def analyze_emotion(text):
    """调用 Hugging Face API 进行情感分析"""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        results = response.json()
        return results[0][0]["label"].lower()
    else:
        print(f"Error: {response.text}")
        return "neutral"

# 加载歌曲数据
song_data = pd.read_csv("song_data.csv")
song_data["emotion"] = song_data["emotion"].str.lower()

# 聊天记录和状态
chat_history = []
user_inputs = []
conversation_stage = 0
emotion_detected = None
selected_option = None

def generate_gpt_response(user_input):
    """调用 GPT 生成对话回复"""
    global chat_history

    # 构建消息历史
    messages = [{"role": "system", "content": "You are a friendly and engaging chatbot for natural conversations."}]
    messages += [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    messages.append({"role": "user", "content": user_input})

    # 调用 OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    bot_response = response.choices[0].message.content

    # 更新聊天历史
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_response})

    return bot_response

def generate_response(user_input):
    """生成聊天回复或推荐歌曲"""
    global chat_history, user_inputs, conversation_stage, emotion_detected, selected_option

    # 记录用户输入
    user_inputs.append(user_input)

    # 如果对话数达到 3，进入情绪分析阶段
    if len(user_inputs) == 3 and conversation_stage == 1:
        conversation_stage += 1
        emotion_detected = analyze_emotion(" ".join(user_inputs))
        return (
            "It seems like your mood is " + emotion_detected + ". Please choose an option for recommending songs:<br>"
            "1. Provide 'song name by artist name'<br>"
            "2. Provide only 'song name'<br>"
            "3. Provide only 'artist name'<br>"
            "4. None (use detected mood to recommend songs)<br>"
            "Please enter the option number (1, 2, 3, or 4)."
        )

    # 用户选择推荐选项
    if conversation_stage == 2 and not selected_option:
        if user_input in ["1", "2", "3", "4"]:
            selected_option = user_input
            if selected_option == "1":
                return "Great! Please provide your favorite song in the format 'song name by artist name'."
            elif selected_option == "2":
                return "Got it! Please provide the 'song name'."
            elif selected_option == "3":
                return "Okay! Please provide the 'artist name'."
            elif selected_option == "4":
                try:
                    recommendations = url_out(
                        recommend_songs_with_main_logic,
                        data=song_data,
                        emotion=emotion_detected,
                        genre=None,
                        n_recommendations=5
                    )
                    if not recommendations:
                        return "Sorry, I couldn't find any recommendations based on your mood."

                    response = "Here are some songs you might like based on your mood:<br>"
                    for song in recommendations:
                        response += f'<a href="{song["url"]}" target="_blank">{song["name"]}</a><br>'
                    return response
                except Exception as e:
                    print(f"Error during recommendation: {e}")
                    return f"An error occurred while processing your request: {e}"
        else:
            return "Please enter a valid option number (1, 2, 3, or 4)."

    # 在歌曲推荐阶段，根据用户输入继续处理
    if conversation_stage == 2 and selected_option in ["1", "2", "3"]:
        if selected_option == "1" and "by" in user_input.lower():
            song_artist_split = user_input.lower().split("by")
            song_name = song_artist_split[0].strip()
            artist_name = song_artist_split[1].strip()
        elif selected_option == "2":
            song_name = user_input.strip()
            artist_name = None
        elif selected_option == "3":
            song_name = None
            artist_name = user_input.strip()
        else:
            return "Invalid format. Please try again."

        try:
            recommendations = url_out(
                recommend_songs_with_main_logic,
                data=song_data,
                emotion=emotion_detected,
                genre=None,
                song_name=song_name,
                artist_name=artist_name,
                n_recommendations=5
            )
            if not recommendations:
                return "Sorry, I couldn't find any recommendations based on your input."

            response = "Here are some songs you might like:<br>"
            for song in recommendations:
                response += f'<a href="{song["url"]}" target="_blank">{song["name"]}</a><br>'
            return response
        except Exception as e:
            print(f"Error during recommendation: {e}")
            return f"An error occurred while processing your request: {e}"

    return generate_gpt_response(user_input)

@app.route("/")
def home():
    """渲染主页面"""
    global chat_history, user_inputs, conversation_stage, emotion_detected, selected_option
    # 初始化对话
    chat_history = [{"role": "assistant", "content": "How are you doing today?"}]
    user_inputs = []
    conversation_stage = 1
    emotion_detected = None
    selected_option = None
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def chatbot_response():
    """处理用户输入"""
    user_message = request.args.get("msg")
    bot_response = generate_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
