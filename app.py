import os
import openai
import pandas as pd
from flask import Flask, render_template, request, jsonify
import requests
from recommendation import url_out, recommend_songs_with_main_logic

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Sentiment analysis using Hugging Face Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_API_KEY = os.getenv("HF_API_KEY")

def analyze_emotion(text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": text}, timeout=10)
        if response.status_code == 200:
            results = response.json()
            print("Emotion API Response:", results) 
            
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0] 


            if isinstance(results, list) and all("label" in r and "score" in r for r in results):

                detected_emotion = max(results, key=lambda x: x["score"])["label"]
                return detected_emotion.lower()
            else:
                print("Unexpected API response format:", results)
                return "neutral" 
        else:
            print(f"Error: {response.text}")
            return "neutral"  
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "neutral" 

# Load song data
song_data = pd.read_csv("song_data.csv")
song_data["emotion"] = song_data["emotion"].str.lower()

# Chat history and status
chat_history = []
user_inputs = []
conversation_stage = 0
emotion_detected = None
selected_option = None

def generate_gpt_response(user_input):
    """调用 GPT 生成对话回复"""
    global chat_history

    # Build message history
    messages = [{"role": "system", "content": "You are a friendly and engaging music recommendation chatbot. All your responses should relate to music, moods, or songs."}]
    messages += [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    messages.append({"role": "user", "content": user_input})

    # Calling OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    bot_response = response.choices[0].message.content

    # Update chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_response})

    return bot_response

def generate_response(user_input):
    """生成聊天回复或推荐歌曲"""
    global chat_history, user_inputs, conversation_stage, emotion_detected, selected_option

    # Recording user input
    user_inputs.append(user_input)

    # If the number of conversations reaches 3, enter the sentiment analysis phase
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

    # User selects recommended options
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

                    response = "Here are some songs you might like:<br>"
                    for song in recommendations:
                        response += f"""
                        <div style="text-align: center; margin-bottom: 20px;">
                            <a href="{song['url']}" target="_blank">
                                <img src="{song['image_url']}" alt="{song['name']}" style="width: 150px; height: 150px; border-radius: 8px;">
                            </a>
                            <br>
                            <a href="{song['url']}" target="_blank">{song['name']}</a>
                        </div>
                        """
                    return response
                except Exception as e:
                    print(f"Error during recommendation: {e}")
                    return f"An error occurred while processing your request: {e}"
        else:
            return "Please enter a valid option number (1, 2, 3, or 4)."

    # In the song recommendation stage, continue processing based on user input
    if conversation_stage == 2 and selected_option in ["1", "2", "3"]:
        song_name, artist_name = None, None  # Initialize variables

        # Parse user input based on options
        if selected_option == "1" and "by" in user_input.lower():
            song_artist_split = user_input.lower().split("by")
            song_name = song_artist_split[0].strip()
            artist_name = song_artist_split[1].strip()
        elif selected_option == "2":
            song_name = user_input.strip()
        elif selected_option == "3":
            artist_name = user_input.strip()
        else:
            return "Invalid format. Please try again."

        if not song_name and not artist_name:
            return "Invalid input. Please provide a valid song or artist name."

        # Make song recommendations
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

            # Build HTML with images and links
            response = "Here are some songs you might like:<br>"
            for song in recommendations:
                response += f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <a href="{song['url']}" target="_blank">
                        <img src="{song['image_url']}" alt="{song['name']}" style="width: 150px; height: 150px; border-radius: 8px;">
                    </a>
                    <br>
                    <a href="{song['url']}" target="_blank">{song['name']}</a>
                </div>
                """
            return response
        except Exception as e:
            print(f"Error during recommendation: {e}")
            return f"An error occurred while processing your request: {e}"

    return generate_gpt_response(user_input)

@app.route("/")
def home():
    """渲染主页面"""
    global chat_history, user_inputs, conversation_stage, emotion_detected, selected_option
    # Initialize the conversation
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
