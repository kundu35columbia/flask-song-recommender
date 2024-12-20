# Music Recommendation Chatbot

This project is a **music recommendation chatbot** that provides personalized song suggestions based on user input and emotional context. The system leverages state-of-the-art models like **EmoRoBERTa** for emotion analysis and integrates item-based collaborative filtering to generate accurate and diverse music recommendations. The chatbot is deployed and accessible at **[flask-song-recommender.onrender.com](https://flask-song-recommender.onrender.com)**.

---

## Features

- **Emotion-Based Recommendations**:  
  Utilizes EmoRoBERTa to analyze user dialogues and infer emotional states (e.g., joy, sadness, anger), aligning recommendations with the user’s mood.

- **User Preference Filtering**:  
  Allows users to provide a favorite song or artist to personalize recommendations. The system also supports filtering by emotion alone when no preferences are provided.

- **Item-Based Collaborative Filtering**:  
  Recommends similar songs by analyzing audio features such as danceability, energy, and tempo, ensuring both personalization and diversity.

- **Dynamic and Scalable Deployment**:  
  The chatbot is hosted on **Render**, ensuring robust scalability and low latency for real-time user interactions.

---

## How It Works

1. **User Interaction**:  
   Users interact with the chatbot through a web interface, providing inputs such as mood, song, or artist preferences.

2. **Emotion Analysis**:  
   User conversations are analyzed using EmoRoBERTa to determine emotional states.

3. **Filtering and Recommendations**:  
   - If a song or artist is provided, the system anchors the recommendations around this input.
   - If no preferences are given, emotion-based filtering drives the recommendations.
   - The final output consists of five songs: one matching the user's input and four similar tracks.

---

## Deployment

The project is deployed and live at:  
**[https://flask-song-recommender.onrender.com](https://flask-song-recommender.onrender.com)**

---

## Technologies Used

- **Backend**: Flask
- **Machine Learning**:
  - EmoRoBERTa (Emotion Detection)
  - Collaborative Filtering
- **APIs**:
  - Hugging Face API (Emotion Detection)
  - Spotify API (Song Metadata)
- **Hosting**: Render

---

## Getting Started Locally

### Prerequisites
- Python 3.7+
- Virtual Environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
