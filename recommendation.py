import pandas as pd
import requests
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import numpy as np
from fuzzywuzzy import process

# 加载数据
song_data = pd.read_csv("song_data.csv")
song_data["emotion"] = song_data["emotion"].str.lower()

def get_song(track_id):
    """从 Spotify API 获取歌曲详细信息"""
    logs = []  # 用于记录日志
    try:
        client_id = "1f7eb600cb0640cd924df4fe69647d69"
        client_secret = "6ac066e8bcbc4c6ea6ab0b72c7ca68bf"

        # 编码凭证
        credentials = f"{client_id}:{client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        logs.append("Spotify API credentials encoded.")

        # 获取访问令牌
        auth_response = requests.post(
            'https://accounts.spotify.com/api/token',
            headers={
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            data={'grant_type': 'client_credentials'}
        )
        if auth_response.status_code != 200:
            logs.append(f"Spotify API authentication failed: {auth_response.text}")
            raise Exception("Failed to authenticate with Spotify API")

        access_token = auth_response.json()['access_token']
        logs.append("Spotify API token retrieved.")

        # 获取歌曲详细信息
        headers = {'Authorization': f'Bearer {access_token}'}
        track_response = requests.get(f'https://api.spotify.com/v1/tracks/{track_id}', headers=headers)
        track_data = track_response.json()

        # 检查 API 响应是否包含必要字段
        if "external_urls" in track_data and "spotify" in track_data["external_urls"]:
            logs.append(f"Song details fetched successfully for track_id: {track_id}")
            return {
                "name": track_data["name"],
                "url": track_data["external_urls"]["spotify"],
                "image_url": track_data['album']['images'][-1]['url']
            }, logs
        else:
            logs.append(f"Invalid track data for track_id: {track_id}.")
            return None, logs
    except Exception as e:
        logs.append(f"Error in get_song: {str(e)}")
        return None, logs
    
def recommend_songs_with_autoencoder(data, emotion, genre=None, n_recommendations=5):
    """
    使用 Autoencoder 推荐歌曲。
    """
    # 筛选符合条件的歌曲
    filtered_data = data[data['emotion'] == emotion]
    if genre:
        filtered_data = filtered_data[filtered_data['genre'] == genre]

    if filtered_data.empty:
        print("No songs found matching the given criteria.")
        return pd.DataFrame()

    # 提取特征列
    feature_columns = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
    ]
    features = filtered_data[feature_columns]

    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 定义 Autoencoder 模型
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    # 设置模型参数
    input_dim = scaled_features.shape[1]
    encoding_dim = 5  # 压缩到 5 维
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 转换数据为 PyTorch 张量
    scaled_features_tensor = torch.FloatTensor(scaled_features)
    dataset = torch.utils.data.TensorDataset(scaled_features_tensor, scaled_features_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练 Autoencoder
    epochs = 50
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, _ = batch
            optimizer.zero_grad()
            _, decoded = model(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()

    # 获取压缩表示
    model.eval()
    with torch.no_grad():
        compressed_features, _ = model(scaled_features_tensor)

    # 计算每首歌与压缩空间原点的距离
    distances = torch.norm(compressed_features, dim=1).numpy()
    filtered_data['autoencoder_distance'] = distances

    # 按距离排序，返回前 n_recommendations 的不同歌曲
    recommended_songs = filtered_data.sort_values('autoencoder_distance').drop_duplicates(subset=['song_title']).head(n_recommendations)

    # 如果推荐数量不足，随机补足
    if len(recommended_songs) < n_recommendations:
        remaining_songs = data[~data['song_title'].isin(recommended_songs['song_title'])]
        additional_songs = remaining_songs.sample(n=n_recommendations - len(recommended_songs), replace=False)
        recommended_songs = pd.concat([recommended_songs, additional_songs])

    return recommended_songs

def filter_data_by_emotion_and_genre(data, emotion, genre):
    """
    筛选符合情绪和音乐类型的歌曲。
    """
    filtered_data = data[data['emotion'] == emotion]
    if genre:
        filtered_data = filtered_data[filtered_data['genre'] == genre]
    return filtered_data

def fuzzy_match(query, choices, limit=1):
    return process.extract(query, choices, limit=limit)

def get_top_artist_songs(data, artist_name, feature_columns, scaler):
    """
    获取指定艺术家最新的三首歌曲及其特征。
    """
    artist_choices = data['artist_name'].unique().tolist()
    best_match = fuzzy_match(artist_name, artist_choices, limit=1)
    if not best_match or best_match[0][1] < 80:  # 设置模糊匹配阈值
        print(f"No good match found for artist: {artist_name}. Proceeding with emotion and genre filtering.")
        return None, None

    matched_artist = best_match[0][0]
    artist_songs = data[data['artist_name'] == matched_artist]
    top_artist_songs = artist_songs.nlargest(3, 'year')
    top_features = scaler.transform(top_artist_songs[feature_columns])
    return top_artist_songs, top_features

def get_target_song_features(data, song_name, feature_columns, scaler):
    """
    获取指定歌曲的特征。
    """
    song_choices = data['song_title'].unique().tolist()
    best_match = fuzzy_match(song_name, song_choices, limit=1)
    if not best_match or best_match[0][1] < 80:  # 设置模糊匹配阈值
        print(f"No good match found for song: {song_name}. Proceeding with emotion and genre filtering.")
        return None, None

    matched_song = best_match[0][0]
    target_song = data[data['song_title'] == matched_song]
    target_features = scaler.transform(target_song[feature_columns].head(1))
    return target_song, target_features

def recommend_songs_by_similarity(filtered_data, top_features, feature_columns, scaler, n_recommendations):
    """
    根据特征相似性推荐歌曲。
    """
    scaled_features = scaler.transform(filtered_data[feature_columns])
    similarity = cosine_similarity(top_features, scaled_features).mean(axis=0)
    filtered_data['similarity'] = similarity
    recommended_songs = filtered_data.nlargest(n_recommendations, 'similarity')

    # 如果推荐数量不足，随机补足
    if len(recommended_songs) < n_recommendations:
        remaining_songs = filtered_data[~filtered_data['song_title'].isin(recommended_songs['song_title'])]
        additional_songs = remaining_songs.sample(n=n_recommendations - len(recommended_songs), replace=False)
        recommended_songs = pd.concat([recommended_songs, additional_songs])

    return recommended_songs

def recommend_songs_with_main_logic(data, emotion, genre=None, artist_name=None, song_name=None, n_recommendations=5):
    """
    主函数：根据输入条件推荐歌曲。
    """

    # Step 1: 筛选符合条件的歌曲
    filtered_data = filter_data_by_emotion_and_genre(data, emotion, genre)
    if filtered_data.empty:
        print("No songs found matching the given criteria.")
        return pd.DataFrame()

    feature_columns = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
    ]
    scaler = StandardScaler()
    scaler.fit(filtered_data[feature_columns])

    if genre:
        genre.lower()

    # Step 3: 如果同时给定 song_name 和 artist_name
    if song_name and artist_name:
        song_name = song_name.lower()
        artist_name = artist_name.lower()

        # 模糊匹配歌手和歌曲
        song_choices = data['song_title'].unique().tolist()
        artist_choices = data['artist_name'].unique().tolist()

        matched_song = fuzzy_match(song_name, song_choices, limit=1)
        matched_artist = fuzzy_match(artist_name, artist_choices, limit=1)

        if matched_song and matched_artist and matched_song[0][1] >= 80 and matched_artist[0][1] >= 80:
            song_match = matched_song[0][0]
            artist_match = matched_artist[0][0]

            target_song = data[(data['song_title'] == song_match) & (data['artist_name'] == artist_match)]

            if not target_song.empty:
                target_features = scaler.transform(target_song[feature_columns])
                recommended_songs = recommend_songs_by_similarity(filtered_data, target_features, feature_columns, scaler, n_recommendations - 1)

                # 如果推荐数量不足，随机补足
                if len(recommended_songs) < n_recommendations:
                    remaining_songs = data[~data['song_title'].isin(recommended_songs['song_title'])]
                    additional_songs = remaining_songs.sample(n=n_recommendations - len(recommended_songs), replace=False)
                    recommended_songs = pd.concat([recommended_songs, additional_songs])

                return pd.concat([target_song, recommended_songs]).drop_duplicates(subset=['song_title']).head(n_recommendations)

        print(f"No good match found for song: {song_name} and artist: {artist_name}. Proceeding with other logic.")

    # Step 4: 如果给定 song_name
    if song_name:
        song_name = song_name.lower()
        target_song, target_features = get_target_song_features(data, song_name, feature_columns, scaler)
        if target_features is not None:
            recommended_songs = recommend_songs_by_similarity(filtered_data, target_features, feature_columns, scaler, n_recommendations - 1)

            # 如果推荐数量不足，随机补足
            if len(recommended_songs) < n_recommendations:
                remaining_songs = data[~data['song_title'].isin(recommended_songs['song_title'])]
                additional_songs = remaining_songs.sample(n=n_recommendations - len(recommended_songs), replace=False)
                recommended_songs = pd.concat([recommended_songs, additional_songs])

            return pd.concat([target_song, recommended_songs]).drop_duplicates(subset=['song_title']).head(n_recommendations)

    # Step 2: 如果给定 artist_name
    if artist_name:
        artist_name = artist_name.lower()
        top_artist_songs, top_features = get_top_artist_songs(data, artist_name, feature_columns, scaler)
        if top_features is not None:
            recommended_songs = recommend_songs_by_similarity(filtered_data, top_features, feature_columns, scaler, n_recommendations + 5)
            top_song = top_artist_songs.iloc[0]

            # 如果推荐数量不足，随机补足
            if len(recommended_songs) < n_recommendations:
                remaining_songs = data[~data['song_title'].isin(recommended_songs['song_title'])]
                additional_songs = remaining_songs.sample(n=n_recommendations - len(recommended_songs), replace=False)
                recommended_songs = pd.concat([recommended_songs, additional_songs])

            recommended_songs = pd.concat([pd.DataFrame([top_song]), recommended_songs]).drop_duplicates(subset=['song_title'])
            return recommended_songs.head(n_recommendations)

    # Step 5: 如果只有 emotion 和 genre
    recommended_songs = recommend_songs_with_autoencoder(data, emotion, genre=None, n_recommendations=n_recommendations)

    # 如果推荐数量不足，随机补足
    if len(recommended_songs) < n_recommendations:
        remaining_songs = data[~data['song_title'].isin(recommended_songs['song_title'])]
        additional_songs = remaining_songs.sample(n=n_recommendations - len(recommended_songs), replace=False)
        recommended_songs = pd.concat([recommended_songs, additional_songs])
    return recommended_songs.head(n_recommendations)

def url_out(recommendfunc, data, emotion, genre=None, artist_name=None, song_name=None, n_recommendations=5):
    recommended = recommendfunc(data, emotion=emotion, genre=genre, artist_name=artist_name, n_recommendations=n_recommendations)
    song_list = []
    for track_id in recommended['id']:
        song_list.append(get_song(track_id)[0])
    return song_list