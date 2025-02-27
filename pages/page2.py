import streamlit as st
import pickle
import numpy as np

# 学習済みモデルの読み込み
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# スケーラーの読み込み
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("住宅価格予測アプリ")

# ユーザー入力
st.sidebar.header("入力データ")
MedInc = st.sidebar.slider("所得の中央値 (MedInc)", 0.5, 15.0, 3.0)
HouseAge = st.sidebar.slider("住宅の築年数 (HouseAge)", 1, 50, 20)
AveRooms = st.sidebar.slider("平均部屋数 (AveRooms)", 1, 10, 5)
AveBedrms = st.sidebar.slider("平均寝室数 (AveBedrms)", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("人口 (Population)", 100, 4000, 1000)
AveOccup = st.sidebar.slider("平均居住者数 (AveOccup)", 1, 5, 2)
Latitude = st.sidebar.slider("緯度 (Latitude)", 32.0, 42.0, 35.0)
Longitude = st.sidebar.slider("経度 (Longitude)", -125.0, -114.0, -120.0)

# 入力データを配列にまとめる
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
input_data_scaled = scaler.transform(input_data)

# 予測
prediction = model.predict(input_data_scaled)

st.write(f"### 予測された住宅価格: **${prediction[0] * 100000:.2f}**")
st.write(f"### 予測された住宅価格: **¥{prediction[0] * 100000 * 150:.0f}**")


