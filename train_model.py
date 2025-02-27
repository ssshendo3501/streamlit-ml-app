import pickle
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# データセットの読み込み
housing = fetch_california_housing()
X, y = housing.data, housing.target  # 特徴量とターゲット
feature_names = housing.feature_names  # 特徴量の名前

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデルの学習
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 学習済みモデルをpickleとして保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# スケーラーも保存
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
