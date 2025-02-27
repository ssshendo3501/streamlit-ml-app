import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from io import BytesIO

st.title("データ前処理アプリ")

# CSV アップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### アップロードされたデータ")
    st.write(df.head())
    
    # 目的変数の選択
    target_col = st.sidebar.selectbox("目的変数を選択", df.columns)
    
    # 目的変数を除外した特徴量リスト
    feature_cols = [col for col in df.columns if col != target_col]
    df_features = df[feature_cols]
    
    st.sidebar.header("前処理の選択")
    
    # 欠損値補完
    num_impute_option = st.sidebar.radio("数値データの欠損値補完方法", ["平均値", "中央値"], index=0)
    cat_impute_option = "最頻値"
    
    # エンコーディング
    encoding_option = st.sidebar.radio("カテゴリデータのエンコーディング方法", ["One-Hot", "Label", "Target"], index=0)
    
    # 特徴量スケーリング
    scaling_option = st.sidebar.radio("数値データのスケーリング方法", ["Standard", "Min-Max"], index=0)
    
    # データ前処理
    df_processed = df_features.copy()
    
    # 数値・カテゴリカラムの分類
    num_cols = df_processed.select_dtypes(include=[np.number]).columns
    cat_cols = df_processed.select_dtypes(exclude=[np.number]).columns
    
    # 欠損値補完
    if num_impute_option == "平均値":
        df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].mean())
    else:
        df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())
    
    df_processed[cat_cols] = df_processed[cat_cols].fillna(df_processed[cat_cols].mode().iloc[0])
    
    # エンコーディング
    if encoding_option == "One-Hot":
        df_processed = pd.get_dummies(df_processed, columns=cat_cols)
    elif encoding_option == "Label":
        for col in cat_cols:
            df_processed[col] = LabelEncoder().fit_transform(df_processed[col])
    elif encoding_option == "Target" and target_col in cat_cols:
        target_mapping = df.groupby(target_col).size() / len(df)
        df_processed[target_col] = df[target_col].map(target_mapping)
    
    # 特徴量スケーリング（目的変数はスケーリングしない）
    if scaling_option == "Standard":
        df_processed[num_cols] = StandardScaler().fit_transform(df_processed[num_cols])
    elif scaling_option == "Min-Max":
        df_processed[num_cols] = MinMaxScaler().fit_transform(df_processed[num_cols])
    
    # 目的変数を最後に結合
    df_processed[target_col] = df[target_col]
    
    st.write("### 前処理後のデータ")
    st.write(df_processed.head())
    st.write(f'shape: {df_processed.shape}')
    
    # CSV ダウンロード
    output = BytesIO()
    df_processed.to_csv(output, index=False)
    output.seek(0)
    st.download_button("前処理済みデータをダウンロード", data=output, file_name="processed_data.csv", mime="text/csv")
