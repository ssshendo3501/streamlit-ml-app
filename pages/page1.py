# EDA & データ可視化アプリ

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("EDA & データ可視化アプリ")

# ファイルアップロード
uploaded_file = st.file_uploader("CSV ファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### データのプレビュー")
    st.dataframe(df.head())
    st.write(f'shape: {df.shape}')

    # データの基本情報
    st.write("### データの情報")
    df_info = pd.DataFrame({
        "データ型": df.dtypes,
        "欠損値の数": df.isnull().sum(),
        "欠損率(%)": (df.isnull().sum() / len(df)) * 100,
        "ユニーク数": df.nunique(),
        "ユニーク値のサンプル(10個)": df.apply(lambda x: list(map(str, x.dropna().unique()[:10])))
    })
    st.dataframe(df_info)
    
    # 基本統計情報
    st.write("### データの基本統計情報")
    st.write(df.describe().T)
    
    # ヒストグラム（Plotly）
    st.write("### ヒストグラム")
    column = st.selectbox("ヒストグラムを表示するカラムを選択してください", df.select_dtypes(include="number").columns)
    fig_hist = px.histogram(df, x=column, nbins=30, title=f"ヒストグラム: {column}")
    st.plotly_chart(fig_hist)
    
    # カテゴリデータの分布
    st.write("### カテゴリデータの分布")
    cat_columns = df.select_dtypes(include="object").columns
    if len(cat_columns) > 0:
        cat_col = st.selectbox("カテゴリ変数を選択してください", cat_columns)
        # カウントデータフレームを作成（カラム名を明示的に設定）
        cat_counts = df[cat_col].value_counts().reset_index(name="count")

        # 可視化
        fig_cat = px.bar(cat_counts, 
                     x=cat_col, y="count", 
                     title=f"{cat_col} の分布", 
                     labels={cat_col: cat_col, "count": "カウント"})
        st.plotly_chart(fig_cat)
        
    # 相関マトリックス
    # 数値カラムのみを対象にする
    numeric_df = df.select_dtypes(include=["number"])
    st.write("### 相関マトリックス")
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.write("数値カラムがありません。")

    # 動的な散布図
    st.write("### 散布図（動的選択）")
    x_var = st.selectbox("X軸の変数", df.select_dtypes(include="number").columns)
    y_var = st.selectbox("Y軸の変数", df.select_dtypes(include="number").columns)
    fig_scatter = px.scatter(df, x=x_var, y=y_var, title=f"{x_var} vs {y_var}")
    st.plotly_chart(fig_scatter)
    
    # ペアプロット
    st.write("### 散布図（ペアプロット）")
    num_columns = df.select_dtypes(include="number").columns
    if len(num_columns) > 1:
        fig_pairplot = sns.pairplot(df[num_columns])
        st.pyplot(fig_pairplot)



