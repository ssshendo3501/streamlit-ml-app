import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_curve, auc, confusion_matrix

# Streamlitアプリのタイトル
st.title("機械学習モデルトレーニング＆評価アプリ")

# CSVファイルのアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("データのプレビュー:")
    st.write(df.head())

    # 目的変数の選択
    target_column = st.selectbox("目的変数を選択", df.columns)
    task_type = st.radio("タスクの種類を選択", ["回帰", "分類"])
    
    # 特徴量の選択
    feature_columns = st.multiselect("特徴量を選択", [col for col in df.columns if col != target_column], default=[col for col in df.columns if col != target_column])
    X = df[feature_columns]
    y = df[target_column]
    
    # 学習データとテストデータの分割
    test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # モデルの選択
    model_options = {
        "線形回帰": LinearRegression(),
        "決定木": DecisionTreeRegressor() if task_type == "回帰" else DecisionTreeClassifier(),
        "ランダムフォレスト": RandomForestRegressor() if task_type == "回帰" else RandomForestClassifier(),
        "GBDT": GradientBoostingRegressor() if task_type == "回帰" else GradientBoostingClassifier()
    }
    model_choice = st.selectbox("モデルを選択", list(model_options.keys()))
    model = model_options[model_choice]
    
    # モデルのトレーニング
    if st.button("モデルをトレーニング"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 評価
        if task_type == "回帰":
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"### 平均二乗誤差 (MSE): {mse:.4f}")
            
            # y-y プロット
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("実際の値")
            ax.set_ylabel("予測値")
            ax.set_title("y-y プロット")
            st.pyplot(fig)
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            st.write(f"### 精度: {acc:.4f}")
            st.write(f"### F1スコア: {f1:.4f}")
            
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
            ax.set_xlabel("予測ラベル")
            ax.set_ylabel("実際のラベル")
            ax.set_title("混同行列")
            st.pyplot(fig)
            
            # ROC曲線
            if len(set(y_test)) == 2:  # 2クラス分類の場合のみ
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig_roc = px.area(x=fpr, y=tpr, title=f"ROC曲線 (AUC={roc_auc:.4f})", labels={"x": "偽陽性率", "y": "真陽性率"})
                st.plotly_chart(fig_roc)
        
        # SHAPによる説明
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        fig_shap, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig_shap)
        
        # 学習済みモデルの保存
        model_filename = "trained_model.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
        
        # ダウンロードリンク
        st.download_button(label="学習済みモデルをダウンロード", data=open(model_filename, "rb").read(), file_name=model_filename, mime="application/octet-stream")
