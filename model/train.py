import argparse
import json
import os
import sys
import pickle
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.preprocess import ChurnPreprocessor

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(data_path):
    print(f"[INFO] загрузка данных из {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[INFO] загружено: {df.shape[0]} записей, {df.shape[1]} колонок")
    return df


def clean_data(df):
    print("[INFO] очистка данных...")
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    print(f"[INFO] после очистки: {len(df)} записей")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/Telco-Customer-Churn.csv')
    parser.add_argument('--output-dir', type=str, default='model')
    args = parser.parse_args()

    print("=" * 50)
    print("обучение модели оттока клиентов")
    print("=" * 50)

    # загрузка
    df = load_data(args.data_path)

    # очистка
    df = clean_data(df)

    # препроцессинг
    print("[INFO] препроцессинг...")
    preprocessor = ChurnPreprocessor()
    df_processed = preprocessor.fit_transform(df)

    feature_names = [c for c in df_processed.columns if c != 'Churn']
    X = df_processed[feature_names]
    y = df_processed['Churn']

    # разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] train: {X_train.shape}, test: {X_test.shape}")

    # обучение
    print("[INFO] обучение LogisticRegression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"  recall:    {recall:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  roc-auc:   {roc_auc:.4f}")

    # сохранение артефактов
    print("[INFO] сохранение артефактов...")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "preprocessor.pkl"), 'wb') as f:
        pickle.dump(preprocessor, f)

    with open(os.path.join(args.output_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(args.output_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_names, f)

    print(f"[INFO] артефакты сохранены в {args.output_dir}/")
    print("[INFO] готово!")


if __name__ == "__main__":
    main()