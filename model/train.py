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

import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.preprocess import ChurnPreprocessor

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT = "churn_prediction"

# загружает сырые данные из CSV
def load_data(data_path: str) -> pd.DataFrame:
    print(f"[INFO] Загрузка данных из {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[INFO] Загружено: {df.shape[0]} записей, {df.shape[1]} колонок")
    return df

# чистка: конвертация TotalCharges, удаление пропусков.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Очистка данных...")
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    print(f"[INFO] После очистки: {len(df)} записей")
    return df


def main():
    parser = argparse.ArgumentParser(description="Обучение модели оттока клиентов")
    parser.add_argument('--data-path', type=str, default='data/Telco-Customer-Churn.csv')
    parser.add_argument('--output-dir', type=str, default='model')
    args = parser.parse_args()

    df = load_data(args.data_path)
    df = clean_data(df)

    preprocessor = ChurnPreprocessor()
    df_processed = preprocessor.fit_transform(df)

    feature_names = [c for c in df_processed.columns if c != 'Churn']
    X = df_processed[feature_names]
    y = df_processed['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    print("Обучение модели")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="LogisticRegression"):
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

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "recall": recall,
            "precision": precision,
            "roc_auc": roc_auc
        })
        mlflow.sklearn.log_model(model, "model")

        print(f"  Recall:    {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

    # сохранение артефактов
    print("Сохранение артефактов")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "preprocessor.pkl"), 'wb') as f:
        pickle.dump(preprocessor, f)

    with open(os.path.join(args.output_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(args.output_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_names, f)

    print(f"Артефакты сохранены в {args.output_dir}/")

if __name__ == "__main__":
    main()
