import os
import json
import pickle
import pandas as pd
import numpy as np


# путь к артефактам относительно корня проекта
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__))


def load_artifacts():
    
    # загружаем препроцессор
    preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # загружаем модель
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # загружаем список признаков
    features_path = os.path.join(ARTIFACTS_DIR, "feature_names.json")
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    
    return preprocessor, model, feature_names


# загружаем один раз при импорте модуля
_preprocessor, _model, _feature_names = load_artifacts()

# предсказывает отток для одного клиента
# на входе raw_data: словарь с сырыми данными клиента
# на выходе словарь с вероятностью, предсказанием и уровнем риска

def predict(raw_data: dict) -> dict:

    # превращаем словарь в dataframe
    input_df = pd.DataFrame([raw_data])
    
    # убираем customerID если есть
    if 'customerID' in input_df.columns:
        input_df = input_df.drop('customerID', axis=1)
    
    # убираем целевую переменную
    if 'Churn' in input_df.columns:
        input_df = input_df.drop('Churn', axis=1)
    
    # преобразуем данные
    processed = input_df.copy()
    
    # обработка категориальных признаквов
    for col in _preprocessor.categorical_cols:
        if col in processed.columns:
            le = _preprocessor.label_encoders[col]
            processed[col] = processed[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # масштабируем числовые признаки
    numeric_in_data = [c for c in _preprocessor.numeric_cols if c in processed.columns]
    if numeric_in_data:
        processed[numeric_in_data] = _preprocessor.scaler.transform(processed[numeric_in_data])
    
    # оставляем только нужные признаки в правильном порядке
    processed = processed[_feature_names]
    
    # предсказание
    proba = _model.predict_proba(processed)[0, 1]
    prediction = int(proba > 0.5)
    
    # определяем уровень риска
    if proba > 0.7:
        risk_level = "high"
    elif proba > 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "churn_probability": float(proba),
        "churn_prediction": prediction,
        "risk_level": risk_level
    }

# предсказание оттока для списка клиенгов
# на входе raw_data_list: список словарей с данными клиентов
# на выходе список словарей с предсказаниями

def predict_batch(raw_data_list: list) -> list:
    results = []
    for raw_data in raw_data_list:
        try:
            result = predict(raw_data)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "churn_probability": None,
                "churn_prediction": None,
                "risk_level": None
            })
    return results
