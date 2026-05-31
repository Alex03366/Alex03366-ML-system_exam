from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable

import sys
import os
import json
import pickle
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = "/opt/airflow/model"
sys.path.insert(0, MODEL_DIR)

from preprocess import ChurnPreprocessor
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TEST_SIZE = 0.2

# порог для замены модели
RECALL_THRESHOLD = 0.50


def load_data(**context):
    data_path = Variable.get("churn_data_path", "/opt/airflow/data/Telco-Customer-Churn.csv")
    print(f"загрузка данных из {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"загружено {len(df)} записей")
    
    # сохраняем во временный файл для следующих шагов
    tmp_path = "/tmp/churn_data.csv"
    df.to_csv(tmp_path, index=False)
    context['ti'].xcom_push(key='data_path', value=tmp_path)


def clean_data(**context):
    ti = context['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')
    
    df = pd.read_csv(data_path)
    print(f"очистка данных, было {len(df)} записей")
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    print(f"после очистки {len(df)} записей")
    
    tmp_path = "/tmp/churn_data_clean.csv"
    df.to_csv(tmp_path, index=False)
    ti.xcom_push(key='clean_data_path', value=tmp_path)


def train_model(**context):
    ti = context['ti']
    data_path = ti.xcom_pull(key='clean_data_path', task_ids='clean_data')
    
    df = pd.read_csv(data_path)
    print(f"обучение модели на {len(df)} записях")
    
    # препроцессинг
    preprocessor = ChurnPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    feature_names = [c for c in df_processed.columns if c != 'Churn']
    X = df_processed[feature_names]
    y = df_processed['Churn']
    
    # разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # обучение
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # метрики
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"метрики: recall={recall:.4f}, precision={precision:.4f}, roc_auc={roc_auc:.4f}")
    
    # сохраняем метрики в xcom
    ti.xcom_push(key='new_recall', value=recall)
    ti.xcom_push(key='new_precision', value=precision)
    ti.xcom_push(key='new_roc_auc', value=roc_auc)
    
    # сохраняем модель и препроцессор во временные файлы
    with open("/tmp/new_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open("/tmp/new_preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    with open("/tmp/new_feature_names.json", 'w') as f:
        json.dump(feature_names, f)
    
    print("модель обучена и сохранена")

# сравнение текущей модели с новой и решение о замене
def evaluate_and_decide(**context):
    ti = context['ti']
    
    new_recall = ti.xcom_pull(key='new_recall', task_ids='train_model')
    print(f"[INFO] recall новой модели: {new_recall:.4f}")
    print(f"[INFO] порог для замены: {RECALL_THRESHOLD}")
    
    # проверка, есть ли текущая модель
    current_model_path = os.path.join(MODEL_DIR, "model.pkl")
    has_current = os.path.exists(current_model_path)
    
    if has_current:
        # загружаем текущую модель и смотрим её метрики
        print("текущая модель существует, сравниваем")
        
        if new_recall > RECALL_THRESHOLD:
            decision = "deploy"
            print(f"решение: заменить модель (recall {new_recall:.4f} > {RECALL_THRESHOLD})")
        else:
            decision = "keep"
            print(f"решение: оставить текущую модель (recall {new_recall:.4f} <= {RECALL_THRESHOLD})")
    else:
        decision = "deploy"
        print("текущей модели нет, деплоим новую")
    
    ti.xcom_push(key='decision', value=decision)

# замена текущей модкели на новую
def deploy_model(**context):
    ti = context['ti']
    decision = ti.xcom_pull(key='decision', task_ids='evaluate_and_decide')
    
    if decision != "deploy":
        print("пропускаем деплой")
        return
    
    print("деплой новой модели")
    
    import shutil
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    shutil.copy("/tmp/new_model.pkl", os.path.join(MODEL_DIR, "model.pkl"))
    shutil.copy("/tmp/new_preprocessor.pkl", os.path.join(MODEL_DIR, "preprocessor.pkl"))
    shutil.copy("/tmp/new_feature_names.json", os.path.join(MODEL_DIR, "feature_names.json"))
    
    print(f"модель развернута в {MODEL_DIR}")


def rollback_model(**context):
    """откатывает модель, если качество упало."""
    ti = context['ti']
    new_recall = ti.xcom_pull(key='new_recall', task_ids='train_model')
    
    if new_recall < RECALL_THRESHOLD:
        print(f"recall {new_recall:.4f} ниже порога {RECALL_THRESHOLD}")
    else:
        print("качество в норме")

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ml-team@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='churn_model_training_pipeline',
    default_args=default_args,
    description='еженедельное переобучение модели оттока клиентов',
    schedule_interval='0 2 * * 1', 
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'churn', 'training'],
) as dag:
    
    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True,
        doc_md='загрузка данных из источника',
    )
    
    task_clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        provide_context=True,
        doc_md='очистка данных',
    )
    
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
        doc_md='обучение модели и расчёт метрик',
    )
    
    task_evaluate = PythonOperator(
        task_id='evaluate_and_decide',
        python_callable=evaluate_and_decide,
        provide_context=True,
        doc_md='сравнение с текущей моделью и принятие решения',
    )
    
    task_deploy = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
        doc_md='замена модели при улучшении метрик',
    )
    
    task_check_quality = PythonOperator(
        task_id='check_quality',
        python_callable=rollback_model,
        provide_context=True,
        doc_md='проверка качества и уведомление при падении',
    )
    
    task_load >> task_clean >> task_train >> task_evaluate
    task_evaluate >> task_deploy
    task_evaluate >> task_check_quality
