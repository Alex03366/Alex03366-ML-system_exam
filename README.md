# ML Churn Prediction Pipeline

Автоматизированный конвейер переобучения модели прогнозирования оттока клиентов.
ML-система уровня 2 по классификации MLOps.

## Бизнес-задача

Прогнозирование оттока клиентов телеком-оператора. Модель определяет клиентов с
высоким риском ухода для запуска превентивных промо-кампаний.

## Архитектура

Система состоит из трёх основных слоёв.

**Слой оркестрации** — Apache Airflow управляет еженедельным пайплайном переобучения:
забирает данные из Object Storage, запускает скрипт обучения, сравнивает метрики
с текущей моделью и принимает решение о замене.

**Слой экспериментов** — MLflow хранит историю запусков, метрики и артефакты моделей.
При каждом переобучении новая версия регистрируется в Model Registry.

**Слой сервинга** — FastAPI принимает запросы от внешних систем, загружает актуальную
модель и препроцессор, возвращает предсказание и уровень риска.

Все компоненты разворачиваются в Yandex Cloud через Terraform (IaC).
Docker-образ сервинга собирается и деплоится через GitHub Actions (CI/CD).

## Компоненты

| Компонент | Назначение | Технология |
|-----------|-----------|------------|
| Оркестратор | Автоматическое переобучение по расписанию | Apache Airflow |
| Реестр моделей | Отслеживание экспериментов и версий | MLflow |
| Сервинг | API для инференса | FastAPI + Docker |
| Хранилище | Данные и артефакты | Yandex Object Storage |
| IaC | Инфраструктура как код | Terraform |
| CI/CD | Автоматическая выкатка | GitHub Actions |

## Структура проекта

```
├── model/ # Код модели
│ ├── preprocess.py # Препроцессинг данных
│ ├── train.py # Скрипт обучения
│ └── predict.py # Инференс
│
├── serving/ # Сервинг модели
│ ├── main.py # FastAPI сервер
│ ├── Dockerfile # Docker-образ
│ └── requirements.txt # Зависимости
│
├── airflow_dags/ # Оркестрация
│ └── churn_pipeline.py # DAG переобучения
│
├── terraform/ # Инфраструктура
│ ├── main.tf # Конфигурация
│ ├── variables.tf # Переменные
│ └── terraform.tfvars.example # Пример
│
├── .github/workflows/ # CI/CD
│ └── deploy.yml # Пайплайн деплоя
│
├── docs/ # Документация
│ ├── manifest.md # ML-манифест
│ ├── adr_latency.md # ADR по оптимизации
│ └── sli_slo.md # Метрики надёжности
│
└── README.md
```

## Быстрый старт


### Клонирование

```bash
git clone https://github.com/rubashnyias/ML-system_exam.git
cd ML-system_exam
```
### Локальное обучение

```bash
pip install pandas numpy scikit-learn
mkdir data
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv' -OutFile 'data\Telco-Customer-Churn.csv'"
python model/train.py --data-path data/Telco-Customer-Churn.csv --output-dir model
```
### Локальный сервинг

```bash
docker build -t churn-api -f serving/Dockerfile .
docker run -d -p 8000:8000 --name churn-container churn-api
```

### Проверка

```bash
# health check
curl http://localhost:8000/health

# предсказание
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

## Развёртывание в Yandex Cloud
```bash
cd terraform
terraform init
terraform apply
```
### Продакшн-сервис

Сервис развёрнут в Yandex Cloud и доступен по адресу:
http://89.169.165.22/health

### Модель

Алгоритм: Logistic Regression

Метрики на тесте: Recall = 0.79, Precision = 0.49, ROC-AUC = 0.83

Датасет: Telco Customer Churn (7043 записи, 19 признаков)

Переобучение: Еженедельно (Airflow DAG)

### Мониторинг

Метрики надёжности описаны в docs/sli_slo.md:

Технический уровень: доступность, latency, ошибки

Модельный уровень: recall, precision, data drift

Бизнес-уровень: churn rate, retention rate

## Документация

- [ML-манифест](docs/manifest.md) — описание системы и целей
- [ADR-001](docs/adr_latency.md) — решение по оптимизации времени отклика
- [SLI/SLO](docs/sli_slo.md) — метрики надёжности

## Уровень зрелости

**Уровень 2** по классификации MLOps:

- Версионирование кода (Git/GitHub)
- CI/CD пайплайн (GitHub Actions)
- Хранилище признаков и реестр моделей (MLflow)
- Сервинг модели через API (FastAPI)
- Мониторинг качества (Grafana)
- Система управления экспериментами (MLflow)
- Оркестратор (Airflow)
