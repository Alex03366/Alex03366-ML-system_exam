import sys
import os

# добавляем корень проекта в путь для импорта model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from model.predict import predict, predict_batch

# описание приложения для документации
app = FastAPI(
    title="churn prediction api",
    description="сервис прогнозирования оттока клиентов телеком-оператора",
    version="1.0.0"
)


# модель данных для одного клиента
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    customerID: Optional[str] = None


# модель для батчевого запроса
class BatchRequest(BaseModel):
    customers: list[CustomerData]


# модель ответа
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str


class BatchResponse(BaseModel):
    predictions: list[dict]


@app.get("/health")
def health_check():
    """проверка работоспособности сервиса."""
    return {
        "status": "healthy",
        "model": "logistic_regression",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerData):
    try:
        result = predict(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchResponse)
def predict_multiple(request: BatchRequest):
    try:
        customers_list = [c.model_dump() for c in request.customers]
        results = predict_batch(customers_list)
        return BatchResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
