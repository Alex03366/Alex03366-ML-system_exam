# Модуль препроцессинга данных для задачи прогнозирования оттока клиентов.
# Используется как на этапе обучения, так и при инференсе.

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ChurnPreprocessor:
    """
    Класс для препроцессинга данных
    
    Обрабатывает:
    - Категориальные признаки (LabelEncoder)
    - Числовые признаки (StandardScaler)
    - Целевую переменную (Yes/No - 1/0)
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_cols = None
        self.target_col = 'Churn'
        self._fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'ChurnPreprocessor':
    
        # Обучает препроцессор на тренировочных данных.
        
        df = df.copy()
        
        # Определяем категориальные колонки
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Убираем customerID и целевую переменную из признаков
        cols_to_remove = ['customerID', self.target_col]
        self.categorical_cols = [c for c in self.categorical_cols if c not in cols_to_remove]
        
        # Обучаем LabelEncoder для каждой категориальной колонки
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Обучаем Scaler для числовых колонок
        self.scaler.fit(df[self.numeric_cols])
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, has_target: bool = False) -> pd.DataFrame:
    
       # Применяет препроцессинг к данным.
        
        if not self._fitted:
            raise RuntimeError("Препроцессор не обучен. Сначала вызовите fit().")
        
        df = df.copy()
        
        # Убираем customerID если есть
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Кодируем категориальные признаки
        for col in self.categorical_cols:
            if col in df.columns:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Масштабируем числовые признаки
        numeric_in_data = [c for c in self.numeric_cols if c in df.columns]
        if numeric_in_data:
            df[numeric_in_data] = self.scaler.transform(df[numeric_in_data])
        
        # Обрабатываем целевую переменную если она есть
        if has_target and self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].map({'Yes': 1, 'No': 0})
        
        return df
      
    #  Обучает препроцессор и сразу применяет к данным.
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:      
   
        self.fit(df)
        return self.transform(df, has_target=True)

    #  Сохраняет препроцессор в файл.
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Препроцессор сохранен в {path}")

  # Загружает препроцессор из файла.
    @staticmethod
    def load(path: str) -> 'ChurnPreprocessor':
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        if not preprocessor._fitted:
            raise RuntimeError("Загружен необученный препроцессор")
        print(f"Препроцессор загружен из {path}")
        return preprocessor
