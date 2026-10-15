import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def train():
    # 1. Поиск и загрузка датасета
    data_path = os.path.join("data", "ai4i2020.csv")

    print(f"Загрузка данных из: {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Выбираем 5 физических признаков телеметрии
    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # Целевая переменная: 0 (норма) или 1 (авария)
    target = "Machine failure"

    X = df[features]
    y = df[target]

    # 3. Разделяем на обучающую (80%) и тестовую (20%) выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Обучающая выборка: {len(X_train)} строк, Тестовая: {len(X_test)} строк.")

    # 4. Обучаем модель Random Forest с балансировкой классов
    # (так как аварии случаются редко, class_weight='balanced' критически важен)
    print("Обучение модели Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Оценка качества модели
    y_pred = model.predict(X_test)
    print("\n--- ОТЧЕТ О КАЧЕСТВЕ МОДЕЛИ ---")
    print(
        classification_report(
            y_test, y_pred, target_names=["Штатный режим (0)", "Авария (1)"]
        )
    )

    print("Матрица ошибок (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    # 6. Сохранение обученной модели
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "model.joblib")
    joblib.dump({"model": model, "features": features}, model_save_path)
    print(f"\nМодель успешно сохранена в: {model_save_path}")


if __name__ == "__main__":
    train()
