import json
import os
import time

import joblib
import pandas as pd
import requests

# 1. Загрузка обученной ML-модели
MODEL_PATH = os.path.join("models", "model.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Сначала обучите модель через src/train_model.py!")

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
features = saved_data["features"]

# 2. Загрузка базы знаний регламентов цеха
RULES_PATH = os.path.join("knowledge_base", "regulations.json")
with open(RULES_PATH, "r", encoding="utf-8") as f:
    regulations = json.load(f)


def call_local_llm(prompt: str, model_name: str = "qwen3.5:9b") -> tuple[str, float]:
    """Отправка запроса в Ollama с отключенным thinking-режимом для прямого ответа"""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": False,  # Отключаем скрытые рассуждения, чтобы сразу шел готовый текст
        "options": {
            "temperature": 0.2,
            "num_predict": 300,  # Запас длины для красивого ответа
        },
    }

    start_llm = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=180)
        llm_time = time.perf_counter() - start_llm

        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "").strip()

            # Страховка: если текст вдруг остался в блоке thinking
            if not content and "thinking" in message:
                content = message.get("thinking", "").strip()

            return content, llm_time
        return f"Ошибка Ollama: код {response.status_code}", llm_time
    except requests.exceptions.ReadTimeout:
        return (
            "Ошибка: Превышено время ожидания.",
            time.perf_counter() - start_llm,
        )
    except requests.exceptions.ConnectionError:
        return "Ошибка: Ollama не запущена.", 0.0


def analyze_sensor_data(sensor_dict: dict) -> dict:
    """Анализ данных датчиков с фиксацией времени каждого этапа"""
    start_total = time.perf_counter()

    # --- ЭТАП 1: Работа предиктивного ML-ядра ---
    start_ml = time.perf_counter()
    df_input = pd.DataFrame([sensor_dict])[features]
    probabilities = model.predict_proba(df_input)[0]
    failure_risk = probabilities[1] * 100
    ml_time = time.perf_counter() - start_ml

    # Штатный режим
    if failure_risk < 50:
        total_time = time.perf_counter() - start_total
        return {
            "status": "NORMAL",
            "risk_percentage": round(failure_risk, 1),
            "recommendation": "Оборудование работает в штатном режиме.",
            "ml_time_ms": round(ml_time * 1000, 2),
            "llm_time_s": 0.0,
            "total_time_s": round(total_time, 2),
        }

    # --- ЭТАП 2: Физический анализ сбоя ---
    temp_diff = (
        sensor_dict["Process temperature [K]"] - sensor_dict["Air temperature [K]"]
    )
    power = sensor_dict["Torque [Nm]"] * (
        sensor_dict["Rotational speed [rpm]"] * 0.1047
    )
    overstrain = sensor_dict["Tool wear [min]"] * sensor_dict["Torque [Nm]"]

    if temp_diff < 8.6 and sensor_dict["Rotational speed [rpm]"] < 1380:
        fault_type = "HDF"
    elif power < 3500 or power > 9000:
        fault_type = "PWF"
    elif overstrain > 11000:
        fault_type = "OSF"
    elif sensor_dict["Tool wear [min]"] >= 200:
        fault_type = "TWF"
    else:
        fault_type = "HDF"

    reg_text = regulations.get(fault_type, "Действовать по общей инструкции цеха.")

    # --- ЭТАП 3: Генерация рекомендации через локальную LLM ---
    prompt = f"""Ты — инженерный СППР-ассистент металлургической линии.
Зафиксирован сбой: {fault_type}. Риск аварии: {round(failure_risk, 1)}%.
Регламент цеха: {reg_text}

Кратко (ровно 2-3 предложения) укажи оператору:
1. Зафиксированную угрозу.
2. Срочное действие по регламенту. Без лишних вводных слов."""

    llm_recommendation, llm_time = call_local_llm(prompt)
    total_time = time.perf_counter() - start_total

    return {
        "status": "WARNING",
        "risk_percentage": round(failure_risk, 1),
        "fault_type": fault_type,
        "recommendation": llm_recommendation,
        "ml_time_ms": round(ml_time * 1000, 2),
        "llm_time_s": round(llm_time, 2),
        "total_time_s": round(total_time, 2),
    }


if __name__ == "__main__":
    print("--- ТЕСТИРОВАНИЕ ИИ-АГЕНТА С ТАЙМЕРОМ ---")

    sample_emergency_sensors = {
        "Air temperature [K]": 298.9,
        "Process temperature [K]": 309.1,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 68.0,
        "Tool wear [min]": 225,
    }

    result = analyze_sensor_data(sample_emergency_sensors)

    print("\n=== ОТВЕТ ИИ-АГЕНТА ОПЕРАТОРУ ===")
    print(f"Статус: {result['status']}")
    print(f"Уровень риска: {result['risk_percentage']}%")
    if "fault_type" in result:
        print(f"Тип сбоя: {result['fault_type']}")
    print(f"\nРекомендация:\n{result['recommendation']}")

    print("\n=== ЗАМЕРЫ ВРЕМЕНИ (ЗАДЕРЖКА ИНФЕРЕНСА) ===")
    print(f"⚡ Предиктивное ядро (ML):  {result['ml_time_ms']} мс")
    print(f"🧠 Генерация совета (LLM): {result['llm_time_s']} сек")
    print(f"⏱️ Общее время реакции:     {result['total_time_s']} сек")
