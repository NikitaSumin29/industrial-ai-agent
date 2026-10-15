import os

import pandas as pd
import streamlit as st

# Импортируем логику нашего агента из src/agent.py
from src.agent import analyze_sensor_data

# 1. Настройка страницы
st.set_page_config(
    page_title="СППР: Советчик оператора технологической линии",
    page_icon="⚙️",
    layout="wide",
)

# Расширяем боковую панель, чтобы текст не обрезался
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 420px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ СППР: Интеллектуальный советчик оператора линии")
st.caption(
    "Система поддержки принятия решений на базе предиктивного ядра (ML) и локального ИИ-агента (Qwen 3.5)"
)

# 2. Боковая панель
st.sidebar.header("Панель телеметрии")

DATA_PATH = os.path.join("data", "ai4i2020.csv")
df_demo = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

# Емкие, красивые названия сценариев
scenario = st.sidebar.selectbox(
    "Выберите технологический сценарий:",
    [
        "🟢 Штатный режим (норма)",
        "🔴 Сбой по мощности (PWF)",
        "🔴 Перегрев теплоотвода (HDF)",
        "🔴 Износ резца (TWF)",
        "🎲 Случайная строка из базы",
    ],
)

# Настройки пресетов под каждый сценарий
if scenario == "🟢 Штатный режим (норма)":
    init_air, init_proc, init_speed, init_torque, init_wear = (
        298.1,
        308.6,
        1551,
        42.8,
        0,
    )
elif scenario == "🔴 Сбой по мощности (PWF)":
    init_air, init_proc, init_speed, init_torque, init_wear = (
        298.9,
        309.1,
        1350,
        68.0,
        45,
    )
elif scenario == "🔴 Перегрев теплоотвода (HDF)":
    init_air, init_proc, init_speed, init_torque, init_wear = (
        302.5,
        310.0,
        1320,
        55.0,
        110,
    )
elif scenario == "🔴 Износ резца (TWF)":
    init_air, init_proc, init_speed, init_torque, init_wear = (
        298.5,
        308.7,
        1400,
        50.0,
        240,
    )
else:
    random_row = df_demo.sample(1).iloc[0] if df_demo is not None else None
    if random_row is not None:
        init_air = float(random_row["Air temperature [K]"])
        init_proc = float(random_row["Process temperature [K]"])
        init_speed = int(random_row["Rotational speed [rpm]"])
        init_torque = float(random_row["Torque [Nm]"])
        init_wear = int(random_row["Tool wear [min]"])
    else:
        init_air, init_proc, init_speed, init_torque, init_wear = (
            298.0,
            308.0,
            1500,
            40.0,
            50,
        )

# Ползунки датчиков
air_temp = st.sidebar.slider(
    "Температура воздуха в цехе [K]", 290.0, 310.0, init_air, step=0.1
)
proc_temp = st.sidebar.slider(
    "Температура в зоне резания [K]", 300.0, 320.0, init_proc, step=0.1
)
speed = st.sidebar.slider("Скорость шпинделя [об/мин]", 1000, 3000, init_speed, step=10)
torque = st.sidebar.slider("Крутящий момент [Н·м]", 0.0, 80.0, init_torque, step=0.5)
wear = st.sidebar.slider("Износ инструмента [мин]", 0, 260, init_wear, step=1)

current_sensors = {
    "Air temperature [K]": air_temp,
    "Process temperature [K]": proc_temp,
    "Rotational speed [rpm]": speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": wear,
}

# 3. Основной экран
col1, col2, col3, col4 = st.columns(4)
temp_diff = round(proc_temp - air_temp, 2)
col1.metric("Разница температур (ΔT)", f"{temp_diff} K")
col2.metric("Обороты привода", f"{speed} rpm")
col3.metric("Крутящий момент", f"{torque} Н·м")
col4.metric("Износ резца", f"{wear} мин")

st.divider()

# Инициализируем память Streamlit (session_state)
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "action_applied" not in st.session_state:
    st.session_state.action_applied = False

# Кнопка запуска анализа
if st.button("🚀 Выполнить анализ технологического режима", type="primary"):
    st.session_state.action_applied = False  # сбрасываем статус прошлой команды
    with st.spinner("Агент анализирует телеметрию и подбирает регламент..."):
        st.session_state.analysis_result = analyze_sensor_data(current_sensors)

# Если анализ уже выполнен — отображаем его из памяти
if st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result

    if result["status"] == "NORMAL":
        st.success(
            f"### 🟢 ШТАТНЫЙ РЕЖИМ (Расчетный риск сбоя: {result['risk_percentage']}%)"
        )
        st.info(result["recommendation"])
    else:
        st.error(
            f"### 🔴 КРИТИЧЕСКОЕ СОСТОЯНИЕ (Расчетный риск аварии: {result['risk_percentage']}%)"
        )
        st.warning(
            f"**Зафиксированный тип угрозы:** {result.get('fault_type', 'Сбой')}"
        )

        st.markdown("#### 🤖 Рекомендация ИИ-агента оператору:")
        st.markdown(
            f"""
            <div style="
                background-color: rgba(128, 128, 128, 0.12);
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-left: 5px solid #ff4b4b;
                border-radius: 8px;
                padding: 16px 20px;
                font-size: 16px;
                line-height: 1.5;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            ">
                {result["recommendation"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Компактная профессиональная кнопка
        if st.button("✅ Подтвердить и передать в АСУ ТП"):
            st.session_state.action_applied = True

        # Вот теперь эта плашка гарантированно горит на экране!
        if st.session_state.action_applied:
            st.success(
                "⚡ **Команда успешно передана в АСУ ТП!** Параметры электропривода скорректированы в соответствии с регламентом. Риск аварии снижен."
            )

    # Замеры скорости
    st.divider()
    st.caption("⏱️ **Метрики быстродействия СППР (Задержка инференса):**")
    t_col1, t_col2, t_col3 = st.columns(3)
    t_col1.metric("Предиктивное ядро (ML)", f"{result['ml_time_ms']} мс")
    t_col2.metric("Локальная LLM (Qwen 3.5)", f"{result['llm_time_s']} сек")
    t_col3.metric("Полный отклик контура", f"{result['total_time_s']} сек")
