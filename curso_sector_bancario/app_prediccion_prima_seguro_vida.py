# =======================================================
# App Streamlit: Predicción de Costos/Premios de Seguro de Vida con AdaBoost
# Incluye: Imagen header, menú lateral, datos históricos, EDA, predicciones (perfil cliente, costo predicho, riesgo),
# evaluación del modelo, análisis SHAP y medición de riesgo + sensibilidad robusta
# Ejecuta con: streamlit run app_premios_vida.py
# =======================================================
# Requisitos: pip install streamlit pandas numpy scikit-learn shap seaborn matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Predicción de Premios Seguro de Vida - Naren Castellón", layout="wide")
st.title("📊 Predicción de Costos/Premios de Seguro de Vida")
st.markdown("**Modelo AdaBoost + SHAP para explicabilidad** | Dictado por Naren Castellón")

# Imagen header (URL pública relevante a seguros de vida)
st.image("https://images.unsplash.com/photo-1450101499163-c88421a3e603?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", 
         caption="Análisis predictivo de premios de seguros de vida con IA", use_column_width=True)

# =======================================================
# Carga de datos y entrenamiento (cacheado)
# =======================================================
@st.cache_resource
def load_data_and_train_model():
    np.random.seed(42)
    n_samples = 10000

    data = {
        'age': np.random.randint(18, 75, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48]),
        'bmi': np.round(np.random.normal(27, 5, n_samples), 1).clip(16, 45),
        'children': np.random.randint(0, 5, n_samples),
        'occupation_risk': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.6, 0.3, 0.1]),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'region': np.random.choice(['Northeast', 'Northwest', 'Southeast', 'Southwest'], n_samples),
        'smoker': np.random.choice(['Yes', 'No'], n_samples, p=[0.18, 0.82]),
        'blood_pressure': np.random.choice(['Normal', 'Elevated', 'High'], n_samples, p=[0.55, 0.3, 0.15]),
        'cholesterol_level': np.round(np.random.normal(200, 45, n_samples)).clip(120, 350).astype(int),
        'diabetes': np.random.choice(['Yes', 'No'], n_samples, p=[0.12, 0.88]),
        'heart_disease_history': np.random.choice(['Yes', 'No'], n_samples, p=[0.08, 0.92]),
        'cancer_family_history': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
        'exercise_frequency': np.random.choice(['None', 'Low', 'Medium', 'High'], n_samples, p=[0.25, 0.35, 0.3, 0.1])
    }

    df = pd.DataFrame(data)

    base_premium = 500
    age_effect = df['age'] ** 1.8 * 10
    smoker_effect = (df['smoker'] == 'Yes') * 15000
    bmi_effect = np.maximum(0, df['bmi'] - 25) * 300
    diabetes_effect = (df['diabetes'] == 'Yes') * 8000
    heart_effect = (df['heart_disease_history'] == 'Yes') * 12000
    blood_pressure_effect = (df['blood_pressure'] == 'High') * 6000 + (df['blood_pressure'] == 'Elevated') * 2500
    cholesterol_effect = np.maximum(0, df['cholesterol_level'] - 200) * 40
    occupation_effect = (df['occupation_risk'] == 'High') * 5000 + (df['occupation_risk'] == 'Medium') * 2000
    exercise_effect = - (df['exercise_frequency'] == 'High') * 3000 - (df['exercise_frequency'] == 'Medium') * 1500
    cancer_effect = (df['cancer_family_history'] == 'Yes') * 4000

    noise = np.random.normal(0, 1500, n_samples)
    df['annual_premium'] = np.round(base_premium + age_effect + smoker_effect + bmi_effect + diabetes_effect +
                                    heart_effect + blood_pressure_effect + cholesterol_effect +
                                    occupation_effect + exercise_effect + cancer_effect + noise).clip(1000)

    # Versión original para mostrar
    df_original = df.copy()

    # Preprocesamiento
    cat_cols = df.select_dtypes(include='object').columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('annual_premium', axis=1)
    y = df['annual_premium']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    estimator = DecisionTreeRegressor(max_depth=5)
    model = AdaBoostRegressor(estimator=estimator, n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)  # Para fácil acceso

    # SHAP con KernelExplainer (para AdaBoost)
    background = shap.kmeans(X_train, 50).data  # Fondo resumido
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test[:100])  # Subset para velocidad

    return df_original, model, scaler, encoders, X_test, y_test_df, y_pred, explainer, shap_values

df_original, model, scaler, encoders, X_test, y_test_df, y_pred, explainer, shap_values = load_data_and_train_model()

feature_names = df_original.drop('annual_premium', axis=1).columns

# =======================================================
# Menú lateral
# =======================================================
st.sidebar.title("Navegación")
section = st.sidebar.radio("Secciones", 
                           ["Datos Históricos", "EDA", "Predicciones", 
                            "Evaluación del Modelo", "Análisis SHAP", 
                            "Medición de Riesgo y Sensibilidad Robusta"])

# =======================================================
# Secciones
# =======================================================
if section == "Datos Históricos":
    st.header("📊 Datos Históricos")
    st.write(f"Dataset sintético con {df_original.shape[0]} asegurados")
    st.dataframe(df_original.head(200))
    st.download_button("Descargar CSV", df_original.to_csv(index=False), "datos_seguros_vida.csv")

elif section == "EDA":
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribución de Premios Anuales")
        fig, ax = plt.subplots()
        sns.histplot(df_original['annual_premium'], kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Fumador vs Premio")
        fig, ax = plt.subplots()
        sns.boxplot(x='smoker', y='annual_premium', data=df_original, ax=ax)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Edad vs Premio")
        fig, ax = plt.subplots()
        sns.scatterplot(x='age', y='annual_premium', data=df_original, alpha=0.5, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Riesgo Ocupacional vs Premio")
        fig, ax = plt.subplots()
        sns.boxplot(x='occupation_risk', y='annual_premium', data=df_original, ax=ax)
        st.pyplot(fig)

elif section == "Predicciones":
    st.header("🔮 Predicciones")
    st.write("Ingresa los datos para predecir el premio anual, riesgo y 'probabilidad' aproximada")

    with st.form("form_prediccion"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Edad", 18, 75, 40)
            gender = st.selectbox("Género", ['Male', 'Female'])
            bmi = st.number_input("BMI", 16.0, 45.0, 27.0)
            children = st.number_input("Hijos", 0, 5, 0)
            occupation_risk = st.selectbox("Riesgo Ocupacional", ['Low', 'Medium', 'High'])
            income_level = st.selectbox("Nivel de Ingresos", ['Low', 'Medium', 'High'])
            marital_status = st.selectbox("Estado Civil", ['Single', 'Married', 'Divorced', 'Widowed'])
            education_level = st.selectbox("Nivel Educativo", ['High School', 'Bachelor', 'Master', 'PhD'])
        with col2:
            region = st.selectbox("Región", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])
            smoker = st.selectbox("Fumador", ['Yes', 'No'])
            blood_pressure = st.selectbox("Presión Arterial", ['Normal', 'Elevated', 'High'])
            cholesterol_level = st.number_input("Colesterol (mg/dL)", 120, 350, 200)
            diabetes = st.selectbox("Diabetes", ['Yes', 'No'])
            heart_disease_history = st.selectbox("Historia Cardíaca", ['Yes', 'No'])
            cancer_family_history = st.selectbox("Historia Familiar Cáncer", ['Yes', 'No'])
            exercise_frequency = st.selectbox("Frecuencia Ejercicio", ['None', 'Low', 'Medium', 'High'])

        submitted = st.form_submit_button("Predecir Premio")

    if submitted:
        # Perfil del cliente
        profile_df = pd.DataFrame({
            'Edad': [age],
            'Género': [gender],
            'BMI': [bmi],
            'Hijos': [children],
            'Riesgo Ocupacional': [occupation_risk],
            'Nivel Ingresos': [income_level],
            'Estado Civil': [marital_status],
            'Nivel Educativo': [education_level],
            'Región': [region],
            'Fumador': [smoker],
            'Presión Arterial': [blood_pressure],
            'Colesterol (mg/dL)': [cholesterol_level],
            'Diabetes': [diabetes],
            'Historia Cardíaca': [heart_disease_history],
            'Historia Familiar Cáncer': [cancer_family_history],
            'Frecuencia Ejercicio': [exercise_frequency]
        }).T
        profile_df.columns = ['Valor']

        # Input para modelo
        input_data = pd.DataFrame({
            'age': [age], 'gender': [gender], 'bmi': [bmi], 'children': [children],
            'occupation_risk': [occupation_risk], 'income_level': [income_level],
            'marital_status': [marital_status], 'education_level': [education_level],
            'region': [region], 'smoker': [smoker], 'blood_pressure': [blood_pressure],
            'cholesterol_level': [cholesterol_level], 'diabetes': [diabetes],
            'heart_disease_history': [heart_disease_history], 'cancer_family_history': [cancer_family_history],
            'exercise_frequency': [exercise_frequency]
        })

        # Encoding y scaling
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
        input_scaled = scaler.transform(input_data)

        # Predicción
        premium = model.predict(input_scaled)[0]

        # Riesgo y probabilidad aproximada
        risk = "Bajo" if premium < 10000 else "Medio" if premium < 25000 else "Alto"
        prob_high_risk = (premium - y_pred.min()) / (y_pred.max() - y_pred.min())

        st.subheader("Resultado")
        col1, col2, col3 = st.columns(3)
        col1.metric("Premio Predicho ($)", f"{premium:,.0f}")
        col2.metric("Nivel de Riesgo", risk)
        col3.metric("'Probabilidad' Alto Riesgo (aprox.)", f"{prob_high_risk:.2%}")

        st.subheader("Perfil del Cliente")
        st.table(profile_df)

elif section == "Evaluación del Modelo":
    st.header("📈 Evaluación del Modelo")
    mae = mean_absolute_error(y_test_df['annual_premium'], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_df['annual_premium'], y_pred))
    r2 = r2_score(y_test_df['annual_premium'], y_test_df['annual_premium'])

    st.text(f"MAE: ${mae:,.2f}\nRMSE: ${rmse:,.2f}\nR²: {r2:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Predicciones vs Reales")
        fig, ax = plt.subplots()
        ax.scatter(y_test_df['annual_premium'], y_pred, alpha=0.5)
        ax.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'r--')
        st.pyplot(fig)

elif section == "Análisis SHAP":
    st.header("🧠 Análisis de Explicabilidad con SHAP")
    st.write("Importancia Global")
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

    st.write("Dirección del Impacto")
    shap.summary_plot(shap_values, X_test[:100], plot_type="violin", feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

elif section == "Medición de Riesgo y Sensibilidad Robusta":
    st.header("⚠️ Medición de Riesgo y Análisis de Sensibilidad Robusta")
    risks = ["Bajo" if p < 10000 else "Medio" if p < 25000 else "Alto" for p in y_pred]
    risk_counts = pd.Series(risks).value_counts()
    st.bar_chart(risk_counts)

    st.write("Distribución de 'Probabilidades' Aproximadas de Alto Riesgo")
    prob_high_risk = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    fig, ax = plt.subplots()
    sns.histplot(prob_high_risk, bins=50, kde=True, ax=ax)
    ax.set_xlabel("'Probabilidad' Alto Riesgo (normalizada)")
    st.pyplot(fig)

    st.write("Análisis de sensibilidad robusta: Ejemplo individual de alto riesgo")
    high_risk_indices = np.where(np.array(risks) == 'Alto')[0]
    if len(high_risk_indices) > 0:
        high_idx = high_risk_indices[0]
        st.write(f"Premio predicho: ${y_pred[high_idx]:,.0f} | Riesgo: Alto")
        shap.force_plot(explainer.expected_value, shap_values[high_idx], X_test[high_idx], feature_names=feature_names, matplotlib=True)
        st.pyplot(plt.gcf())
    else:
        st.write("No hay casos de alto riesgo en el set de prueba para mostrar ejemplo.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Creado por @NarenCastellon** | Especialización en Forecasting & IA 2026")

#este va con el note Prediccion_costo_premios de seguro de vida.ipynb