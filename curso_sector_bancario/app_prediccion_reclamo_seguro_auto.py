# =======================================================
# App Streamlit: Predicción de Probabilidad de Reclamo en Seguros de Auto con Naive Bayes
# Incluye: Imagen header, menú lateral, datos históricos, EDA, predicciones (perfil cliente, predicción, probabilidad, riesgo),
# evaluación del modelo, análisis SHAP y medición de riesgo + sensibilidad robusta
# Ejecuta con: streamlit run app_reclamos_auto.py
# =======================================================
# Requisitos: pip install streamlit pandas numpy scikit-learn shap seaborn matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score)
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Predicción de Reclamos en Seguros de Auto - Naren Castellón", layout="wide")
st.title("🚗 Predicción de Probabilidad de Reclamo en Seguros de Auto")
st.markdown("**Modelo Naive Bayes + SHAP para explicabilidad** | Dictado por Naren Castellón")

# Imagen header (URL pública relevante a seguros de auto)
st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", 
         caption="Análisis predictivo de reclamos en seguros de auto con IA")

# =======================================================
# Carga de datos y entrenamiento (cacheado)
# =======================================================
@st.cache_resource
def load_data_and_train_model():
    np.random.seed(42)
    n_samples = 10000

    data = {
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        'residence_type': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'vehicle_age': np.random.randint(0, 20, n_samples),
        'vehicle_type': np.random.choice(['Sedan', 'SUV', 'Truck', 'Sports Car'], n_samples),
        'annual_miles_driven': np.random.normal(12000, 5000, n_samples).clip(1000, 30000).astype(int),
        'driving_experience_years': np.random.randint(0, 60, n_samples),
        'traffic_violations': np.random.randint(0, 5, n_samples),
        'previous_accidents': np.random.randint(0, 3, n_samples),
        'alcohol_consumption': np.random.choice(['None', 'Low', 'Moderate', 'High'], n_samples),
        'health_condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
        'credit_score': np.random.normal(700, 100, n_samples).clip(300, 850).astype(int),
        'driving_habits': np.random.choice(['Safe', 'Average', 'Risky'], n_samples, p=[0.5, 0.3, 0.2])
    }

    df = pd.DataFrame(data)

    claim_prob = (
        0.05 + 
        0.20 * (df['age'] < 25) + 0.10 * (df['age'] > 65) + 
        0.08 * (df['gender'] == 'Male') + 
        0.15 * (df['traffic_violations'] > 1) + 
        0.25 * (df['previous_accidents'] > 0) + 
        0.12 * (df['alcohol_consumption'] == 'High') + 0.06 * (df['alcohol_consumption'] == 'Moderate') + 
        0.10 * (df['driving_habits'] == 'Risky') + 
        0.08 * (df['health_condition'] == 'Poor') + 0.04 * (df['health_condition'] == 'Fair') + 
        0.07 * (df['annual_miles_driven'] > 20000) + 
        0.05 * (df['vehicle_type'] == 'Sports Car')
    )
    claim_prob = claim_prob.clip(0, 0.95)
    df['claim'] = np.random.binomial(1, claim_prob)

    # Versión original para mostrar
    df_original = df.copy()

    # Preprocesamiento
    cat_cols = df.select_dtypes(include='object').columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('claim', axis=1)
    y = df['claim']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)

    # SHAP con KernelExplainer
    def model_predict(data):
        return model.predict_proba(data)[:, 1]

    background = shap.kmeans(X_train, 50).data
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_test[:100])  # Subset para velocidad

    return df_original, model, scaler, encoders, X_test, y_test_df, y_pred, y_prob, explainer, shap_values

df_original, model, scaler, encoders, X_test, y_test_df, y_pred, y_prob, explainer, shap_values = load_data_and_train_model()

feature_names = df_original.drop('claim', axis=1).columns

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
    st.write(f"Dataset sintético con {df_original.shape[0]} conductores")
    st.dataframe(df_original.head(200))
    st.download_button("Descargar CSV", df_original.to_csv(index=False), "datos_reclamos_auto.csv")

elif section == "EDA":
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribución de Reclamos")
        fig, ax = plt.subplots()
        sns.countplot(x='claim', data=df_original, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Hábitos de Conducción vs Reclamo")
        fig, ax = plt.subplots()
        sns.countplot(x='driving_habits', hue='claim', data=df_original, ax=ax)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Edad vs Reclamo")
        fig, ax = plt.subplots()
        sns.boxplot(x='claim', y='age', data=df_original, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Consumo Alcohol vs Reclamo")
        fig, ax = plt.subplots()
        sns.countplot(x='alcohol_consumption', hue='claim', data=df_original, ax=ax)
        st.pyplot(fig)

elif section == "Predicciones":
    st.header("🔮 Predicciones")
    st.write("Ingresa los datos para predecir reclamo, probabilidad y riesgo")

    with st.form("form_prediccion"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Edad", 18, 80, 40)
            gender = st.selectbox("Género", ['Male', 'Female'])
            marital_status = st.selectbox("Estado Civil", ['Single', 'Married', 'Divorced'])
            education_level = st.selectbox("Nivel Educativo", ['High School', 'Bachelor', 'Master', 'PhD'])
            income_level = st.selectbox("Nivel de Ingresos", ['Low', 'Medium', 'High'])
            residence_type = st.selectbox("Tipo Residencia", ['Urban', 'Suburban', 'Rural'])
            vehicle_age = st.number_input("Edad Vehículo", 0, 20, 5)
            vehicle_type = st.selectbox("Tipo Vehículo", ['Sedan', 'SUV', 'Truck', 'Sports Car'])
        with col2:
            annual_miles_driven = st.number_input("Millas Anuales", 1000, 30000, 12000)
            driving_experience_years = st.number_input("Años Experiencia", 0, 60, 10)
            traffic_violations = st.number_input("Violaciones Tráfico", 0, 5, 0)
            previous_accidents = st.number_input("Accidentes Previos", 0, 3, 0)
            alcohol_consumption = st.selectbox("Consumo Alcohol", ['None', 'Low', 'Moderate', 'High'])
            health_condition = st.selectbox("Condición Salud", ['Excellent', 'Good', 'Fair', 'Poor'])
            credit_score = st.number_input("Puntaje Crediticio", 300, 850, 700)
            driving_habits = st.selectbox("Hábitos Conducción", ['Safe', 'Average', 'Risky'])

        submitted = st.form_submit_button("Predecir Reclamo")

    if submitted:
        # Perfil del cliente
        profile_df = pd.DataFrame({
            'Edad': [age],
            'Género': [gender],
            'Estado Civil': [marital_status],
            'Nivel Educativo': [education_level],
            'Nivel de Ingresos': [income_level],
            'Tipo Residencia': [residence_type],
            'Edad Vehículo': [vehicle_age],
            'Tipo Vehículo': [vehicle_type],
            'Millas Anuales': [annual_miles_driven],
            'Años Experiencia': [driving_experience_years],
            'Violaciones Tráfico': [traffic_violations],
            'Accidentes Previos': [previous_accidents],
            'Consumo Alcohol': [alcohol_consumption],
            'Condición Salud': [health_condition],
            'Puntaje Crediticio': [credit_score],
            'Hábitos Conducción': [driving_habits]
        }).T
        profile_df.columns = ['Valor']

        # Input para modelo
        input_data = pd.DataFrame({
            'age': [age], 'gender': [gender], 'marital_status': [marital_status],
            'education_level': [education_level], 'income_level': [income_level],
            'residence_type': [residence_type], 'vehicle_age': [vehicle_age],
            'vehicle_type': [vehicle_type], 'annual_miles_driven': [annual_miles_driven],
            'driving_experience_years': [driving_experience_years], 'traffic_violations': [traffic_violations],
            'previous_accidents': [previous_accidents], 'alcohol_consumption': [alcohol_consumption],
            'health_condition': [health_condition], 'credit_score': [credit_score],
            'driving_habits': [driving_habits]
        })

        # Encoding y scaling
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
        input_scaled = scaler.transform(input_data)

        # Predicción
        prob = model.predict_proba(input_scaled)[0][1]
        pred = "Reclamo" if prob >= 0.5 else "No Reclamo"

        # Riesgo
        risk = "Bajo" if prob < 0.3 else "Medio" if prob < 0.7 else "Alto"

        st.subheader("Resultado")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilidad de Reclamo", f"{prob:.2%}")
        col2.metric("Predicción", pred)
        col3.metric("Nivel de Riesgo", risk)

        st.subheader("Perfil del Cliente")
        st.table(profile_df)

elif section == "Evaluación del Modelo":
    st.header("📈 Evaluación del Modelo")
    st.text(classification_report(y_test_df['claim'], y_pred))

    st.write(f"AUC-ROC: {roc_auc_score(y_test_df['claim'], y_prob):.4f}")
    st.write(f"Average Precision: {average_precision_score(y_test_df['claim'], y_prob):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Curva ROC")
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(y_test_df['claim'], y_prob)
        ax.plot(fpr, tpr, label='ROC')
        ax.plot([0,1], [0,1], 'k--')
        st.pyplot(fig)
    with col2:
        st.write("Curva Precision-Recall")
        fig, ax = plt.subplots()
        precision, recall, _ = precision_recall_curve(y_test_df['claim'], y_prob)
        ax.plot(recall, precision)
        st.pyplot(fig)

    st.write("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test_df['claim'], y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

elif section == "Análisis SHAP":
    st.header("🧠 Análisis de Explicabilidad con SHAP")
    st.write("Importancia Global para Reclamo")
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

    st.write("Dirección del Impacto para Reclamo")
    shap.summary_plot(shap_values, X_test[:100], plot_type="violin", feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

elif section == "Medición de Riesgo y Sensibilidad Robusta":
    st.header("⚠️ Medición de Riesgo y Análisis de Sensibilidad Robusta")
    risks = ["Bajo" if p < 0.3 else "Medio" if p < 0.7 else "Alto" for p in y_prob]
    risk_counts = pd.Series(risks).value_counts()
    st.bar_chart(risk_counts)

    st.write("Distribución de Probabilidades de Reclamo")
    fig, ax = plt.subplots()
    sns.histplot(y_prob, bins=50, kde=True, ax=ax)
    ax.set_xlabel("Probabilidad de Reclamo")
    st.pyplot(fig)

    st.write("Análisis de sensibilidad robusta: Ejemplo individual de alto riesgo")
    high_risk_indices = np.where(np.array(risks) == 'Alto')[0]
    if len(high_risk_indices) > 0:
        high_idx = high_risk_indices[0]
        st.write(f"Probabilidad de reclamo: {y_prob[high_idx]:.2%} | Riesgo: Alto")
        shap.force_plot(explainer.expected_value, shap_values[high_idx], X_test[high_idx], feature_names=feature_names, matplotlib=True)
        st.pyplot(plt.gcf())
    else:
        st.write("No hay casos de alto riesgo en el set de prueba para mostrar ejemplo.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Creado por @NarenCastellon** | Especialización en Forecasting & IA 2026")