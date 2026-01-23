# =======================================================
# App Streamlit CORREGIDA: Detección de Fraude en Reclamos de Seguros con XGBoost
# Corrección clave: explainer y shap_values calculados GLOBALMENTE después del entrenamiento
# Esto evita NameError al acceder desde cualquier sección
# Todo funciona sin errores
# Ejecuta con: streamlit run app_fraude_seguros.py
# =======================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Detección de Fraude en Seguros - Naren Castellón", layout="wide")
st.title("🛡️ Detección de Fraude en Reclamos de Seguros")
st.markdown("**Modelo XGBoost + SHAP para explicabilidad** | Dictado por Naren Castellón")

# Imagen header
st.image("https://images.unsplash.com/photo-1554224155-6726b3b3ff858f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", 
         caption="Análisis inteligente de reclamos de seguros con IA", use_column_width=True)

# =======================================================
# Carga de datos y entrenamiento (cacheado)
# =======================================================
@st.cache_resource
def load_data_and_train_model():
    np.random.seed(42)
    n_samples = 10000

    data = {
        'age': np.random.randint(18, 85, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        'policy_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_samples),
        'policy_amount': np.round(np.random.uniform(5000, 100000, n_samples), 2),
        'claim_amount': np.round(np.random.uniform(1000, 80000, n_samples), 2),
        'incident_type': np.random.choice(['Collision', 'Theft', 'Natural', 'Other'], n_samples),
        'days_to_report': np.random.randint(0, 60, n_samples),
        'previous_claims': np.random.randint(0, 6, n_samples),
        'vehicle_age': np.random.randint(0, 25, n_samples),
        'vehicle_value': np.round(np.random.uniform(5000, 120000, n_samples), 2),
        'driver_experience_years': np.random.randint(0, 50, n_samples),
        'location_risk': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2]),
        'police_report': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'num_witnesses': np.random.randint(0, 6, n_samples),
        'injury_severity': np.random.choice(['None', 'Minor', 'Moderate', 'Severe'], n_samples),
    }

    df = pd.DataFrame(data)

    fraud_prob = (
        0.02 + 
        0.15 * (df['claim_amount'] > df['policy_amount'] * 0.8) +
        0.10 * (df['days_to_report'] > 15) +
        0.20 * (df['police_report'] == 0) +
        0.08 * (df['num_witnesses'] == 0) +
        0.12 * (df['location_risk'] == 'High')
    )
    fraud_prob = fraud_prob.clip(0, 0.95)
    df['fraud'] = np.random.binomial(1, fraud_prob)

    # Guardar versión original (para mostrar datos y EDA)
    df_original = df.copy()

    # Preprocesamiento
    cat_cols = df.select_dtypes(include='object').columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('fraud', axis=1)
    y = df['fraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='auc', scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )
    model.fit(X_train_res, y_train_res)

    # Predicciones en test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # SHAP global (calculado una vez para todo X_test)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return (df_original, model, scaler, encoders, X_test, y_test, y_pred, y_prob,
            explainer, shap_values)

(df_original, model, scaler, encoders, X_test, y_test, y_pred, y_prob,
 explainer, shap_values) = load_data_and_train_model()

feature_names = df_original.drop('fraud', axis=1).columns

# =======================================================
# Menú lateral
# =======================================================
st.sidebar.title("Navegación")
section = st.sidebar.radio("Secciones", 
                           ["Datos Históricos", "EDA", "Predicciones", 
                            "Evaluación del Modelo", "Análisis SHAP", 
                            "Medición de Riesgo y Sensibilidad"])

# =======================================================
# Secciones
# =======================================================
if section == "Datos Históricos":
    st.header("📊 Datos Históricos")
    st.write(f"Dataset sintético con {df_original.shape[0]} reclamos")
    st.dataframe(df_original.head(200))
    st.download_button("Descargar CSV", df_original.to_csv(index=False), "datos_fraude_seguros.csv")

elif section == "EDA":
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribución de Fraude")
        fig, ax = plt.subplots()
        sns.countplot(x='fraud', data=df_original.replace({'fraud': {0: 'Legítimo', 1: 'Fraudulento'}}), ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Reporte Policial vs Fraude")
        fig, ax = plt.subplots()
        sns.countplot(x='police_report', hue='fraud', data=df_original, ax=ax)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Monto Reclamado vs Fraude")
        fig, ax = plt.subplots()
        sns.boxplot(x='fraud', y='claim_amount', data=df_original, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Días para Reportar vs Fraude")
        fig, ax = plt.subplots()
        sns.boxplot(x='fraud', y='days_to_report', data=df_original, ax=ax)
        st.pyplot(fig)

elif section == "Predicciones":
    st.header("🔮 Predicciones")
    st.write("Ingresa los datos del reclamo para obtener predicción, probabilidad y riesgo")

    with st.form("form_prediccion"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Edad", 18, 85, 40)
            gender = st.selectbox("Género", ['Male', 'Female'])
            income_level = st.selectbox("Nivel de Ingresos", ['Low', 'Medium', 'High'])
            policy_type = st.selectbox("Tipo de Póliza", ['Basic', 'Standard', 'Premium'])
            policy_amount = st.number_input("Monto Póliza ($)", 5000.0, 100000.0, 30000.0)
            claim_amount = st.number_input("Monto Reclamado ($)", 1000.0, 80000.0, 15000.0)
            incident_type = st.selectbox("Tipo de Incidente", ['Collision', 'Theft', 'Natural', 'Other'])
            days_to_report = st.number_input("Días para Reportar", 0, 60, 5)
        with col2:
            previous_claims = st.number_input("Reclamos Previos", 0, 6, 0)
            vehicle_age = st.number_input("Edad Vehículo", 0, 25, 5)
            vehicle_value = st.number_input("Valor Vehículo ($)", 5000.0, 120000.0, 30000.0)
            driver_experience_years = st.number_input("Años Experiencia Conductor", 0, 50, 10)
            location_risk = st.selectbox("Riesgo Ubicación", ['Low', 'Medium', 'High'])
            police_report = st.selectbox("Reporte Policial", ['No', 'Sí'])
            num_witnesses = st.number_input("Número Testigos", 0, 6, 1)
            injury_severity = st.selectbox("Severidad Lesión", ['None', 'Minor', 'Moderate', 'Severe'])

        submitted = st.form_submit_button("Predecir Fraude")

    if submitted:
        # DataFrame con valores originales (para mostrar perfil)
        profile_df = pd.DataFrame({
            'Edad': [age],
            'Género': [gender],
            'Nivel Ingresos': [income_level],
            'Tipo Póliza': [policy_type],
            'Monto Póliza ($)': [policy_amount],
            'Monto Reclamado ($)': [claim_amount],
            'Tipo Incidente': [incident_type],
            'Días para Reportar': [days_to_report],
            'Reclamos Previos': [previous_claims],
            'Edad Vehículo': [vehicle_age],
            'Valor Vehículo ($)': [vehicle_value],
            'Experiencia Conductor (años)': [driver_experience_years],
            'Riesgo Ubicación': [location_risk],
            'Reporte Policial': [police_report],
            'Número Testigos': [num_witnesses],
            'Severidad Lesión': [injury_severity]
        }).T
        profile_df.columns = ['Valor']

        # Input para modelo
        input_data = pd.DataFrame({
            'age': [age], 'gender': [gender], 'income_level': [income_level],
            'policy_type': [policy_type], 'policy_amount': [policy_amount],
            'claim_amount': [claim_amount], 'incident_type': [incident_type],
            'days_to_report': [days_to_report], 'previous_claims': [previous_claims],
            'vehicle_age': [vehicle_age], 'vehicle_value': [vehicle_value],
            'driver_experience_years': [driver_experience_years],
            'location_risk': [location_risk], 'police_report': [1 if police_report == 'Sí' else 0],
            'num_witnesses': [num_witnesses], 'injury_severity': [injury_severity]
        })

        # Encoding y scaling
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
        input_scaled = scaler.transform(input_data)

        # Predicción
        prob = model.predict_proba(input_scaled)[0][1]
        pred = "Fraude" if prob >= 0.5 else "Legítimo"

        # Nivel de riesgo
        if prob < 0.3:
            risk = "Bajo"
        elif prob < 0.7:
            risk = "Medio"
        else:
            risk = "Alto"

        st.subheader("Resultado")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilidad de Fraude", f"{prob:.2%}")
        col2.metric("Predicción", pred)
        col3.metric("Nivel de Riesgo", risk)

        st.subheader("Perfil del Cliente")
        st.table(profile_df)

elif section == "Evaluación del Modelo":
    st.header("📈 Evaluación del Modelo")
    st.text(classification_report(y_test, y_pred))

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label='ROC')
        ax.plot([0,1], [0,1], 'k--')
        st.pyplot(fig)
    with col2:
        st.write(f"Average Precision: {average_precision_score(y_test, y_prob):.4f}")
        fig, ax = plt.subplots()
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(recall, precision)
        st.pyplot(fig)

    st.write("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

elif section == "Análisis SHAP":
    st.header("🧠 Análisis de Explicabilidad con SHAP")
    st.write("Importancia Global")
    shap.summary_plot(shap_values[:500], X_test[:500], feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

    st.write("Dirección del Impacto")
    shap.summary_plot(shap_values[:500], X_test[:500], plot_type="violin", feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

elif section == "Medición de Riesgo y Sensibilidad":
    st.header("⚠️ Medición de Riesgo y Análisis de Sensibilidad Robusta")
    risks = ["Bajo" if p < 0.3 else "Medio" if p < 0.7 else "Alto" for p in y_prob]
    risk_counts = pd.Series(risks).value_counts()
    st.bar_chart(risk_counts)

    st.write("Distribución de probabilidades de fraude en datos de prueba")
    fig, ax = plt.subplots()
    sns.histplot(y_prob, bins=50, kde=True, ax=ax)
    ax.set_xlabel("Probabilidad de Fraude")
    st.pyplot(fig)

    st.write("Análisis de sensibilidad robusta: Ejemplo individual de un caso fraudulento")
    fraud_indices = np.where(y_pred == 1)[0]
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]  # Primer caso fraudulento predicho
        st.write(f"Probabilidad de fraude: {y_prob[fraud_idx]:.2%} | Riesgo: {'Alto' if y_prob[fraud_idx] >= 0.7 else 'Medio' if y_prob[fraud_idx] >= 0.3 else 'Bajo'}")
        shap.force_plot(explainer.expected_value, shap_values[fraud_idx], 
                        X_test[fraud_idx], feature_names=feature_names, matplotlib=True)
        st.pyplot(plt.gcf())
    else:
        st.write("No hay casos predichos como fraude en el set de prueba para mostrar ejemplo.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Creado por @NarenCastellon** | Especialización en Forecasting & IA 2026")