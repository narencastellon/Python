# =======================================================
# App Streamlit: Credit Scoring Integrado con Riesgo de Seguros usando Stacked Ensemble
# Predice 'default' (incumplimiento crédito) y 'claim' (siniestro seguro)
# Incluye: Imagen header, menú lateral, datos históricos, EDA, predicciones (perfil cliente, preds default + claim, riesgo integrado),
# evaluación del modelo, análisis SHAP y medición de riesgo + sensibilidad robusta
# Ejecuta con: streamlit run app_credit_scoring_seguros.py
# =======================================================
# Requisitos: pip install streamlit pandas numpy scikit-learn xgboost shap seaborn matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score)
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Credit Scoring Integrado con Riesgo de Seguros - Naren Castellón", layout="wide")
st.title("💳 Credit Scoring Integrado con Riesgo de Seguros")
st.markdown("**Modelo Stacked Ensemble para predecir default y siniestros** | Dictado por Naren Castellón")

# Imagen header (URL pública relevante a finanzas/seguros)
st.image("https://images.unsplash.com/photo-1554224154-22dec7ec8818?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", 
         caption="Análisis integrado de crédito y riesgos de seguros con IA", )

# =======================================================
# Carga de datos y entrenamiento (cacheado)
# =======================================================
@st.cache_resource
def load_data_and_train_model():
    np.random.seed(42)
    n_samples = 10000

    data = {
        'age': np.random.randint(18, 75, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'income': np.random.normal(60000, 20000, n_samples).clip(20000, 150000).astype(int),
        'debt_to_income_ratio': np.round(np.random.uniform(0.1, 0.6, n_samples), 2),
        'credit_score': np.random.normal(700, 100, n_samples).clip(300, 850).astype(int),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
        'smoker': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'bmi': np.round(np.random.normal(27, 5, n_samples), 1).clip(15, 45),
        'blood_pressure_category': np.random.choice(['Normal', 'Elevated', 'High'], n_samples),
        'diabetes': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'heart_disease': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9]),
        'exercise_frequency': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'alcohol_consumption': np.random.choice(['Low', 'Moderate', 'High'], n_samples),
        'driving_record': np.random.choice(['Clean', 'Minor Violations', 'Major Violations'], n_samples)
    }

    df = pd.DataFrame(data)

    default_prob = (
        0.05 + 
        0.15 * (df['debt_to_income_ratio'] > 0.4) + 
        0.20 * (df['credit_score'] < 600) + 
        0.10 * (df['employment_status'] == 'Unemployed') + 
        0.08 * (df['dependents'] > 3) + 
        0.05 * (df['age'] < 30)
    )
    default_prob = default_prob.clip(0, 0.95)
    df['default'] = np.random.binomial(1, default_prob)

    claim_prob = (
        0.10 + 
        0.15 * (df['smoker'] == 'Yes') + 
        0.10 * (df['bmi'] > 30) + 
        0.12 * (df['diabetes'] == 'Yes') + 
        0.18 * (df['heart_disease'] == 'Yes') + 
        0.08 * (df['blood_pressure_category'] == 'High') + 
        0.05 * (df['alcohol_consumption'] == 'High') + 
        0.15 * (df['driving_record'] == 'Major Violations') + 
        0.07 * (df['driving_record'] == 'Minor Violations') + 
        0.05 * (df['age'] > 65)
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

    X = df.drop(['default', 'claim'], axis=1)
    y = df[['default', 'claim']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def build_stacked_classifier():
        base_estimators = [
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42))
        ]
        meta = XGBClassifier(random_state=42)
        return StackingClassifier(estimators=base_estimators, final_estimator=meta, cv=3)

    model_default = build_stacked_classifier()
    model_default.fit(X_train, y_train['default'])

    model_claim = build_stacked_classifier()
    model_claim.fit(X_train, y_train['claim'])

    y_pred_default = model_default.predict(X_test)
    y_prob_default = model_default.predict_proba(X_test)[:, 1]

    y_pred_claim = model_claim.predict(X_test)
    y_prob_claim = model_claim.predict_proba(X_test)[:, 1]

    y_prob_integrated = (y_prob_default + y_prob_claim) / 2
    y_test_df = y_test.reset_index(drop=True)

    # SHAP con KernelExplainer para default (como ejemplo)
    background = shap.kmeans(X_train, 50).data
    def predict_default(data):
        return model_default.predict_proba(data)[:, 1]
    explainer = shap.KernelExplainer(predict_default, background)
    shap_values = explainer.shap_values(X_test[:100])

    return df_original, model_default, model_claim, scaler, encoders, X_test, y_test_df, y_pred_default, y_pred_claim, y_prob_default, y_prob_claim, y_prob_integrated, explainer, shap_values

(df_original, model_default, model_claim, scaler, encoders, X_test, y_test_df, y_pred_default, y_pred_claim, y_prob_default, y_prob_claim, y_prob_integrated, explainer, shap_values) = load_data_and_train_model()

feature_names = df_original.drop(['default', 'claim'], axis=1).columns

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
    st.write(f"Dataset sintético con {df_original.shape[0]} clientes")
    st.dataframe(df_original.head(200))
    st.download_button("Descargar CSV", df_original.to_csv(index=False), "datos_credit_scoring_seguros.csv")

elif section == "EDA":
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribución de Default")
        fig, ax = plt.subplots()
        sns.countplot(x='default', data=df_original, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Distribución de Siniestros (Claim)")
        fig, ax = plt.subplots()
        sns.countplot(x='claim', data=df_original, ax=ax)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Credit Score vs Default")
        fig, ax = plt.subplots()
        sns.boxplot(x='default', y='credit_score', data=df_original, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("BMI vs Claim")
        fig, ax = plt.subplots()
        sns.boxplot(x='claim', y='bmi', data=df_original, ax=ax)
        st.pyplot(fig)

elif section == "Predicciones":
    st.header("🔮 Predicciones")
    st.write("Ingresa los datos para predecir default, siniestro, riesgo integrado y probabilidades")

    with st.form("form_prediccion"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Edad", 18, 75, 40)
            gender = st.selectbox("Género", ['Male', 'Female'])
            income = st.number_input("Ingreso Anual ($)", 20000, 150000, 60000)
            debt_to_income_ratio = st.number_input("Ratio Deuda/Ingreso", 0.1, 0.6, 0.3)
            credit_score = st.number_input("Puntaje Crediticio", 300, 850, 700)
            employment_status = st.selectbox("Estatus Empleo", ['Employed', 'Unemployed', 'Self-Employed'])
            marital_status = st.selectbox("Estado Civil", ['Single', 'Married', 'Divorced'])
            dependents = st.number_input("Dependientes", 0, 5, 0)
        with col2:
            smoker = st.selectbox("Fumador", ['Yes', 'No'])
            bmi = st.number_input("BMI", 15.0, 45.0, 27.0)
            blood_pressure_category = st.selectbox("Categoría Presión Arterial", ['Normal', 'Elevated', 'High'])
            diabetes = st.selectbox("Diabetes", ['Yes', 'No'])
            heart_disease = st.selectbox("Enfermedad Cardíaca", ['Yes', 'No'])
            exercise_frequency = st.selectbox("Frecuencia Ejercicio", ['Low', 'Medium', 'High'])
            alcohol_consumption = st.selectbox("Consumo Alcohol", ['Low', 'Moderate', 'High'])
            driving_record = st.selectbox("Registro Conducción", ['Clean', 'Minor Violations', 'Major Violations'])

        submitted = st.form_submit_button("Predecir Riesgos")

    if submitted:
        # Perfil del cliente
        profile_df = pd.DataFrame({
            'Edad': [age],
            'Género': [gender],
            'Ingreso Anual ($)': [income],
            'Ratio Deuda/Ingreso': [debt_to_income_ratio],
            'Puntaje Crediticio': [credit_score],
            'Estatus Empleo': [employment_status],
            'Estado Civil': [marital_status],
            'Dependientes': [dependents],
            'Fumador': [smoker],
            'BMI': [bmi],
            'Categoría Presión Arterial': [blood_pressure_category],
            'Diabetes': [diabetes],
            'Enfermedad Cardíaca': [heart_disease],
            'Frecuencia Ejercicio': [exercise_frequency],
            'Consumo Alcohol': [alcohol_consumption],
            'Registro Conducción': [driving_record]
        }).T
        profile_df.columns = ['Valor']

        # Input para modelo
        input_data = pd.DataFrame({
            'age': [age], 'gender': [gender], 'income': [income], 'debt_to_income_ratio': [debt_to_income_ratio],
            'credit_score': [credit_score], 'employment_status': [employment_status],
            'marital_status': [marital_status], 'dependents': [dependents],
            'smoker': [smoker], 'bmi': [bmi], 'blood_pressure_category': [blood_pressure_category],
            'diabetes': [diabetes], 'heart_disease': [heart_disease], 'exercise_frequency': [exercise_frequency],
            'alcohol_consumption': [alcohol_consumption], 'driving_record': [driving_record]
        })

        # Encoding y scaling
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
        input_scaled = scaler.transform(input_data)

        # Predicciones
        prob_default = model_default.predict_proba(input_scaled)[0][1]
        pred_default = "Default" if prob_default >= 0.5 else "No Default"

        prob_claim = model_claim.predict_proba(input_scaled)[0][1]
        pred_claim = "Siniestro" if prob_claim >= 0.5 else "No Siniestro"

        prob_integrated = (prob_default + prob_claim) / 2
        risk = "Bajo" if prob_integrated < 0.3 else "Medio" if prob_integrated < 0.7 else "Alto"

        st.subheader("Resultado")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilidad Default", f"{prob_default:.2%}", delta=pred_default)
        col2.metric("Probabilidad Siniestro", f"{prob_claim:.2%}", delta=pred_claim)
        col3.metric("Riesgo Integrado", risk, delta=f"{prob_integrated:.2%}")

        st.subheader("Perfil del Cliente")
        st.table(profile_df)

elif section == "Evaluación del Modelo":
    st.header("📈 Evaluación del Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Classification Report para Default")
        st.text(classification_report(y_test_df['default'], y_pred_default))
        st.write(f"AUC-ROC Default: {roc_auc_score(y_test_df['default'], y_prob_default):.4f}")
        st.write(f"AP Default: {average_precision_score(y_test_df['default'], y_prob_default):.4f}")
    with col2:
        st.write("Classification Report para Siniestro (Claim)")
        st.text(classification_report(y_test_df['claim'], y_pred_claim))
        st.write(f"AUC-ROC Claim: {roc_auc_score(y_test_df['claim'], y_prob_claim):.4f}")
        st.write(f"AP Claim: {average_precision_score(y_test_df['claim'], y_prob_claim):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Matriz de Confusión para Default")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test_df['default'], y_pred_default), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Matriz de Confusión para Siniestro")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test_df['claim'], y_pred_claim), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    st.write("Curvas ROC")
    fig, ax = plt.subplots()
    fpr_d, tpr_d, _ = roc_curve(y_test_df['default'], y_prob_default)
    ax.plot(fpr_d, tpr_d, label=f'Default AUC={roc_auc_score(y_test_df["default"], y_prob_default):.2f}')
    fpr_c, tpr_c, _ = roc_curve(y_test_df['claim'], y_prob_claim)
    ax.plot(fpr_c, tpr_c, label=f'Claim AUC={roc_auc_score(y_test_df["claim"], y_prob_claim):.2f}')
    ax.plot([0,1], [0,1], 'k--')
    ax.legend()
    st.pyplot(fig)

elif section == "Análisis SHAP":
    st.header("🧠 Análisis de Explicabilidad con SHAP")
    st.write("Importancia Global para Default")
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

    st.write("Dirección del Impacto para Default")
    shap.summary_plot(shap_values, X_test[:100], plot_type="violin", feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())

elif section == "Medición de Riesgo y Sensibilidad Robusta":
    st.header("⚠️ Medición de Riesgo y Análisis de Sensibilidad Robusta")
    risks = ["Bajo" if p < 0.3 else "Medio" if p < 0.7 else "Alto" for p in y_prob_integrated]
    risk_counts = pd.Series(risks).value_counts()
    st.bar_chart(risk_counts)

    st.write("Distribución de Probabilidades Integradas de Riesgo")
    fig, ax = plt.subplots()
    sns.histplot(y_prob_integrated, bins=50, kde=True, ax=ax)
    ax.set_xlabel("Probabilidad Integrada de Riesgo")
    st.pyplot(fig)

    st.write("Análisis de sensibilidad robusta: Ejemplo individual de alto riesgo")
    high_risk_indices = np.where(np.array(risks) == 'Alto')[0]
    if len(high_risk_indices) > 0:
        high_idx = high_risk_indices[0]
        st.write(f"Probabilidad integrada: {y_prob_integrated[high_idx]:.2%} | Riesgo: Alto")
        shap.force_plot(explainer.expected_value, shap_values[high_idx], X_test[high_idx], feature_names=feature_names, matplotlib=True)
        st.pyplot(plt.gcf())
    else:
        st.write("No hay casos de alto riesgo en el set de prueba para mostrar ejemplo.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Creado por @NarenCastellon** | Especialización en Forecasting & IA 2026")