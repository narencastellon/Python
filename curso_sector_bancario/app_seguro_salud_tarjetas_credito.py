import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
import shap
import os

# Configuración de la página para estilo profesional (similar a Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Detección de Alto Uso Médico para Seguros de Salud", layout="wide", initial_sidebar_state="expanded")


# Cargar o generar datos (cacheado)
@st.cache_data
def load_or_generate_data():
    if os.path.exists('health_insurance_risk_data.csv'):
        df = pd.read_csv('health_insurance_risk_data.csv')
    else:
        # Generar si no existe (función del código anterior)
        df = generate_health_insurance_data(num_clients=500, num_days=200)
        df.to_csv('health_insurance_risk_data.csv', index=False)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

health_insurance_df = load_or_generate_data()

# Preprocesamiento y entrenamiento del modelo Random Forest (cacheado)
@st.cache_resource
def preprocess_and_train():
    # Codificar categóricas
    cat_cols = ['gender', 'smoking_status', 'occupation_risk_level', 'location_risk', 'policy_type_requested']
    le_dict = {col: LabelEncoder().fit(health_insurance_df[col]) for col in cat_cols}
    for col in cat_cols:
        health_insurance_df[col] = le_dict[col].transform(health_insurance_df[col])
    
    # Escalar numéricas
    num_cols = ['age', 'annual_income', 'credit_limit', 'card_usage_frequency', 'bmi', 'coverage_requested', 'health_visits_last_year']
    scaler = StandardScaler()
    health_insurance_df[num_cols] = scaler.fit_transform(health_insurance_df[num_cols])
    
    # Dividir train/test
    X = health_insurance_df.drop(columns=['client_id', 'ds', 'is_high_medical_usage'])
    y = health_insurance_df['is_high_medical_usage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Métricas
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    cm_matrix = confusion_matrix(y_test, y_pred)
    
    return rf_model, X.columns, scaler, le_dict, report, roc_auc, pr_auc, cm_matrix, X_train, X_test, y_test, y_prob

model, features, scaler, le_dict, report, roc_auc, pr_auc, cm, X_train, X_test, y_test, y_prob = preprocess_and_train()

# Función para predecir probabilidad de alto uso médico
def predict_high_usage(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Codificar categóricas usando encoders
    for col in le_dict:
        if col in input_df.columns:
            input_df[col] = le_dict[col].transform([input_df[col][0]])
    
    # Escalar numéricas
    num_cols = ['age', 'annual_income', 'credit_limit', 'card_usage_frequency', 'bmi', 'coverage_requested', 'health_visits_last_year']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Alinear features
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Predicción
    prob = model.predict_proba(input_df)[0][1]
    high_usage = 1 if prob > 0.5 else 0
    
    return high_usage, prob

# Función para medir riesgo basado en probabilidad
def measure_risk(prob):
    if prob > 0.7:
        return "Riesgo Alto", prob
    elif prob > 0.3:
        return "Riesgo Medio", prob
    else:
        return "Riesgo Bajo", prob

# SHAP explainer (cacheado)
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

# Sidebar para navegación (menú tipo Power BI)
with st.sidebar:
    #Mostrar imagen/banner en la parte superior
    st.image("./imagen/seguro_salud.png")
    st.sidebar.title("Menú")
    section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio (EDA)", "Predicciones", "Evaluación del Modelo", "Análisis con SHAP", "Medir Riesgo de Clientes", "Análisis de Sensibilidad"])


# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Clientes y Tarjetas")
    st.markdown("Visualización del dataset completo de clientes con tarjetas de crédito y riesgo médico.")
    
    #st.dataframe(health_insurance_df.style.background_gradient(cmap='viridis', subset=['annual_income', 'bmi']))
    st.dataframe(health_insurance_df)
    st.dataframe(X_test)

# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual del dataset para entender patrones en alto uso médico.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Alto Uso Médico")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x='is_high_medical_usage', data=health_insurance_df, palette='viridis')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("BMI por Alto Uso Médico")
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='is_high_medical_usage', y='bmi', data=health_insurance_df, palette='viridis')
        st.pyplot(fig_box)
    
    st.subheader("Matriz de Correlación (Variables Numéricas)")
    numeric_cols = health_insurance_df.select_dtypes(include=[np.number]).columns
    corr_matrix = health_insurance_df[numeric_cols].corr()
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig_corr)

# Sección 3: Predicciones
elif section == "Predicciones":
    st.title("Predicción de Alto Uso Médico")
    st.markdown("Ingrese los datos del cliente para predecir la probabilidad de alto uso médico y ofrecer seguro de salud vinculado a tarjeta.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad del Cliente", 18, 80, 40)
            gender = st.selectbox("Género", ['male', 'female'])
            annual_income = st.number_input("Ingreso Anual (USD)", min_value=0.0, value=70000.0)
            credit_limit = st.number_input("Límite de Crédito en Tarjeta (USD)", min_value=0.0, value=20000.0)
            card_usage_frequency = st.slider("Frecuencia de Uso de Tarjeta (trans/mes)", 0, 100, 30)
            bmi = st.number_input("BMI", min_value=18.0, max_value=40.0, value=28.0)
            smoking_status = st.selectbox("Fumador", ['yes', 'no'])
            has_chronic_disease = st.selectbox("Enfermedad Crónica", [0, 1])
        
        with col2:
            family_medical_history = st.selectbox("Historia Familiar Médica de Riesgo", [0, 1])
            occupation_risk_level = st.selectbox("Nivel de Riesgo Ocupacional", ['low', 'medium', 'high'])
            location_risk = st.selectbox("Riesgo por Ubicación", ['urban', 'rural'])
            policy_type_requested = st.selectbox("Tipo de Seguro Solicitado", ['basic', 'premium', 'family'])
            coverage_requested = st.number_input("Cobertura Solicitada (USD)", min_value=10000.0, value=50000.0)
            policy_term_months = st.selectbox("Plazo del Seguro (meses)", [12, 24, 36])
            previous_medical_claims = st.selectbox("Reclamos Médicos Previos", [0, 1])
            health_visits_last_year = st.slider("Visitas Médicas Último Año", 0, 20, 5)
        
        submit = st.form_submit_button("Predecir Riesgo")

    if submit:
        input_data = {
            'age': age,
            'gender': gender,
            'annual_income': annual_income,
            'credit_limit': credit_limit,
            'card_usage_frequency': card_usage_frequency,
            'bmi': bmi,
            'smoking_status': smoking_status,
            'has_chronic_disease': has_chronic_disease,
            'family_medical_history': family_medical_history,
            'occupation_risk_level': occupation_risk_level,
            'location_risk': location_risk,
            'policy_type_requested': policy_type_requested,
            'coverage_requested': coverage_requested,
            'policy_term_months': policy_term_months,
            'previous_medical_claims': previous_medical_claims,
            'health_visits_last_year': health_visits_last_year
        }
        
        high_usage, prob = predict_high_usage(input_data)
        risk, _ = measure_risk(prob)
        
        st.subheader("Resultados de Predicción")
        st.markdown(f"**Probabilidad de Alto Uso Médico:** {prob:.2%}")
        st.markdown(f"**Nivel de Riesgo:** {risk}")
        
        st.subheader("Perfil del Cliente y Seguro Solicitado")
        st.markdown(f"**Tipo de Seguro Solicitado:** {policy_type_requested}")
        st.markdown(f"**Plazo del Seguro:** {policy_term_months} meses")
        st.markdown(f"**Cobertura Solicitada:** ${coverage_requested:.2f}")

# Sección 4: Evaluación del Modelo
elif section == "Evaluación del Modelo":
    st.title("Evaluación del Modelo")
    st.markdown("Métricas y gráficos de rendimiento del modelo Random Forest para detección de alto uso médico.")
    
    st.subheader("Reporte de Clasificación")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='viridis'))
    
    st.subheader("ROC-AUC y PR-AUC")
    st.markdown(f"**ROC-AUC:** {roc_auc:.4f}")
    st.markdown(f"**PR-AUC:** {pr_auc:.4f}")
    
    st.subheader("Matriz de Confusión")
    fig_cm = plt.figure(figsize=(16, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig_cm)

# Sección 5: Análisis con SHAP
elif section == "Análisis con SHAP":
    st.title("Análisis con SHAP para Explicabilidad")
    st.markdown("SHAP explica la contribución de cada variable a las predicciones del modelo.")
    
    sample_size = st.slider("Tamaño de Muestra para SHAP", 100, 1000, 500)
    
    X_sample = X_test.sample(sample_size)
    shap_values = explainer.shap_values(X_sample)
    
    st.subheader("Resumen SHAP (Importancia Global - Clase Alto Uso)")
    fig_summary = plt.figure(figsize=(18, 8))
    shap.summary_plot(shap_values[1], X_sample, plot_type="bar")
    st.pyplot(fig_summary)
    
    st.subheader("Resumen Detallado SHAP")
    fig_detail = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[1], X_sample)
    st.pyplot(fig_detail)
    
    st.subheader("Force Plot para una Muestra Específica")
    sample_idx = st.slider("Selecciona Muestra", 0, sample_size - 1, 0)
    fig_force = plt.figure(figsize=(10, 4))
    shap.initjs()
    st.write(shap.force_plot(explainer.expected_value[1], shap_values[1][sample_idx], X_sample.iloc[sample_idx]))

# Sección 6: Medir Riesgo de Clientes
elif section == "Medir Riesgo de Clientes":
    st.title("Medir Riesgo de Clientes")
    st.markdown("Categoriza el riesgo de alto uso médico: Bajo (<30%), Medio (30-70%), Alto (>70%). Evalúa una muestra de clientes.")
    
    num_samples = st.slider("Número de Clientes a Evaluar", 10, 100, 50)
    X_risk_sample = X_test.sample(num_samples)
    y_risk_prob = model.predict_proba(X_risk_sample)[:, 1]
    
    risk_levels = []
    for prob in y_risk_prob:
        if prob < 0.3:
            risk_levels.append("Riesgo Bajo")
        elif prob < 0.7:
            risk_levels.append("Riesgo Medio")
        else:
            risk_levels.append("Riesgo Alto")
    
    risk_df = pd.DataFrame({
        'Cliente ID (Muestra)': range(1, num_samples + 1),
        'Probabilidad de Alto Uso Médico': y_risk_prob,
        'Nivel de Riesgo': risk_levels
    })
    
    st.dataframe(risk_df.style.background_gradient(cmap='viridis', subset=['Probabilidad de Alto Uso Médico']))
    
    st.subheader("Distribución de Riesgo")
    fig_risk = plt.figure(figsize=(18, 6))
    sns.countplot(y='Nivel de Riesgo', data=risk_df, palette='viridis')
    st.pyplot(fig_risk)

# Sección 7: Análisis de Sensibilidad
else:
    st.title("Análisis de Sensibilidad")
    st.markdown("Evalúa cómo cambia la predicción al variar features (±10% numéricas, cambio categórico). Muestra features más sensibles.")
    
    sample_size = st.slider("Tamaño de Muestra para Sensibilidad", 100, 500, 200)
    sensitivity_sample = X_test.sample(sample_size)
    base_pred = model.predict_proba(sensitivity_sample)[:, 1]
    
    sensitivity_results = []
    
    # Perturbación numérica
    num_features = ['age', 'annual_income', 'credit_limit', 'card_usage_frequency', 'bmi', 'coverage_requested', 'health_visits_last_year']
    for feat in num_features:
        delta_plus = sensitivity_sample.copy()
        delta_plus[feat] *= 1.1
        pred_plus = model.predict_proba(delta_plus)[:, 1]
        
        delta_minus = sensitivity_sample.copy()
        delta_minus[feat] *= 0.9
        pred_minus = model.predict_proba(delta_minus)[:, 1]
        
        avg_delta = np.mean(np.abs(pred_plus - base_pred) + np.abs(pred_minus - base_pred)) / 2
        sensitivity_results.append({'Feature': feat, 'Sensibilidad Promedio (Delta Prob)': avg_delta})
    
    # Perturbación categórica
    cat_features = ['gender', 'smoking_status', 'occupation_risk_level', 'location_risk', 'policy_type_requested']
    for feat in cat_features:
        perturbed = sensitivity_sample.copy()
        #unique_vals = health_insurance_df[feat].unique()
        unique_vals = X_train[feat].unique()
        perturbed[feat] = perturbed[feat].apply(lambda x: np.random.choice([v for v in unique_vals if v != x]))
        pred_perturbed = model.predict_proba(perturbed)[:, 1]
        
        avg_delta = np.mean(np.abs(pred_perturbed - base_pred))
        sensitivity_results.append({'Feature': feat, 'Sensibilidad Promedio (Delta Prob)': avg_delta})
    
    sensitivity_df = pd.DataFrame(sensitivity_results).sort_values(by='Sensibilidad Promedio (Delta Prob)', ascending=False)
    
    st.dataframe(sensitivity_df.style.background_gradient(cmap='viridis'))
    
    st.subheader("Gráfico de Sensibilidad")
    fig_sens = plt.figure(figsize=(18, 8))
    sns.barplot(x='Sensibilidad Promedio (Delta Prob)', y='Feature', data=sensitivity_df, palette='viridis')
    st.pyplot(fig_sens)

# Pie de página
st.markdown("---")
st.caption("App para detección de alto uso médico y oferta de seguros de salud vinculados a tarjeta de crédito. Datos sintéticos para demostración.")