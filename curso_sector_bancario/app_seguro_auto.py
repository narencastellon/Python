import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
import shap
import os

# Configuración de la página para estilo profesional (similar a Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Predicción de Riesgo en Préstamos de Autos", layout="wide", initial_sidebar_state="expanded")


# Cargar o generar datos (cacheado)
@st.cache_data
def load_or_generate_data():
    if os.path.exists('auto_loan_risk_data.csv'):
        df = pd.read_csv('auto_loan_risk_data.csv')
    else:
        # Generar si no existe (función del código anterior)
        df = generate_auto_loan_risk_data(num_clients=500, num_days=200)
        df.to_csv('auto_loan_risk_data.csv', index=False)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

auto_loan_df = load_or_generate_data()

# Preprocesamiento y entrenamiento del modelo XGBoost (cacheado)
@st.cache_resource
def preprocess_and_train():
    # Codificar categóricas
    cat_cols = ['gender', 'vehicle_type', 'location_risk', 'insurance_coverage_level']
    le_dict = {col: LabelEncoder().fit(auto_loan_df[col]) for col in cat_cols}
    for col in cat_cols:
        auto_loan_df[col] = le_dict[col].transform(auto_loan_df[col])
    
    # Escalar numéricas
    num_cols = ['age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount', 'vehicle_value', 'down_payment', 'interest_rate', 'annual_mileage', 'vehicle_year']
    scaler = StandardScaler()
    auto_loan_df[num_cols] = scaler.fit_transform(auto_loan_df[num_cols])
    
    # Dividir train/test
    X = auto_loan_df.drop(columns=['client_id', 'ds', 'is_high_risk'])
    y = auto_loan_df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Calcular scale_pos_weight para desbalance
    pos_weight = (y == 0).sum() / (y == 1).sum()
    
    # Modelo XGBoost
    gb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1
    )
    gb_model.fit(X_train, y_train)
    
    # Métricas
    y_pred = gb_model.predict(X_test)
    y_prob = gb_model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    cm_matrix = confusion_matrix(y_test, y_pred)
    
    return gb_model, X.columns, scaler, le_dict, report, roc_auc, pr_auc, cm_matrix, X_train, X_test, y_test, y_prob

model, features, scaler, le_dict, report, roc_auc, pr_auc, cm, X_train, X_test, y_test, y_prob = preprocess_and_train()

# Función para predecir probabilidad de alto riesgo
def predict_high_risk(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Codificar categóricas usando encoders
    for col in le_dict:
        if col in input_df.columns:
            input_df[col] = le_dict[col].transform([input_df[col][0]])
    
    # Escalar numéricas
    num_cols = ['age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount', 'vehicle_value', 'down_payment', 'interest_rate', 'annual_mileage', 'vehicle_year']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Alinear features
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Predicción
    prob = model.predict_proba(input_df)[0][1]
    high_risk = 1 if prob > 0.5 else 0
    
    return high_risk, prob

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
    st.image("./imagen/seguro_auto.png")
    st.sidebar.title("Menú")
    section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio (EDA)", "Predicciones", "Evaluación del Modelo", "Análisis con SHAP", "Medir Riesgo de Clientes", "Análisis de Sensibilidad Robusta"])


# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Préstamos de Autos")
    st.markdown("Visualización del dataset completo de clientes con préstamos vehiculares.")
    
    #st.dataframe(auto_loan_df.style.background_gradient(cmap='viridis', subset=['annual_income', 'credit_score']))
    st.dataframe(auto_loan_df)

# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual del dataset para entender patrones en riesgo de siniestros/incumplimiento.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Alto Riesgo")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x='is_high_risk', data=auto_loan_df, palette='viridis')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("Edad por Alto Riesgo")
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='is_high_risk', y='age', data=auto_loan_df, palette='viridis')
        st.pyplot(fig_box)
    
    st.subheader("Matriz de Correlación (Variables Numéricas)")
    numeric_cols = auto_loan_df.select_dtypes(include=[np.number]).columns
    corr_matrix = auto_loan_df[numeric_cols].corr()
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig_corr)

# Sección 3: Predicciones
elif section == "Predicciones":
    st.title("Predicción de Riesgo en Préstamos de Autos")
    st.markdown("Ingrese los datos del cliente para predecir la probabilidad de siniestro o incumplimiento.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad del Cliente", 18, 70, 35)
            gender = st.selectbox("Género", ['male', 'female'])
            annual_income = st.number_input("Ingreso Anual (USD)", min_value=0.0, value=60000.0)
            credit_score = st.slider("Score Crediticio", 300, 850, 650)
            debt_to_income_ratio = st.number_input("Ratio Deuda/Ingreso", min_value=0.0, max_value=0.6, value=0.35)
            loan_amount = st.number_input("Monto del Préstamo (USD)", min_value=10000.0, value=30000.0)
            vehicle_value = st.number_input("Valor del Vehículo (USD)", min_value=10000.0, value=35000.0)
            down_payment = st.number_input("Pago Inicial (USD)", min_value=0.0, value=5000.0)
        
        with col2:
            vehicle_type = st.selectbox("Tipo de Vehículo", ['sedan', 'SUV', 'truck', 'luxury', 'sports'])
            vehicle_year = st.slider("Año del Vehículo", 2010, 2024, 2020)
            annual_mileage = st.number_input("Kilometraje Anual (km)", min_value=0.0, value=15000.0)
            driving_history_accidents = st.slider("Accidentes Últimos 5 Años", 0, 5, 1)
            location_risk = st.selectbox("Riesgo por Ubicación", ['urban_high', 'urban_low', 'rural'])
            loan_term_months = st.selectbox("Plazo del Préstamo (meses)", [36, 48, 60, 72, 84])
            interest_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, value=7.5)
            payment_history_late = st.selectbox("Pagos Atrasados Previos", [0, 1])
            insurance_coverage_level = st.selectbox("Nivel de Cobertura", ['basic', 'comprehensive'])
        
        submit = st.form_submit_button("Predecir Riesgo")

    if submit:
        input_data = {
            'age': age,
            'gender': gender,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_amount': loan_amount,
            'vehicle_value': vehicle_value,
            'down_payment': down_payment,
            'vehicle_type': vehicle_type,
            'vehicle_year': vehicle_year,
            'annual_mileage': annual_mileage,
            'driving_history_accidents': driving_history_accidents,
            'location_risk': location_risk,
            'loan_term_months': loan_term_months,
            'interest_rate': interest_rate,
            'payment_history_late': payment_history_late,
            'insurance_coverage_level': insurance_coverage_level
        }
        
        high_risk, prob = predict_high_risk(input_data)
        risk, _ = measure_risk(prob)
        
        st.subheader("Resultados de Predicción")
        st.markdown(f"**Probabilidad de Siniestro o Incumplimiento:** {prob:.2%}")
        st.markdown(f"**Nivel de Riesgo:** {risk}")
        
        st.subheader("Perfil del Cliente y Préstamo")
        st.markdown(f"**Tipo de Vehículo:** {vehicle_type}")
        st.markdown(f"**Año del Vehículo:** {vehicle_year}")
        st.markdown(f"**Plazo del Préstamo:** {loan_term_months} meses")
        st.markdown(f"**Monto del Préstamo:** ${loan_amount:.2f}")

# Sección 4: Evaluación del Modelo
elif section == "Evaluación del Modelo":
    st.title("Evaluación del Modelo")
    st.markdown("Métricas y gráficos de rendimiento del modelo Gradient Boosting (XGBoost).")
    
    st.subheader("Reporte de Clasificación")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='viridis'))
    
    st.subheader("ROC-AUC y PR-AUC")
    st.markdown(f"**ROC-AUC:** {roc_auc:.4f}")
    st.markdown(f"**PR-AUC:** {pr_auc:.4f}")
    
    st.subheader("Matriz de Confusión")
    fig_cm = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig_cm)

# Sección 5: Análisis con SHAP
elif section == "Análisis con SHAP":
    st.title("Análisis con SHAP para Explicabilidad")
    st.markdown("SHAP explica la contribución de cada variable a las predicciones del modelo.")
    
    sample_size = st.slider("Tamaño de Muestra para SHAP", 100, 1000, 500)
    
    X_sample = X_test.sample(sample_size)
    shap_values = explainer.shap_values(X_sample)
    
    st.subheader("Resumen SHAP (Importancia Global)")
    fig_summary = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    st.pyplot(fig_summary)
    
    st.subheader("Resumen Detallado SHAP")
    fig_detail = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample)
    st.pyplot(fig_detail)
    
    st.subheader("Force Plot para una Muestra Específica")
    sample_idx = st.slider("Selecciona Muestra", 0, sample_size - 1, 0)
    fig_force = plt.figure(figsize=(10, 4))
    shap.initjs()
    st.write(shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_sample.iloc[sample_idx]))

# Sección 6: Medir Riesgo de Clientes
elif section == "Medir Riesgo de Clientes":
    st.title("Medir Riesgo de Clientes")
    st.markdown("Categoriza el riesgo de siniestro/incumplimiento: Bajo (<30%), Medio (30-70%), Alto (>70%). Evalúa una muestra de clientes.")
    
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
        'Probabilidad de Siniestro/Incumplimiento': y_risk_prob,
        'Nivel de Riesgo': risk_levels
    })
    
    st.dataframe(risk_df.style.background_gradient(cmap='viridis', subset=['Probabilidad de Siniestro/Incumplimiento']))
    
    st.subheader("Distribución de Riesgo")
    fig_risk = plt.figure(figsize=(8, 6))
    sns.countplot(y='Nivel de Riesgo', data=risk_df, palette='viridis')
    st.pyplot(fig_risk)

# Sección 7: Análisis de Sensibilidad Robusta
else:
    st.title("Análisis de Sensibilidad Robusta")
    st.markdown("Evalúa variabilidad en predicciones con perturbaciones Monte Carlo (±20% en features clave).")
    
    sample_size = st.slider("Tamaño de Muestra para Sensibilidad", 100, 300, 200)
    sensitivity_sample = X_test.sample(sample_size)
    base_pred = model.predict_proba(sensitivity_sample)[:, 1]
    
    num_perturbations = 100
    perturbed_preds = np.zeros((len(sensitivity_sample), num_perturbations))
    
    key_features = ['age', 'credit_score', 'debt_to_income_ratio', 'annual_mileage', 'driving_history_accidents', 'vehicle_year']
    
    for p in range(num_perturbations):
        perturbed = sensitivity_sample.copy()
        for feat in key_features:
            perturbation = np.random.uniform(0.8, 1.2)  # ±20%
            perturbed[feat] *= perturbation
        perturbed_preds[:, p] = model.predict_proba(perturbed)[:, 1]
    
    variance_per_sample = np.var(perturbed_preds, axis=1)
    max_delta_per_sample = np.max(perturbed_preds, axis=1) - np.min(perturbed_preds, axis=1)
    avg_variance = np.mean(variance_per_sample)
    avg_max_delta = np.mean(max_delta_per_sample)
    
    st.markdown(f"**Varianza Promedio en Probabilidad:** {avg_variance:.4f}")
    st.markdown(f"**Delta Máximo Promedio en Probabilidad:** {avg_max_delta:.4f}")
    
    st.subheader("Distribución de Varianza")
    fig_var = plt.figure(figsize=(10, 6))
    plt.hist(variance_per_sample, bins=30, color='#0078d4')
    st.pyplot(fig_var)
    
    st.subheader("Probabilidad Base vs Delta Máximo")
    fig_delta = plt.figure(figsize=(10, 6))
    plt.scatter(base_pred, max_delta_per_sample, alpha=0.6, color='#0078d4')
    plt.xlabel('Probabilidad Base')
    plt.ylabel('Delta Máximo')
    st.pyplot(fig_delta)

# Pie de página
st.markdown("---")
st.caption("App para predicción de riesgo en préstamos de autos y seguros vinculados. Datos sintéticos para demostración.")