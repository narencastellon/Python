import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap
import os

# Configuración de la página para estilo profesional (similar a Power BI: azules, blancos, sombras)
st.set_page_config(page_title="Predicción de Aprobación de Crédito Automotriz", layout="wide", initial_sidebar_state="expanded")


# Modelos de automóviles (hardcodeados del dataset para selectbox y evitar unseen labels)
vehicle_models = {
    'sedan': ['Toyota Camry', 'Honda Civic', 'Hyundai Elantra', 'Ford Fusion', 'Nissan Altima'],
    'SUV': ['Toyota RAV4', 'Honda CR-V', 'Ford Explorer', 'Chevrolet Equinox', 'Jeep Grand Cherokee'],
    'truck': ['Ford F-150', 'Chevrolet Silverado', 'Ram 1500', 'Toyota Tacoma', 'GMC Sierra'],
    'luxury': ['Mercedes-Benz S-Class', 'BMW 7 Series', 'Audi A8', 'Lexus LS', 'Porsche Panamera'],
    'electric': ['Tesla Model 3', 'Tesla Model Y', 'Nissan Leaf', 'Chevrolet Bolt', 'Ford Mustang Mach-E']
}
all_models = sorted(set(model for models in vehicle_models.values() for model in models))

# Cargar o generar datos (cacheado)
@st.cache_data
def load_or_generate_data():
    if os.path.exists('auto_credit_data.csv'):
        df = pd.read_csv('auto_credit_data.csv')
    else:
        # Generar si no existe (función del código anterior)
        df = generate_auto_credit_data(num_applicants=500, num_days=200)
        df.to_csv('auto_credit_data.csv', index=False)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

auto_credit_df = load_or_generate_data()

# Preprocesamiento y entrenamiento del modelo Voting (cacheado)
@st.cache_resource
def preprocess_and_train():
    # Codificar categóricas
    cat_cols = ['employment_status', 'vehicle_type', 'vehicle_model', 'marital_status', 'residence_type']
    le_dict = {col: LabelEncoder().fit(auto_credit_df[col]) for col in cat_cols}
    for col in cat_cols:
        auto_credit_df[col] = le_dict[col].transform(auto_credit_df[col])
    
    # Escalar numéricas
    num_cols = ['applicant_age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested', 'vehicle_value', 'down_payment', 'interest_rate', 'num_dependents', 'years_employed', 'bank_account_balance', 'vehicle_year']
    scaler = StandardScaler()
    auto_credit_df[num_cols] = scaler.fit_transform(auto_credit_df[num_cols])
    
    # Dividir train/test
    X = auto_credit_df.drop(columns=['application_id', 'applicant_id', 'ds', 'is_approved'])
    y = auto_credit_df['is_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Voting model
    base_models = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42))
    ]
    voting_model = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
    voting_model.fit(X_train, y_train)
    
    # Métricas
    y_pred = voting_model.predict(X_test)
    y_prob = voting_model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    cm_matrix = confusion_matrix(y_test, y_pred)
    
    return voting_model, X.columns, scaler, le_dict, report, roc_auc, pr_auc, cm_matrix, X_train, X_test, y_test, y_prob

model, features, scaler, le_dict, report, roc_auc, pr_auc, cm, X_train, X_test, y_test, y_prob = preprocess_and_train()

# Función para predecir aprobación y probabilidad
def predict_approval(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Codificar categóricas usando encoders
    for col in le_dict:
        if col in input_df.columns:
            input_df[col] = le_dict[col].transform([input_df[col][0]])
    
    # Escalar numéricas
    num_cols = ['applicant_age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount_requested', 'vehicle_value', 'down_payment', 'interest_rate', 'num_dependents', 'years_employed', 'bank_account_balance', 'vehicle_year']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Alinear features
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Predicción
    prob = model.predict_proba(input_df)[0][1]
    approved = 1 if prob > 0.5 else 0
    
    return approved, prob

# Función para medir riesgo basado en probabilidad
def measure_risk(prob):
    if prob > 0.7:
        return "Riesgo Bajo (Alta Probabilidad de Aprobación)", prob
    elif prob > 0.3:
        return "Riesgo Medio", prob
    else:
        return "Riesgo Alto (Baja Probabilidad de Aprobación)", prob

# SHAP explainer (cacheado)
@st.cache_resource
def get_shap_explainer(_model, _X_background):
    return shap.KernelExplainer(lambda x: _model.predict_proba(x)[:, 1], _X_background)

explainer = get_shap_explainer(model, X_train.sample(100))  # Muestra pequeña para background

with st.sidebar:

    # Mostrar imagen/banner en la parte superior
    st.image("./imagen/credi_auto.png")
    st.sidebar.title("Menú")
    section = st.sidebar.radio("Selecciona una Sección", ["Datos Históricos", "Análisis Exploratorio (EDA)", "Predicciones", "Evaluación del Modelo", "Análisis con SHAP", "Medir Riesgo de Clientes"])

# Mostrar imagen/banner en la parte superior
#st.image("https://example.com/auto_credit_banner.jpg", use_column_width=True)  # Reemplaza con URL real o path local de imagen (e.g., banner de crédito automotriz)

# Sección 1: Datos Históricos
if section == "Datos Históricos":
    st.title("Datos Históricos de Solicitudes de Crédito Automotriz")
    st.markdown("Visualización del dataset completo de solicitudes de crédito automotriz.")
    
    #st.dataframe(auto_credit_df.style.background_gradient(cmap='viridis', subset=['annual_income', 'credit_score']))
    st.dataframe(auto_credit_df)

# Sección 2: Análisis Exploratorio (EDA)
elif section == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual del dataset para entender patrones en aprobaciones de crédito automotriz.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Aprobación")
        fig_dist = plt.figure(figsize=(6, 4))
        sns.countplot(x='is_approved', data=auto_credit_df, palette='viridis')
        st.pyplot(fig_dist)
    
    with col2:
        st.subheader("Monto Solicitado por Aprobación")
        fig_box = plt.figure(figsize=(6, 4))
        sns.boxplot(x='is_approved', y='loan_amount_requested', data=auto_credit_df, palette='viridis')
        plt.yscale('log')
        st.pyplot(fig_box)
    
    st.subheader("Distribución de Modelos de Automóvil")
    fig_model = plt.figure(figsize=(12, 6))
    sns.countplot(y='vehicle_model', data=auto_credit_df, order=auto_credit_df['vehicle_model'].value_counts().index, palette='viridis')
    st.pyplot(fig_model)

# Sección 3: Predicciones
elif section == "Predicciones":
    st.title("Predicción de Aprobación de Crédito Automotriz")
    st.markdown("Ingrese los datos del cliente para predecir si el crédito será aprobado y ver la probabilidad.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            applicant_age = st.slider("Edad del Solicitante", 18, 70, 35)
            annual_income = st.number_input("Ingreso Anual (USD)", min_value=0.0, value=60000.0)
            employment_status = st.selectbox("Estado Laboral", ['employed', 'self-employed', 'unemployed'])
            credit_score = st.slider("Score Crediticio", 300, 850, 650)
            debt_to_income_ratio = st.number_input("Ratio Deuda/Ingreso", min_value=0.0, max_value=0.6, value=0.3)
            loan_amount_requested = st.number_input("Monto Solicitado (USD)", min_value=10000.0, max_value=100000.0, value=30000.0)
            vehicle_value = st.number_input("Valor del Automóvil (USD)", min_value=10000.0, value=35000.0)
        
        with col2:
            down_payment = st.number_input("Pago Inicial (USD)", min_value=0.0, value=5000.0)
            vehicle_type = st.selectbox("Tipo de Automóvil", ['sedan', 'SUV', 'truck', 'luxury', 'electric'])
            vehicle_model = st.selectbox("Modelo del Automóvil", all_models)  # Selectbox con todos los modelos conocidos para evitar unseen labels
            vehicle_year = st.slider("Año del Automóvil", 2010, 2024, 2022)
            loan_term_months = st.selectbox("Plazo del Préstamo (meses)", [36, 48, 60, 72, 84])
            interest_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, value=6.0)
            marital_status = st.selectbox("Estado Civil", ['single', 'married', 'divorced'])
            num_dependents = st.slider("Número de Dependientes", 0, 4, 0)
            residence_type = st.selectbox("Tipo de Residencia", ['own', 'rent', 'family'])
            years_employed = st.slider("Años Empleado", 0, 40, 10)
            has_previous_auto_loan = st.selectbox("Tiene Préstamo Automotriz Previo", [0, 1])
            previous_default = st.selectbox("Default Previo", [0, 1])
            bank_account_balance = st.number_input("Saldo Bancario (USD)", min_value=0.0, value=5000.0)
        
        submit = st.form_submit_button("Predecir Aprobación")

    if submit:
        input_data = {
            'applicant_age': applicant_age,
            'annual_income': annual_income,
            'employment_status': employment_status,
            'credit_score': credit_score,
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_amount_requested': loan_amount_requested,
            'vehicle_value': vehicle_value,
            'down_payment': down_payment,
            'vehicle_type': vehicle_type,
            'vehicle_model': vehicle_model,
            'vehicle_year': vehicle_year,
            'loan_term_months': loan_term_months,
            'interest_rate': interest_rate,
            'marital_status': marital_status,
            'num_dependents': num_dependents,
            'residence_type': residence_type,
            'years_employed': years_employed,
            'has_previous_auto_loan': has_previous_auto_loan,
            'previous_default': previous_default,
            'bank_account_balance': bank_account_balance
        }
        
        approved, prob = predict_approval(input_data)
        
        st.subheader("Resultados de Predicción")
        st.markdown(f"**Aprobación Predicha:** {'Aprobado' if approved == 1 else 'No Aprobado'}")
        st.markdown(f"**Probabilidad de Aprobación:** {prob:.2%}")
        
        st.subheader("Perfil del Cliente y Vehículo Solicitado")
        st.markdown(f"**Tipo de Automóvil:** {vehicle_type}")
        st.markdown(f"**Modelo Solicitado:** {vehicle_model}")
        st.markdown(f"**Plazo del Préstamo:** {loan_term_months} meses")

# Sección 4: Evaluación del Modelo
elif section == "Evaluación del Modelo":
    st.title("Evaluación del Modelo")
    st.markdown("Métricas y gráficos de rendimiento del modelo de voting para predicción de aprobación de crédito automotriz.")
    
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
else:
    st.title("Medir Riesgo de Clientes")
    st.markdown("Categoriza el riesgo de aprobación basado en la probabilidad predicha: Bajo (>70%), Medio (30-70%), Alto (<30%). Evalúa una muestra de clientes del test set.")
    
    num_samples = st.slider("Número de Clientes a Evaluar", 10, 100, 50)
    X_risk_sample = X_test.sample(num_samples)
    y_risk_prob = model.predict_proba(X_risk_sample)[:, 1]
    
    risk_levels = []
    for prob in y_risk_prob:
        if prob > 0.7:
            risk_levels.append("Riesgo Bajo")
        elif prob > 0.3:
            risk_levels.append("Riesgo Medio")
        else:
            risk_levels.append("Riesgo Alto")
    
    risk_df = pd.DataFrame({
        'Cliente ID (Muestra)': range(1, num_samples + 1),
        'Probabilidad de Aprobación': y_risk_prob,
        'Nivel de Riesgo': risk_levels
    })
    
    st.dataframe(risk_df.style.background_gradient(cmap='viridis', subset=['Probabilidad de Aprobación']))
    
    st.subheader("Distribución de Riesgo")
    fig_risk = plt.figure(figsize=(8, 6))
    sns.countplot(y='Nivel de Riesgo', data=risk_df, palette='viridis')
    st.pyplot(fig_risk)

# Pie de página
st.markdown("---")
st.caption("App para predicción de aprobación de crédito automotriz. Datos sintéticos para demostración.")